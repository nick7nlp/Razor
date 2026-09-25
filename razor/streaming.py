"""Layer-resident native forwards and checkpoint-faithful scoring decoders.

Only the active decoder layer is materialized. Native model forwards are
suspended at layer boundaries, preserving their own masks, rotary embeddings,
indexer state, hyper-connections and latent attention residuals.
"""
from __future__ import annotations

import copy
import json
import queue
import re
import threading
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from .loader import (_auto_model, _require_complete_checkpoint, _scoring_config,
                     load_config, quantization_config, resolve_checkpoint)


_FP8 = tuple(getattr(torch, name) for name in
             ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
             if hasattr(torch, name))


def _e8m0(scale):
    allowed = (torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8))
    if scale.dtype not in allowed:
        raise ValueError("MXFP scales must be E8M0 bytes, not numeric float scales")
    raw = scale.view(torch.uint8)
    if bool((raw == 255).any()):
        raise ValueError("invalid E8M0 NaN scale")
    return torch.ldexp(torch.ones_like(raw, dtype=torch.float32), raw.int() - 127)


def unpack_mxfp4(packed, scale, dtype=torch.bfloat16):
    """Decode two E2M1 codes per byte, low nibble first, in groups of 32."""
    if packed.dtype not in (torch.uint8, torch.int8) or packed.ndim < 2:
        raise ValueError("MXFP4 weights require a packed uint8/int8 matrix")
    width = packed.shape[-1] * 2
    if (scale.shape[:-1] != packed.shape[:-1] or width % 32 or
            scale.shape[-1] != width // 32):
        raise ValueError("MXFP4 scale grid must contain one E8M0 scale per 32 codes")
    raw = packed.view(torch.uint8)
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                          -0., -.5, -1., -1.5, -2., -3., -4., -6.],
                         device=packed.device, dtype=torch.float32)
    result = torch.empty((*packed.shape[:-1], width), device=packed.device,
                         dtype=torch.float32)
    result[..., 0::2] = table[(raw & 15).long()]
    result[..., 1::2] = table[(raw >> 4).long()]
    result *= _e8m0(scale).repeat_interleave(32, dim=-1)
    return result.to(dtype)


def dequantize_fp8(weight, scale, *, block_size=(128, 128), dtype=torch.bfloat16):
    """Decode FP8 block scales; float scales and E8M0 storage are distinct."""
    if weight.dtype not in _FP8 or weight.ndim < 2:
        raise ValueError("FP8 block decoding requires floating FP8 weights")
    if len(block_size) != 2 or any(int(value) < 1 for value in block_size):
        raise ValueError("FP8 weight_block_size must contain two positive integers")
    expected = (*weight.shape[:-2],
                (weight.shape[-2] + block_size[0] - 1) // block_size[0],
                (weight.shape[-1] + block_size[1] - 1) // block_size[1])
    if tuple(scale.shape) != expected:
        raise ValueError(f"FP8 scale shape {tuple(scale.shape)} != {expected}")
    if scale.dtype in (torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8)):
        numeric = _e8m0(scale)
    elif scale.is_floating_point() and scale.dtype not in _FP8:
        numeric = scale.float()
    else:
        raise ValueError("unsupported FP8 scale storage dtype")
    if not bool(torch.isfinite(numeric).all()) or bool((numeric <= 0).any()):
        raise ValueError("FP8 scales must be positive and finite")
    expanded = numeric.repeat_interleave(block_size[0], -2).repeat_interleave(block_size[1], -1)
    return (weight.float() * expanded[..., :weight.shape[-2], :weight.shape[-1]]).to(dtype)


def _natural(key):
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", key))


def _assign(model, name, value):
    owner, _, leaf = name.rpartition(".")
    module = model.get_submodule(owner) if owner else model
    if leaf in module._parameters:
        module._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
    elif leaf in module._buffers:
        module._buffers[leaf] = value
    else:
        raise KeyError(f"{name} is not a parameter or buffer")


class CheckpointWeights:
    """Lazy safetensors reads, indexed by official native conversion targets."""
    def __init__(self, root, model, *, dtype=torch.bfloat16):
        _require_complete_checkpoint(root)
        from safetensors import safe_open
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key
        self.root = Path(root)
        self.dtype, self.model = dtype, model
        self.quant = copy.deepcopy(getattr(model, "_razor_source_quantization", None))
        if self.quant is None:
            self.quant = quantization_config(model.config)
        self.slots = dict(model.state_dict())
        aliases = defaultdict(list)
        for name, parameter in model.named_parameters(remove_duplicate=False):
            aliases[id(parameter)].append(name)
        self.aliases = {name: group for group in aliases.values() for name in group if len(group) > 1}
        index = self.root / "model.safetensors.index.json"
        if index.is_file():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            single = self.root / "model.safetensors"
            if not single.is_file():
                raise FileNotFoundError("streaming requires local safetensors weights and config")
            with safe_open(str(single), framework="pt") as handle:
                self.weight_map = {name: single.name for name in handle.keys()}
        for shard in set(self.weight_map.values()):
            reference = Path(shard)
            if reference.is_absolute() or ".." in reference.parts:
                raise ValueError("checkpoint index references a shard outside its directory")
        mappings = get_model_conversion_mapping(model)
        renamings = [rule for rule in mappings if isinstance(rule, WeightRenaming)]
        converters = [rule for rule in mappings if isinstance(rule, WeightConverter)]
        by_pattern = {pattern: rule for rule in converters for pattern in rule.source_patterns}
        self.groups, self.rules, self.scales = defaultdict(list), {}, {}
        scale_keys = set()
        for source in self.weight_map:
            candidates = []
            if source.endswith(".weight_packed"):
                candidates.append(source[:-len("weight_packed")] + "weight_scale")
            if source.endswith(".weight"):
                candidates += [source + "_scale_inv", source[:-len("weight")] + "scale"]
            found = [key for key in candidates if key in self.weight_map]
            if len(found) > 1:
                raise ValueError(f"ambiguous quantization scales for {source}")
            if found:
                self.scales[source] = found[0]
                scale_keys.add(found[0])
        self.unmapped = []
        for source in sorted(self.weight_map, key=_natural):
            if source in scale_keys:
                continue
            canonical = source[:-len("weight_packed")] + "weight" if source.endswith(".weight_packed") else source
            target, pattern = rename_source_key(canonical, renamings, converters,
                                                 model.base_model_prefix, self.slots)
            if target not in self.slots and canonical in self.slots:
                target, pattern = canonical, None
            if target not in self.slots:
                self.unmapped.append(source)
                continue
            self.groups[target].append((source, pattern))
            if pattern is not None:
                self.rules[target] = by_pattern[pattern]
        self.read_count = defaultdict(int)

    def raw(self, name, device="cpu"):
        from safetensors import safe_open
        with safe_open(str(self.root / self.weight_map[name]), framework="pt", device=str(device)) as handle:
            self.read_count[name] += 1
            value = handle.get_tensor(name)
            return value.clone() if value.device.type == "cpu" else value

    def tensor(self, name, device, *, dtype=None):
        dtype = self.dtype if dtype is None else dtype
        value = self.raw(name, device)
        scale_name = self.scales.get(name)
        if scale_name is not None:
            scale = self.raw(scale_name, device)
            if value.dtype in (torch.int8, torch.uint8):
                return unpack_mxfp4(value, scale, dtype)
            if value.dtype in _FP8:
                return dequantize_fp8(value, scale,
                                     block_size=tuple(self.quant.get("weight_block_size") or (128, 128)),
                                     dtype=dtype)
            raise ValueError(f"scale attached to unsupported tensor dtype: {name} ({value.dtype})")
        if (value.dtype in _FP8 or name.endswith(".weight_packed") or
                (value.dtype in (torch.int8, torch.uint8) and name.endswith(".weight"))):
            raise ValueError(f"quantized tensor lacks its scale: {name}")
        return value.to(dtype) if value.is_floating_point() else value

    def _merged(self, target, entries, rule, device):
        from transformers.core_model_loading import Concatenate, MergeModulelist
        ops = rule.operations
        if not (len(ops) in (1, 2) and isinstance(ops[0], MergeModulelist) and ops[0].dim == 0):
            return None
        if len(ops) == 2 and not (isinstance(ops[1], Concatenate) and ops[1].dim == 1):
            return None
        patterns = rule.source_patterns
        grouped = {pattern: [source for source, match in entries if match == pattern] for pattern in patterns}
        expected = self.slots[target].shape
        if any(len(values) != expected[0] for values in grouped.values()):
            raise ValueError(f"{target}: incomplete expert bank")
        for values in grouped.values():
            ids = [int(re.search(r"\.experts\.(\d+)\.", source).group(1))
                   for source in values if re.search(r"\.experts\.(\d+)\.", source)]
            if ids and ids != list(range(expected[0])):
                raise ValueError(f"{target}: expert ids are not contiguous from zero")
        result = None
        offset = 0
        for pattern in patterns:
            for expert, source in enumerate(grouped[pattern]):
                value = self.tensor(source, device, dtype=self.slots[target].dtype)
                if result is None:
                    result = torch.empty(expected, device=device, dtype=value.dtype)
                if value.dtype != result.dtype:
                    raise ValueError(f"{target}: fusion source dtypes disagree")
                destination = result[expert] if len(ops) == 1 else result[expert, offset:offset + value.shape[0]]
                if destination.shape != value.shape:
                    raise ValueError(f"{source}: converted expert shape disagrees with native module")
                destination.copy_(value)
                width = value.shape[0]
                del value
            offset += width
        if len(ops) == 2 and offset != expected[1]:
            raise ValueError(f"{target}: incomplete concatenation")
        return {target: result}

    def converted(self, target, device):
        entries = self.groups[target]
        rule = self.rules.get(target)
        dtype = self.slots[target].dtype
        if rule is None:
            if len(entries) != 1:
                raise ValueError(f"multiple unconverted sources for {target}")
            return {target: self.tensor(entries[0][0], device, dtype=dtype)}
        merged = self._merged(target, entries, rule, device)
        if merged is not None:
            return merged
        conversion = copy.deepcopy(rule)
        for source, pattern in entries:
            conversion.add_tensor(target, source, pattern,
                                  lambda source=source: self.tensor(source, device, dtype=dtype))
        return conversion.convert(target, model=self.model, config=self.model.config)

    def _tied(self, name, device):
        """Resolve a tied slot from a resident alias or from the alias source itself.

        Materialization proceeds one prefix at a time, so an alias group can be
        split across calls. A resident alias is reused as before, which keeps
        tied storage shared; otherwise the checkpoint entry that the official
        conversion mapped to the alias is read, so resolution no longer depends
        on the order in which prefixes are requested.
        """
        aliases = self.aliases.get(name, ())
        for alias in aliases:
            value = self.model.get_parameter(alias)
            if not value.is_meta:
                return value.to(device)
        expected = self.slots[name]
        for alias in aliases:
            if alias == name or alias not in self.groups:
                continue
            value = self.converted(alias, device).get(alias)
            if isinstance(value, list):
                if len(value) != 1:
                    raise ValueError(f"official conversion returned multiple tensors for {alias}")
                value = value[0]
            if value is None:
                continue
            if value.shape != expected.shape:
                raise ValueError(f"{name}: tied source {alias} shape {tuple(value.shape)} "
                                 f"!= native {tuple(expected.shape)}")
            return value.to(expected.dtype) if value.is_floating_point() else value
        return None

    def materialize(self, prefix, device, *, exclude=(), allow_missing=()):
        loaded = set()
        wanted = {name for name in self.slots if name.startswith(prefix)
                  and not any(name.startswith(item) for item in exclude)}
        try:
            for target in self.groups:
                if target not in wanted:
                    continue
                for name, value in self.converted(target, device).items():
                    if isinstance(value, list):
                        if len(value) != 1:
                            raise ValueError(f"official conversion returned multiple tensors for {name}")
                        value = value[0]
                    if name not in wanted:
                        raise ValueError(f"conversion escaped requested module: {name}")
                    expected = self.slots[name]
                    if value.shape != expected.shape:
                        owner = self.model.get_submodule(name.rpartition(".")[0])
                        kimi_alog = (name.endswith(".A_log") and type(owner).__name__ == "KimiDeltaAttention"
                                     and value.ndim == 1 and value.shape[0] == getattr(owner, "head_k_dim", None)
                                     and expected.shape == (getattr(owner, "num_heads", None),))
                        if kimi_alog:
                            self.slots[name] = value.to(device="meta", dtype=expected.dtype)
                        else:
                            raise ValueError(f"{name}: checkpoint shape {tuple(value.shape)} != native {tuple(expected.shape)}")
                    if value.is_floating_point():
                        value = value.to(expected.dtype)
                    _assign(self.model, name, value)
                    loaded.add(name)
            for name in wanted - loaded:
                value = self._tied(name, device)
                if value is not None:
                    _assign(self.model, name, value)
                    loaded.add(name)
            missing = wanted - loaded
            bad = [name for name in missing if not any(re.search(pattern, name) for pattern in allow_missing)]
            if bad:
                raise RuntimeError(f"checkpoint is missing native weights: {sorted(bad)[:8]}")
            for name in missing:
                _assign(self.model, name, torch.zeros_like(self.slots[name], device=device))
                loaded.add(name)
            return loaded
        except BaseException:
            self.release(loaded)
            raise

    def release(self, names):
        for name in names:
            _assign(self.model, name, torch.empty_like(self.slots[name], device="meta"))


def _skeleton(model_path, config, dtype, trust_remote_code, **kwargs):
    _require_complete_checkpoint(model_path)
    from accelerate import init_empty_weights
    quant = quantization_config(config)
    clean = _scoring_config(config)
    clean.name_or_path = str(model_path)
    with init_empty_weights(include_buffers=False):
        model = _auto_model("from_config", config=clean, dtype=dtype,
                            trust_remote_code=trust_remote_code, **kwargs).eval()
    model.tie_weights()
    model._razor_source_quantization = copy.deepcopy(quant)
    return model


def _decoder(model):
    choices = []
    for name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and layers:
            if any(part in name.split(".") for part in ("visual", "vision_model", "vision_tower", "mtp")):
                continue
            config = getattr(module, "config", None)
            depth = getattr(config, "num_hidden_layers", None)
            if depth is not None and len(layers) >= depth:
                choices.append((name, module, list(layers[:depth])))
    if len(choices) != 1:
        raise ValueError(f"cannot unambiguously locate native text decoder: {[name for name, _, _ in choices]}")
    return choices[0]


class _StopForward(BaseException):
    pass


class _NativeWorker:
    def __init__(self, model, batch, layers, device, callback, index):
        self.events = queue.Queue()
        self.resume = threading.Event()
        self.cancel = threading.Event()
        self.index, self.output = index, None
        self.callback = callback
        self.layer_count = len(layers)
        self.thread = threading.Thread(target=self._run, args=(model, batch, device), daemon=True)

    def _run(self, model, batch, device):
        try:
            with torch.no_grad():
                if torch.device(device).type == "cuda":
                    torch.cuda.set_device(device)
                arguments = {name: value.to(device) for name, value in batch.items()
                             if name not in ("loss_mask", "labels")}
                model(**arguments, use_cache=False, return_dict=True)
            self.events.put(("error", RuntimeError("native forward bypassed the final decoder boundary")))
        except _StopForward:
            self.events.put(("finished", None))
        except BaseException as error:
            self.events.put(("error", error))

    def before(self, layer, args, kwargs, layer_index):
        self.events.put(("layer", layer_index))
        self.resume.wait()
        self.resume.clear()
        if self.cancel.is_set():
            raise _StopForward()
        if self.callback is not None:
            self.callback("input", layer_index, self.index, args, kwargs)

    def after(self, layer, args, kwargs, output, layer_index):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if torch.is_tensor(hidden) and not bool(torch.isfinite(hidden).all()):
            raise RuntimeError(f"native layer {layer_index}, batch {self.index}: non-finite hidden states")
        if self.callback is not None:
            self.callback("output", layer_index, self.index, output, kwargs)
        if layer_index == self.layer_count - 1:
            self.output = output
            raise _StopForward()

    def wait(self, expected):
        event, value = self.events.get()
        if event == "error":
            raise value
        if event != expected[0] or value != expected[1]:
            raise RuntimeError(f"native decoder order changed: {(event, value)} != {expected}")


@contextmanager
def _layer_collection(collector, model, index, layer):
    if collector is None:
        yield
        return
    if hasattr(collector, "layer_context"):
        with collector.layer_context(f"main_{index}", layer):
            yield
        return
    blocks = [block for block in collector.adapter.moe_blocks(model)
              if block.trunk == "main" and block.index == index]
    collector.adapter.validate_blocks(blocks)
    try:
        collector.install_blocks(blocks)
        yield
    finally:
        collector.remove()
        collector.blocks = []


def stream_forward(model, weights, batches, *, device="cpu", collector=None,
                   layer_callback=None, max_layers=None):
    """Execute original model control flow, with one layer and one batch active.

    ``layer_callback(event, layer_index, batch_index, value, kwargs)`` can audit
    layer inputs/outputs. The returned values are final decoder outputs, before
    the output head. No scores are implemented in this engine.
    """
    batches = list(batches)
    if not batches:
        raise ValueError("streaming requires at least one calibration batch")
    trunk_name, trunk, layers = _decoder(model)
    if max_layers is not None:
        if isinstance(max_layers, bool) or not isinstance(max_layers, int) or max_layers < 1:
            raise ValueError("max_layers must be a positive integer")
        layers = layers[:max_layers]
    trunk_prefix = trunk_name + "." if trunk_name else ""
    layer_prefix = trunk_prefix + "layers."
    prelude = {name for name in weights.slots
               if name.startswith(trunk_prefix) and not name.startswith(layer_prefix)}
    handles = []
    active = set()
    started = []
    try:
        weights.materialize(trunk_prefix, device, exclude=(layer_prefix,))
        for name, buffer in list(model.named_buffers()):
            if name not in weights.slots and not buffer.is_meta:
                _assign(model, name, buffer.to(device))
        workers = [_NativeWorker(model, batch, layers, device, layer_callback, index)
                   for index, batch in enumerate(batches)]
        local = threading.local()
        for index, layer in enumerate(layers):
            def before(module, args, kwargs, index=index):
                local.worker.before(module, args, kwargs, index)
            handles.append(layer.register_forward_pre_hook(before, with_kwargs=True))
        for worker in workers:
            def target(worker=worker):
                local.worker = worker
                worker._run(model, batches[worker.index], device)
            worker.thread = threading.Thread(target=target, daemon=True)
        for worker in workers:
            worker.thread.start()
            started.append(worker)
            worker.wait(("layer", 0))
        for index, layer in enumerate(layers):
            prefix = f"{layer_prefix}{index}."
            text_config = getattr(trunk, "config", model.config)
            indexer_types = getattr(text_config, "indexer_types", None) or []
            allow = ()
            if index < len(indexer_types) and indexer_types[index] == "shared":
                allow = (re.escape(prefix) + r"self_attn\.indexer\.",)
            active = {name for name in weights.slots if name.startswith(prefix)}
            weights.materialize(prefix, device, allow_missing=allow)
            with _layer_collection(collector, model, index, layer):
                def after(module, args, kwargs, output, index=index):
                    local.worker.after(module, args, kwargs, output, index)
                handle = layer.register_forward_hook(after, with_kwargs=True)
                try:
                    for worker, batch in zip(workers, batches):
                        if collector:
                            if hasattr(collector, "set_batch"):
                                collector.set_batch(batch.get("attention_mask"), batch["input_ids"])
                            else:
                                collector.set_attention_mask(batch.get("attention_mask"))
                        worker.resume.set()
                        expected = ("finished", None) if index == len(layers) - 1 else ("layer", index + 1)
                        worker.wait(expected)
                finally:
                    handle.remove()
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            weights.release(active)
            active = set()
        return [worker.output for worker in workers]
    finally:
        for worker in started:
            worker.cancel.set()
            worker.resume.set()
        for worker in started:
            worker.thread.join()
        for handle in handles:
            handle.remove()
        weights.release(active | prelude)


def collect_streaming(model_path, batches, *, expert_chunk=4, dtype=torch.bfloat16,
                      device="cuda:0", trust_remote_code=False, adapter=None,
                      collector_factory=None, layer_callback=None, attn_implementation=None,
                      attention_chunk=None, verbose=True, batch_window=1):
    """Collect saliency with bounded active calibration batches.

    Each window reads the decoder weights once. Increasing batch_window trades
    activation memory for fewer checkpoint reads; the default keeps one batch's
    native forward state alive rather than retaining the complete corpus.
    """
    from itertools import islice
    from .adapters import get_adapter
    from .collect import SaliencyCollector
    if isinstance(batch_window, bool) or not isinstance(batch_window, int) or batch_window < 1:
        raise ValueError("batch_window must be a positive integer")
    if expert_chunk < 1:
        raise ValueError("expert_chunk must be positive")
    model_path = resolve_checkpoint(model_path, trust_remote_code=trust_remote_code)
    config = load_config(model_path, trust_remote_code=trust_remote_code)
    quantization_config(config)
    kwargs = {"attn_implementation": attn_implementation} if attn_implementation else {}
    model = _skeleton(model_path, config, dtype, trust_remote_code, **kwargs)
    ad = get_adapter(model.config, name=adapter)
    weights = CheckpointWeights(model_path, model, dtype=dtype)
    factory = collector_factory or SaliencyCollector
    collector = factory(ad, expert_chunk=expert_chunk)
    context = nullcontext()
    if attention_chunk is not None:
        from ._streaming_attention import native_chunked
        context = native_chunked(model, chunk=attention_chunk)
    batches = iter(batches)
    offset = 0
    with context:
        while window := list(islice(batches, batch_window)):
            callback = None
            if layer_callback is not None:
                def callback(event, layer, batch, value, kwargs, offset=offset):
                    return layer_callback(event, layer, offset + batch, value, kwargs)
            stream_forward(model, weights, window, device=device, collector=collector,
                           layer_callback=callback)
            offset += len(window)
    result = collector.finalize()
    if not result:
        raise RuntimeError("no saliency was collected from the native decoder")
    if verbose:
        print(f"[stream] adapter={ad.name}; one decoder layer resident")
    return result


def load_decoded_model(model_path, *, config, dtype=torch.bfloat16, device_map="auto",
                       trust_remote_code=False, **kwargs):
    """Materialize known packed checkpoints through the same scoring decoder."""
    max_memory = kwargs.pop("max_memory", None)
    model = _skeleton(model_path, config, dtype, trust_remote_code, **kwargs)
    weights = CheckpointWeights(model_path, model, dtype=dtype)
    if device_map == "auto":
        from accelerate import infer_auto_device_map
        no_split = getattr(model, "_no_split_modules", None) or []
        device_map = infer_auto_device_map(model, dtype=dtype, no_split_module_classes=no_split,
                                           max_memory=max_memory)
        if "disk" in device_map.values():
            raise RuntimeError("decoded model does not fit resident memory; select layer-stream mode")
    if isinstance(device_map, dict):
        loaded = set()
        for prefix, device in sorted(device_map.items(), key=lambda item: -len(item[0])):
            if device == "disk":
                raise ValueError("disk offloading is incompatible with resident expert collection")
            device = f"cuda:{device}" if isinstance(device, int) else device
            key = prefix + "." if prefix else ""
            loaded |= weights.materialize(key, device, exclude=tuple(loaded))
        from accelerate import dispatch_model
        model = dispatch_model(model, device_map=device_map)
    else:
        weights.materialize("", device_map or "cpu")
        for name, buffer in list(model.named_buffers()):
            if name not in weights.slots and not buffer.is_meta:
                _assign(model, name, buffer.to(device_map or "cpu"))
    return model.eval()
