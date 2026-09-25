"""Verify checkpoint tensors or sampled outputs against a sliced reference."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch

from . import calibration, loader, verify
from .pipeline import resolve_data
from .prune import _unique_object


def _load_keep(pruned: str, keep: Optional[str]) -> Dict[str, List[int]]:
    path = Path(keep) if keep else Path(pruned) / "kept_expert_indices.json"
    with path.open() as handle:
        raw = json.load(handle, object_pairs_hook=_unique_object)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("keep-set must be a nonempty object keyed by layer tag")
    result = {}
    for tag, record in raw.items():
        ids = record.get("keep") if isinstance(record, dict) else record
        if not re.fullmatch(r"(?:main|mtp)_(?:0|[1-9]\d*)", tag):
            raise ValueError(f"invalid layer tag: {tag!r}")
        if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError(f"{tag}: keep must be a nonempty list of nonnegative integers")
        if len(ids) != len(set(ids)):
            raise ValueError(f"{tag}: duplicate expert ids")
        result[tag] = ids
    return result


def run(src: str, pruned: str, keep: Optional[str] = None,
        tensors_only: bool = False, data: Optional[str] = None,
        num_batches: int = 4, max_len: int = 2048,
        how: str = "slice",
        adapter: Optional[str] = None, verbose: bool = True,
        trust_remote_code: bool = False, mode: str = "model",
        device: str = "cuda:0", max_layers: Optional[int] = None,
        atol: float = 0.0, rtol: float = 1e-3) -> bool:
    """Verify storage, resident sampled logprobs, or streamed native layer outputs."""
    if how != "slice":
        raise ValueError("checkpoint verification requires how='slice'")
    if mode not in ("model", "tensors", "stream"):
        raise ValueError("verification mode must be model, tensors, or stream")
    if tensors_only and mode == "stream":
        raise ValueError("tensors_only cannot be combined with streamed forward verification")
    if max_layers is not None and (type(max_layers) is not int or max_layers < 1):
        raise ValueError("max_layers must be a positive integer")
    keep_sets = _load_keep(pruned, keep)
    if tensors_only or mode == "tensors":
        return _tensor_level(src, pruned, keep_sets, verbose=verbose)
    native_storage = _uses_native_storage(src, pruned)
    tensor_verified = mode == "stream" or native_storage
    if tensor_verified and not _tensor_level(src, pruned, keep_sets, verbose=verbose):
        return False
    tensor_only_tags = ()
    if native_storage:
        from .checkpoint import inspect_checkpoint

        tensor_only_tags = tuple(inspect_checkpoint(src)["mtp_layers"])
    tokenizer = loader.load_tokenizer(src, trust_remote_code=trust_remote_code)
    batches = calibration.build_batches(
        resolve_data(data), tokenizer, num_batches=num_batches,
        batch_size=1, max_len=max_len, verbose=verbose)
    if mode == "stream":
        return _stream_reference(
            src, pruned, keep_sets, batches, adapter=adapter, device=device,
            trust_remote_code=trust_remote_code, max_layers=max_layers,
            atol=atol, rtol=rtol, verbose=verbose, tensor_only_tags=tensor_only_tags)
    net, _ = loader.load_model(pruned, device_map="auto", adapter=adapter,
                               verbose=verbose, trust_remote_code=trust_remote_code)
    target_config = copy.deepcopy(net.config)
    try:
        test = verify.score_batches(net, batches)
    finally:
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    net, ad = loader.load_model(src, device_map="auto", adapter=adapter,
                                verbose=verbose, trust_remote_code=trust_remote_code)
    try:
        _slice_reference(net, ad, keep_sets, verbose=verbose,
                         target_config=target_config, tensor_verified=tensor_verified,
                         tensor_only_tags=tensor_only_tags)
        ref = verify.score_batches(net, batches)
    finally:
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return verify.compare_logprobs(ref, test, verbose=verbose)


def _stream_reference(src, pruned, keep_sets, batches, *, adapter=None,
                      device="cuda:0", trust_remote_code=False, max_layers=None,
                      atol=0.0, rtol=1e-3, verbose=True, tensor_only_tags=()):
    """Slice original native layers, then audit two unmodified model control flows."""
    from .adapters import get_adapter
    try:
        from .streaming import CheckpointWeights, _decoder, _skeleton
    except ImportError as exc:
        raise ImportError(
            "native streaming verification requires the selected model backend and "
            "razor.streaming; tensor-only verification remains available"
        ) from exc
    source_config = loader.load_config(src, trust_remote_code=trust_remote_code)
    target_config = loader.load_config(pruned, trust_remote_code=trust_remote_code)
    text_config = getattr(source_config, "text_config", source_config)
    dtype = getattr(text_config, "dtype", None) or getattr(source_config, "dtype", None) or torch.bfloat16
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError("native stream requires a floating-point compute dtype")
    source_model = _skeleton(src, source_config, dtype, trust_remote_code,
                             attn_implementation="eager")
    target_model = _skeleton(pruned, target_config, dtype, trust_remote_code,
                             attn_implementation="eager")
    source_ad = get_adapter(source_model.config, name=adapter)
    target_ad = get_adapter(target_model.config, name=adapter)
    source_blocks = {block.tag: block.module for block in source_ad.moe_blocks(source_model)
                     if block.tag.startswith("main_")}
    target_blocks = {block.tag: block.module for block in target_ad.moe_blocks(target_model)
                     if block.tag.startswith("main_")}
    if set(source_blocks) != set(target_blocks):
        raise ValueError("source and target instantiated MoE layers differ")
    _, _, layers = _decoder(source_model)
    depth = min(max_layers, len(layers)) if max_layers is not None else len(layers)
    selected = {tag for tag in keep_sets if tag in source_blocks and tag.startswith("main_")
                and int(tag[5:]) < depth}
    if not selected:
        raise ValueError("no selected MoE layer is exercised; reference comparison would be vacuous")
    for tag, ids in keep_sets.items():
        if tag not in source_blocks:
            if tag.startswith("mtp_") or tag in tensor_only_tags:
                continue
            raise ValueError(f"keep-set names an uninstantiated main MoE layer: {tag}")
        verify.validate_keep(source_ad.gate_module(source_blocks[tag]).weight.shape[0], ids)
        if target_ad.gate_module(target_blocks[tag]).weight.shape[0] != len(ids):
            raise ValueError(f"{tag}: target native expert count differs from keep-set")
    source_weights = CheckpointWeights(src, source_model, dtype=dtype)
    target_weights = CheckpointWeights(pruned, target_model, dtype=dtype)
    sliced = set()

    def transform(index, layer):
        tag = f"main_{index}"
        if tag not in selected:
            return None
        block = source_blocks[tag]
        bank = source_ad.expert_bank(block)
        restore = None
        if isinstance(bank, torch.nn.ModuleList):
            path = next(name for name, module in block.named_modules() if module is bank)
            parent, _, leaf = path.rpartition(".")
            owner = block.get_submodule(parent) if parent else block
            restore = lambda: setattr(owner, leaf, bank)
        try:
            gate = source_ad.gate_module(block)
            ids = torch.tensor(keep_sets[tag], dtype=torch.long, device=gate.weight.device)
            options = {}
            if source_ad.is_hash(block) and not torch.equal(ids.cpu(), torch.arange(gate.weight.shape[0])):
                options["hash_policy"] = "remap"
            top_k = target_ad.router_spec(target_blocks[tag]).top_k
            source_ad.prune_block(block, ids, target_top_k=top_k, **options)
            if source_ad.gate_module(block).weight.shape[0] != len(keep_sets[tag]):
                raise ValueError(f"{tag}: native reference did not slice the requested router rows")
            sliced.add(tag)
        except BaseException:
            if restore is not None:
                restore()
            raise
        return restore

    report = verify.verify_native_streams(
        source_model, source_weights, target_model, target_weights, batches,
        transform_reference=transform, device=device, max_layers=max_layers,
        atol=atol, rtol=rtol, verbose=verbose)
    if sliced != selected:
        raise ValueError("native stream did not exercise every selected reference layer")
    if verbose:
        excluded = sorted(set(keep_sets) - selected)
        if excluded:
            print(f"[verify] tensor-only coverage outside executed native layers: {excluded}")
    return report["ok"]


def _slice_reference(model, adapter, keep: Dict[str, List[int]],
                     verbose: bool = True, target_config=None,
                     tensor_verified: bool = False, tensor_only_tags=()) -> int:
    """Apply the same structural pruning and routing configuration as the target."""
    from .prune import prune_model

    blocks = {block.tag: block for block in adapter.moe_blocks(model)}
    absent = set(keep) - set(blocks)
    if tensor_verified and absent and all(
            re.fullmatch(r"mtp_(?:0|[1-9]\d*)", tag) or tag in tensor_only_tags for tag in absent):
        if verbose:
            print(f"[verify] tensor-only MTP coverage (not executed by model forward): {sorted(absent)}")
        keep = {tag: ids for tag, ids in keep.items() if tag in blocks}
    if tensor_verified and target_config is not None and any(
            adapter.is_hash(block.module) for block in blocks.values()):
        return _slice_hash_reference(adapter, blocks, keep, target_config, verbose=verbose)
    if not keep or set(keep) != set(blocks):
        raise ValueError("reference keep-set must name every source MoE layer exactly once")
    for tag, ids in keep.items():
        verify.validate_keep(adapter.gate_module(blocks[tag].module).weight.shape[0], ids)
    counts = {len(ids) for ids in keep.values()}
    if len(counts) != 1:
        raise ValueError("reference requires a uniform expert count")
    count = next(iter(counts))
    top_k = adapter.top_k
    if target_config is not None:
        target = type(adapter)(target_config)
        if target.num_experts != count:
            raise ValueError("target config expert count does not match the keep-set")
        top_k = target.top_k
        for config in (target.config, target.text_config):
            for names, expected in ((target.CONFIG_NUM_EXPERTS, count), (target.CONFIG_TOP_K, top_k)):
                for name in names:
                    value = getattr(config, name, None)
                    if value is not None and (type(value) is not int or value != expected):
                        raise ValueError(f"target config has inconsistent expert-count or top-k alias: {name}")
        fields = ("n_group", "topk_group", "scoring_func", "norm_topk_prob",
                  "routed_scaling_factor", "topk_method",
                  "num_hash_layers", "n_hash_layers", "first_k_hash_replace")
        for field in fields:
            source_value = adapter._cfg((field,), "__absent__")
            target_value = target._cfg((field,), "__absent__")
            if source_value != target_value:
                raise ValueError(f"target routing config differs at {field}")
    applied = prune_model(model, adapter, keep, target_top_k=top_k, verbose=False)
    if applied != {tag: list(ids) for tag, ids in keep.items()}:
        raise ValueError("reference pruning changed the requested keep order")
    if adapter.num_experts != count or adapter.top_k != top_k:
        raise ValueError("reference pruning did not update the routing configuration")
    if verbose:
        print(f"[verify] sliced reference: {len(applied)} MoE layers, {count} experts, top-k {top_k}")
    return len(applied)


def _slice_hash_reference(adapter, blocks, keep, target_config, *, verbose=True):
    """Apply independently verified V4 remap/preserve policy to resident modules."""
    from .adapters.base import MoEAdapter

    target = type(adapter)(target_config)
    if not keep or set(keep) - set(blocks):
        raise ValueError("hash reference contains unknown or empty keep sets")
    if not 1 <= target.top_k <= adapter.top_k:
        raise ValueError("target top-k must not exceed the source top-k")
    plans = []
    for tag, item in blocks.items():
        block = item.module
        gate = adapter.gate_module(block)
        is_hash = adapter.is_hash(block)
        expected = target._cfg(("hash_n_routed_experts",), target.num_experts) if is_hash else target.num_experts
        if type(expected) is not int or expected < 1:
            raise ValueError("invalid target hash expert count")
        ids = keep.get(tag)
        if ids is None:
            if not is_hash or expected != gate.weight.shape[0]:
                raise ValueError(f"{tag}: missing keep-set for a non-preserved layer")
            ids = list(range(gate.weight.shape[0]))
        ids = verify.validate_keep(gate.weight.shape[0], ids)
        if len(ids) != expected:
            raise ValueError(f"{tag}: target expert count differs from reference keep-set")
        rows = torch.tensor(ids, dtype=torch.long, device=gate.weight.device)
        top_k = adapter.router_spec(block).top_k if is_hash else target.top_k
        if is_hash:
            MoEAdapter.validate_keep(adapter, block, rows, target_top_k=top_k)
            if gate.tid2eid.ndim != 2 or gate.tid2eid.shape[1] != top_k:
                raise ValueError(f"{tag}: native hash table width differs from top-k")
        else:
            adapter.validate_keep(block, rows, target_top_k=top_k)
        options = {}
        if is_hash and ids != list(range(gate.weight.shape[0])):
            options["hash_policy"] = "remap"
        plans.append((tag, block, rows, top_k, options))
    for tag, block, rows, top_k, options in plans:
        adapter.prune_block(block, rows, target_top_k=top_k, **options)
    adapter.set_num_experts(target.num_experts)
    adapter.set_top_k(target.top_k)
    for source_cfg, target_cfg in ((adapter.config, target.config),
                                   (adapter.text_config, target.text_config)):
        for name in ("hash_n_routed_experts", "hash_layer_indices"):
            if hasattr(target_cfg, name):
                setattr(source_cfg, name, copy.deepcopy(getattr(target_cfg, name)))
            elif hasattr(source_cfg, name):
                delattr(source_cfg, name)
    if verbose:
        print(f"[verify] native hash-aware reference: {len(plans)} layers; "
              "source-only hash remapping, preserved native hash top-k")
    return len(plans)


class _TensorCheckpoint:
    """Read safetensors metadata eagerly and tensor contents on demand."""

    def __init__(self, root: str):
        from safetensors import safe_open

        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError(f"local checkpoint directory required: {root}")
        indices = sorted(root.glob("*.safetensors.index.json"))
        files = sorted(root.glob("*.safetensors"))
        self.paths, self.shapes = {}, {}
        declared = None
        if indices:
            if len(indices) != 1:
                raise ValueError("ambiguous safetensors index files")
            with indices[0].open() as handle:
                index = json.load(handle, object_pairs_hook=_unique_object)
            declared = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(declared, dict) or not declared:
                raise ValueError("safetensors index requires a nonempty weight_map")
            for name, shard in declared.items():
                if not isinstance(name, str) or not isinstance(shard, str):
                    raise ValueError("invalid safetensors weight_map")
                path = (root / shard).resolve()
                if path.parent != root or path.suffix != ".safetensors" or not path.is_file():
                    raise ValueError(f"invalid or missing checkpoint shard: {shard}")
                self.paths[name] = path
            if {path.resolve() for path in files} != set(self.paths.values()):
                raise ValueError("safetensors files disagree with the index")
        elif len(files) != 1:
            raise ValueError("expected one safetensors shard or an explicit safetensors index")
        observed = {}
        for path in files:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name in observed:
                        raise ValueError(f"duplicate tensor across shards: {name}")
                    observed[name] = path.resolve()
                    self.shapes[name] = tuple(handle.get_slice(name).get_shape())
        if not observed:
            raise ValueError("empty safetensors checkpoint")
        if declared is not None and observed != self.paths:
            raise ValueError("safetensors index keys or shard assignments disagree with shard contents")
        self.paths = observed

    def tensor(self, name):
        from safetensors import safe_open

        with safe_open(str(self.paths[name]), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)


_EXPERT_KEY = re.compile(
    r"^(?P<root>(?:model\.)?(?:language_model\.)?)"
    r"(?P<trunk>layers|mtp(?:\.layers)?|mtp_layers(?:\.layers)?)\."
    r"(?P<index>0|[1-9]\d*)\.(?P<block>mlp|ffn)\.experts\.(?P<tail>.+)$")
_ROUTER_PARTS = ("gate", "router", "mlp_gate", "wg")
_BIASES = {"bias", "e_score_correction_bias", "expert_bias", "correction_bias"}
_SCALES = {"scale", "scale_inv", "weight_scale", "weight_scale_inv", "input_scale"}
_FUSED_TAIL = re.compile(
    r"(?:gate_up_proj|gate_proj|up_proj|down_proj|w1|w2|w3)"
    r"(?:(?:\.|_)(?:weight|bias|scale|scale_inv|weight_scale|weight_scale_inv|input_scale))?")


def _tensor_plan(shapes, keep_sets):
    """Map each expected target key to a source key and optional expert rows."""
    if not isinstance(keep_sets, dict) or not keep_sets:
        raise ValueError("keep-set must be nonempty")
    layers = {}
    for name in shapes:
        if "experts" not in name.split("."):
            continue
        match = _EXPERT_KEY.fullmatch(name)
        if match is None:
            raise ValueError(f"unsupported expert tensor path: {name}")
        tag = ("main" if match["trunk"] == "layers" else "mtp") + "_" + match["index"]
        base = name.rsplit(".experts.", 1)[0] + "."
        layer = layers.setdefault(tag, {"base": base, "experts": {}})
        if layer["base"] != base:
            raise ValueError(f"ambiguous checkpoint paths for {tag}")
        layer["experts"][name] = match["tail"]
    unknown = set(keep_sets) - set(layers)
    if unknown:
        raise ValueError(f"keep-set names missing MoE layers: {sorted(unknown)}")
    transforms, dropped = {}, set()
    for tag, kept in keep_sets.items():
        layer = layers[tag]
        base, experts = layer["base"], layer["experts"]
        router_weights = []
        for first in _ROUTER_PARTS:
            for path in (first,) + tuple(f"{first}.{second}" for second in _ROUTER_PARTS):
                name = base + path + ".weight"
                if name in shapes:
                    router_weights.append(name)
        if len(router_weights) != 1 or len(shapes[router_weights[0]]) != 2:
            raise ValueError(f"{tag}: expected one explicit (experts, hidden) router weight")
        router_weight = router_weights[0]
        router_base = router_weight.rsplit(".", 1)[0] + "."
        n_experts = shapes[router_weight][0]
        kept = verify.validate_keep(n_experts, kept)
        listed = [bool(re.fullmatch(r"(?:0|[1-9]\d*)\..+", tail)) for tail in experts.values()]
        if any(listed) and not all(listed):
            raise ValueError(f"{tag}: mixed or unsupported expert layout")
        if all(listed):
            suffixes = {}
            for name, tail in experts.items():
                old, suffix = tail.split(".", 1)
                suffixes.setdefault(int(old), set()).add(suffix)
                if int(old) in kept:
                    target = base + f"experts.{kept.index(int(old))}.{suffix}"
                    transforms[name] = (target, None)
                else:
                    dropped.add(name)
            if set(suffixes) != set(range(n_experts)) or any(
                    value != suffixes[0] for value in suffixes.values()):
                raise ValueError(f"{tag}: incomplete or inconsistent source expert tensors")
        else:
            for name, tail in experts.items():
                shape = shapes[name]
                if _FUSED_TAIL.fullmatch(tail) is None:
                    raise ValueError(f"unsupported fused tensor: {name}")
                if not shape or shape[0] != n_experts:
                    raise ValueError(f"{name}: unsupported fused expert axis or scale layout")
                transforms[name] = (name, kept)
        transforms[router_weight] = (router_weight, kept)
        for name, shape in shapes.items():
            if not name.startswith(base):
                continue
            suffix = name[len(base):]
            if suffix in _BIASES - {"bias"}:
                if shape != (n_experts,):
                    raise ValueError(f"{name}: unsupported correction bias shape")
                transforms[name] = (name, kept)
            elif name.startswith(router_base) and name != router_weight:
                leaf = name[len(router_base):]
                if leaf in _BIASES and shape == (n_experts,):
                    transforms[name] = (name, kept)
                elif leaf in _SCALES and shape and shape[0] == n_experts:
                    transforms[name] = (name, kept)
                elif leaf in _SCALES and shape in ((), (1,)):
                    pass
                else:
                    raise ValueError(f"unsupported router tensor or scale layout: {name}")
            elif suffix.split(".", 1)[0] in _ROUTER_PARTS and name not in transforms:
                raise ValueError(f"unsupported router tensor path: {name}")
    plan = {}
    for source in shapes:
        if source in dropped:
            continue
        target, ids = transforms.get(source, (source, None))
        if target in plan:
            raise ValueError(f"duplicate target tensor mapping: {target}")
        plan[target] = (source, ids)
    return plan


def _compare_plan(source, target, plan, verbose):
    expected_keys, target_keys = set(plan), set(target.shapes)
    if expected_keys != target_keys:
        if verbose:
            print(f"FAIL: missing tensors={sorted(expected_keys - target_keys)[:8]}; "
                  f"unexpected tensors={sorted(target_keys - expected_keys)[:8]}")
        return False
    failures = []
    for target_name, (source_name, ids) in plan.items():
        expected = source.tensor(source_name)
        if ids is not None:
            rows = torch.tensor(ids, dtype=torch.long)
            shape, dtype = expected.shape, expected.dtype
            raw = expected.contiguous().view(torch.uint8).reshape(shape[0], -1)
            expected = raw.index_select(0, rows).view(dtype).reshape((len(ids),) + shape[1:])
            del raw
        actual = target.tensor(target_name)
        if not verify.tensors_identical(expected, actual):
            failures.append(target_name)
        del expected, actual
    if verbose:
        print(f"{'FAIL' if failures else 'PASS'}: {len(plan)} checkpoint tensors checked")
        for name in failures[:10]:
            print(f"  tensor mismatch: {name}")
    return not failures


def _declared_storage(pruned: str) -> Optional[bool]:
    """Report whether the output declares raw storage, or None when undeclared."""
    info = Path(pruned) / "pruning_info.json"
    if not info.is_file():
        return None
    with info.open() as handle:
        metadata = json.load(handle, object_pairs_hook=_unique_object)
    if not isinstance(metadata, dict):
        raise ValueError("pruning metadata must be an object")
    return (metadata.get("storage_profile") == "safetensors_raw_v1" or
            metadata.get("tensor_only") is True)


def _native_source(src: str) -> bool:
    """Detect native-storage families and quantization without importing checkpoint code."""
    path = Path(src) / "config.json"
    if not path.is_file():
        return False
    with path.open() as handle:
        config = json.load(handle, object_pairs_hook=_unique_object)
    if not isinstance(config, dict):
        raise ValueError("checkpoint config must be an object")
    for cfg in (config, config.get("text_config") or {}):
        if not isinstance(cfg, dict):
            raise ValueError("text_config must be an object")
        kind = str(cfg.get("model_type", "")).lower()
        if cfg.get("quantization_config") is not None or kind.startswith(
                ("hy_v3", "hyv3", "deepseek_v4", "glm_moe_dsa", "glm5_next",
                 "qwen3_5_moe", "qwen3_6_moe", "gemma4", "gemma_4", "kimi")):
            return True
    return False


def _uses_native_storage(src: str, pruned: str) -> bool:
    """Dispatch storage verification on the output's actual format.

    Raw exports declare ``storage_profile``/``tensor_only`` and keep the strict
    native verifier, including its manifest and configuration checks. A
    ``save_pretrained`` output declares neither, so it is compared with the
    ordinary tensor plan even when the source is a native-storage family: its
    stored layout is the renumbered one that plan describes. An output that
    declares no storage format at all cannot be classified, so the stricter
    source-side family and quantization heuristic still decides.
    """
    declared = _declared_storage(pruned)
    return _native_source(src) if declared is None else declared


def _tensor_level(src: str, pruned: str, keep_sets: Dict[str, List[int]],
                  verbose: bool = True) -> bool:
    """Check storage independently, including native packed scales and hash tables."""
    from safetensors import SafetensorError

    try:
        if _uses_native_storage(src, pruned):
            from .checkpoint import verify_checkpoint

            result = verify_checkpoint(src, pruned, keep_sets)
            if (not isinstance(result, dict) or result.get("ok") is not True or
                    not isinstance(result.get("checked_tensors"), int) or
                    result["checked_tensors"] < 1):
                raise ValueError("native checkpoint verifier returned no verified tensors")
            if verbose:
                print(f"PASS: {result['checked_tensors']} native checkpoint tensors checked")
                print("[verify] native storage/config checks do not execute model forward or validate routing")
                if result.get("auxiliary_files_verified") is False:
                    print("[verify] tokenizer and copied custom assets are not verified; test output loading separately")
            return True
        source, target = _TensorCheckpoint(src), _TensorCheckpoint(pruned)
        plan = _tensor_plan(source.shapes, keep_sets)
        ok = _compare_plan(source, target, plan, verbose)
    except (OSError, ValueError, TypeError, KeyError, IndexError, RuntimeError, SafetensorError) as exc:
        if verbose:
            print(f"FAIL: tensor verification unavailable: {exc}")
        return False
    if verbose:
        print("[verify] tensor checks do not validate config, loading, or forward execution")
    return ok


def _check_renumbered(src_state: Dict[str, torch.Tensor],
                      dst_state: Dict[str, torch.Tensor],
                      listed: Dict[str, List[int]],
                      verbose: bool = True) -> bool:
    """Compatibility wrapper for in-memory checkpoint comparisons."""
    class State:
        def __init__(self, tensors):
            self.shapes = {name: tuple(tensor.shape) for name, tensor in tensors.items()}
            self.tensor = tensors.__getitem__

    try:
        source, target = State(src_state), State(dst_state)
        return _compare_plan(source, target, _tensor_plan(source.shapes, listed), verbose)
    except (ValueError, KeyError, IndexError, RuntimeError) as exc:
        if verbose:
            print(f"FAIL: {exc}")
        return False
