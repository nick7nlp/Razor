"""Public collection, pruning and sweep APIs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from . import calibration, geometry, loader, metrics
from .adapters import MoEAdapter, RouterSpec, get_adapter
from .collect import SALIENCY_FILENAME, collect as _collect
from .collect import find_saliency, verify_routing
from .prune import (_unique_object, build_info, fill_unobserved_layers, prune_model,
                    resolve_target_experts, save_pruned, select_keep_indices,
                    validate_keep_indices)

DEFAULT_DATA = str(Path(__file__).resolve().parent.parent / "data" / "RazorCal.json")
_LFS_MAGIC = b"version https://git-lfs.github.com/spec/"
COLLECT_DEFAULTS = {"num_batches": geometry.EXHAUST, "batch_size": 1,
                    "max_len": 32768, "seed": geometry.DEFAULT_SEED}
_CACHE_SCHEMA = 1
_CHECKPOINT_TYPES = frozenset({
    "hy_v3", "deepseek_v4", "deepseek_v4_mixed", "glm_moe_dsa", "glm4_moe_lite",
    "glm5_next", "glm5_next_text", "qwen3_5_moe", "qwen3_5_moe_text",
    "qwen3_6_moe", "qwen3_6_moe_text", "qwen4_exp", "qwen4_exp_text",
    "gemma4", "gemma4_text", "kimi_k3", "kimi_linear",
})


def _raw_config(model):
    path = Path(model).expanduser() / "config.json"
    if not path.is_file():
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(str(model), "config.json"))
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream, object_pairs_hook=_unique_object)
    if not isinstance(config, dict):
        raise ValueError("checkpoint config must be an object")
    return config


def _needs_checkpoint_backend(config):
    text = config.get("text_config") or config
    return (config.get("model_type") in _CHECKPOINT_TYPES or
            text.get("model_type") in _CHECKPOINT_TYPES or
            bool(config.get("quantization_config") or text.get("quantization_config")))


def _local_checkpoint(model):
    return loader.resolve_checkpoint(model)


def _storage_metadata(model, export_mode):
    if export_mode not in ("auto", "checkpoint", "model"):
        raise ValueError("export_mode must be auto, checkpoint or model")
    if export_mode == "model":
        config = _raw_config(model)
        text = config.get("text_config") or config
        if {config.get("model_type"), text.get("model_type")} & {"qwen4_exp", "qwen4_exp_text"}:
            from .checkpoint import inspect_checkpoint
            if inspect_checkpoint(_local_checkpoint(model))["mtp_layers"]:
                raise ValueError("Qwen3.8 native model export does not retain stored MTP weights; "
                                 "use export_mode='checkpoint' (or 'auto')")
        return None
    if export_mode == "auto" and not _needs_checkpoint_backend(_raw_config(model)):
        return None
    from .checkpoint import inspect_checkpoint
    path = _local_checkpoint(model)
    return path, inspect_checkpoint(path)


def resolve_data(data: Optional[str] = None) -> str:
    """Resolve a local calibration path or a Hugging Face dataset specification."""
    data = str(data) if data is not None else DEFAULT_DATA
    if data.startswith("hf:"):
        return data
    p = Path(data).expanduser()
    if p.is_dir():
        return str(p)
    if not p.is_file():
        raise FileNotFoundError(
            "calibration data is missing; retrieve RazorCal with git lfs pull "
            "in a source checkout, or supply --data")
    with p.open("rb") as f:
        if f.read(len(_LFS_MAGIC)) == _LFS_MAGIC:
            raise FileNotFoundError("calibration file is a Git LFS pointer; run git lfs pull")
    return str(p)


def _identity(source: str, content: bool = False) -> str:
    p = Path(source).expanduser()
    digest = hashlib.sha256()
    digest.update(str(p.resolve() if p.exists() else source).encode())
    files = sorted(p.rglob("*")) if p.is_dir() else [p]
    text_suffixes = {".json", ".jsonl", ".py", ".jinja", ".model", ".txt", ".tiktoken", ".vocab", ".merges"}
    for f in files:
        if not f.is_file():
            continue
        if p.is_dir() and f.suffix not in text_suffixes | {".safetensors", ".bin"}:
            continue
        st = f.stat()
        digest.update(str(f.relative_to(p) if p.is_dir() else f.name).encode())
        digest.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
        if content or f.suffix in text_suffixes:
            with f.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _outside_source(model: str, out: Path) -> None:
    source = Path(model).expanduser()
    if source.is_dir() and (out.resolve() == source.resolve() or source.resolve() in out.resolve().parents):
        raise ValueError("output must be outside the source checkpoint directory")


def _validate_budget(adapter: MoEAdapter, kept: int, top_k: int) -> int:
    """Check config-level group constraints before collection or checkpoint writes."""
    groups = int(adapter._cfg(("n_group",), 1))
    selected_groups = int(adapter._cfg(("topk_group",), 1))
    if groups < 1 or adapter.num_experts % groups or not 1 <= selected_groups <= groups:
        raise ValueError("invalid routing group configuration")
    if kept < top_k:
        raise ValueError("target_experts must be at least target_top_k; set a smaller target_top_k")
    if kept % groups:
        raise ValueError("expert budget must be divisible by the routing group count")
    quota = kept // groups
    if quota < adapter.MIN_EXPERTS_PER_GROUP:
        raise ValueError("expert budget is too small for the native group scorer")
    if groups > 1 and (quota < RouterSpec().group_score_topk or quota * selected_groups < top_k):
        raise ValueError("expert budget is too small for grouped routing and top_k")
    return groups


def _cache_key(model: str, data: str, num_batches: int, batch_size: int,
               max_len: int, seed: int, **options) -> str:
    import transformers

    defaults = dict(adapter=None, verify=True, dtype=str(torch.bfloat16),
                    expert_chunk=4, device_map="auto", trust_remote_code=False,
                    calibration_limit=None, execution="auto", stream_device=None,
                    chunk_attn=None, stream_batch_window=1)
    options = {key: options.get(key, value) for key, value in defaults.items()}
    implementation = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        implementation.update(str(path.relative_to(Path(__file__).parent)).encode())
        implementation.update(path.read_bytes())
    raw = [_CACHE_SCHEMA, implementation.hexdigest(), torch.__version__, transformers.__version__,
           _identity(model), _identity(data, content=True), num_batches,
           batch_size, max_len, seed, options]
    return hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()[:24]


def saliency_dir(model: str, data: str = DEFAULT_DATA,
                 out_dir: Optional[str] = None, **collect_kwargs) -> Path:
    """Return the collection directory for a set of inputs."""
    if out_dir is not None:
        return Path(out_dir)
    args = {k: collect_kwargs.pop(k, v) for k, v in COLLECT_DEFAULTS.items()}
    return Path("outputs") / ("saliency_" + _cache_key(model, data, **args, **collect_kwargs))


def _restore_score_contract(record, layer, header):
    choices = {"counterfactual_semantics": ("fixed_selected_set", "frozen_weights"),
               "pruning_policy": ("preserve", "score"), "score_space": ("latent", "expert_output")}
    for key in geometry.NON_ADDITIVE_CONST:
        values = []
        for source in (header, layer):
            if key not in source:
                continue
            value = source[key]
            if torch.is_tensor(value):
                if value.ndim != 0:
                    raise ValueError(f"{key} must be a scalar")
                value = value.item()
            if key in ("razor_supported", "refill_supported", "renormalized") and type(value) is not bool:
                raise ValueError(f"{key} must be boolean")
            if key == "top_k" and (type(value) is not int or value < 1):
                raise ValueError("top_k must be a positive integer")
            if key == "router_scaling" and (type(value) not in (int, float) or
                    not bool(torch.isfinite(torch.tensor(value))) or value <= 0):
                raise ValueError("router_scaling must be positive and finite")
            if key in choices and value not in choices[key]:
                raise ValueError(f"unsupported {key}")
            values.append(value)
        if values:
            if any(type(value) is not type(values[0]) or value != values[0] for value in values[1:]):
                raise ValueError(f"pack and layer disagree on {key}")
            record[key] = values[0]
    if (record.get("razor_supported") is False or record.get("top_k") == 1 or
            record.get("pruning_policy") == "preserve"):
        record["razor_supported"] = False
        for key in list(record):
            if key.startswith(("counterfactual_delta_", "rcs_", "rcs_refill_")):
                del record[key]
    # A pack that carries no refill moments cannot serve RAZOR, and must say so
    # rather than letting selection fall through to a different criterion.
    if not any(key.startswith("rcs_refill_") for key in record):
        record["refill_supported"] = False
    return record


def _load_saliency(path) -> Dict[str, dict]:
    """Load finalized scores or a compatible additive statistics pack."""
    found = find_saliency(path)
    if found is None:
        raise FileNotFoundError("saliency file not found")
    packed = torch.load(found, map_location="cpu", weights_only=True)
    value = packed.get("state", packed) if isinstance(packed, dict) else packed
    if not isinstance(value, dict) or not value or not all(isinstance(v, dict) for v in value.values()):
        raise ValueError("saliency must be a non-empty layer mapping")
    out = {}
    header = packed if "state" in packed else {}
    for tag, rec in value.items():
        if "count" not in rec:
            out[tag] = _restore_score_contract(dict(rec), rec, header)
            continue
        raw_count = metrics._vector(rec["count"], "count")
        count = raw_count.double()
        if raw_count.dtype == torch.bool or not torch.equal(count, count.round()):
            raise ValueError("routed counts must be integers")
        normalized = {"routed_count": count, "expert_frequency": count.long()}
        pairs = (("ean_sum", "ean_sum", "ean_mean"),
                 ("reap_sum", "weighted_ean_sum", "reap"),
                 ("d1_sum", "counterfactual_delta_sum", "counterfactual_delta_mean"),
                 ("rcs_sum", "rcs_sum", "rcs_mean"),
                 ("refill_sum", "rcs_refill_sum", "rcs_refill_mean"))
        for source, total, mean in pairs:
            if source not in rec:
                continue
            sums = metrics._vector(rec[source], source).double()
            if sums.shape != count.shape or bool(((count == 0) & (sums != 0)).any()):
                raise ValueError("statistics sums and routed counts disagree")
            normalized[total] = sums
            normalized[mean] = (sums / count.clamp_min(1)).float()
        for source, square, rms in (
            ("ean_sq", "ean_square_sum", "ean_rms"),
            ("reap_sq", "weighted_ean_square_sum", "reap_rms"),
            ("d1_sq", "counterfactual_delta_square_sum", "counterfactual_delta_rms"),
            ("rcs_sq", "rcs_square_sum", "rcs_rms"),
            ("refill_sq", "rcs_refill_square_sum", "rcs_refill_rms"),
        ):
            if source not in rec:
                continue
            sums = metrics._vector(rec[source], source).double()
            if sums.shape != count.shape or bool(((count == 0) & (sums != 0)).any()):
                raise ValueError("statistics squared sums and routed counts disagree")
            normalized[square] = sums
            normalized[rms] = (sums / count.clamp_min(1)).sqrt().float()
        if "n_tokens" in header:
            normalized["total_tokens"] = header["n_tokens"]
        out[tag] = _restore_score_contract(normalized, rec, header)
    return out


def collect_saliency(
    model: str, data: str = DEFAULT_DATA, out_dir: Optional[str] = None,
    num_batches: int = geometry.EXHAUST, batch_size: int = 1,
    max_len: int = 32768, seed: int = geometry.DEFAULT_SEED,
    expert_chunk: int = 4, device_map: str = "auto",
    adapter: Optional[str] = None, verify: bool = True, cache: bool = True,
    verbose: bool = True, trust_remote_code: bool = False,
    dtype: torch.dtype = torch.bfloat16, calibration_limit: Optional[int] = None,
    execution: str = "auto", stream_device: Optional[str] = None,
    chunk_attn: Optional[int] = None, stream_batch_window: int = 1,
) -> Dict[str, dict]:
    """Collect supported scores in resident or native layer-streaming mode."""
    if execution not in ("auto", "resident", "streaming"):
        raise ValueError("execution must be auto, resident or streaming")
    if chunk_attn is not None and (type(chunk_attn) is not int or chunk_attn < 1):
        raise ValueError("chunk_attn must be a positive integer")
    if num_batches != geometry.EXHAUST and (isinstance(num_batches, bool) or not isinstance(num_batches, int) or num_batches < 1):
        raise ValueError("num_batches must be -1 or a positive integer")
    for name, value in (("batch_size", batch_size), ("max_len", max_len),
                        ("expert_chunk", expert_chunk), ("stream_batch_window", stream_batch_window)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    data = resolve_data(data)
    options = dict(adapter=adapter, verify=verify, dtype=str(dtype),
                   expert_chunk=expert_chunk, device_map=device_map,
                   trust_remote_code=trust_remote_code, calibration_limit=calibration_limit,
                   execution=execution, stream_device=stream_device, chunk_attn=chunk_attn,
                   stream_batch_window=stream_batch_window)
    key = _cache_key(model, data, num_batches, batch_size, max_len, seed, **options)
    out = Path(out_dir).expanduser() if out_dir is not None else Path("outputs") / f"saliency_{key}"
    _outside_source(model, out)
    manifest = out / "collection.json"
    expected = {"schema": _CACHE_SCHEMA, "fingerprint": key}
    if cache and Path(model).is_dir() and not data.startswith("hf:") and manifest.is_file():
        with manifest.open() as f:
            same = json.load(f) == expected
        if same and find_saliency(out) is not None:
            if verbose:
                print("[saliency] reusing verified cache")
            return _load_saliency(out)
    native_storage = _needs_checkpoint_backend(_raw_config(model))
    streaming = execution == "streaming" or (execution == "auto" and native_storage)
    source = _local_checkpoint(model) if streaming or native_storage else model
    _outside_source(source, out)
    tokenizer = loader.load_tokenizer(source, trust_remote_code=trust_remote_code)
    batches = calibration.build_batches(
        data, tokenizer, num_batches=num_batches, batch_size=batch_size,
        max_len=max_len, seed=seed, limit=calibration_limit, verbose=verbose)
    if not batches or not any(bool(b["attention_mask"].sum() > 0) for b in batches):
        raise ValueError("calibration contains no valid tokens")
    if streaming:
        from .streaming import collect_streaming
        device = stream_device or (device_map if device_map != "auto" else
                                   ("cuda:0" if torch.cuda.is_available() else "cpu"))
        if not isinstance(device, (str, torch.device)):
            raise ValueError("streaming requires a single stream_device")
        saliency = collect_streaming(
            source, batches, dtype=dtype, device=device, adapter=adapter,
            trust_remote_code=trust_remote_code, expert_chunk=expert_chunk,
            attention_chunk=chunk_attn, batch_window=stream_batch_window, verbose=verbose)
    else:
        from contextlib import nullcontext
        net, ad = loader.load_model(source, device_map=device_map, dtype=dtype,
                                    adapter=adapter, trust_remote_code=trust_remote_code,
                                    verbose=verbose)
        try:
            context = nullcontext()
            if chunk_attn is not None:
                from ._streaming_attention import native_chunked
                context = native_chunked(net, chunk=chunk_attn)
            with context:
                if verify and not verify_routing(net, ad, batches[0]):
                    raise RuntimeError("routing verification failed; collection aborted")
                saliency = _collect(net, ad, batches, out_dir=str(out),
                                    expert_chunk=expert_chunk, verbose=verbose)
        finally:
            del net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    import os
    import tempfile
    out.mkdir(parents=True, exist_ok=True)
    manifest.unlink(missing_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".scores-", suffix=".pt", dir=out)
    os.close(fd)
    try:
        torch.save(saliency, temporary)
        os.replace(temporary, out / SALIENCY_FILENAME)
        manifest.write_text(json.dumps(expected, indent=2) + "\n")
    except BaseException:
        manifest.unlink(missing_ok=True)
        raise
    finally:
        Path(temporary).unlink(missing_ok=True)
    return saliency


def _select_stored(saliency, metadata, method, kept, aggregation):
    ignored = set(metadata["hash_layers"]) | set(metadata["mtp_layers"])
    unexpected = set(saliency) - set(metadata["layers"])
    if unexpected:
        raise ValueError("saliency layer identities do not match the checkpoint")
    ordinary = {tag: rec for tag, rec in saliency.items() if tag not in ignored}
    expected = set(metadata["layers"]) - ignored
    if set(ordinary) != expected:
        raise ValueError("saliency is missing ordinary decoder MoE layers")
    if any(metrics.score(rec, method, aggregation=aggregation).numel() != metadata["counts"][tag]
           for tag, rec in ordinary.items()):
        raise ValueError("saliency expert counts do not match the checkpoint")
    if not ordinary:
        raise ValueError("checkpoint has no scored ordinary decoder experts")
    for tag, record in ordinary.items():
        if record.get("pruning_policy") == "preserve":
            raise ValueError(f"{tag}: scoring conflicts with preserve policy")
        count = record.get("routed_count", record.get("expert_frequency"))
        if count is not None and not bool(metrics._vector(count, "routed_count").sum() > 0):
            raise ValueError(f"{tag}: ordinary decoder experts were not observed")
    return select_keep_indices(ordinary, method, kept, verbose=False,
                               n_group=metadata["n_group"], aggregation=aggregation)


def _prune_stored(model, out, storage, *, method, target_experts, ratio, saliency,
                  keep_indices, data, target_top_k, hash_policy, mtp_policy,
                  aggregation, adapter, trust_remote_code, verbose,
                  max_shard_size, collect_kwargs):
    from .checkpoint import prune_checkpoint, validate_budget
    source, metadata = storage
    k = metadata["top_k"] if target_top_k is None else target_top_k
    if keep_indices:
        with open(keep_indices, encoding="utf-8") as stream:
            raw = json.load(stream, object_pairs_hook=_unique_object)
        if not isinstance(raw, dict) or not raw:
            raise ValueError("keep_indices must be a non-empty layer mapping")
        keep = {tag: rec.get("keep") if isinstance(rec, dict) else rec for tag, rec in raw.items()}
        budget_excluded = set(metadata["hash_layers"])
        if mtp_policy == "drop":
            budget_excluded.update(metadata["mtp_layers"])
        sizes = [len(ids) for tag, ids in keep.items()
                 if tag not in budget_excluded and isinstance(ids, list)]
        if not sizes:
            raise ValueError("explicit keep-set must include ordinary decoder experts")
        kept = sizes[0]
    else:
        kept = resolve_target_experts(metadata["num_experts"], target_experts, ratio)
        if metrics.canonical(method).startswith("rcs") and metadata["top_k"] < 2:
            raise ValueError("the RCS criteria require top_k >= 2; use reap, ean or frequency")
    validate_budget(metadata, kept, k)
    if not keep_indices:
        sal = _load_saliency(saliency) if saliency else collect_saliency(
            source, data=data, adapter=adapter, verbose=verbose,
            trust_remote_code=trust_remote_code, **collect_kwargs)
        keep = _select_stored(sal, metadata, method, kept, aggregation)
    result = prune_checkpoint(
        source, str(out), keep, method="external" if keep_indices else method,
        target_experts=kept, target_top_k=k, hash_policy=hash_policy,
        mtp_policy=mtp_policy, trust_remote_code=trust_remote_code, verbose=verbose,
        max_shard_size=max_shard_size, aggregation=aggregation)
    return str(result["out_dir"])


def prune(
    model: str, out_dir: str, method: str = "razor",
    ratio: Optional[float] = None, target_experts: Optional[int] = None,
    saliency: Optional[str] = None, keep_indices: Optional[str] = None,
    data: str = DEFAULT_DATA, target_top_k: Optional[int] = None,
    skip_layers: Optional[Sequence[str]] = None, max_shard_size: str = "5GB",
    adapter: Optional[str] = None, verbose: bool = True,
    trust_remote_code: bool = False, export_mode: str = "auto",
    hash_policy: str = "remap", mtp_policy: str = "router_norm",
    aggregation: str = metrics.DEFAULT_AGGREGATION, **collect_kwargs,
) -> str:
    """Prune a model, preserving native checkpoint storage where supported."""
    if aggregation not in metrics.AGGREGATIONS:
        raise ValueError("aggregation must be rms, mean or sum")
    if hash_policy not in ("remap", "preserve") or mtp_policy not in ("router_norm", "drop", "error"):
        raise ValueError("invalid hash_policy or mtp_policy")
    metrics.canonical(method)
    if skip_layers:
        raise NotImplementedError("skipping layers requires per-layer expert-count support")
    if saliency and keep_indices:
        raise ValueError("saliency and keep_indices are mutually exclusive")
    if keep_indices and (ratio is not None or target_experts is not None):
        raise ValueError("keep_indices cannot be combined with a pruning budget")
    if not keep_indices and (ratio is None) == (target_experts is None):
        raise ValueError("specify exactly one of target_experts / ratio")
    out = Path(out_dir).expanduser()
    if out.exists():
        raise FileExistsError("output directory already exists; choose a new directory")
    _outside_source(model, out)
    storage = _storage_metadata(model, export_mode)
    if storage is not None:
        return _prune_stored(
            model, out, storage, method=method, target_experts=target_experts,
            ratio=ratio, saliency=saliency, keep_indices=keep_indices, data=data,
            target_top_k=target_top_k, hash_policy=hash_policy, mtp_policy=mtp_policy,
            aggregation=aggregation, adapter=adapter, trust_remote_code=trust_remote_code,
            verbose=verbose, max_shard_size=max_shard_size, collect_kwargs=collect_kwargs)
    if hash_policy != "remap" or mtp_policy != "router_norm":
        raise ValueError("hash and MTP policies require export_mode='checkpoint'")
    cfg = loader.load_config(model, trust_remote_code=trust_remote_code)
    ad = get_adapter(cfg, name=adapter)
    orig_e, orig_k = ad.num_experts, ad.top_k
    k = orig_k if target_top_k is None else target_top_k
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= orig_k:
        raise ValueError("target_top_k must be an integer in [1, original top_k]")
    external = None
    if keep_indices:
        with open(keep_indices) as f:
            raw = json.load(f, object_pairs_hook=_unique_object)
        if not isinstance(raw, dict) or not raw:
            raise ValueError("keep_indices must be a non-empty layer mapping")
        external = {t: (v.get("keep") if isinstance(v, dict) else v) for t, v in raw.items()}
        external = validate_keep_indices(external, {t: orig_e for t in external}, k)
        kept = len(next(iter(external.values())))
    else:
        kept = resolve_target_experts(orig_e, target_experts, ratio)
        if metrics.canonical(method).startswith("rcs") and orig_k < 2:
            raise ValueError("the RCS criteria require top_k >= 2; use reap, ean or frequency")
    n_group = _validate_budget(ad, kept, k)
    if external is not None:
        sal = None
        keep = external
    else:
        sal = (_load_saliency(saliency) if saliency else collect_saliency(
            model, data=data, adapter=adapter, verbose=verbose,
            trust_remote_code=trust_remote_code, **collect_kwargs))
        if any(metrics.score(rec, method, aggregation=aggregation).numel() != orig_e for rec in sal.values()):
            raise ValueError("saliency expert counts do not match the model")
        keep = select_keep_indices(sal, method, kept, verbose=verbose,
                                   n_group=n_group, aggregation=aggregation)
    net, ad = loader.load_model(model, device_map="cpu", adapter=adapter,
                                dtype=collect_kwargs.get("dtype", torch.bfloat16),
                                trust_remote_code=trust_remote_code, verbose=verbose)
    try:
        ad = get_adapter(net.config, name=adapter)
        tags = [b.tag for b in ad.moe_blocks(net)]
        keep = fill_unobserved_layers(keep, tags, sal, method, kept,
                                      verbose=verbose, n_group=n_group, aggregation=aggregation)
        applied = prune_model(net, ad, keep, verbose=verbose, target_top_k=k)
        info = build_info(method="external" if external is not None else method,
                          src_model=model, orig_experts=orig_e, kept_experts=kept,
                          orig_top_k=orig_k, top_k=k, n_layers=len(applied), adapter_name=ad.name)
        info["routing_groups"] = n_group
        info["aggregation"] = aggregation
        save_pruned(net, ad, out, model, info, applied,
                    max_shard_size=max_shard_size, remote_code=trust_remote_code, verbose=verbose)
    finally:
        del net
    return str(out)


def sweep(
    model: str, out_root: str, methods: Sequence[str] = metrics.METHODS,
    ratios: Sequence[float] = (0.25, 0.5, 0.75), data: str = DEFAULT_DATA,
    saliency: Optional[str] = None, adapter: Optional[str] = None,
    verbose: bool = True, trust_remote_code: bool = False,
    export_mode: str = "auto", hash_policy: str = "remap", mtp_policy: str = "router_norm",
    aggregation: str = metrics.DEFAULT_AGGREGATION, **collect_kwargs,
) -> List[str]:
    """Reuse one collection across a grid of methods and removal ratios."""
    if aggregation not in metrics.AGGREGATIONS:
        raise ValueError("aggregation must be rms, mean or sum")
    if hash_policy not in ("remap", "preserve") or mtp_policy not in ("router_norm", "drop", "error"):
        raise ValueError("invalid hash_policy or mtp_policy")
    methods, ratios = list(methods), list(ratios)
    if not methods or not ratios or any(m not in metrics.CHOICES for m in methods):
        raise ValueError("sweep needs non-empty valid methods and ratios")
    root = Path(out_root).expanduser()
    _outside_source(model, root)
    storage = _storage_metadata(model, export_mode)
    if storage is None:
        cfg = loader.load_config(model, trust_remote_code=trust_remote_code)
        ad = get_adapter(cfg, name=adapter)
        original, top_k = ad.num_experts, ad.top_k
        budgets = [resolve_target_experts(original, None, r) for r in ratios]
        for kept in budgets:
            n_group = _validate_budget(ad, kept, top_k)
    else:
        from .checkpoint import validate_budget
        source, metadata = storage
        _outside_source(source, root)
        original, top_k = metadata["num_experts"], metadata["top_k"]
        n_group = metadata["n_group"]
        budgets = [resolve_target_experts(original, None, r) for r in ratios]
        for kept in budgets:
            validate_budget(metadata, kept, top_k)
    if any(metrics.canonical(m).startswith("rcs") for m in methods) and top_k < 2:
        raise ValueError("top-1 models support sweeps with reap, ean and frequency only")
    targets = [root / f"{m}-{format(r, '.12g')}" for m in methods for r in ratios]
    if len(set(targets)) != len(targets) or any(p.exists() for p in targets):
        raise ValueError("sweep output paths overlap or already exist")
    for target in targets:
        _outside_source(model, target)
    if saliency is None:
        options = dict(collect_kwargs)
        if options.get("out_dir") is None:
            options["out_dir"] = str(root / "saliency")
        collection_out = Path(options["out_dir"]).expanduser()
        _outside_source(model, collection_out)
        for target in targets:
            if collection_out.resolve() == target.resolve() or target.resolve() in collection_out.resolve().parents:
                raise ValueError("collection directory overlaps a sweep checkpoint output")
        sal = collect_saliency(model, data=data, adapter=adapter, verbose=verbose,
                               trust_remote_code=trust_remote_code, **options)
        saliency = str(collection_out / SALIENCY_FILENAME)
    else:
        sal = _load_saliency(saliency)
    for method in methods:
        if storage is None:
            if any(metrics.score(rec, method, aggregation=aggregation).numel() != original for rec in sal.values()):
                raise ValueError("saliency expert counts do not match the model")
            for kept in budgets:
                select_keep_indices(sal, method, kept, verbose=False,
                                     n_group=n_group, aggregation=aggregation)
        else:
            for kept in budgets:
                _select_stored(sal, metadata, method, kept, aggregation)
    made = []
    for (m, r), out in zip(((m, r) for m in methods for r in ratios), targets):
        made.append(prune(model, str(out), method=m, ratio=r, saliency=saliency,
                           data=data, adapter=adapter, verbose=verbose,
                           trust_remote_code=trust_remote_code, export_mode=export_mode,
                           hash_policy=hash_policy, mtp_policy=mtp_policy,
                           aggregation=aggregation,
                           dtype=collect_kwargs.get("dtype", torch.bfloat16)))
    return made
