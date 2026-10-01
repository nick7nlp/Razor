"""Prune local safetensors checkpoints without materializing or converting a model.

Storage profiles describe checkpoint keys, not the modules a loader constructs.
Unmodified tensors and selected expert rows are copied as raw byte ranges; FP8,
MXFP4, and their scale tensors therefore retain their original representation.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from . import metrics

_COUNT = ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts")
_TOPK = ("num_experts_per_tok", "num_experts_per_token", "top_k_experts", "moe_topk")
_GROUP = ("n_group", "num_expert_group", "num_expert_groups")
_MTP_COUNT = ("num_nextn_predict_layers", "num_mtp_layers", "mtp_num_hidden_layers", "num_next_token_prediction_layers")
_CHUNK = 8 * 1024 * 1024
_LAYER = re.compile(r"^(?P<prefix>(?:language_model\.)?model\.(?:language_model\.)?layers|layers|mtp(?:\.layers)?)\.(?P<layer>\d+)\.(?P<tail>.+)$")
_EXPERT = re.compile(r"^(?P<block>mlp|ffn|block_sparse_moe)\.experts\.(?P<expert>\d+)\.(?P<role>.+)$")
_SIBLING = re.compile(r"^experts\.(?P<expert>\d+)\.(?P<role>.+)$")
_FUSED = re.compile(r"^(?:(?:mlp|ffn|block_sparse_moe)\.)?experts\.(?:gate_up_proj|gate_proj|up_proj|down_proj|w[123])(?:\.(?:weight|scale|weight_scale|weight_scale_inv|weight_packed))?$")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_unique)


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _alias(config, fields, default=None):
    values = [config[f] for f in fields if f in config and config[f] is not None]
    if not values:
        if default is None:
            raise ValueError(f"configuration has none of {fields}")
        return default
    if len(set(values)) != 1:
        raise ValueError(f"configuration aliases disagree: {fields}")
    return _integer(values[0], fields[0])


def _contained(root, relative):
    root = Path(root).resolve()
    relative = os.fspath(relative)
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or Path(relative).is_absolute() or ".." in relative.split("/")):
        raise ValueError("checkpoint contains an unsafe relative path")
    path = root / relative
    if any(parent.is_symlink() for parent in path.parents if parent != root and parent.is_relative_to(root)):
        raise ValueError("checkpoint contains an unsafe symlink directory")
    resolved = path.resolve()
    if resolved.is_relative_to(root):
        return path
    blobs = root.parent.parent / "blobs"
    if (root.parent.name == "snapshots" and path.is_symlink()
            and not blobs.is_symlink() and resolved.parent == blobs
            and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", resolved.name)
            and not (blobs / resolved.name).is_symlink() and resolved.is_file()
            and os.readlink(path) == os.path.relpath(resolved, path.parent)):
        return path
    raise ValueError("checkpoint contains an unsafe symlink or relative path")


@dataclass(frozen=True)
class _Tensor:
    file: Path
    dtype: str
    shape: tuple
    offset: int
    size: int


def _complete_root(root):
    root = Path(root).resolve()
    marker = root / ".razor-incomplete"
    if marker.exists() or marker.is_symlink():
        raise RuntimeError(f"checkpoint is incomplete: {marker}")
    return root


def _headers(root, *, require_complete=False):
    """Read all source shards; exported checkpoints require a complete standard index."""
    root = _complete_root(root)
    if not root.is_dir():
        raise ValueError("checkpoint must be a local directory")
    indexes = sorted(root.glob("*.safetensors.index.json"))
    if require_complete and indexes and [path.name for path in indexes] != ["model.safetensors.index.json"]:
        raise ValueError("output requires a single standard model.safetensors.index.json")
    declared = {}
    for path in indexes:
        mapping = _json(_contained(root, path.relative_to(root))).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("checkpoint index requires a nonempty weight_map")
        for key, value in mapping.items():
            if not isinstance(key, str) or not isinstance(value, str) or not value.endswith(".safetensors"):
                raise ValueError("checkpoint index requires tensor names and safetensors shard paths")
            _contained(root, value)
            if key in declared and declared[key] != value:
                raise ValueError(f"conflicting indexes for {key}")
            declared[key] = value
    files = {_contained(root, path.relative_to(root)) for path in root.glob("*.safetensors")}
    files.update(_contained(root, value) for value in set(declared.values()))
    if not files:
        raise ValueError("checkpoint has no safetensors weights")
    if require_complete and not indexes and files != {root / "model.safetensors"}:
        raise ValueError("output requires a complete standard index or a single model.safetensors")
    tensors, weight_map = {}, {}
    for path in sorted(files):
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError("truncated safetensors header")
            length = struct.unpack("<Q", prefix)[0]
            if length > file_size - 8 or length > 100_000_000:
                raise ValueError("invalid safetensors header size")
            header = json.loads(handle.read(length), object_pairs_hook=_unique)
        intervals = []
        for key, record in header.items():
            if key == "__metadata__":
                continue
            if key in tensors:
                raise ValueError(f"duplicate tensor across shards: {key}")
            start, end = record["data_offsets"]
            shape = tuple(record["shape"])
            if any(isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in shape):
                raise ValueError(f"invalid tensor shape: {key}")
            if not 0 <= start <= end <= file_size - 8 - length:
                raise ValueError(f"invalid tensor offsets: {key}")
            dtype = record["dtype"]
            bytes_per_element = {"BOOL": 1, "I8": 1, "U8": 1, "I16": 2, "U16": 2,
                                 "I32": 4, "U32": 4, "I64": 8, "U64": 8,
                                 "F16": 2, "BF16": 2, "F32": 4, "F64": 8}.get(dtype)
            if dtype.startswith("F8_"):
                bytes_per_element = 1
            if bytes_per_element is None or end - start != math.prod(shape) * bytes_per_element:
                raise ValueError(f"unsupported dtype or invalid tensor byte count: {key}")
            intervals.append((start, end))
            relative = str(path.relative_to(root))
            if key in declared and declared[key] != relative:
                raise ValueError(f"index points to the wrong shard: {key}")
            tensors[key] = _Tensor(path, record["dtype"], shape, 8 + length + start, end - start)
            weight_map[key] = relative
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValueError("overlapping or non-contiguous safetensors offsets")
            cursor = end
        if cursor != file_size - 8 - length:
            raise ValueError("unindexed bytes in safetensors shard")
        if require_complete:
            with safe_open(path, framework="pt", device="cpu") as handle:
                if (handle.metadata() or {}).get("format") != "pt":
                    raise ValueError("output safetensors requires standard PyTorch format metadata")
    if set(declared) - set(tensors):
        raise ValueError("checkpoint index refers to missing tensors")
    if require_complete and indexes:
        if set(declared) != set(tensors) or set(declared.values()) != {str(path.relative_to(root)) for path in files}:
            raise ValueError("output index must cover every tensor and shard completely")
    return tensors, weight_map


def _family(config, keys):
    text = config.get("text_config") or config
    model_types = {
        "hy3": {"hy_v3"},
        "deepseek_v4": {"deepseek_v4"},
        "gemma4": {"gemma4", "gemma4_text"},
        "kimi": {"kimi_k3", "kimi_linear"},
        "glm": {"glm4_moe", "glm4_moe_lite", "glm_moe_dsa", "glm5_next", "glm5_next_text"},
        "qwen": {"qwen2_moe", "qwen3_moe", "qwen3_next", "qwen3_5_moe", "qwen3_5_moe_text",
                 "qwen3_6_moe", "qwen3_6_moe_text", "qwen4_exp", "qwen4_exp_text"},
    }
    declared_types = {value for value in (config.get("model_type"), text.get("model_type")) if value}
    matches = {family for family, names in model_types.items() if declared_types & names}
    if len(matches) == 1:
        return matches.pop()
    if declared_types:
        raise ValueError("unrecognized checkpoint family or conflicting model types")
    architectures = {
        "hy3": {"HYV3ForCausalLM"},
        "deepseek_v4": {"DeepseekV4ForCausalLM", "DeepseekV4MixedForCausalLM"},
        "gemma4": {"Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"},
        "kimi": {"KimiK3ForConditionalGeneration", "KimiLinearForCausalLM"},
        "glm": {"Glm4MoeForCausalLM", "Glm4MoeLiteForCausalLM", "GlmMoeDsaForCausalLM",
                "Glm5NextForConditionalGeneration", "Glm5NextForCausalLM"},
        "qwen": {"Qwen2MoeForCausalLM", "Qwen3MoeForCausalLM", "Qwen3NextForCausalLM",
                 "Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM",
                 "Qwen3_6MoeForConditionalGeneration", "Qwen3_6MoeForCausalLM",
                 "Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM"},
    }
    declared_arch = set(config.get("architectures") or []) | set(text.get("architectures") or [])
    matches = {family for family, names in architectures.items() if declared_arch & names}
    if len(matches) == 1:
        return matches.pop()
    if not declared_arch:
        signatures = {
            "deepseek_v4": r"^layers\.\d+\.ffn\.gate\.tid2eid$",
            "hy3": r"^model\.layers\.\d+\.mlp\.router\.gate\.weight$",
            "gemma4": r"^model\.(?:language_model\.)?layers\.\d+\.router\.per_expert_scale$",
            "kimi": r"^language_model\.model\.layers\.\d+\.block_sparse_moe\.experts\.\d+\.w[123]\.weight_packed$",
        }
        matches = {family for family, pattern in signatures.items() if any(re.fullmatch(pattern, key) for key in keys)}
        if len(matches) == 1:
            return matches.pop()
    raise ValueError("unrecognized checkpoint family; configuration and tensor names are required")


def _classify(key, family):
    match = _LAYER.fullmatch(key)
    if not match:
        return None, None, None
    tag = ("mtp_" if match["prefix"].startswith("mtp") else "main_") + match["layer"]
    tail = match["tail"]
    expert = _EXPERT.fullmatch(tail) or (_SIBLING.fullmatch(tail) if family == "gemma4" else None)
    if expert:
        return "expert", tag, int(expert["expert"])
    if _FUSED.fullmatch(tail):
        return "fused", tag, None
    routers = {
        "hy3": {"mlp.router.gate.weight", "mlp.router.gate.bias", "mlp.expert_bias",
                "mlp.gate.weight", "mlp.gate.bias", "mlp.e_score_correction_bias"},
        "deepseek_v4": {"ffn.gate.weight", "ffn.gate.bias", "mlp.gate.weight", "mlp.gate.bias", "mlp.gate.e_score_correction_bias"},
        "glm": {"mlp.gate.weight", "mlp.gate.bias", "mlp.gate.e_score_correction_bias"},
        "qwen": {"mlp.gate.weight", "mlp.gate.bias"},
        "gemma4": {"router.proj.weight", "router.proj.bias", "router.per_expert_scale"},
        "kimi": {"block_sparse_moe.gate.weight", "block_sparse_moe.gate.e_score_correction_bias"},
    }
    if tail in routers[family]:
        return "router", tag, None
    if family == "deepseek_v4" and tail in ("ffn.gate.tid2eid", "mlp.gate.tid2eid"):
        return "hash", tag, None
    return None, tag, None


def _renumber(key, expert):
    return re.sub(r"(?<=experts\.)\d+(?=\.)", str(expert), key, count=1)


def _scan(model_path, *, require_complete=False):
    root = _complete_root(model_path)
    config = _json(_contained(root, "config.json"))
    tensors, weight_map = _headers(root, require_complete=require_complete)
    family = _family(config, tensors)
    text = config.get("text_config") or config
    count = _alias(text, _COUNT)
    top_k = _alias(text, _TOPK)
    groups = _alias(text, _GROUP, 1)
    n_layers = _alias(text, ("num_hidden_layers", "n_layers"), 0)
    classes, widths, roles, hashes, gates, mtp = {}, {}, {}, {}, {}, set()
    for key, tensor in tensors.items():
        kind, tag, expert = _classify(key, family)
        classes[key] = (kind, tag, expert)
        if tag and (tag.startswith("mtp_") or (n_layers and int(tag.split("_")[1]) >= n_layers)):
            mtp.add(tag)
        if kind in ("fused", "router"):
            if not tensor.shape:
                raise ValueError(f"expert-axis tensor is scalar: {key}")
            widths.setdefault(tag, set()).add(tensor.shape[0])
            if kind == "router" and tensor.dtype not in {"F16", "BF16", "F32", "F64"}:
                raise ValueError(f"quantized router tensors are unsupported; refusing unsafe row slicing: {key}")
            if kind == "router" and key.endswith(".weight"):
                if len(tensor.shape) != 2:
                    raise ValueError(f"router weight must be a matrix: {key}")
                gates[tag] = key
        elif kind == "expert":
            roles.setdefault((tag, _renumber(key, 0)), set()).add(expert)
        elif kind == "hash":
            if len(tensor.shape) != 2 or tensor.shape[1] != top_k:
                raise ValueError(f"hash table width differs from configured top-k: {key}")
            hashes[tag] = key
    for (tag, role), experts in roles.items():
        if experts != set(range(len(experts))):
            raise ValueError(f"incomplete per-expert tensor role: {role}")
        widths.setdefault(tag, set()).add(len(experts))
    if not widths:
        raise ValueError("no expert tensors match the checkpoint profile")
    for tag, gate in gates.items():
        prefix = gate.rsplit(".", 1)[0] + "."
        unknown = [key for key in tensors if key.startswith(prefix) and classes[key][0] not in {"router", "hash"}]
        if unknown:
            raise ValueError(f"unsupported router state or quantization scale: {unknown[0]}")
    if set(hashes) - set(widths):
        raise ValueError("hash table has no matching expert tensors")
    counts = {}
    for tag, values in widths.items():
        expected = _integer(text.get("hash_n_routed_experts", count), "hash_n_routed_experts") if tag in hashes else count
        if values != {expected}:
            raise ValueError(f"{tag}: tensor expert widths {values} disagree with configuration {expected}")
        if tag not in gates:
            raise ValueError(f"{tag}: expert tensors have no matching router weight")
        counts[tag] = expected
    for key, tensor in tensors.items():
        kind = classes[key][0]
        if ".experts." in key and kind is None:
            raise ValueError(f"unrecognized expert tensor layout: {key}")
        if kind in ("expert", "fused"):
            base, _, suffix = key.rpartition(".")
            if suffix in ("scale", "weight_scale", "weight_scale_inv"):
                if not any(candidate in tensors for candidate in (base + ".weight", base + ".weight_packed", base)):
                    raise ValueError(f"orphan expert quantization scale: {key}")
            if suffix == "weight_packed" or tensor.dtype.startswith("F8_") and suffix == "weight":
                if not any(base + "." + scale in tensors for scale in ("scale", "weight_scale", "weight_scale_inv")):
                    raise ValueError(f"quantized expert weight has no paired scale: {key}")
    metadata = dict(family=family, num_experts=count, top_k=top_k, n_group=groups,
                    topk_group=_alias(text, ("topk_group", "top_k_group"), groups),
                    layers=sorted(counts), counts=counts, hash_layers=sorted(hashes),
                    mtp_layers=sorted(set(counts) & mtp), config=config, weight_map=weight_map)
    return metadata, tensors, classes, gates, hashes, mtp


def inspect_checkpoint(model_path):
    """Describe local storage, expert counts, routing, and on-disk MTP layers."""
    return _scan(model_path)[0]


def validate_budget(metadata, kept, k=None):
    """Validate a scalar budget without I/O; return the resolved routing top-k.

    Actual keep maps additionally require equal contiguous quotas in each router
    group. Hash-routed checkpoints retain their stored table width.
    """
    count = _integer(metadata["num_experts"], "num_experts")
    kept = _integer(kept, "target_experts")
    original_k = _integer(metadata["top_k"], "original top-k")
    k = original_k if k is None else _integer(k, "target_top_k")
    groups = _integer(metadata.get("n_group", 1), "n_group")
    selected_groups = _integer(metadata.get("topk_group", groups), "topk_group")
    if kept > count:
        raise ValueError("target_experts exceeds the source expert count")
    if k > original_k or k > kept:
        raise ValueError("target_top_k exceeds original top-k or retained expert count")
    if count % groups or kept % groups:
        raise ValueError("source and retained expert counts must be divisible by routing groups")
    if selected_groups > groups:
        raise ValueError("topk_group exceeds the routing group count")
    if k > (kept // groups) * selected_groups:
        raise ValueError("retained selected routing groups cannot supply target_top_k experts")
    if groups > selected_groups and metadata.get("family") in ("glm", "kimi") and kept // groups < 2:
        raise ValueError("group-limited routing requires at least two experts per group")
    if metadata.get("hash_layers") and k != original_k:
        raise ValueError("hash routing requires preserving the original top-k/table width")
    return k


def _load(tensor, key):
    with safe_open(tensor.file, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def _rank(scores, budget, groups):
    if scores.numel() % groups or budget % groups:
        raise ValueError("expert counts and retained budget must be divisible by routing groups")
    width = scores.numel() // groups
    values = torch.argsort(scores.reshape(groups, width), descending=True, stable=True)[:, :budget // groups]
    return sorted((values + torch.arange(groups)[:, None] * width).flatten().tolist())


def remap_hash_table(table, gate_weight, keep):
    """Choose cosine-nearest unused survivors, reserving original survivors first."""
    if table.ndim != 2 or len(keep) < table.shape[1]:
        raise ValueError("retained experts cannot fill the hash table without duplicates")
    if table.numel() and (table.min() < 0 or table.max() >= gate_weight.shape[0]):
        raise ValueError("hash table contains out-of-range expert IDs")
    weights = gate_weight.float()
    if not torch.isfinite(weights).all():
        raise ValueError("non-finite router weights cannot define a hash remap")
    weights = weights / weights.norm(dim=1, keepdim=True).clamp_min(1e-12)
    order = torch.argsort(weights @ weights[keep].T, dim=1, descending=True, stable=True).tolist()
    inverse = {old: new for new, old in enumerate(keep)}
    output = torch.empty_like(table)
    for i, row in enumerate(table.tolist()):
        used, holes = set(), []
        for j, old in enumerate(row):
            new = inverse.get(old)
            if new is None or new in used:
                holes.append(j)
            else:
                output[i, j] = new
                used.add(new)
        for j in holes:
            new = next(candidate for candidate in order[row[j]] if candidate not in used)
            output[i, j] = new
            used.add(new)
    return output


def _selection(scan, keep, target_top_k, hash_policy, mtp_policy):
    meta, tensors, classes, gates, hashes, mtp = scan
    if isinstance(keep, (str, os.PathLike)):
        keep = _json(keep)
    if not isinstance(keep, dict) or not keep:
        raise ValueError("keep must be a nonempty mapping of layer tags to expert IDs")
    if hash_policy not in ("remap", "preserve"):
        raise ValueError("hash_policy must be 'remap' or 'preserve'")
    if mtp_policy not in ("router_norm", "require", "error", "preserve", "drop"):
        raise ValueError("mtp_policy must be 'router_norm', 'require'/'error', or 'drop'")
    selected = {}
    for raw_tag, values in keep.items():
        tag = f"main_{raw_tag}" if isinstance(raw_tag, int) else raw_tag
        if tag not in meta["counts"]:
            raise ValueError(f"unknown keep-set layer: {tag}")
        if isinstance(values, dict):
            values = values.get("keep")
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{tag}: keep must be a nonempty integer list")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in values):
            raise ValueError(f"{tag}: keep IDs must be integers")
        if len(set(values)) != len(values) or min(values) < 0 or max(values) >= meta["counts"][tag]:
            raise ValueError(f"{tag}: duplicate or out-of-range expert IDs")
        selected[tag] = list(values)
    full = set(hashes) if hash_policy == "preserve" else set()
    for tag in full & set(selected):
        if selected[tag] != list(range(meta["counts"][tag])):
            raise ValueError("preserve hash policy conflicts with a pruned or reordered hash keep-set")
    sizes = {len(v) for tag, v in selected.items() if tag not in full and not (mtp_policy == "drop" and tag in mtp)}
    if len(sizes) != 1:
        raise ValueError("non-hash keep sets must have one common retained expert count")
    budget = sizes.pop()
    top_k = validate_budget(meta, budget, target_top_k)
    groups = meta["n_group"]
    for tag, count in meta["counts"].items():
        if mtp_policy == "drop" and tag in mtp:
            selected.pop(tag, None)
            continue
        if tag in full:
            selected[tag] = list(range(count))
        elif tag not in selected:
            if tag in hashes:
                table = _load(tensors[hashes[tag]], hashes[tag])
                if table.numel() and (table.min() < 0 or table.max() >= count):
                    raise ValueError("hash table contains out-of-range IDs")
                selected[tag] = _rank(torch.bincount(table.flatten().long(), minlength=count), budget, groups)
            elif tag in mtp and mtp_policy == "router_norm":
                weights = _load(tensors[gates[tag]], gates[tag]).float()
                selected[tag] = _rank(weights.norm(dim=1), budget, groups)
            else:
                raise ValueError(f"missing keep set for {'MTP' if tag in mtp else 'decoder'} layer {tag}")
        if len(selected[tag]) < top_k:
            raise ValueError(f"{tag}: retained expert count is below top-k")
        if groups > 1:
            if count % groups or len(selected[tag]) % groups:
                raise ValueError("expert budget must be divisible by routing groups")
            width = count // groups
            group_ids = [expert // width for expert in selected[tag]]
            expected = [g for g in range(groups) for _ in range(len(selected[tag]) // groups)]
            if group_ids != expected:
                raise ValueError("keep IDs must preserve equal contiguous routing-group quotas")
    return selected, budget, top_k, full


def _config(meta, budget, top_k, hash_policy, mtp_policy):
    from .prune import _public_config

    config = _public_config(copy.deepcopy(meta["config"]), defer_auto_map=True)
    targets = [config]
    if isinstance(config.get("text_config"), dict):
        targets.append(config["text_config"])
    for target in targets:
        for field in _COUNT:
            if field in target:
                target[field] = budget
        for field in _TOPK:
            if field in target:
                target[field] = top_k
        if mtp_policy == "drop":
            for field in _MTP_COUNT:
                if field in target:
                    target[field] = 0
        if hash_policy == "remap":
            if "hash_n_routed_experts" in target:
                target["hash_n_routed_experts"] = budget
            auto_map = dict(target.get("auto_map") or {})
            for name, reference in (("AutoConfig", "configuration_deepseek_v4_mixed.DeepseekV4MixedConfig"),
                                    ("AutoModelForCausalLM", "modeling_deepseek_v4_mixed.DeepseekV4MixedForCausalLM")):
                if auto_map.get(name) == reference:
                    del auto_map[name]
            if auto_map:
                target["auto_map"] = auto_map
            elif "auto_map" in target:
                del target["auto_map"]
            if target.get("architectures"):
                target["architectures"] = ["DeepseekV4ForCausalLM" if name == "DeepseekV4MixedForCausalLM" else name
                                           for name in target["architectures"]]
    if hash_policy == "preserve" and meta["hash_layers"]:
        target = targets[-1]
        target["hash_n_routed_experts"] = meta["counts"][meta["hash_layers"][0]]
        target["hash_layer_indices"] = [int(tag.split("_")[1]) for tag in meta["hash_layers"]]
        auto_map = dict(config.get("auto_map") or {})
        auto_map.update(AutoConfig="configuration_deepseek_v4_mixed.DeepseekV4MixedConfig",
                        AutoModelForCausalLM="modeling_deepseek_v4_mixed.DeepseekV4MixedForCausalLM")
        config["auto_map"] = auto_map
        config["architectures"] = ["DeepseekV4MixedForCausalLM"]
    return config


def _copy_aux(root, stage, *, trust_remote_code=False, config=None):
    from .prune import copy_aux_files

    copy_aux_files(root, stage, remote_code=trust_remote_code, verbose=False, config=config)


def _shard_bytes(value):
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMGT]?I?B)", str(value).upper())
    if not match:
        raise ValueError("max_shard_size must be positive bytes or a size such as '5GB'")
    suffix = match[2]
    exponent = "BKMGT".index(suffix[0]) if suffix[0] != "B" else 0
    size = int(float(match[1]) * (1024 if "I" in suffix else 1000) ** exponent)
    if size <= 0:
        raise ValueError("max_shard_size must be positive")
    return size


def _stream_range(source, offset, length, target):
    source.seek(offset)
    while length:
        data = source.read(min(length, _CHUNK))
        if not data:
            raise ValueError("source checkpoint changed or was truncated during export")
        target.write(data)
        length -= len(data)


def _write_shard(path, entries):
    header, position = {"__metadata__": {"format": "pt"}}, 0
    for key, tensor, ids, replacement in entries:
        shape = list(tensor.shape)
        size = tensor.size
        if ids is not None:
            size = tensor.size // tensor.shape[0] * len(ids)
            shape[0] = len(ids)
        header[key] = dict(dtype=tensor.dtype, shape=shape, data_offsets=[position, position + size])
        position += size
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    with path.open("xb") as output:
        output.write(struct.pack("<Q", len(raw)))
        output.write(raw)
        for key, tensor, ids, replacement in entries:
            if replacement is not None:
                data = replacement.contiguous().view(torch.uint8).numpy().tobytes()
                if len(data) != tensor.size:
                    raise ValueError("hash replacement changed the stored tensor size")
                output.write(data)
            else:
                with tensor.file.open("rb") as source:
                    if ids is None:
                        _stream_range(source, tensor.offset, tensor.size, output)
                    else:
                        row = tensor.size // tensor.shape[0]
                        for expert in ids:
                            _stream_range(source, tensor.offset + expert * row, row, output)
    return position


def _publish_reserved(stage, destination):
    """Reserve a private output and commit config last; visibility is not atomic."""
    import warnings

    stage, destination = Path(stage), Path(destination)
    if not (stage / "config.json").is_file() or (stage / "config.json").is_symlink():
        raise ValueError("reserved publication requires a regular config.json")
    destination.mkdir(mode=0o700, exist_ok=False)
    marker = destination / ".razor-incomplete"
    with marker.open("x", encoding="utf-8") as handle:
        handle.write("Checkpoint publication is incomplete; do not load or reuse this directory.\n")
    warnings.warn("atomic no-overwrite directory rename is unavailable; using an exclusively reserved "
                  "directory with .razor-incomplete and config.json committed last. Directory visibility "
                  "is NOT atomic; interrupted exports retain the incomplete marker.", RuntimeWarning, stacklevel=2)

    def link(source, target):
        if source.is_symlink():
            raise ValueError("reserved publication refuses symbolic links in the staging directory")
        if source.is_dir():
            target.mkdir(mode=0o700, exist_ok=False)
            for child in sorted(source.iterdir()):
                link(child, target / child.name)
        elif source.is_file():
            os.link(source, target, follow_symlinks=False)
        else:
            raise ValueError("reserved publication requires regular files and directories")

    for path in sorted(stage.iterdir(), key=lambda path: (path.name == "config.json", path.name)):
        link(path, destination / path.name)
    try:
        marker.unlink()
    except FileNotFoundError:
        if marker.exists() or marker.is_symlink():
            raise


def _publish(stage, destination):
    """Never replace output; prefer atomic rename, otherwise reserve an incomplete output."""
    if os.name == "nt":
        os.rename(stage, destination)
        return
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        result = function(-100, os.fsencode(stage), -100, os.fsencode(destination), 1)
    elif hasattr(libc, "renamex_np"):
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = function(os.fsencode(stage), os.fsencode(destination), 4)
    else:
        raise OSError("atomic no-overwrite directory rename is unavailable on this platform")
    if result:
        import errno

        error = ctypes.get_errno()
        if error in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}:
            _publish_reserved(stage, destination)
            return
        raise OSError(error, os.strerror(error), str(destination))


def _check_publish(parent):
    """Probe safe publication on the destination filesystem before copying weights."""
    import warnings

    directory = Path(tempfile.mkdtemp(prefix=".razor-probe-", dir=parent))
    try:
        source, destination = directory / "source", directory / "destination"
        source.mkdir()
        (source / "config.json").write_text("{}", encoding="utf-8")
        destination.mkdir()
        try:
            _publish(source, destination)
        except FileExistsError:
            pass
        else:
            raise OSError("target filesystem does not enforce no-overwrite publication")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="atomic no-overwrite directory rename is unavailable", category=RuntimeWarning)
            _publish(source, directory / "published")
    finally:
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            if directory.exists() or directory.is_symlink():
                raise


def prune_checkpoint(model_path, out_dir, keep, method="unknown", *, target_experts=None,
                     target_top_k=None, hash_policy="remap", mtp_policy="router_norm", trust_remote_code=False,
                     verbose=False, max_shard_size="5GB", verify=True,
                     aggregation=metrics.DEFAULT_AGGREGATION):
    """Export without replacement; return metadata and the completed keep map.

    Publication uses atomic no-replace rename where supported. Other supported
    filesystems use an exclusive directory and incomplete marker, with config
    committed last; that fallback is explicitly not atomically visible.
    Missing ordinary decoder layers always fail. Missing hash layers use table
    frequency; missing MTP layers use router row norms only with ``router_norm``.
    ``preserve`` hash mode bundles a local Transformers mixed-width implementation
    and requires explicit remote-code consent. No code is executed during export.
    """
    source, destination = Path(model_path).resolve(), Path(out_dir).absolute()
    resolved = destination.resolve()
    if resolved == source or resolved.is_relative_to(source) or source.is_relative_to(resolved):
        raise ValueError("source and output directories must not contain one another")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("output already exists; choose a new directory")
    if not isinstance(aggregation, str) or not aggregation or not re.fullmatch(r"[A-Za-z0-9_-]+", aggregation):
        raise ValueError("aggregation must be a nonempty public method name")
    if isinstance(keep, dict) and not keep:
        raise ValueError("keep must be a nonempty mapping; target_experts alone cannot select hash/MTP-only checkpoints")
    scan = _scan(source)
    meta, tensors, classes, gates, hashes, mtp = scan
    selected, budget, top_k, full = _selection(scan, keep, target_top_k, hash_policy, mtp_policy)
    if target_experts is not None and _integer(target_experts, "target_experts") != budget:
        raise ValueError("target_experts disagrees with the retained keep-set size")
    if full and (meta["family"] != "deepseek_v4" or not trust_remote_code):
        raise ValueError("preserve hash policy needs DeepSeek-V4 and trust_remote_code=True")
    config = _config(meta, budget, top_k, hash_policy, mtp_policy)
    limit = _shard_bytes(max_shard_size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _check_publish(destination.parent)
    remaps = {}
    for tag, key in hashes.items():
        if tag not in full and not (mtp_policy == "drop" and tag in mtp):
            remaps[tag] = remap_hash_table(_load(tensors[key], key), _load(tensors[gates[tag]], gates[tag]), selected[tag])
    entries, new_keys = [], set()
    for key, tensor in tensors.items():
        kind, tag, expert = classes[key]
        if mtp_policy == "drop" and (tag in mtp or key.startswith("mtp.")):
            continue
        new_key, ids, replacement = key, None, None
        if kind and tag not in full:
            if kind == "expert":
                if expert not in selected[tag]:
                    continue
                new_key = _renumber(key, selected[tag].index(expert))
            elif kind in ("router", "fused"):
                if tensor.size % tensor.shape[0]:
                    raise ValueError(f"tensor rows are not byte aligned: {key}")
                ids = selected[tag]
            elif kind == "hash":
                replacement = remaps[tag]
        if new_key in new_keys:
            raise ValueError("renumbering generated duplicate tensor keys")
        new_keys.add(new_key)
        entries.append((new_key, tensor, ids, replacement))
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".razor-", dir=destination.parent))
    info = dict(method=method, aggregation=aggregation, family=meta["family"], original_experts=meta["num_experts"],
                kept_experts=budget, experts_per_tok=top_k, hash_policy=hash_policy,
                mtp_policy=mtp_policy, keep=selected, tensor_only=True,
                storage_profile="safetensors_raw_v1")
    try:
        from .prune import _public_config

        aux_config = _config(meta, budget, top_k, "remap", mtp_policy) if full else config
        _copy_aux(source, stage, trust_remote_code=trust_remote_code, config=aux_config)
        if full:
            (stage / "configuration_deepseek_v4_mixed.py").write_text(_MIXED_CONFIG, encoding="utf-8")
            (stage / "modeling_deepseek_v4_mixed.py").write_text(_MIXED_MODEL, encoding="utf-8")
        config = _public_config(config, code_root=stage)
        weight_map, buffer, size, total, index = {}, [], 0, 0, 0

        def flush():
            nonlocal buffer, size, total, index
            if not buffer:
                return
            index += 1
            name = f"model-{index:05d}.safetensors"
            total += _write_shard(stage / name, buffer)
            weight_map.update({entry[0]: name for entry in buffer})
            buffer, size = [], 0

        for entry in entries:
            tensor, ids = entry[1], entry[2]
            length = tensor.size if ids is None else tensor.size // tensor.shape[0] * len(ids)
            if buffer and size + length > limit:
                flush()
            buffer.append(entry)
            size += length
        flush()
        for name, value in (("config.json", config), ("kept_expert_indices.json", selected),
                            ("pruning_info.json", info), ("model.safetensors.index.json", {"metadata": {"total_size": total}, "weight_map": weight_map})):
            (stage / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        output_meta = _scan(stage, require_complete=True)[0]
        if (output_meta["counts"] != {tag: len(ids) for tag, ids in selected.items()}
                or output_meta["num_experts"] != budget or output_meta["top_k"] != top_k):
            raise ValueError("output configuration and stored expert widths disagree with the keep map")
        result = verify_checkpoint(source, stage, selected, hash_policy=hash_policy, mtp_policy=mtp_policy, target_top_k=top_k) if verify else None
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("output appeared during export")
        _publish(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    if verbose:
        print(f"[checkpoint] wrote {len(weight_map)} tensors in {index} shards")
    return dict(info, out_dir=str(destination), verification=result)


def _equal_ranges(left, left_offset, right, right_offset, length):
    with left.open("rb") as a, right.open("rb") as b:
        a.seek(left_offset)
        b.seek(right_offset)
        while length:
            count = min(length, _CHUNK)
            x, y = a.read(count), b.read(count)
            if len(x) != count or x != y:
                return False
            length -= count
    return True


def _verify_hash(original, output, weights, keep):
    """Reference check intentionally independent of the writer's remapping routine."""
    if output.shape != original.shape or output.dtype != original.dtype:
        return False
    weights = weights.float()
    weights = weights / weights.norm(dim=1, keepdim=True).clamp_min(1e-12)
    similarities = (weights @ weights[keep].T).tolist()
    inverse = {old: new for new, old in enumerate(keep)}
    for before, after in zip(original.tolist(), output.tolist()):
        if len(set(after)) != len(after) or any(x < 0 or x >= len(keep) for x in after):
            return False
        expected, occupied = [None] * len(before), set()
        for slot, old in enumerate(before):
            candidate = inverse.get(old)
            if candidate is not None and candidate not in occupied:
                expected[slot] = candidate
                occupied.add(candidate)
        for slot, old in enumerate(before):
            if expected[slot] is None:
                available = [new for new in range(len(keep)) if new not in occupied]
                best = max(available, key=lambda new: (similarities[old][new], -new))
                expected[slot] = best
                occupied.add(best)
        if after != expected:
            return False
    return True


def verify_checkpoint(model_path, out_dir, keep=None, *, hash_policy=None, mtp_policy=None,
                      target_top_k=None):
    """Verify tensor storage, configuration, keep manifests and generated code.

    The on-disk keep manifest is required and must be complete; any external keep
    and pruning_info.keep must agree with it without automatic selection. Missing
    pruning_info is allowed: supply nondefault policies/top-k explicitly then.
    When present, its count, family, policy and storage declarations are checked;
    method/aggregation provenance is not inferred from checkpoint bytes.

    Tensor rows and hash remaps are checked independently of writer transforms.
    Generated mixed-width templates are compared as bytes, never executed. This
    does not verify tokenizer/other auxiliary files (including copied custom
    code), loadability, or model-forward numerical equivalence.
    """
    source, destination = _complete_root(model_path), _complete_root(out_dir)
    scan = _scan(source)
    meta, tensors, classes, gates, hashes, mtp = scan
    info_path = _contained(destination, "pruning_info.json")
    has_info = info_path.exists() or info_path.is_symlink()
    info = _json(info_path) if has_info else {}
    if not isinstance(info, dict):
        raise ValueError("pruning_info must be an object")
    hash_policy = info.get("hash_policy", "remap") if hash_policy is None else hash_policy
    mtp_policy = info.get("mtp_policy", "router_norm") if mtp_policy is None else mtp_policy
    if target_top_k is None:
        target_top_k = info.get("experts_per_tok")

    def manifest(value, name):
        expected_layers = set(meta["counts"]) - (mtp if mtp_policy == "drop" else set())
        if (not isinstance(value, dict) or set(value) != expected_layers
                or any(not isinstance(ids, list) for ids in value.values())):
            raise ValueError(f"{name}: keep manifest must list every retained layer and no dropped layers")
        return _selection(scan, value, target_top_k, hash_policy, mtp_policy)

    keep_path = _contained(destination, "kept_expert_indices.json")
    if not keep_path.is_file():
        raise ValueError("output keep manifest kept_expert_indices.json is required")
    selected, budget, top_k, full = manifest(_json(keep_path), "output")
    if isinstance(keep, (str, os.PathLike)):
        keep = _json(keep)
    if keep is not None and manifest(keep, "external")[0] != selected:
        raise ValueError("external keep manifest disagrees with the output keep manifest")
    if has_info:
        if manifest(info.get("keep"), "pruning_info")[0] != selected:
            raise ValueError("pruning_info keep manifest disagrees with the output keep manifest")
        expected_info = dict(family=meta["family"], original_experts=meta["num_experts"],
                             kept_experts=budget, experts_per_tok=top_k, hash_policy=hash_policy,
                             mtp_policy=mtp_policy, tensor_only=True, storage_profile="safetensors_raw_v1")
        for field, value in expected_info.items():
            if type(info.get(field)) is not type(value) or info.get(field) != value:
                raise ValueError(f"pruning_info {field} disagrees with the verified storage selection")
    checked_code = 0
    if full:
        for name, template in (("configuration_deepseek_v4_mixed.py", _MIXED_CONFIG),
                               ("modeling_deepseek_v4_mixed.py", _MIXED_MODEL)):
            path = _contained(destination, name)
            if not path.is_file():
                raise ValueError(f"mixed-width checkpoint is missing generated source: {name}")
            expected = template.encode("utf-8")
            with path.open("rb") as handle:
                if handle.read(len(expected) + 1) != expected:
                    raise ValueError(f"mixed-width generated source differs from the required template: {name}")
            checked_code += 1
    output_scan = _scan(destination, require_complete=True)
    output_meta, output = output_scan[:2]
    if (output_meta["counts"] != {tag: len(ids) for tag, ids in selected.items()}
            or output_meta["num_experts"] != budget or output_meta["top_k"] != top_k):
        raise ValueError("output configuration and stored expert widths disagree with the keep map")
    expected_keys, checked = set(), 0
    for key, tensor in tensors.items():
        kind, tag, expert = classes[key]
        if mtp_policy == "drop" and (tag in mtp or key.startswith("mtp.")):
            continue
        new_key = key
        if kind == "expert" and tag not in full:
            if expert not in selected[tag]:
                continue
            pieces = key.split(".")
            position = pieces.index("experts") + 1
            pieces[position] = str(selected[tag].index(expert))
            new_key = ".".join(pieces)
        expected_keys.add(new_key)
        if new_key not in output:
            raise ValueError(f"missing output tensor: {new_key}")
        actual = output[new_key]
        shape = list(tensor.shape)
        row_select = kind in ("router", "fused") and tag not in full
        if row_select:
            shape[0] = len(selected[tag])
        if tuple(shape) != actual.shape or actual.dtype != tensor.dtype:
            raise ValueError(f"output tensor shape/dtype mismatch: {new_key}")
        if kind == "hash" and tag not in full:
            equal = _verify_hash(_load(tensor, key), _load(actual, new_key), _load(tensors[gates[tag]], gates[tag]), selected[tag])
        elif row_select:
            row_bytes = tensor.size // tensor.shape[0]
            equal = actual.size == row_bytes * len(selected[tag]) and all(
                _equal_ranges(tensor.file, tensor.offset + old * row_bytes, actual.file, actual.offset + new * row_bytes, row_bytes)
                for new, old in enumerate(selected[tag]))
        else:
            equal = tensor.size == actual.size and _equal_ranges(tensor.file, tensor.offset, actual.file, actual.offset, tensor.size)
        if not equal:
            raise ValueError(f"output tensor bytes differ from expected source: {new_key}")
        checked += 1
    if set(output) != expected_keys:
        raise ValueError("output contains unexpected tensor keys")
    from .prune import _public_config

    config = _json(_contained(destination, "config.json"))
    expected_config = _public_config(_config(meta, budget, top_k, hash_policy, mtp_policy), code_root=destination)
    if config != expected_config:
        raise ValueError("output config differs from permitted count/top-k/MTP changes")
    return {"ok": True, "checked_tensors": checked, "checked_pruning_info": has_info,
            "checked_generated_code": checked_code, "auxiliary_files_verified": False,
            "scope": "all tensor bytes, config, complete keep manifests, storage declarations when present, "
                     "generated mixed-width code; excludes tokenizer/other auxiliary files and forward execution"}


_MIXED_CONFIG = '''"""Mixed-width DeepSeek-V4 configuration."""
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config


class DeepseekV4MixedConfig(DeepseekV4Config):
    model_type = "deepseek_v4"

    def __init__(self, hash_n_routed_experts=None, hash_layer_indices=None, **kwargs):
        super().__init__(**kwargs)
        self.hash_n_routed_experts = self.n_routed_experts if hash_n_routed_experts is None else hash_n_routed_experts
        self.hash_layer_indices = hash_layer_indices
'''

_MIXED_MODEL = '''"""Upstream DeepSeek-V4 operations with layer-specific expert widths."""
import contextlib
import torch
from transformers.integrations import use_experts_implementation
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Experts, DeepseekV4ForCausalLM, DeepseekV4HashRouter,
    DeepseekV4MLP, DeepseekV4Model, DeepseekV4SparseMoeBlock, DeepseekV4TopKRouter,
)
from .configuration_deepseek_v4_mixed import DeepseekV4MixedConfig


@use_experts_implementation
class DeepseekV4MixedExperts(DeepseekV4Experts):
    pass


@contextlib.contextmanager
def _as_width(config, count):
    old = config.n_routed_experts
    config.n_routed_experts = count
    try:
        yield config
    finally:
        config.n_routed_experts = old


class DeepseekV4MixedSparseMoeBlock(DeepseekV4SparseMoeBlock):
    def __init__(self, config, layer_idx):
        torch.nn.Module.__init__(self)
        self.is_hash = config.mlp_layer_types[layer_idx] == "hash_moe"
        indices = config.hash_layer_indices
        if indices is not None and self.is_hash != (layer_idx in indices):
            raise ValueError("hash layer configuration disagrees with checkpoint tensor layout")
        count = config.hash_n_routed_experts if self.is_hash else config.n_routed_experts
        with _as_width(config, count):
            self.gate = DeepseekV4HashRouter(config) if self.is_hash else DeepseekV4TopKRouter(config)
            self.experts = DeepseekV4MixedExperts(config)
        self.shared_experts = DeepseekV4MLP(config)


class DeepseekV4MixedModel(DeepseekV4Model):
    config_class = DeepseekV4MixedConfig

    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.layers):
            if isinstance(getattr(layer, "mlp", None), DeepseekV4SparseMoeBlock):
                reference = layer.mlp.experts.gate_up_proj
                layer.mlp = DeepseekV4MixedSparseMoeBlock(config, i).to(device=reference.device, dtype=reference.dtype)


class DeepseekV4MixedForCausalLM(DeepseekV4ForCausalLM):
    config_class = DeepseekV4MixedConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekV4MixedModel(config)
        self.post_init()
'''
