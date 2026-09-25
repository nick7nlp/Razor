'Expert selection and checkpoint export.'
from __future__ import annotations

import ast
import ipaddress
import json
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from urllib.parse import parse_qsl, urlsplit

import torch

from . import metrics
from .adapters import MoEAdapter


def resolve_target_experts(num_experts: int, target_experts: Optional[int],
                           ratio: Optional[float]) -> int:
    'Resolve a removal ratio or an integer number of experts to retain.'
    if (target_experts is None) == (ratio is None):
        raise ValueError("specify exactly one of target_experts / ratio")
    if ratio is not None:
        if isinstance(ratio, bool) or not 0.0 < ratio < 1.0:
            raise ValueError("ratio must be in (0, 1)")
        target_experts = max(1, round(num_experts * (1.0 - ratio)))
    if (isinstance(target_experts, bool) or not isinstance(target_experts, int)
            or not 0 < target_experts < num_experts):
        raise ValueError(f"target_experts must be an integer in [1, {num_experts - 1}]")
    return target_experts


def validate_keep_indices(keep, counts: Dict[str, int], top_k: int = 1):
    'Validate all layer selections before changing any tensor.'
    if not isinstance(keep, dict) or not keep:
        raise ValueError("keep indices must be a non-empty layer mapping")
    if set(keep) != set(counts):
        raise ValueError("keep-set layers must exactly match the model's MoE layers")
    checked = {}
    for tag, values in keep.items():
        if (not isinstance(values, (list, tuple)) or not values
                or any(isinstance(i, bool) or not isinstance(i, int) for i in values)):
            raise ValueError(f"{tag}: keep indices must be non-empty integer lists")
        if len(set(values)) != len(values) or min(values) < 0 or max(values) >= counts[tag]:
            raise ValueError(f"{tag}: duplicate or out-of-range expert index")
        if len(values) < top_k:
            raise ValueError(f"{tag}: retained expert count must be at least top_k={top_k}")
        checked[tag] = list(values)
    if len({len(v) for v in checked.values()}) != 1:
        raise NotImplementedError("non-uniform expert counts require a dedicated adapter")
    return checked


def select_keep_indices(saliency: Dict[str, dict], method: str,
                        target_experts: int,
                        verbose: bool = True, n_group: int = 1,
                        aggregation: str = metrics.DEFAULT_AGGREGATION) -> Dict[str, List[int]]:
    """Select experts, retaining an equal quota in each routing group."""
    if not isinstance(saliency, dict) or not saliency:
        raise ValueError("saliency must contain at least one layer")
    out: Dict[str, List[int]] = {}
    for tag, rec in saliency.items():
        if "total_tokens" in rec and int(rec["total_tokens"]) <= 0:
            raise ValueError(f"{tag}: no valid calibration tokens")
        if "expert_frequency" in rec and not bool(rec["expert_frequency"].sum() > 0):
            raise ValueError(f"{tag}: no routed calibration tokens")
        selected = metrics.keep_indices(rec, method, target_experts, aggregation=aggregation)
        if n_group > 1:
            scores = metrics.score(rec, method, aggregation=aggregation)
            if scores.numel() % n_group or target_experts % n_group:
                raise ValueError("expert budget must be divisible by the routing group count")
            width = scores.numel() // n_group
            local = torch.argsort(scores.reshape(n_group, width), dim=1,
                                  descending=True, stable=True)[:, :target_experts // n_group]
            ids = local + torch.arange(n_group, device=local.device).unsqueeze(1) * width
            selected = sorted(ids.flatten().tolist())
        out[tag] = selected
    return out


def fill_unobserved_layers(keep: Dict[str, List[int]], tags: List[str],
                           saliency: Optional[Dict[str, dict]], method: str,
                           target_experts: int,
                           skip: Sequence[str] = (),
                           verbose: bool = True, n_group: int = 1,
                           aggregation: str = metrics.DEFAULT_AGGREGATION) -> Dict[str, List[int]]:
    'Give a keep-set to MoE layers the calibration pass never exercised.'
    skipped = set(skip)
    missing = [t for t in tags if t not in keep and t not in skipped]
    if not missing:
        return keep
    if any(not t.startswith("mtp_") for t in missing):
        raise ValueError("saliency is missing decoder MoE layers; refusing automatic selection")

    if not saliency:
        raise KeyError(
            f"{len(missing)} MoE layers have no keep-set and no saliency is "
            f"available to derive one from (first few: {missing[:5]}). "
            "Supply --saliency, or extend the keep-indices JSON to cover them."
        )

    scores = []
    for rec in saliency.values():
        try:
            scores.append(metrics.score(rec, method, aggregation=aggregation).double())
        except KeyError:
            continue
    if not scores:
        raise KeyError(
            f"cannot derive a fallback score for {missing}: the saliency file "
            f"has no usable '{method}' field"
        )
    width = scores[0].numel()
    scores = [s for s in scores if s.numel() == width]
    mean = torch.stack(scores).mean(dim=0)
    proxy = select_keep_indices({"mtp": {metrics.resolve(method, aggregation): mean}}, method,
                                target_experts, verbose=False, n_group=n_group,
                                aggregation=aggregation)["mtp"]

    keep = dict(keep)
    for tag in missing:
        keep[tag] = proxy
    if verbose:
        print(f"[prune] {len(missing)} MoE layer(s) were never exercised by "
              f"the calibration pass ({', '.join(missing[:4])}"
              f"{' ...' if len(missing) > 4 else ''}); pruning them with the "
              f"mean '{method}' score over the {len(scores)} observed layers. "
              "They must be pruned for the checkpoint to load.")
    return keep


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


_AUX_ASSETS = frozenset({
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "chat_template.jinja", "chat_template.json", "generation_config.json",
    "vocab.json", "vocab.txt", "merges.txt", "added_tokens.json", "tokenizer.model",
    "spiece.model", "sentencepiece.bpe.model", "tokenizer.tiktoken",
    "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json",
})
_LICENSE_FILES = frozenset({"LICENSE", "LICENSE.txt", "LICENSE.md", "NOTICE", "NOTICE.txt", "COPYING"})
_ASSET_SUBDIRECTORIES = ("tokenizer", "processor")


def _auxiliary_assets(root):
    assets = {Path(name) for name in _AUX_ASSETS | _LICENSE_FILES}
    assets.update(Path(directory) / name for directory in _ASSET_SUBDIRECTORIES
                  for name in _AUX_ASSETS | _LICENSE_FILES)
    for directory in (Path("."), *(Path(name) for name in _ASSET_SUBDIRECTORIES)):
        assets.update(path.relative_to(root) for path in (root / directory / "chat_templates").glob("*.jinja")
                      if not path.name.startswith("."))
    for suffix in ("*.tiktoken", "*.model"):
        assets.update(path.relative_to(root) for path in (root / "encoding").glob(suffix)
                      if not path.name.startswith("."))
    return assets


def _asset_path(root, relative, *, output=False):
    from .checkpoint import _contained

    path = _contained(Path(root), relative)
    if output and (path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve())):
        raise ValueError("checkpoint output asset must not follow a symlink")
    if path.exists() and not path.is_file():
        raise ValueError("checkpoint asset must be a regular file")
    return path


def _write_aux_asset(root, relative, content):
    """Publish a fresh inode without following mutable output directory links."""
    if os.name != "posix" or not all(hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW")):
        raise OSError("safe auxiliary export requires POSIX directory-descriptor operations")
    root, relative = Path(root), Path(relative)
    _asset_path(root, relative, output=True)
    root.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open(root, flags)
    temporary = None
    try:
        for part in relative.parent.parts:
            try:
                os.mkdir(part, dir_fd=directory)
            except FileExistsError:
                pass
            child = os.open(part, flags, dir_fd=directory)
            os.close(directory)
            directory = child
        try:
            existing = os.stat(relative.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ValueError("checkpoint output asset must be a regular file, not a symlink")
        name = ".razor-asset-" + uuid.uuid4().hex
        descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o666, dir_fd=directory)
        temporary = name
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        if existing is None:
            os.link(temporary, relative.name, src_dir_fd=directory, dst_dir_fd=directory,
                    follow_symlinks=False)
        else:
            os.replace(temporary, relative.name, src_dir_fd=directory, dst_dir_fd=directory)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
        os.close(directory)


def _source_tree(root, relative):
    path = _asset_path(root, relative)
    try:
        return ast.parse(path.read_bytes())
    except (SyntaxError, UnicodeError) as error:
        raise ValueError("checkpoint Python source cannot be inspected safely") from error


def _code_reference(reference):
    if not isinstance(reference, str):
        raise ValueError("auto_map entries must name a local Python class")
    local = reference.rsplit("--", 1)[-1]
    if not re.fullmatch(r"[A-Za-z_]\w*\.[A-Za-z_]\w*", local):
        raise ValueError("auto_map must name a supported local module.Class")
    module, name = local.split(".")
    return local, Path(module + ".py"), name


def _localize_auto_map(value, code_root):
    if not isinstance(value, dict):
        raise ValueError("auto_map must be a mapping")

    def localize(reference):
        if reference is None:
            return None
        if code_root is None:
            raise ValueError("auto_map requires explicitly authorized, copied local Python classes")
        local, relative, name = _code_reference(reference)
        path = _asset_path(code_root, relative)
        if not path.is_file() or not any(isinstance(node, ast.ClassDef) and node.name == name
                                         for node in _source_tree(code_root, relative).body):
            raise ValueError("auto_map class is not available in copied local source")
        closure = _code_closure(code_root, [relative])
        direct = {relative}
        for node in ast.walk(_source_tree(code_root, relative)):
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.level != 1 or not node.module or not node.module.isidentifier():
                    raise ValueError("auto_map dependency layout is unsupported by local Transformers loading")
                direct.add(Path(node.module + ".py"))
        if not closure.issubset(direct):
            raise ValueError("auto_map requires its static dependencies as direct root-module imports; source review required")
        return local

    return {key: [localize(item) for item in value] if isinstance(value, (list, tuple))
            else localize(value) for key, value in value.items()}


def _code_roots(config):
    if isinstance(config, dict):
        for key, value in config.items():
            if key == "auto_map":
                if not isinstance(value, dict):
                    raise ValueError("auto_map must be a mapping")
                for references in value.values():
                    for reference in references if isinstance(references, (list, tuple)) else [references]:
                        if reference is not None:
                            yield _code_reference(reference)
            elif key not in {"metadata", "provenance", "credentials"}:
                yield from _code_roots(value)
    elif isinstance(config, list):
        for value in config:
            yield from _code_roots(value)


def _code_closure(root, roots):
    """Resolve static relative imports without importing or rewriting checkpoint code."""
    selected, pending = set(), list(roots)

    def module_path(parts):
        if not parts:
            initializer = Path("__init__.py")
            return initializer if _asset_path(root, initializer).is_file() else None
        if any(not part.isidentifier() for part in parts):
            return None
        for candidate in (Path(*parts).with_suffix(".py"), Path(*parts) / "__init__.py"):
            if _asset_path(root, candidate).is_file():
                return candidate
        return None

    while pending:
        relative = pending.pop()
        if relative in selected:
            continue
        tree = _source_tree(root, relative)
        selected.add(relative)
        for parent in relative.parents:
            if parent == Path("."):
                continue
            initializer = parent / "__init__.py"
            if _asset_path(root, initializer).is_file():
                pending.append(initializer)
        package = relative.parent.parts
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (
                    isinstance(node.func, ast.Name) and node.func.id == "__import__"
                    or isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"):
                raise ValueError("dynamic checkpoint imports need explicit source review; static export is unsupported")
            absolute = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                        else [node.module] if isinstance(node, ast.ImportFrom) and not node.level else [])
            if any(module_path(name.split(".")) is not None for name in absolute if name):
                raise ValueError("local absolute checkpoint imports need explicit source review; use package-relative imports")
            if not isinstance(node, ast.ImportFrom) or not node.level:
                continue
            if node.level > len(package) + 1:
                raise ValueError("checkpoint Python import escapes its source package")
            base = package[:len(package) - node.level + 1]
            parts = (*base, *(node.module.split(".") if node.module else ()))
            dependency = module_path(parts)
            if node.module and dependency is None:
                raise ValueError("checkpoint Python source has an unavailable relative dependency")
            if dependency is not None:
                pending.append(dependency)
            for alias in node.names:
                child = module_path((*parts, alias.name)) if alias.name != "*" else None
                if child is not None:
                    pending.append(child)
                elif not node.module and dependency is None:
                    raise ValueError("checkpoint Python source has an unavailable relative dependency")
    return selected


def copy_aux_files(src: Path, dst: Path, adapter: Optional[MoEAdapter] = None,
                   remote_code: bool = False, verbose: bool = True, *, config=None) -> None:
    """Copy defined assets and explicitly authorized static Python dependencies.

    Required code, token text and licenses are preserved byte-for-byte, not
    advertised as sanitized arbitrary prose. Unsupported code references fail
    closed rather than leaving a checkpoint pointing to unavailable sources.
    """
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        from huggingface_hub import snapshot_download

        patterns = sorted(_AUX_ASSETS | _LICENSE_FILES | {"config.json", "chat_templates/*.jinja",
                                                        "encoding/*.tiktoken", "encoding/*.model"})
        patterns.extend(f"{directory}/{name}" for directory in _ASSET_SUBDIRECTORIES
                        for name in _AUX_ASSETS | _LICENSE_FILES | {"chat_templates/*.jinja"})
        if remote_code:
            patterns.extend(["*.py", "**/*.py"])
        src = Path(snapshot_download(str(src), allow_patterns=patterns))
    source_root, target_root = src.resolve(), dst.resolve()
    if source_root == target_root or source_root.is_relative_to(target_root) or target_root.is_relative_to(source_root):
        raise ValueError("source and auxiliary output directories must not contain one another")
    assets = _auxiliary_assets(src)
    configs = {}
    for relative in sorted(assets):
        path = _asset_path(src, relative)
        if path.is_file() and relative.name.endswith("_config.json"):
            value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
            if relative.parent != Path(".") and list(_code_roots(_public_config(value, defer_auto_map=True))):
                raise ValueError("nested auxiliary auto_map is unsupported; use a root-level configuration")
            configs[str(relative)] = value
    if config is None:
        path = _asset_path(src, "config.json")
        config = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object) if path.is_file() else {}
    declarations = [_public_config(value, defer_auto_map=True) for value in [config, *configs.values()]]
    if any(not isinstance(value, dict) for value in declarations):
        raise ValueError("checkpoint configurations must be JSON objects")
    references = [reference for value in declarations for reference in _code_roots(value)]
    if references and not remote_code:
        raise ValueError("checkpoint auto_map requires trust_remote_code=True to copy its Python classes")
    roots = set()
    encoder = Path("encoding/encoding_dsv4.py")
    if _asset_path(src, encoder).is_file():
        if not remote_code:
            raise ValueError("native checkpoint encoder requires trust_remote_code=True for export")
        roots.add(encoder)
        if _asset_path(src, "__init__.py").is_file():
            roots.add(Path("__init__.py"))
    if remote_code:
        for reference, relative, _ in references:
            _localize_auto_map({"entry": reference}, src)
            roots.add(relative)
    architecture_names = set()
    for value in declarations:
        for current in (value, value.get("text_config", {})):
            if not isinstance(current, dict):
                continue
            architecture_names.update(current.get("architectures") or [])
            architecture_names.update(current[key] for key in ("tokenizer_class", "processor_class")
                                      if isinstance(current.get(key), str))
    if architecture_names:
        matches = {}
        for path in sorted(src.glob("*.py")):
            if not path.name.startswith(("modeling_", "configuration_", "tokenization_", "processing_")):
                continue
            relative = path.relative_to(src)
            for node in _source_tree(src, relative).body:
                if isinstance(node, ast.ClassDef) and node.name in architecture_names:
                    matches.setdefault(node.name, set()).add(relative)
        if matches and not remote_code:
            raise ValueError("checkpoint-declared local classes require trust_remote_code=True for export")
        if any(len(paths) > 1 for paths in matches.values()):
            raise ValueError("checkpoint architecture has ambiguous local Python declarations")
        roots.update(relative for paths in matches.values() for relative in paths)
    code = _code_closure(src, roots)
    for relative in sorted(code):
        _write_aux_asset(dst, relative, _asset_path(src, relative).read_bytes())
    copied = []
    assets.update(parent / name for relative in code for parent in relative.parents for name in _LICENSE_FILES)
    for relative in sorted(assets):
        source = _asset_path(src, relative)
        if not source.is_file():
            continue
        if str(relative) in configs:
            value = _public_config(configs[str(relative)], code_root=dst)
            content = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        else:
            content = source.read_bytes()
        _write_aux_asset(dst, relative, content)
        copied.append(relative)
    _public_config(config, code_root=dst)
    if verbose and (copied or code):
        print(f"[prune] copied {len(copied)} public auxiliary assets and {len(code)} required source files")


def prune_model(model, adapter: MoEAdapter, keep: Dict[str, List[int]],
                skip: Sequence[str] = (), verbose: bool = True,
                target_top_k: Optional[int] = None) -> Dict[str, List[int]]:
    'Validate and slice every MoE block, then synchronize its configuration.'
    if skip:
        raise NotImplementedError("skipping layers requires per-layer expert-count support")
    blocks = adapter.moe_blocks(model)
    if not blocks:
        raise RuntimeError("no MoE blocks found in the model")
    k = adapter.top_k if target_top_k is None else target_top_k
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= adapter.top_k:
        raise ValueError("target_top_k must be an integer between 1 and the original top_k")
    counts = {b.tag: int(adapter.gate_module(b.module).weight.shape[0]) for b in blocks}
    checked = validate_keep_indices(keep, counts, k)
    n = len(next(iter(checked.values())))
    indices = {}
    for b in blocks:
        idx = torch.tensor(checked[b.tag], dtype=torch.long,
                           device=adapter.gate_module(b.module).weight.device)
        adapter.validate_keep(b.module, idx, target_top_k=k)
        indices[b.tag] = idx
    for b in blocks:
        adapter.prune_block(b.module, indices[b.tag], target_top_k=k)
        for module in b.module.modules():
            for attr in ("top_k", "num_experts_per_tok", "moe_topk"):
                if isinstance(getattr(module, attr, None), int):
                    setattr(module, attr, k)
        if hasattr(adapter.gate_module(b.module), "out_features"):
            adapter.gate_module(b.module).out_features = n
    adapter.set_num_experts(n)
    adapter.set_top_k(k)
    if verbose:
        print(f"[prune] sliced {len(checked)} MoE layers to {n} experts, top_k={k}")
    return checked


def _private_location(value):
    if (value.startswith(("/", "~", "./", "../", "file://", "\\\\", ".\\", "..\\"))
            or re.match(r"^[A-Za-z]:[\\/]", value)
            or re.search(r"(?:^|[/\\])\.\.(?:[/\\]|$)", value)
            or re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", value)):
        return True
    if "://" not in value:
        return False
    try:
        url = urlsplit(value)
        host = (url.hostname or "").lower().rstrip(".")
        if url.username or url.password or not host or "." not in host:
            return True
        if host.endswith((".internal", ".local", ".localhost", ".corp", ".intranet", ".lan")):
            return True
        if any(re.sub(r"[^a-z0-9]", "", key.lower()) in {
                "token", "accesstoken", "apikey", "signature", "credential", "password", "secret"}
               for key, _ in parse_qsl(url.query)):
            return True
        try:
            return not ipaddress.ip_address(host).is_global
        except ValueError:
            return False
    except ValueError:
        return True


def _public_config(value, *, literal_text: bool = False, code_root=None,
                   defer_auto_map: bool = False):
    """Remove provenance and credentials, retaining algorithm and literal token data.

    ``defer_auto_map`` is for in-memory planning only. Exported configurations
    must validate references against ``code_root`` after source files are copied.
    """
    private_keys = {"nameorpath", "pruninginfo", "sourcemodel", "saliencyfile",
                    "keepindicesfile", "calibrationdata", "token", "accesstoken",
                    "apikey", "password", "secret", "secrets", "credential", "credentials",
                    "hftoken", "authtoken", "apitoken", "bearertoken", "secretkey", "clientsecret",
                    "accesskey", "accesskeyid", "secretaccesskey", "sessiontoken", "refreshtoken",
                    "metadata", "provenance", "auth", "authorization", "authentication",
                    "headers", "email", "contact", "username", "owner", "createdby",
                    "author", "authors", "checkpoint",
                    "cachedir", "checkpointpath", "trainingargs", "trainerstate",
                    "source", "origin", "repoid", "repositoryid", "commit", "commithash"}
    text_keys = {"chat_template", "added_tokens_decoder", "additional_special_tokens",
                 "extra_special_tokens", "special_tokens_map", "bos_token", "eos_token",
                 "unk_token", "pad_token", "sep_token", "cls_token", "mask_token"}
    if literal_text:
        return value
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            if normalized in private_keys or normalized.endswith(("metadata", "provenance", "credentials")):
                continue
            if key == "auto_map":
                result[key] = item if defer_auto_map else _localize_auto_map(item, code_root)
            else:
                result[key] = _public_config(item, literal_text=key in text_keys,
                                              code_root=code_root, defer_auto_map=defer_auto_map)
        return result
    if isinstance(value, list):
        return [_public_config(item, code_root=code_root, defer_auto_map=defer_auto_map) for item in value]
    if isinstance(value, str) and _private_location(value):
        return None
    return value


def save_pruned(model, adapter: MoEAdapter, out_dir: Path, src_model: str,
                info: dict, keep: Dict[str, List[int]],
                max_shard_size: str = "5GB", remote_code: bool = False,
                verbose: bool = True) -> None:
    'Save a checkpoint atomically without overwriting an existing output.'
    from .checkpoint import _publish

    out_dir = Path(out_dir)
    if out_dir.exists() or out_dir.is_symlink():
        raise FileExistsError("output directory already exists; choose a new directory")
    adapter.set_num_experts(info["kept_experts"])
    adapter.set_top_k(info["experts_per_tok"])
    adapter.postprocess_config(model.config, info)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".razor-", dir=out_dir.parent))
    try:
        model.save_pretrained(stage, safe_serialization=True, max_shard_size=max_shard_size)
        saved_config = json.loads(_asset_path(stage, "config.json").read_text(encoding="utf-8"),
                                  object_pairs_hook=_unique_object)
        for path in stage.rglob("*.py"):
            _asset_path(stage, path.relative_to(stage)).unlink()
        copy_aux_files(Path(src_model), stage, adapter, remote_code=remote_code,
                       verbose=verbose, config=saved_config)
        public_info = _public_config(info)
        for name in ("config.json", "generation_config.json"):
            path = _asset_path(stage, name)
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as f:
                value = _public_config(json.load(f, object_pairs_hook=_unique_object), code_root=stage)
            if path.name == "config.json":
                value["_pruning_info"] = public_info
            _write_aux_asset(stage, name, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        _write_aux_asset(stage, "kept_expert_indices.json", (json.dumps(keep, indent=2) + "\n").encode("utf-8"))
        _write_aux_asset(stage, "pruning_info.json", (json.dumps(public_info, indent=2) + "\n").encode("utf-8"))
        _publish(stage, out_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    if verbose:
        print("[prune] checkpoint saved")


def build_info(method: str, src_model: str, orig_experts: int,
               kept_experts: int, orig_top_k: int, top_k: int,
               n_layers: int, saliency_path: Optional[str] = None,
               keep_path: Optional[str] = None,
               calibration: Optional[str] = None,
               adapter_name: str = "",
               skipped_layers: Sequence[str] = ()) -> dict:
    'The ``_pruning_info`` block stamped into the output config.'
    return {
        "method": method,


        "metric": method,
        "adapter": adapter_name,
        "source_model": None,
        "original_experts": orig_experts,
        "kept_experts": kept_experts,
        "prune_ratio": round(1.0 - kept_experts / orig_experts, 4),
        "original_experts_per_tok": orig_top_k,
        "experts_per_tok": top_k,
        "pruned_layers": n_layers,
        "skipped_layers": list(skipped_layers),
        "saliency_file": None,
        "keep_indices_file": None,
        "calibration_data": None,
        "razor_version": __import__("razor").__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
