"""Model loading with explicit remote-code opt-in."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import json
import sys
import types
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

import torch

from .adapters import MoEAdapter, get_adapter


def _require_complete_checkpoint(model):
    marker = Path(model).expanduser() / ".razor-incomplete"
    if marker.exists() or marker.is_symlink():
        raise RuntimeError(f"checkpoint is incomplete: {marker}")


def resolve_checkpoint(model: str, trust_remote_code: bool = False, *, revision=None,
                       cache_dir=None, local_files_only=False, include_weights=True) -> str:
    """Resolve local weights or a minimal HF snapshot without executing code."""
    path = Path(model).expanduser()
    _require_complete_checkpoint(path)
    if path.is_dir():
        return str(path.resolve())
    if path.is_absolute() or str(model).startswith(("./", "../", "~")):
        raise FileNotFoundError(f"checkpoint directory does not exist: {model}")
    from huggingface_hub import snapshot_download

    from .prune import _ASSET_SUBDIRECTORIES, _AUX_ASSETS, _LICENSE_FILES

    assets = set(_AUX_ASSETS | _LICENSE_FILES)
    patterns = set(MoEAdapter.AUX_FILES) | assets | {
        "config.json", "tokenizer*", "tokenizer/**", "vocab.*", "*.model", "chat_template*",
        "chat_templates/*.jinja", "*.py", "encoding/**", "processor_config.json",
        "preprocessor_config.json", "video_preprocessor_config.json"}
    for directory in _ASSET_SUBDIRECTORIES:
        patterns.update(f"{directory}/{name}" for name in assets)
        patterns.add(f"{directory}/chat_templates/*.jinja")
    patterns = sorted(patterns)
    if include_weights:
        patterns += ["*.safetensors", "model.safetensors.index.json"]
    resolved = snapshot_download(repo_id=str(model), revision=revision, cache_dir=cache_dir,
                                 local_files_only=local_files_only, allow_patterns=patterns)
    _require_complete_checkpoint(resolved)
    return resolved


def _scoring_config(config):
    clean = copy.deepcopy(config)
    for current in (clean, getattr(clean, "text_config", clean)):
        if hasattr(current, "quantization_config"):
            delattr(current, "quantization_config")
    return clean


def load_checkpoint_class(model, *, class_name=None, model_type=None, kind="modeling",
                          trust_remote_code=False):
    """Import one explicitly trusted checkpoint class, including relative imports.

    AST inspection locates declarations without executing other modeling files.
    Configuration files are selected by their declared model_type, model files
    by the architecture named in the checkpoint metadata.
    """
    if not trust_remote_code:
        raise ValueError("checkpoint Python classes require trust_remote_code=True")
    _require_complete_checkpoint(model)
    root = Path(model).resolve()
    candidates = []
    for path in sorted(root.glob(f"{kind}_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            match = node.name == class_name if class_name else False
            if model_type is not None:
                for statement in node.body:
                    if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                        if any(isinstance(target, ast.Name) and target.id == "model_type" for target in targets):
                            value = statement.value
                            match |= isinstance(value, ast.Constant) and value.value == model_type
            if match:
                candidates.append((path, node.name))
    if len(candidates) != 1:
        raise ImportError(f"expected one {kind} class for {class_name or model_type!r} in checkpoint; "
                          f"found {len(candidates)}. Install its native Transformers implementation "
                          "or provide the official configuration/modeling Python files.")
    path, name = candidates[0]
    package_name = "_razor_checkpoint_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(root)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    try:
        module = importlib.import_module(f"{package_name}.{path.stem}")
    except ImportError as error:
        raise ImportError(f"cannot import trusted checkpoint class {name}: {error}") from error
    result = getattr(module, name)
    from transformers import PretrainedConfig, PreTrainedModel
    base = PretrainedConfig if kind == "configuration" else PreTrainedModel
    if not isinstance(result, type) or not issubclass(result, base):
        raise TypeError(f"checkpoint class {name} does not implement {base.__name__}")
    if kind == "modeling":
        classes = {cls for imported_name, imported in tuple(sys.modules.items())
                   if imported_name.startswith(package_name + ".")
                   for cls in vars(imported).values()
                   if isinstance(cls, type) and issubclass(cls, PreTrainedModel)
                   and cls.__module__.startswith(package_name + ".")}
        for cls in classes:
            if cls.__dict__.get("_supports_flash_attn_2") is True and "_supports_flash_attn" not in cls.__dict__:
                cls._supports_flash_attn = True
            original = cls.__dict__.get("tie_weights")
            if original is not None and tuple(inspect.signature(original).parameters) == ("self",):
                def tie_weights(self, recompute_mapping=True, original=original):
                    return original(self)
                cls.tie_weights = tie_weights
    return result


def load_config(model: str, trust_remote_code: bool = False):
    """Load a config; custom checkpoint code is disabled by default."""
    _require_complete_checkpoint(model)
    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
    except ValueError as error:
        unknown = any(marker in str(error).lower() for marker in
                      ("does not recognize this architecture", "unrecognized model", "unrecognized configuration"))
        if not trust_remote_code or not unknown:
            raise
        root = resolve_checkpoint(model, trust_remote_code=True, include_weights=False)
        raw = json.loads((Path(root) / "config.json").read_text())
        try:
            config_class = load_checkpoint_class(root, model_type=raw.get("model_type"),
                                                  kind="configuration", trust_remote_code=True)
        except ImportError as missing:
            raise ImportError(f"native config is unavailable: {missing}") from error
        config = config_class.from_dict(raw)
        config.name_or_path = root
    text = getattr(config, "text_config", None) or config
    if getattr(text, "model_type", "") == "glm_moe_dsa":
        path = Path(model) / "config.json"
        if path.is_file():
            raw = json.loads(path.read_text())
            raw = raw.get("text_config") or raw
            if "qk_rope_head_dim" in raw:
                text.qk_rope_head_dim = int(raw["qk_rope_head_dim"])
        if (hasattr(text, "qk_head_dim") and
                text.qk_nope_head_dim + text.qk_rope_head_dim != text.qk_head_dim):
            raise ValueError("GLM DSA head dimensions disagree with the checkpoint")
    return config


def needs_remote_code(model: str) -> bool:
    """Report custom-code metadata without executing it."""
    path = Path(model) / "config.json"
    if not path.is_file():
        return False
    with path.open() as stream:
        return bool(json.load(stream).get("auto_map"))


def _auto_model(method: str, model=None, **kwargs):
    """Only retry an unsupported config, never a failed weight load or forward."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    argument = kwargs.pop("config") if method == "from_config" else model
    config = argument if method == "from_config" else kwargs.get("config")
    text = getattr(config, "text_config", None) or config
    requested_attention = kwargs.get("attn_implementation") or getattr(text, "_attn_implementation", None)
    is_kimi = "kimi" in str(getattr(text, "model_type", ""))
    def finish(net):
        if is_kimi and requested_attention is not None:
            if requested_attention not in ("eager", "flash_attention_2"):
                raise ValueError("Kimi MLA scoring supports its native eager or flash_attention_2 path")
            for module in net.modules():
                current = getattr(module, "config", None)
                if current is not None and getattr(current, "model_type", "") == "kimi_linear":
                    current._attn_implementation = requested_attention
                    if hasattr(module, "_use_flash_attention_2"):
                        module._use_flash_attention_2 = requested_attention == "flash_attention_2"
        return net
    shim = None
    if kwargs.get("trust_remote_code", False) and is_kimi:
        import transformers.utils.generic as generic
        if not hasattr(generic, "OutputRecorder"):
            from transformers.utils.output_capturing import OutputRecorder
            generic.OutputRecorder = OutputRecorder
            shim = generic
    def local_model():
        architectures = getattr(config, "architectures", None) or []
        if len(architectures) != 1:
            raise ValueError("trusted local loading requires one declared architecture")
        root = model if method == "from_pretrained" else config.name_or_path
        root = resolve_checkpoint(root, trust_remote_code=True, include_weights=method == "from_pretrained")
        model_class = load_checkpoint_class(root, class_name=architectures[0], trust_remote_code=True)
        direct_kwargs = {key: value for key, value in kwargs.items() if key != "trust_remote_code"}
        if method == "from_config":
            return finish(model_class._from_config(config, **direct_kwargs))
        return finish(model_class.from_pretrained(root, **direct_kwargs))
    try:
        root = model if method == "from_pretrained" else getattr(config, "name_or_path", "")
        if is_kimi and kwargs.get("trust_remote_code", False) and list(Path(root).glob("modeling_kimi*.py")):
            return local_model()
        failure = None
        for auto_class in (AutoModelForCausalLM, AutoModelForImageTextToText):
            try:
                return finish(getattr(auto_class, method)(argument, **kwargs))
            except ValueError as error:
                if "Unrecognized configuration class" not in str(error):
                    raise
                failure = error
        if not kwargs.get("trust_remote_code", False):
            raise failure
        return local_model()
    finally:
        if shim is not None:
            del shim.OutputRecorder


def quantization_config(config):
    """Validate only formats for which the scoring loader has a decoder."""
    text = getattr(config, "text_config", None) or config
    quant = getattr(config, "quantization_config", None)
    if quant is None:
        quant = getattr(text, "quantization_config", None)
    if quant is None:
        return {}
    quant = quant.to_dict() if hasattr(quant, "to_dict") else dict(quant)
    if not quant:
        return {}
    method = str(quant.get("quant_method", "")).lower()
    if method in ("fp8", "mxfp4"):
        return quant
    if method == "compressed-tensors":
        groups = quant.get("config_groups") or {}
        valid = bool(groups)
        for group in groups.values():
            weights = group.get("weights") or {}
            valid &= (weights.get("num_bits") == 4 and
                      weights.get("type") == "float" and
                      weights.get("group_size") == 32)
        if valid:
            return quant
    raise NotImplementedError(f"unverified checkpoint quantization: {method!r}")


def load_model(model: str, device_map: str = "auto", dtype: torch.dtype = torch.bfloat16,
               trust_remote_code: bool = False, adapter: Optional[str] = None,
               verbose: bool = True, **kwargs) -> Tuple[object, MoEAdapter]:
    """Load native causal/conditional models; checkpoint code requires opt-in."""
    config = load_config(model, trust_remote_code=trust_remote_code)
    quant = quantization_config(config)
    get_adapter(_scoring_config(config), name=adapter)
    if quant:
        from .streaming import load_decoded_model
        model = resolve_checkpoint(model, trust_remote_code=trust_remote_code)
        net = load_decoded_model(model, config=config, dtype=dtype,
                                 device_map=device_map,
                                 trust_remote_code=trust_remote_code, **kwargs)
    else:
        if verbose:
            print(f"[load] loading model (device_map={device_map}, dtype={dtype})")
        net = _auto_model("from_pretrained", model, config=config, dtype=dtype,
                          device_map=device_map, low_cpu_mem_usage=True,
                          trust_remote_code=trust_remote_code, **kwargs).eval()
    ad = get_adapter(net.config, name=adapter)
    ad.validate_model(net)
    if any(value.is_meta for value in net.parameters()):
        raise RuntimeError("resident scoring cannot read offloaded meta expert weights; use streaming")
    if verbose:
        print(f"[load] adapter={ad.name} | experts={ad.num_experts} top_k={ad.top_k}")
    return net, ad


class _EncoderSourceLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        return self.source_to_code(self.get_data(self.path), self.path)


class _EncoderPackageFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.namespace = "_razor_dsv4_" + uuid.uuid4().hex

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.namespace and not fullname.startswith(self.namespace + "."):
            return None
        from .checkpoint import _contained

        parts = fullname[len(self.namespace):].lstrip(".").split(".") if fullname != self.namespace else []
        relative = Path(*parts)
        initializer = _contained(self.root, relative / "__init__.py")
        source = _contained(self.root, relative.with_suffix(".py")) if parts else None
        filename = initializer if initializer.is_file() else source
        if filename is not None and filename.is_file():
            loader = _EncoderSourceLoader(fullname, str(filename))
            return importlib.util.spec_from_file_location(
                fullname, filename, loader=loader,
                submodule_search_locations=[str(filename.parent)] if filename == initializer else None)
        directory = _contained(self.root, relative)
        if directory.is_dir():
            spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = [str(directory)]
            return spec
        return None


@contextmanager
def _encoder_package(root):
    """Keep successful namespaces for lazy imports; roll back every failed load."""
    finder = _EncoderPackageFinder(root)
    sys.meta_path.insert(0, finder)
    try:
        yield importlib.import_module(finder.namespace + ".encoding.encoding_dsv4")
    except BaseException:
        for name in tuple(sys.modules):
            if name == finder.namespace or name.startswith(finder.namespace + "."):
                sys.modules.pop(name, None)
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        raise


class DSV4Tokenizer:
    """Use the checkpoint's official encoder without reimplementing its format."""
    def __init__(self, tokenizer, model: str, *, trust_remote_code=False,
                 thinking_mode="chat"):
        if not trust_remote_code:
            raise ValueError("DSV4 encoding/encoding_dsv4.py requires trust_remote_code=True")
        root = Path(model).expanduser().resolve()
        _require_complete_checkpoint(root)
        path = root / "encoding" / "encoding_dsv4.py"
        if not path.is_file():
            raise FileNotFoundError(f"official DSV4 prompt encoder is missing: {path}")
        with _encoder_package(root) as module:
            if not callable(getattr(module, "encode_messages", None)):
                raise RuntimeError("official DSV4 encoder lacks encode_messages")
            marker = "\x00RAZOR_USER_PROBE\x00"
            rendered = module.encode_messages([{"role": "user", "content": marker}],
                                              thinking_mode=thinking_mode)
            if marker not in rendered:
                raise RuntimeError("cannot recover the official assistant opener")
            self.assistant_open_str = rendered.split(marker, 1)[1]
            self._tokenizer, self._encoder = tokenizer, module
            self._mode = thinking_mode
            self.chat_template = f"[official encoder: {path}]"

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def __len__(self):
        return len(self._tokenizer)

    def __call__(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs)

    def render_record(self, record):
        """Render a complete calibration record, preserving DSML tool metadata."""
        if "messages" not in record:
            if "text" in record:
                return record["text"]
            raise ValueError("DSV4 calibration records require messages or text")
        if not isinstance(record["messages"], list):
            raise TypeError("record.messages must be a list")
        return self.apply_chat_template(record["messages"], tokenize=False,
                                        add_generation_prompt=False, tools=record.get("tools"))

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, tools=None, **kwargs):
        messages = copy.deepcopy(messages)
        for message in messages:
            content = message.get("content")
            if content is None and not message.get("tool_calls"):
                message["content"] = ""
            elif content is not None and not isinstance(content, str):
                message["content"] = json.dumps(content, ensure_ascii=False)
            for call in message.get("tool_calls") or []:
                function = call.get("function")
                if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                    try:
                        function["arguments"] = json.loads(function["arguments"])
                    except json.JSONDecodeError:
                        pass
        if tools:
            for message in messages:
                if message.get("role") in ("system", "developer"):
                    message["tools"] = copy.deepcopy(tools)
                    break
            else:
                messages.insert(0, {"role": "system", "content": "", "tools": copy.deepcopy(tools)})
        if add_generation_prompt:
            while messages and messages[-1].get("role") == "assistant":
                messages.pop()
        text = self._encoder.encode_messages(messages, thinking_mode=self._mode)
        if not tokenize:
            return text
        token_kwargs = {name: kwargs[name] for name in
                        ("return_tensors", "padding", "truncation", "max_length")
                        if name in kwargs}
        encoded = self._tokenizer(text, add_special_tokens=False, **token_kwargs)
        return encoded if kwargs.get("return_dict") else encoded["input_ids"]


def load_tokenizer(model: str, trust_remote_code: bool = False,
                   thinking_mode: str = "chat"):
    """Load native tokenization or the explicitly trusted checkpoint renderer."""
    from transformers import AutoTokenizer
    model = resolve_checkpoint(model, trust_remote_code=trust_remote_code, include_weights=False)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
    if not getattr(tokenizer, "chat_template", None):
        encoder = Path(model) / "encoding" / "encoding_dsv4.py"
        raw_path = Path(model) / "config.json"
        raw = json.loads(raw_path.read_text()) if raw_path.is_file() else {}
        text = raw.get("text_config") or raw
        if encoder.is_file() or text.get("model_type") == "deepseek_v4":
            return DSV4Tokenizer(tokenizer, model, trust_remote_code=trust_remote_code,
                                 thinking_mode=thinking_mode)
    return tokenizer
