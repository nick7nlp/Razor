"""CPU regressions for checkpoint verification and reference construction."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from razor import verify
from razor.verify_cmd import _load_keep, _slice_reference, _tensor_level, run


def _result(n=500, shift=0.0):
    lp = -torch.rand(n, generator=torch.Generator().manual_seed(0)) * 4 - 0.1 - shift
    valid = torch.ones(n, dtype=torch.bool)
    return {"logprobs": lp, "valid": valid, "n": n,
            "nll": float(-lp.double().mean())}


def test_logprob_comparison():
    result = _result()
    assert verify.compare_logprobs(result, copy.deepcopy(result), verbose=False)
    assert verify.compare_logprobs(result, _result(shift=1e-5), verbose=False)
    assert not verify.compare_logprobs(result, _result(shift=0.01), verbose=False)
    assert not verify.compare_logprobs(result, _result(n=499), verbose=False)
    outliers = copy.deepcopy(result)
    outliers["logprobs"][:15] -= 0.1
    outliers["logprobs"][15:30] += 0.1
    outliers["nll"] = float(-outliers["logprobs"].double().mean())
    assert abs(outliers["nll"] - result["nll"]) < verify.NLL_TOL
    assert not verify.compare_logprobs(result, outliers, verbose=False)


@pytest.mark.parametrize("defect", ["empty", "nan", "infinity", "bad_mask", "shape", "count", "nll", "all_padding"])
def test_invalid_logprob_results_fail(defect):
    bad = _result()
    if defect == "empty":
        bad = {"logprobs": torch.empty(0), "valid": torch.empty(0, dtype=torch.bool), "n": 0, "nll": 0.0}
    elif defect in ("nan", "infinity"):
        bad["logprobs"][0] = float("nan" if defect == "nan" else "inf")
    elif defect == "bad_mask":
        bad["valid"] = bad["valid"].long()
    elif defect == "shape":
        bad["logprobs"] = bad["logprobs"].unsqueeze(0)
    elif defect == "count":
        bad["n"] += 1
    elif defect == "nll":
        bad["nll"] = float("nan")
    else:
        bad["valid"].zero_()
        bad["n"] = 0
    assert not verify.compare_logprobs(bad, bad, verbose=False)


class _LanguageModel(torch.nn.Module):
    def __init__(self, tuple_output=False, nonfinite=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(4))
        self.tuple_output = tuple_output
        self.nonfinite = nonfinite

    def forward(self, input_ids, attention_mask):
        logits = self.weight[input_ids]
        if self.nonfinite:
            logits = logits * float("nan")
        return (logits, "cache") if self.tuple_output else SimpleNamespace(logits=logits)


def test_score_batches_accepts_tuple_and_restores_training():
    batch = {"input_ids": torch.tensor([[0, 1, 2, 3]]),
             "attention_mask": torch.tensor([[0, 1, 1, 1]])}
    model = _LanguageModel().train()
    expected = verify.score_batches(model, [batch])
    actual = verify.score_batches(_LanguageModel(tuple_output=True), [batch])
    assert model.training
    assert expected["valid"].tolist() == [False, True, True]
    assert expected["n"] == 2
    assert verify.compare_logprobs(expected, actual, verbose=False)


@pytest.mark.parametrize("defect", ["empty", "short", "mask_shape", "bad_mask", "padding", "nan"])
def test_score_batches_rejects_invalid_inputs(defect):
    model = _LanguageModel(nonfinite=defect == "nan").train()
    batch = {"input_ids": torch.tensor([[0, 1, 2]]), "attention_mask": torch.ones(1, 3)}
    batches = [batch]
    if defect == "empty":
        batches = []
    elif defect == "short":
        batch = {"input_ids": torch.tensor([[0]]), "attention_mask": torch.ones(1, 1)}
        batches = [batch]
    elif defect == "mask_shape":
        batch["attention_mask"] = torch.ones(3)
    elif defect == "bad_mask":
        batch["attention_mask"][0, 1] = 2
    elif defect == "padding":
        batch["attention_mask"].zero_()
    with pytest.raises(ValueError):
        verify.score_batches(model, batches)
    assert model.training


@pytest.mark.parametrize("ids", [[], [0, 0], [-1], [8], [1.5], [True], ["1"]])
def test_keep_mask_rejects_invalid_ids(ids):
    with pytest.raises(ValueError):
        verify.keep_mask(8, ids)


def test_keep_mask_preserves_selection():
    assert verify.keep_mask(4, [3, 1]).tolist() == [False, True, False, True]


def test_tensor_helpers_check_dtype_and_extra_keys():
    source = {"x": torch.arange(8).reshape(4, 2).float()}
    target = {"x": source["x"][[3, 1]]}
    assert verify.compare_expert_tensors(source, target, {"x": [3, 1]}, verbose=False)
    assert not verify.compare_expert_tensors(source, {"x": target["x"].double()}, {"x": [3, 1]}, verbose=False)
    assert not verify.check_untouched(source, dict(source, extra=torch.zeros(1)), verbose=False)


def _states(layout="fused"):
    generator = torch.Generator().manual_seed(7)
    source = {"model.embed_tokens.weight": torch.randn(8, 4, generator=generator),
              "model.norm.weight": torch.randn(4, generator=generator)}
    kept = {"main_0": [3, 1], "mtp_0": [0, 2]}
    bases = {"main_0": "model.layers.0.mlp.",
             "mtp_0": "model.mtp.layers.0.mlp.",
             "main_12": "model.layers.12.mlp."}
    for tag, base in bases.items():
        source[base + "gate.weight"] = torch.randn(4, 4, generator=generator)
        for suffix in ("gate.bias", "gate.e_score_correction_bias", "expert_bias"):
            source[base + suffix] = torch.randn(4, generator=generator)
        source[base + "gate.weight_scale_inv"] = torch.randn(4, 1, generator=generator)
        source[base + "shared_experts.down_proj.weight"] = torch.randn(4, 4, generator=generator)
        source[base + "shared_expert_gate.weight"] = torch.randn(1, 4, generator=generator)
        if layout == "list":
            for expert in range(4):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    prefix = base + f"experts.{expert}.{projection}."
                    source[prefix + "weight"] = torch.randn(4, 4, generator=generator)
                    source[prefix + "bias"] = torch.randn(4, generator=generator)
                    source[prefix + "weight_scale_inv"] = torch.randn(1, 1, generator=generator)
        else:
            source[base + "experts.gate_up_proj"] = torch.randn(4, 6, 4, generator=generator)
            source[base + "experts.down_proj"] = torch.randn(4, 4, 3, generator=generator)
            source[base + "experts.gate_up_proj_bias"] = torch.randn(4, 6, generator=generator)
            source[base + "experts.down_proj_scale_inv"] = torch.randn(4, 1, 1, generator=generator)
    target = {}
    for name, tensor in source.items():
        match = next(((tag, base) for tag, base in bases.items() if name.startswith(base) and tag in kept), None)
        if match is None:
            target[name] = tensor.clone()
            continue
        tag, base = match
        suffix = name[len(base):]
        ids = kept[tag]
        if suffix.startswith("experts.") and layout == "list":
            _, old, rest = suffix.split(".", 2)
            if int(old) in ids:
                target[base + f"experts.{ids.index(int(old))}.{rest}"] = tensor.clone()
        elif suffix.startswith(("experts.", "gate.")) or suffix == "expert_bias":
            target[name] = tensor[ids].clone()
        else:
            target[name] = tensor.clone()
    return source, target, kept


def _write_checkpoint(root, state, indexed=False):
    root.mkdir(parents=True, exist_ok=True)
    if not indexed:
        save_file(state, str(root / "model.safetensors"))
        return
    keys = list(state)
    weight_map = {}
    for i in range(2):
        shard = f"model-{i + 1:05d}-of-00002.safetensors"
        tensors = {name: state[name] for name in keys[i::2]}
        save_file(tensors, str(root / shard))
        weight_map.update({name: shard for name in tensors})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def _check_files(tmp_path, source, target, keep, indexed=False):
    _write_checkpoint(tmp_path / "source", source, indexed)
    _write_checkpoint(tmp_path / "target", target, indexed)
    return _tensor_level(str(tmp_path / "source"), str(tmp_path / "target"), keep, verbose=False)


@pytest.mark.parametrize("layout", ["fused", "list"])
@pytest.mark.parametrize("indexed", [False, True])
def test_tensor_level_reads_real_checkpoints(tmp_path, layout, indexed, monkeypatch):
    source, target, keep = _states(layout)
    import safetensors.torch

    def forbidden(*args, **kwargs):
        raise AssertionError("whole-shard loading is not allowed")

    monkeypatch.setattr(safetensors.torch, "load_file", forbidden)
    assert _check_files(tmp_path, source, target, keep, indexed)


@pytest.mark.parametrize("layout", ["fused", "list"])
def test_tensor_level_compares_fp8_storage(tmp_path, layout):
    source, target, keep = _states(layout)
    source = {name: value.to(torch.float8_e4m3fn) if ".experts." in name else value
              for name, value in source.items()}
    target = {name: value.to(torch.float8_e4m3fn) if ".experts." in name else value
              for name, value in target.items()}
    assert _check_files(tmp_path, source, target, keep)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_tensor_level_rejects_nonfinite_weights_on_both_sides(tmp_path, value):
    source, target, keep = _states()
    source["model.norm.weight"][0] = value
    target["model.norm.weight"][0] = value
    assert not _check_files(tmp_path, source, target, keep)


@pytest.mark.parametrize("layout", ["fused", "list"])
@pytest.mark.parametrize("defect", ["router", "router_bias", "correction_bias", "block_bias", "scale",
                                    "shared", "missing", "extra", "wrong_layer", "wrong_keep", "dtype", "omitted_layer"])
def test_tensor_level_rejects_corrupt_checkpoints(tmp_path, layout, defect):
    source, target, keep = _states(layout)
    base = "model.layers.0.mlp."
    perturb = {"router": "gate.weight", "router_bias": "gate.bias",
               "correction_bias": "gate.e_score_correction_bias", "block_bias": "expert_bias",
               "scale": "experts.down_proj_scale_inv" if layout == "fused" else "experts.0.down_proj.weight_scale_inv",
               "shared": "shared_experts.down_proj.weight"}
    if defect in perturb:
        target[base + perturb[defect]] += 1
    elif defect == "missing":
        del target[base + "gate.weight"]
    elif defect == "extra":
        target[base + "experts.7.down_proj.weight"] = torch.ones(4, 4)
    elif defect == "wrong_layer":
        keep["main_999"] = keep.pop("main_0")
    elif defect == "wrong_keep":
        keep["main_0"] = [1, 3]
    elif defect == "dtype":
        target[base + "gate.weight"] = target[base + "gate.weight"].double()
    else:
        del keep["mtp_0"]
    assert not _check_files(tmp_path, source, target, keep)


@pytest.mark.parametrize("ids", [[], [1, 1], [-1, 1], [1, 4], [1.0, 3], [True, 3]])
def test_tensor_level_rejects_bad_keep(tmp_path, ids):
    source, target, keep = _states()
    keep["main_0"] = ids
    assert not _check_files(tmp_path, source, target, keep)


def test_tensor_level_does_not_confuse_main_and_mtp(tmp_path):
    source, target, keep = _states()
    keep["mtp_0"] = keep["main_0"]
    assert not _check_files(tmp_path, source, target, keep)


def test_tensor_level_validates_nested_router(tmp_path):
    source, target, keep = _states("list")
    source = {name.replace(".mlp.gate.", ".mlp.router.gate."): value for name, value in source.items()}
    target = {name.replace(".mlp.gate.", ".mlp.router.gate."): value for name, value in target.items()}
    assert _check_files(tmp_path, source, target, keep)


@pytest.mark.parametrize("defect", ["extra_index_key", "missing_index_key", "wrong_shard", "extra_shard", "no_index", "unsupported_axis", "unsupported_path"])
def test_tensor_level_refuses_ambiguous_formats(tmp_path, defect):
    source, target, keep = _states()
    if defect == "unsupported_axis":
        key = "model.layers.0.mlp.experts.down_proj_scale_inv"
        source[key] = torch.ones(2, 1)
    elif defect == "unsupported_path":
        key = "model.layers.0.mlp.experts.gate_up_proj"
        source[key.replace("model.layers", "other.layers")] = source.pop(key)
    _write_checkpoint(tmp_path / "source", source, indexed=True)
    _write_checkpoint(tmp_path / "target", target, indexed=True)
    index_path = tmp_path / "source" / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    first = next(iter(index["weight_map"]))
    if defect == "extra_index_key":
        index["weight_map"]["missing.weight"] = index["weight_map"][first]
    elif defect == "missing_index_key":
        del index["weight_map"][first]
    elif defect == "wrong_shard":
        index["weight_map"][first] = "model-00002-of-00002.safetensors"
    elif defect == "extra_shard":
        save_file({"extra": torch.ones(1)}, str(tmp_path / "source" / "extra.safetensors"))
    index_path.write_text(json.dumps(index))
    if defect == "no_index":
        index_path.unlink()
    assert not _tensor_level(str(tmp_path / "source"), str(tmp_path / "target"), keep, verbose=False)


@pytest.mark.parametrize("raw", [{}, {"main_0": []}, {"wrong_0": [0]}, {"main_0": [True]}, {"main_0": [1, 1]}])
def test_load_keep_rejects_bad_json(tmp_path, raw):
    (tmp_path / "kept_expert_indices.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        _load_keep(str(tmp_path), None)


def test_slice_is_only_public_verification_mode():
    from razor.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["verify", "--src", "source", "--pruned", "target"])
    assert getattr(args, "how", "slice") == "slice"
    for how in ("mask", "unknown"):
        with pytest.raises(SystemExit):
            parser.parse_args(["verify", "--src", "source", "--pruned", "target", "--how", how])
        with pytest.raises(ValueError, match="slice"):
            run("source", "target", how=how)


def test_slice_reference_updates_configuration_and_rejects_unknown_layers():
    from razor.adapters import get_adapter
    from test_end_to_end import Config, TinyMoE

    config = Config()
    net = TinyMoE("fused", config)
    adapter = get_adapter(config, name="qwen")
    target_config = Config(num_experts=4, num_experts_per_tok=1)
    keep = {f"main_{i}": [0, 2, 4, 6] for i in range(3)}
    assert _slice_reference(net, adapter, keep, verbose=False, target_config=target_config) == 3
    assert net.config.num_experts == 4
    assert net.config.num_experts_per_tok == 1
    for block in adapter.moe_blocks(net):
        assert block.module.experts.gate_up_proj.shape[0] == 4
        assert adapter.gate_module(block.module).weight.shape[0] == 4
    other = TinyMoE("fused", Config())
    with pytest.raises(ValueError, match="every source MoE layer"):
        _slice_reference(other, get_adapter(other.config, name="qwen"), {"wrong_0": [0, 1]}, verbose=False)


@pytest.mark.parametrize("defect", ["count", "top_k", "alias", "grouping"])
def test_slice_reference_rejects_invalid_target_config(defect):
    from razor.adapters import get_adapter
    from test_end_to_end import Config, TinyMoE

    source = TinyMoE("fused", Config())
    target_config = Config(num_experts=4, num_experts_per_tok=1)
    if defect == "count":
        target_config.num_experts = 3
    elif defect == "top_k":
        target_config.num_experts_per_tok = 3
    elif defect == "alias":
        target_config.n_routed_experts = 8
    else:
        target_config.n_group = 2
    before = {name: tensor.clone() for name, tensor in source.state_dict().items()}
    keep = {f"main_{i}": [0, 2, 4, 6] for i in range(3)}
    with pytest.raises(ValueError):
        _slice_reference(source, get_adapter(source.config, name="qwen"), keep,
                         target_config=target_config, verbose=False)
    assert verify.check_untouched(before, source.state_dict(), verbose=False)


@pytest.mark.parametrize("family", ["qwen", "deepseek"])
def test_slice_reference_matches_local_save_reload(tmp_path, family, monkeypatch):
    from razor.adapters import get_adapter
    from razor.prune import prune_model

    if family == "qwen":
        from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

        config = Qwen2MoeConfig(
            vocab_size=32, hidden_size=16, intermediate_size=24,
            moe_intermediate_size=8, shared_expert_intermediate_size=8,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            num_experts=8, num_experts_per_tok=2, max_position_embeddings=32)
        model_class, top_k = Qwen2MoeForCausalLM, 1
    else:
        from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

        config = DeepseekV3Config(
            vocab_size=32, hidden_size=16, intermediate_size=24,
            moe_intermediate_size=8, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, n_routed_experts=8, n_shared_experts=1,
            num_experts_per_tok=2, first_k_dense_replace=0, n_group=2,
            topk_group=1, q_lora_rank=4, kv_lora_rank=4,
            qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4,
            norm_topk_prob=True, max_position_embeddings=32)
        model_class, top_k = DeepseekV3ForCausalLM, 2
    config._attn_implementation = "eager"
    torch.manual_seed(17)
    source = model_class(config).eval()
    source.save_pretrained(tmp_path / "source", safe_serialization=True)
    target = copy.deepcopy(source)
    adapter = get_adapter(target.config, name=family)
    keep = {block.tag: [0, 2, 4, 6] for block in adapter.moe_blocks(target)}
    prune_model(target, adapter, keep, target_top_k=top_k, verbose=False)
    target.save_pretrained(tmp_path / "target", safe_serialization=True)
    loaded = model_class.from_pretrained(tmp_path / "target", local_files_only=True,
                                        dtype=torch.float32, attn_implementation="eager").eval()
    assert _tensor_level(str(tmp_path / "source"), str(tmp_path / "target"), keep, verbose=False)
    _slice_reference(source, get_adapter(source.config, name=family), keep,
                     target_config=loaded.config, verbose=False)
    batch = {"input_ids": torch.tensor([[1, 7, 3, 9, 4]]),
             "attention_mask": torch.ones(1, 5, dtype=torch.long)}
    assert verify.compare_logprobs(verify.score_batches(source, [batch]),
                                  verify.score_batches(loaded, [batch]), verbose=False)
    assert loaded.config.num_experts_per_tok == top_k

    import razor.verify_cmd as command

    trust_settings = []

    def load_cpu(path, **kwargs):
        trust_settings.append(kwargs["trust_remote_code"])
        model = model_class.from_pretrained(path, local_files_only=True,
                                           dtype=torch.float32, attn_implementation="eager").eval()
        return model, get_adapter(model.config, name=family)

    def tokenizer(path, **kwargs):
        trust_settings.append(kwargs["trust_remote_code"])
        return object()

    monkeypatch.setattr(command.loader, "load_model", load_cpu)
    monkeypatch.setattr(command.loader, "load_tokenizer", tokenizer)
    monkeypatch.setattr(command.calibration, "build_batches", lambda *args, **kwargs: [batch])
    monkeypatch.setattr(command, "resolve_data", lambda data: data)
    (tmp_path / "target" / "kept_expert_indices.json").write_text(json.dumps(keep))
    assert run(str(tmp_path / "source"), str(tmp_path / "target"), data="fixed", verbose=False)
    assert trust_settings == [False, False, False]


@pytest.mark.parametrize("shape", [(2, 5, 4), (2, 5, 3, 4)])
def test_native_layer_outputs_compare_all_stream_axes(shape):
    reference = torch.ones(shape, dtype=torch.bfloat16)
    assert verify.compare_layer_outputs(reference, reference.clone())["ok"]
    changed = reference.clone()
    changed.reshape(-1)[-1] = 2
    result = verify.compare_layer_outputs(reference, changed)
    assert not result["ok"]
    assert result["max_abs"] == 1.0
    assert not verify.compare_layer_outputs(reference, reference.float())["ok"]
    assert not verify.compare_layer_outputs(reference, reference.reshape(-1))["ok"]
    changed.reshape(-1)[0] = float("nan")
    assert not verify.compare_layer_outputs(changed, changed)["ok"]


@pytest.mark.parametrize("tol", [-1.0, float("inf"), float("nan"), True])
def test_native_layer_outputs_reject_invalid_tolerance(tol):
    with pytest.raises(ValueError, match="tolerances"):
        verify.compare_layer_outputs(torch.ones(2), torch.ones(2), rtol=tol)


@pytest.mark.parametrize("chunk_size", [1, 2, 1024])
def test_streamed_logprobs_match_resident_head(chunk_size):
    batch = {"input_ids": torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]]),
             "attention_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 0, 1]])}
    model = _LanguageModel().train()
    hidden = torch.nn.functional.one_hot(batch["input_ids"], 4).float()
    head = torch.nn.Linear(4, 4, bias=False).train()
    head.weight = model.weight
    seen = []

    def epilogue(h):
        seen.append(tuple(h.shape))
        return h.mean(dim=2)

    result = verify.score_hidden_batches(
        [hidden.unsqueeze(2).expand(-1, -1, 3, -1)], [batch], head,
        finalize=epilogue, chunk_size=chunk_size)
    assert seen == [(2, 4, 3, 4)]
    assert head.training
    assert result["n"] == 3
    assert verify.compare_logprobs(verify.score_batches(model, [batch]), result, verbose=False)


@pytest.mark.parametrize("defect", ["empty", "missing_epilogue", "nan", "mask", "padding", "chunk", "vocab"])
def test_streamed_logprobs_fail_closed(defect):
    batch = {"input_ids": torch.tensor([[0, 1, 2]]), "attention_mask": torch.ones(1, 3)}
    h = torch.ones(1, 3, 4)
    head = torch.nn.Linear(4, 4).train()
    hidden, batches, chunk = [h], [batch], 2
    if defect == "empty":
        hidden = []
    elif defect == "missing_epilogue":
        hidden = [h.unsqueeze(2)]
    elif defect == "nan":
        h[0, 0, 0] = float("nan")
    elif defect == "mask":
        batch["attention_mask"][0, 0] = 2
    elif defect == "padding":
        batch["attention_mask"].zero_()
    elif defect == "chunk":
        chunk = 0
    else:
        batch["input_ids"][0, 1] = 8
    with pytest.raises(ValueError):
        verify.score_hidden_batches(hidden, batches, head, chunk_size=chunk)
    assert head.training


class _NativeQwenStream:
    """Exercise actual Qwen3.5/3.6 MoE code with independently sliced states."""
    layer_ids = [0, 1]

    def __init__(self, source_states, target=False, defect=None):
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

        self.block_class = Qwen3_5MoeSparseMoeBlock
        self.config = Qwen3_5MoeTextConfig(
            vocab_size=16, hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=2, head_dim=4, num_experts=2, num_experts_per_tok=1,
            moe_intermediate_size=6, shared_expert_intermediate_size=6)
        self.config._experts_implementation = "eager"
        self.states = source_states
        self.target, self.defect = target, defect
        self.loaded, self.released, self.closed = [], [], False

    def prepare(self, batch):
        hidden = torch.nn.functional.one_hot(batch["input_ids"] % 8, 8).float()
        return hidden, {"carry": 0, "mask": batch["attention_mask"]}

    def load_layer(self, index, keep=None):
        self.loaded.append((index, keep))
        ids = [3, 1] if self.target else keep
        if self.defect == "order" and self.target:
            ids = [1, 2]
        sd = {}
        for name, value in self.states[index].items():
            if name in ("gate.weight", "experts.gate_up_proj", "experts.down_proj"):
                value = value[ids]
            sd[name] = value.clone()
        if self.defect == "weights" and self.target and index == 1:
            sd["experts.down_proj"] += 2
        layer = self.block_class(copy.deepcopy(self.config))
        layer.load_state_dict(sd, strict=True)
        layer._test_index = index
        return layer

    def step(self, layer, hidden, state, index):
        state["carry"] += 1
        out = layer(hidden) + state["carry"] * 0.01
        out = out * state["mask"].unsqueeze(-1)
        if self.target and self.defect == "carry":
            out += 1
        if self.target and self.defect == "nan":
            out *= float("nan")
        if self.target and self.defect == "dtype":
            out = out.double()
        if self.target and self.defect == "exception":
            raise RuntimeError("native forward failed")
        return out, state

    def release_layer(self, layer):
        self.released.append(layer._test_index)

    def close(self):
        self.closed = True


def _native_stream_pair(defect=None):
    template = _NativeQwenStream([])
    config = copy.deepcopy(template.config)
    config.num_experts = 4
    states = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        for _ in template.layer_ids:
            block = template.block_class(config)
            for parameter in block.parameters():
                torch.nn.init.normal_(parameter, std=0.2)
            states.append({name: value.clone() for name, value in block.state_dict().items()})
    return _NativeQwenStream(states), _NativeQwenStream(states, target=True, defect=defect)


@pytest.mark.parametrize("defect", [None, "weights", "order", "carry", "nan"])
def test_native_stream_reference_executes_real_qwen35_moe(defect):
    ref, target = _native_stream_pair(defect)
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
             "attention_mask": torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]])}
    keep = {"main_0": [3, 1], "main_1": [3, 1], "mtp_0": [3, 1]}
    result = verify.verify_streamed_reference(ref, target, [batch], keep, atol=0, rtol=0, verbose=False)
    assert result["ok"] == (defect is None)
    assert result["layers_checked"] == 2
    assert result["sliced_layers"] == 2
    assert result["unexecuted_keep_tags"] == ["mtp_0"]
    assert ref.loaded == [(0, [3, 1]), (1, [3, 1])]
    assert target.loaded == [(0, None), (1, None)]
    assert ref.released == target.released == [0, 1]
    if defect == "weights":
        assert result["first_bad_layer"] == "main_1"
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0], [0, 1, 1, 1]]


def test_native_stream_partial_scope_and_empty_selection():
    ref, target = _native_stream_pair()
    batch = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3)}
    keep = {"main_0": [3, 1], "main_1": [3, 1]}
    result = verify.verify_streamed_reference(ref, target, [batch], keep, max_layers=1, verbose=False)
    assert result["ok"] and result["partial"]
    assert result["scope"] == "partial layer outputs"
    assert result["unexecuted_keep_tags"] == ["main_1"]
    with pytest.raises(ValueError, match="vacuous"):
        verify.verify_streamed_reference(ref, target, [batch], {"main_1": [3, 1]}, max_layers=1)
    with pytest.raises(ValueError, match="absent"):
        verify.verify_streamed_reference(ref, target, [batch], {"main_7": [3, 1]})


def test_native_stream_releases_layer_after_forward_exception():
    ref, target = _native_stream_pair("exception")
    batch = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3)}
    with pytest.raises(RuntimeError, match="native forward failed"):
        verify.verify_streamed_reference(ref, target, [batch], {"main_0": [3, 1]}, verbose=False)
    assert ref.released == target.released == [0]


def test_slice_reference_only_allows_tensor_verified_uninstantiated_mtp():
    from razor.adapters import get_adapter
    from test_end_to_end import Config, TinyMoE

    keep = {f"main_{i}": [0, 2, 4, 6] for i in range(3)}
    keep["mtp_0"] = [0, 2, 4, 6]
    source = TinyMoE("fused", Config())
    with pytest.raises(ValueError, match="every source MoE layer"):
        _slice_reference(source, get_adapter(source.config, name="qwen"), keep, verbose=False)
    assert _slice_reference(source, get_adapter(source.config, name="qwen"), keep,
                            verbose=False, tensor_verified=True) == 3


def test_slice_reference_appended_main_mtp_requires_explicit_verified_tag():
    from razor.adapters import get_adapter
    from test_end_to_end import Config, TinyMoE

    keep = {f"main_{i}": [0, 2, 4, 6] for i in range(4)}
    source = TinyMoE("fused", Config())
    adapter = get_adapter(source.config, name="qwen")
    with pytest.raises(ValueError, match="every source MoE layer"):
        _slice_reference(source, adapter, keep, verbose=False, tensor_verified=True)
    with pytest.raises(ValueError, match="every source MoE layer"):
        _slice_reference(source, adapter, keep, verbose=False, tensor_only_tags=("main_3",))
    assert _slice_reference(source, adapter, keep, verbose=False, tensor_verified=True,
                            tensor_only_tags=("main_3",)) == 3


class _MemoryCheckpointWeights:
    """Keep fixture bytes in RAM while testing real meta-slot materialization."""
    def __init__(self, model, values, tracker, side):
        self.model, self.values, self.tracker, self.side = model, values, tracker, side
        self.slots = {name: value.to("meta") for name, value in model.state_dict().items()}
        self.active = None

    def materialize(self, prefix, device, *, exclude=(), allow_missing=()):
        from razor.streaming import _assign

        if prefix.startswith("model.layers.") and prefix.count(".") == 3:
            assert self.tracker["active"] is None, "both native layers resident simultaneously"
            self.active = (self.side, int(prefix.split(".")[2]))
            self.tracker["active"] = self.active
            self.tracker["order"].append(self.active)
        loaded = set()
        for name in self.slots:
            if name.startswith(prefix) and not any(name.startswith(item) for item in exclude):
                _assign(self.model, name, self.values[name].clone().to(device))
                loaded.add(name)
        return loaded

    def release(self, names):
        from razor.streaming import _assign

        for name in names:
            _assign(self.model, name, torch.empty_like(self.slots[name], device="meta"))
        if self.active is not None:
            assert self.tracker["active"] == self.active
            self.tracker["active"], self.active = None, None


@pytest.mark.parametrize("corrupt", [False, True, "missing"])
@pytest.mark.parametrize("family", ["qwen3", "glm52", "deepseek_v4"])
def test_stream_command_uses_native_full_forward_and_alternates_layers(monkeypatch, corrupt, family):
    from transformers import AutoModelForCausalLM, Qwen3MoeConfig
    from razor import streaming
    import razor.verify_cmd as command

    common = dict(vocab_size=32, hidden_size=16, intermediate_size=24, moe_intermediate_size=8,
                  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
    if family == "glm52":
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
        config = GlmMoeDsaConfig(
            **common, n_routed_experts=4, num_experts_per_tok=2, q_lora_rank=8,
            kv_lora_rank=4, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=8,
            index_head_dim=8, index_n_heads=2, index_topk=2,
            mlp_layer_types=["sparse", "sparse"], indexer_types=["full", "shared"])
    elif family == "deepseek_v4":
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
        common.pop("intermediate_size")
        config = DeepseekV4Config(
            **common, n_routed_experts=4, num_experts_per_tok=2, head_dim=16,
            partial_rotary_factor=.5, q_lora_rank=8, o_lora_rank=8, o_groups=1,
            index_n_heads=2, index_head_dim=16, index_topk=2, sliding_window=4,
            mlp_layer_types=["hash_moe", "moe"],
            layer_types=["sliding_attention", "compressed_sparse_attention"],
            compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4})
    else:
        config = Qwen3MoeConfig(**common, num_experts=4, num_experts_per_tok=1, head_dim=8)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    config.dtype = torch.float32
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        original = AutoModelForCausalLM.from_config(config).eval()
    source = {name: value.clone() for name, value in original.state_dict().items()}
    target_config = copy.deepcopy(config)
    setattr(target_config, "num_experts" if family == "qwen3" else "n_routed_experts", 2)
    target = {}
    for name, value in source.items():
        if name.endswith(("mlp.gate.weight", "mlp.gate.bias", "mlp.gate.e_score_correction_bias",
                          "experts.gate_up_proj", "experts.down_proj")):
            value = value[[3, 1]]
        elif name.endswith(".tid2eid"):
            from razor.checkpoint import remap_hash_table
            value = remap_hash_table(value, source[name.removesuffix("tid2eid") + "weight"], [3, 1])
        target[name] = value.clone()
    if corrupt == "missing":
        del target["model.layers.0.mlp.experts.down_proj"]
    elif corrupt:
        target["model.layers.1.mlp.experts.down_proj"] += 0.5
    tracker = {"active": None, "order": []}
    states = {"source": source, "target": target}
    monkeypatch.setattr(command.loader, "load_config", lambda path, **kw:
                        copy.deepcopy(config if path == "source" else target_config))
    monkeypatch.setattr(streaming, "CheckpointWeights", lambda path, model, **kw:
                        _MemoryCheckpointWeights(model, states[path], tracker, path))
    batches = [{"input_ids": torch.tensor([[1, 3, 4, 5]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0]])},
               {"input_ids": torch.tensor([[7, 3, 2]]),
                "attention_mask": torch.tensor([[0, 1, 1]])}]
    def check(max_layers=None):
        return command._stream_reference(
            "source", "target", {"main_0": [3, 1], "main_1": [3, 1]}, batches,
            device="cpu", atol=0, rtol=0, verbose=False, max_layers=max_layers)

    if corrupt == "missing":
        with pytest.raises(KeyError, match="down_proj"):
            check()
    else:
        assert check() is not corrupt
        assert tracker["order"] == [("source", 0), ("target", 0), ("source", 1), ("target", 1)]
    assert tracker["active"] is None
    if corrupt is False:
        tracker["order"].clear()
        assert check(max_layers=1)
        assert tracker["active"] is None
        assert tracker["order"] == [("source", 0), ("target", 0)]
    if family == "deepseek_v4" and corrupt is False:
        from razor.adapters import get_adapter

        resident_ref = copy.deepcopy(original)
        resident_target = AutoModelForCausalLM.from_config(copy.deepcopy(target_config)).eval()
        resident_target.load_state_dict(target, strict=True)
        command._slice_reference(
            resident_ref, get_adapter(resident_ref.config), {"main_0": [3, 1], "main_1": [3, 1]},
            target_config=resident_target.config, tensor_verified=True, verbose=False)
        assert verify.check_untouched(resident_ref.state_dict(), resident_target.state_dict(), verbose=False)
        assert verify.compare_logprobs(verify.score_batches(resident_ref, batches),
                                      verify.score_batches(resident_target, batches), verbose=False)

        mixed_target = copy.deepcopy(original)
        mixed_adapter = get_adapter(mixed_target.config)
        mixed_adapter.prune_block(mixed_target.model.layers[1].mlp, torch.tensor([3, 1]), target_top_k=1)
        mixed_adapter.set_num_experts(2)
        mixed_adapter.set_top_k(1)
        mixed_target.config.hash_n_routed_experts = 4
        mixed_ref = copy.deepcopy(original)
        before = mixed_ref.model.layers[0].mlp.gate.tid2eid.clone()
        command._slice_reference(mixed_ref, get_adapter(mixed_ref.config), {"main_1": [3, 1]},
                                 target_config=mixed_target.config, tensor_verified=True, verbose=False)
        assert mixed_ref.model.layers[0].mlp.gate.top_k == 2
        assert mixed_ref.model.layers[1].mlp.gate.top_k == 1
        assert torch.equal(mixed_ref.model.layers[0].mlp.gate.tid2eid, before)
        assert verify.compare_logprobs(verify.score_batches(mixed_ref, batches),
                                      verify.score_batches(mixed_target, batches), verbose=False)


_MODEL_FORMAT_INFO = {"method": "razor", "kept_experts": 2, "experts_per_tok": 1}
_RAW_STORAGE_INFO = {"tensor_only": True, "storage_profile": "safetensors_raw_v1",
                     "kept_experts": 2, "experts_per_tok": 1}


def _storage_pair(tmp_path, *, model_type="qwen2_moe", output_info=None, quantized=False):
    """Write a save_pretrained-shaped pair plus the metadata each side declares."""
    source, target, keep = _states()
    _write_checkpoint(tmp_path / "source", source)
    _write_checkpoint(tmp_path / "target", target)
    config = {"model_type": model_type}
    if quantized:
        config["quantization_config"] = {"quant_method": "fp8"}
    (tmp_path / "source" / "config.json").write_text(json.dumps(config))
    info = tmp_path / "target" / "pruning_info.json"
    if output_info is None:
        info.unlink(missing_ok=True)
    else:
        info.write_text(json.dumps(output_info))
    return str(tmp_path / "source"), str(tmp_path / "target"), keep


@pytest.mark.parametrize("model_type", ["hy_v3", "deepseek_v4", "glm_moe_dsa", "kimi_linear"])
def test_model_format_output_uses_the_tensor_plan_for_native_family_sources(tmp_path, model_type, monkeypatch):
    from razor import checkpoint
    import razor.verify_cmd as command

    src, dst, keep = _storage_pair(tmp_path, model_type=model_type, output_info=_MODEL_FORMAT_INFO)

    def forbidden(*args, **kwargs):
        raise AssertionError("save_pretrained output must not use the raw checkpoint verifier")

    monkeypatch.setattr(checkpoint, "verify_checkpoint", forbidden)
    monkeypatch.setattr(checkpoint, "inspect_checkpoint", forbidden)
    assert not command._uses_native_storage(src, dst)
    assert _tensor_level(src, dst, keep, verbose=False)
    keep["main_0"] = [1, 3]
    assert not _tensor_level(src, dst, keep, verbose=False)


@pytest.mark.parametrize("model_type", ["qwen2_moe", "deepseek_v4"])
@pytest.mark.parametrize("info", [{"tensor_only": True}, {"storage_profile": "safetensors_raw_v1"},
                                  _RAW_STORAGE_INFO])
def test_raw_storage_output_keeps_the_strict_native_verifier(tmp_path, model_type, info, monkeypatch):
    from razor import checkpoint
    import razor.verify_cmd as command

    src, dst, keep = _storage_pair(tmp_path, model_type=model_type, output_info=info)
    assert command._uses_native_storage(src, dst)
    assert not _tensor_level(src, dst, keep, verbose=False)
    calls = []

    def independent(source, destination, keep_sets):
        calls.append((source, destination, keep_sets))
        return {"ok": True, "checked_tensors": 5}

    monkeypatch.setattr(checkpoint, "verify_checkpoint", independent)
    assert _tensor_level(src, dst, keep, verbose=False)
    assert calls == [(src, dst, keep)]


@pytest.mark.parametrize("model_type,quantized,native", [
    ("qwen2_moe", False, False), ("hy_v3", False, True),
    ("qwen2_moe", True, True), ("deepseek_v4", True, True)])
def test_undeclared_output_keeps_the_source_based_native_dispatch(tmp_path, model_type, quantized, native):
    import razor.verify_cmd as command

    src, dst, _ = _storage_pair(tmp_path, model_type=model_type, quantized=quantized)
    assert command._uses_native_storage(src, dst) is native


def test_native_family_source_verifies_its_model_format_export(tmp_path, monkeypatch):
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM
    from razor import checkpoint
    from razor.adapters import get_adapter
    from razor.prune import prune_model
    import razor.verify_cmd as command

    config = Qwen2MoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24,
        moe_intermediate_size=8, shared_expert_intermediate_size=8,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        num_experts=8, num_experts_per_tok=2, max_position_embeddings=32)
    config._attn_implementation = "eager"
    torch.manual_seed(17)
    source = Qwen2MoeForCausalLM(config).eval()
    source.save_pretrained(tmp_path / "source", safe_serialization=True)
    target = copy.deepcopy(source)
    adapter = get_adapter(target.config, name="qwen")
    keep = {block.tag: [0, 2, 4, 6] for block in adapter.moe_blocks(target)}
    prune_model(target, adapter, keep, target_top_k=1, verbose=False)
    target.save_pretrained(tmp_path / "target", safe_serialization=True)
    declared = json.loads((tmp_path / "source" / "config.json").read_text())
    declared["model_type"] = "hy_v3"
    (tmp_path / "source" / "config.json").write_text(json.dumps(declared))
    (tmp_path / "target" / "kept_expert_indices.json").write_text(json.dumps(keep))
    (tmp_path / "target" / "pruning_info.json").write_text(json.dumps(
        dict(_MODEL_FORMAT_INFO, kept_experts=4, keep=keep)))

    def forbidden(*args, **kwargs):
        raise AssertionError("save_pretrained output must not use the raw checkpoint verifier")

    monkeypatch.setattr(checkpoint, "verify_checkpoint", forbidden)
    monkeypatch.setattr(checkpoint, "inspect_checkpoint", forbidden)
    configs = {str(tmp_path / "source"): config, str(tmp_path / "target"): target.config}
    batch = {"input_ids": torch.tensor([[1, 7, 3, 9, 4]]),
             "attention_mask": torch.ones(1, 5, dtype=torch.long)}

    def load_cpu(path, **kwargs):
        model = Qwen2MoeForCausalLM.from_pretrained(
            path, config=copy.deepcopy(configs[str(path)]), local_files_only=True,
            dtype=torch.float32, attn_implementation="eager").eval()
        return model, get_adapter(model.config, name="qwen")

    monkeypatch.setattr(command.loader, "load_model", load_cpu)
    monkeypatch.setattr(command.loader, "load_tokenizer", lambda path, **kwargs: object())
    monkeypatch.setattr(command.calibration, "build_batches", lambda *args, **kwargs: [batch])
    monkeypatch.setattr(command, "resolve_data", lambda data: data)
    for mode in ("tensors", "model"):
        assert run(str(tmp_path / "source"), str(tmp_path / "target"), data="fixed",
                   verbose=False, mode=mode)


def _allow_patterns(monkeypatch, tmp_path, *, include_weights=False):
    """Capture one HF snapshot request without downloading anything."""
    import huggingface_hub
    from razor.loader import resolve_checkpoint

    captured = {}

    def snapshot(**kwargs):
        captured.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    assert resolve_checkpoint("org/model", include_weights=include_weights) == str(tmp_path)
    assert captured["repo_id"] == "org/model"
    return list(captured["allow_patterns"])


def _requested(patterns, names):
    from huggingface_hub.utils import filter_repo_objects

    return set(filter_repo_objects(sorted(names), allow_patterns=patterns))


def test_resolve_checkpoint_requests_every_exported_auxiliary_asset(tmp_path, monkeypatch):
    from razor.prune import _ASSET_SUBDIRECTORIES, _AUX_ASSETS, _LICENSE_FILES

    patterns = _allow_patterns(monkeypatch, tmp_path)
    assets = set(_AUX_ASSETS | _LICENSE_FILES)
    required = assets | {"config.json", "chat_templates/custom.jinja",
                         "encoding/encoding_dsv4.py", "encoding/tokenizer.tiktoken",
                         "encoding/tokenizer.model", "__init__.py"}
    required |= {f"{directory}/{name}" for directory in _ASSET_SUBDIRECTORIES for name in assets}
    required |= {f"{directory}/chat_templates/custom.jinja" for directory in _ASSET_SUBDIRECTORIES}
    assert _requested(patterns, required) == required


@pytest.mark.parametrize("include_weights", [False, True])
def test_resolve_checkpoint_never_requests_non_safetensors_weights(tmp_path, monkeypatch, include_weights):
    patterns = _allow_patterns(monkeypatch, tmp_path, include_weights=include_weights)
    unrelated = {"pytorch_model.bin", "pytorch_model-00001-of-00002.bin", "adapter_model.bin",
                 "tf_model.h5", "flax_model.msgpack", "consolidated.00.pth", "weights.gguf"}
    assert not any(pattern.endswith(".bin") for pattern in patterns)
    assert _requested(patterns, unrelated) == set()
    weights = {"model.safetensors", "model-00001-of-00002.safetensors",
               "model.safetensors.index.json"}
    assert _requested(patterns, weights) == (weights if include_weights else set())


@pytest.mark.parametrize("result", [{"ok": True, "checked_tensors": 7},
                                    {"ok": False, "checked_tensors": 7},
                                    {"ok": True, "checked_tensors": 0}])
def test_native_tensor_dispatch_requires_successful_independent_checks(monkeypatch, result):
    from razor import checkpoint
    import razor.verify_cmd as command

    calls = []
    monkeypatch.setattr(command, "_uses_native_storage", lambda *args: True)

    def independent(src, dst, keep):
        calls.append((src, dst, keep))
        return result

    monkeypatch.setattr(checkpoint, "verify_checkpoint", independent)
    keep = {"main_0": [3, 1]}
    assert command._tensor_level("source", "target", keep, verbose=False) == (
        result["ok"] and result["checked_tensors"] > 0)
    assert calls == [("source", "target", keep)]
