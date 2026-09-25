"""Storage-level pruning tests use real, small safetensors checkpoints."""
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from razor.checkpoint import inspect_checkpoint, prune_checkpoint, remap_hash_table, verify_checkpoint


def _hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


def _get(root, key):
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(root / index[key], framework="pt") as handle:
        return handle.get_tensor(key)


def _checkpoint(root, family, *, quantized=False, mtp=False, hash_layer=False):
    root.mkdir()
    count = 4
    config = {"model_type": family, "num_hidden_layers": 1,
              "num_experts": count, "num_local_experts": count,
              "num_experts_per_tok": 2, "rope_scaling": {"type": "custom", "factor": 1.75},
              "custom_model_field": {"literal": "preserve"}, "_name_or_path": "/private/model"}
    if family == "gemma4":
        config["top_k_experts"] = config.pop("num_experts_per_tok")
    if family == "kimi_k3":
        config["num_experts_per_token"] = config.pop("num_experts_per_tok")
    if family in ("glm_moe_dsa", "glm5_next", "deepseek_v4"):
        config["n_routed_experts"] = config.pop("num_experts")
    if family == "deepseek_v4" and hash_layer:
        config["num_hidden_layers"] = 2
        config["mlp_layer_types"] = ["hash_moe", "moe"]
    if mtp:
        config["num_nextn_predict_layers"] = 1
    wrapped = family in ("qwen3_5_moe", "qwen3_6_moe", "glm5_next", "gemma4", "kimi_k3")
    raw = {"model_type": family, "text_config": config} if wrapped else config
    (root / "config.json").write_text(json.dumps(raw))
    values = {"embed.weight": torch.tensor([float("nan"), -0.0, 2.0]), "dense.weight": torch.arange(12).reshape(3, 4).float()}
    prefix = "model.language_model.layers.0" if wrapped else "model.layers.0"
    if family == "deepseek_v4":
        prefix = "layers.0"
    if family == "kimi_k3":
        prefix = "language_model.model.layers.0"
    block = "block_sparse_moe" if family == "kimi_k3" else "ffn" if family == "deepseek_v4" else "mlp"
    gate = "router.proj.weight" if family == "gemma4" else "mlp.router.gate.weight" if family == "hy_v3" else f"{block}.gate.weight"
    values[f"{prefix}.{gate}"] = torch.tensor([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.], [4., 0., 0.]])
    if family == "gemma4":
        values[f"{prefix}.router.per_expert_scale"] = torch.arange(count).float()
        values[f"{prefix}.router.scale"] = torch.arange(count).float() + 11
    elif family == "hy_v3":
        values[f"{prefix}.mlp.expert_bias"] = torch.arange(count).float()
    elif family in ("glm_moe_dsa", "glm5_next", "kimi_k3"):
        values[f"{prefix}.{block}.gate.e_score_correction_bias"] = torch.arange(count).float()
    elif family == "deepseek_v4":
        values[f"{prefix}.ffn.gate.bias"] = torch.arange(count).float()
    fused = family in ("gemma4", "qwen3_5_moe", "qwen3_6_moe", "qwen3_moe")
    if fused:
        expert_prefix = f"{prefix}.experts" if family == "gemma4" else f"{prefix}.mlp.experts"
        for role in ("gate_up_proj", "down_proj"):
            values[f"{expert_prefix}.{role}"] = torch.arange(count * 6).reshape(count, 2, 3).float()
    else:
        roles = ("w1", "w2", "w3") if family in ("deepseek_v4", "kimi_k3") else ("gate_proj", "up_proj", "down_proj")
        for expert in range(count):
            for role in roles:
                base = f"{prefix}.{block}.experts.{expert}.{role}"
                if family == "kimi_k3":
                    values[f"{base}.weight_packed"] = torch.full((2, 3), expert + 1, dtype=torch.uint8)
                    values[f"{base}.weight_scale"] = torch.full((1, 1), expert + 1.5)
                elif quantized:
                    values[f"{base}.weight"] = torch.full((2, 3), expert + 1).to(torch.float8_e4m3fn)
                    scale = "scale" if family == "deepseek_v4" else "weight_scale_inv"
                    values[f"{base}.{scale}"] = torch.full((1, 1), expert + 1.5)
                else:
                    values[f"{base}.weight"] = torch.full((2, 3), expert + 0.25)
    values[f"{prefix}.{block}.shared_experts.weight"] = torch.arange(count).float()
    if hash_layer:
        values["layers.0.ffn.gate.tid2eid"] = torch.tensor([[0, 3], [1, 2], [2, 3], [0, 2]], dtype=torch.int32)
        for key, value in list(values.items()):
            if key.startswith("layers.0.") and not key.endswith("tid2eid"):
                values[key.replace("layers.0.", "layers.1.", 1)] = value.clone()
    if mtp:
        if family == "deepseek_v4":
            mtp_prefix = "mtp.0"
        elif family.startswith("qwen"):
            mtp_prefix = "mtp.layers.0"
        else:
            mtp_prefix = prefix.rsplit(".", 1)[0] + ".1"
        for key, value in list(values.items()):
            if key.startswith(prefix + ".") and not key.endswith("tid2eid"):
                values[mtp_prefix + key[len(prefix):]] = value.clone()
        if mtp_prefix.startswith("mtp."):
            values["mtp.norm.weight"] = torch.arange(3).float()
    keys = list(values)
    halves = [keys[::2], keys[1::2]]
    mapping = {}
    for i, subset in enumerate(halves):
        name = f"model-{i}.safetensors"
        save_file({key: values[key] for key in subset}, root / name)
        mapping.update({key: name for key in subset})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    (root / "tokenizer").mkdir()
    (root / "tokenizer" / "tokenizer.json").write_text('{"literal": "/token/text"}')
    (root / "encoding").mkdir()
    (root / "encoding" / "tokens.tiktoken").write_text("YQ== 0\n")
    (root / "tokenizer_config.json").write_text(json.dumps({"chat_template": "/literal/template", "_name_or_path": "/secret"}))
    (root / "cache_receipt.json").write_text('{"path": "/private/cache"}')
    (root / "modeling_custom.py").write_text("VALUE = 1\n")
    return values, prefix, gate


@pytest.mark.parametrize("family,quantized", [
    ("hy_v3", False), ("deepseek_v4", True), ("glm_moe_dsa", False),
    ("glm5_next", True), ("qwen3_5_moe", False), ("qwen3_6_moe", False),
    ("gemma4", False), ("kimi_k3", True),
])
def test_raw_storage_profiles(tmp_path, family, quantized):
    source, output = tmp_path / "source", tmp_path / "output"
    values, prefix, gate = _checkpoint(source, family, quantized=quantized, mtp=True)
    original = _hashes(source)
    meta = inspect_checkpoint(source)
    assert meta["num_experts"] == 4 and meta["top_k"] == 2
    assert meta["mtp_layers"]
    result = prune_checkpoint(source, output, {"main_0": [3, 1]}, max_shard_size="50B")
    assert result["verification"]["ok"]
    assert _hashes(source) == original
    assert torch.equal(_get(output, f"{prefix}.{gate}"), values[f"{prefix}.{gate}"][[3, 1]])
    assert (output / "tokenizer/tokenizer.json").read_bytes() == (source / "tokenizer/tokenizer.json").read_bytes()
    assert (output / "encoding/tokens.tiktoken").exists()
    assert not (output / "modeling_custom.py").exists()
    assert not (output / "cache_receipt.json").exists()
    config = json.loads((output / "config.json").read_text())
    text = config.get("text_config", config)
    assert text["num_local_experts"] == 2
    assert text["rope_scaling"] == {"type": "custom", "factor": 1.75}
    assert text["custom_model_field"] == {"literal": "preserve"}
    assert "_name_or_path" not in text
    if family == "gemma4":
        assert torch.equal(_get(output, prefix + ".router.scale"), values[prefix + ".router.scale"])
    assert verify_checkpoint(source, output)["ok"]


def test_hash_collision_and_automatic_frequency(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "deepseek_v4", quantized=True, hash_layer=True, mtp=True)
    result = prune_checkpoint(source, output, {"main_1": [1, 3]})
    table = _get(output, "layers.0.ffn.gate.tid2eid")
    assert all(len(set(row)) == 2 for row in table.tolist())
    assert table.min() >= 0 and table.max() < 2
    assert "main_0" in result["keep"] and "mtp_0" in result["keep"]
    assert verify_checkpoint(source, output)["ok"]
    weights = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [1., 0.]])
    original = torch.tensor([[2, 3], [2, 0], [1, 3]], dtype=torch.int32)
    remapped = remap_hash_table(original, weights, [0, 1])
    assert remapped.tolist() == [[0, 1], [1, 0], [1, 0]]


def test_preserve_hash_requires_consent_and_bundles_patch(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    values, _, _ = _checkpoint(source, "deepseek_v4", hash_layer=True)
    with pytest.raises(ValueError, match="trust_remote_code"):
        prune_checkpoint(source, output, {"main_1": [0, 2]}, hash_policy="preserve")
    assert not output.exists()
    prune_checkpoint(source, output, {"main_1": [0, 2]}, hash_policy="preserve", trust_remote_code=True)
    config = json.loads((output / "config.json").read_text())
    assert config["hash_n_routed_experts"] == 4
    assert config["n_routed_experts"] == config["num_local_experts"] == 2
    for key, value in values.items():
        if key.startswith("layers.0."):
            assert torch.equal(_get(output, key), value)
    for name in ("configuration_deepseek_v4_mixed.py", "modeling_deepseek_v4_mixed.py"):
        assert (output / name).is_file()
        compile((output / name).read_text(), name, "exec")
    assert verify_checkpoint(source, output)["ok"]


def test_mtp_require_and_drop(tmp_path):
    source = tmp_path / "source"
    _checkpoint(source, "qwen3_5_moe", mtp=True)
    with pytest.raises(ValueError, match="missing keep set for MTP"):
        prune_checkpoint(source, tmp_path / "require", {"main_0": [0, 2]}, mtp_policy="require")
    output = tmp_path / "drop"
    prune_checkpoint(source, output, {"main_0": [0, 2]}, mtp_policy="drop")
    config = json.loads((output / "config.json").read_text())
    assert config["text_config"]["num_nextn_predict_layers"] == 0
    keys = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    assert not any(key.startswith("mtp.") for key in keys)


def test_missing_main_rejects_and_source_is_unchanged(tmp_path):
    source = tmp_path / "source"
    _checkpoint(source, "deepseek_v4", hash_layer=True)
    original = _hashes(source)
    with pytest.raises(ValueError, match="missing keep set for decoder"):
        prune_checkpoint(source, tmp_path / "bad", {"main_0": [0, 1]})
    assert not (tmp_path / "bad").exists()
    assert _hashes(source) == original
    with pytest.raises(ValueError, match="contain"):
        prune_checkpoint(source, source / "output", {"main_0": [0, 1]})


def test_independent_verifier_detects_corruption(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "hy_v3")
    prune_checkpoint(source, output, {"main_0": [0, 2]})
    key = "model.layers.0.mlp.experts.0.gate_proj.weight"
    index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
    path = output / index[key]
    with safe_open(path, framework="pt") as handle:
        data = {name: handle.get_tensor(name).clone() for name in handle.keys()}
    data[key].add_(1)
    save_file(data, path, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="bytes differ"):
        verify_checkpoint(source, output)


def test_unindexed_mtp_is_discovered(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "qwen3_6_moe", mtp=True)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"] = {key: value for key, value in index["weight_map"].items() if not key.startswith("mtp.")}
    index_path.write_text(json.dumps(index))
    assert inspect_checkpoint(source)["mtp_layers"] == ["mtp_0"]
    assert "mtp_0" in prune_checkpoint(source, output, {"main_0": [0, 1]})["keep"]


def test_existing_output_is_not_overwritten(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "glm_moe_dsa")
    output.mkdir()
    (output / "sentinel").write_text("unchanged")
    with pytest.raises(FileExistsError):
        prune_checkpoint(source, output, {"main_0": [0, 1]})
    assert (output / "sentinel").read_text() == "unchanged"


def test_group_quota_validation(tmp_path):
    source = tmp_path / "source"
    _checkpoint(source, "glm_moe_dsa")
    path = source / "config.json"
    config = json.loads(path.read_text())
    config["n_group"] = 2
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="routing-group quotas"):
        prune_checkpoint(source, tmp_path / "bad", {"main_0": [0, 1]})
    assert prune_checkpoint(source, tmp_path / "good", {"main_0": [0, 2]})["verification"]["ok"]


def test_pro_mxfp4_storage_and_scale_pairing(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    values, _, _ = _checkpoint(source, "deepseek_v4", quantized=True)
    for path in source.glob("*.safetensors"):
        with safe_open(path, framework="pt") as handle:
            data = {key: handle.get_tensor(key).clone() for key in handle.keys()}
        for key in data:
            if ".experts." in key and key.endswith(".weight"):
                data[key] = data[key].to(torch.uint8)
        save_file(data, path)
    prune_checkpoint(source, output, {"main_0": [3, 1]})
    assert _get(output, "layers.0.ffn.experts.0.w1.weight").dtype == torch.uint8
    assert _get(output, "layers.0.ffn.experts.0.w1.scale").item() == 4.5
    assert verify_checkpoint(source, output)["ok"]


def test_orphan_packed_scale_rejects_before_writing(tmp_path):
    source = tmp_path / "source"
    _checkpoint(source, "kimi_k3")
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    removed = [key for key in index["weight_map"] if key.endswith("w1.weight_scale")]
    for path in source.glob("*.safetensors"):
        with safe_open(path, framework="pt") as handle:
            data = {key: handle.get_tensor(key).clone() for key in handle.keys() if key not in removed}
        save_file(data, path)
    for key in removed:
        index["weight_map"].pop(key)
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="paired scale"):
        prune_checkpoint(source, tmp_path / "bad", {"main_0": [0, 2]})
    assert not (tmp_path / "bad").exists()


def test_write_failure_cleans_stage(tmp_path, monkeypatch):
    import razor.checkpoint as checkpoint

    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "hy_v3")
    original = _hashes(source)

    def fail(*args, **kwargs):
        raise OSError("simulated storage failure")

    monkeypatch.setattr(checkpoint, "_write_shard", fail)
    with pytest.raises(OSError, match="simulated"):
        prune_checkpoint(source, output, {"main_0": [0, 2]})
    assert not output.exists()
    assert not list(tmp_path.glob(".razor-*"))
    assert _hashes(source) == original


def test_mixed_code_builds_upstream_blocks(tmp_path):
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "deepseek_v4", hash_layer=True)
    config = DeepseekV4Config(vocab_size=8, hidden_size=8, moe_intermediate_size=4,
                              num_hidden_layers=2, n_routed_experts=4, num_experts_per_tok=2,
                              mlp_layer_types=["hash_moe", "moe"])
    (source / "config.json").write_text(json.dumps(config.to_dict()))
    prune_checkpoint(source, output, {"main_1": [0, 2]}, hash_policy="preserve", trust_remote_code=True)
    mixed_config = AutoConfig.from_pretrained(output, local_files_only=True, trust_remote_code=True)
    block_class = get_class_from_dynamic_module(
        "modeling_deepseek_v4_mixed.DeepseekV4MixedSparseMoeBlock", str(output), local_files_only=True)
    with torch.device("meta"):
        hash_block = block_class(mixed_config, 0)
        regular_block = block_class(mixed_config, 1)
    assert hash_block.gate.weight.shape[0] == 4
    assert regular_block.gate.weight.shape[0] == 2
    assert hash_block.experts.gate_up_proj.shape[0] == 4
    assert regular_block.experts.gate_up_proj.shape[0] == 2
    assert hash_block.experts.config is regular_block.experts.config is mixed_config
    assert mixed_config.n_routed_experts == 2
    model_class = get_class_from_dynamic_module(
        "modeling_deepseek_v4_mixed.DeepseekV4MixedForCausalLM", str(output), local_files_only=True)
    with torch.device("meta"):
        model = model_class(mixed_config)
    assert model.model.layers[0].mlp.experts.gate_up_proj.shape[0] == 4
    assert model.model.layers[1].mlp.experts.gate_up_proj.shape[0] == 2


def test_mixed_reload_converts_original_per_expert_storage(tmp_path):
    from transformers import AutoModelForCausalLM
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

    config = DeepseekV4Config(vocab_size=8, hidden_size=8, moe_intermediate_size=4,
                              num_hidden_layers=2, n_routed_experts=4, num_experts_per_tok=2,
                              num_attention_heads=2, head_dim=4, q_lora_rank=4,
                              o_groups=1, o_lora_rank=4, index_n_heads=1, index_head_dim=4,
                              hc_mult=2, num_nextn_predict_layers=0, partial_rotary_factor=0.5,
                              mlp_layer_types=["hash_moe", "moe"])
    original = DeepseekV4ForCausalLM(config).eval()
    original.model.layers[0].mlp.gate.tid2eid.copy_(torch.tensor([[0, 3]] * 8))
    state, raw = original.state_dict(), {}
    for key, tensor in state.items():
        if ".mlp.experts." in key:
            layer = key.split(".")[2]
            for expert in range(4):
                prefix = f"layers.{layer}.ffn.experts.{expert}"
                if key.endswith("gate_up_proj"):
                    raw[prefix + ".w1.weight"] = tensor[expert, :4].contiguous()
                    raw[prefix + ".w3.weight"] = tensor[expert, 4:].contiguous()
                else:
                    raw[prefix + ".w2.weight"] = tensor[expert].contiguous()
        elif ".mlp.gate." in key:
            raw[key.removeprefix("model.").replace(".mlp.", ".ffn.").replace(".e_score_correction_bias", ".bias")] = tensor
        else:
            raw[key] = tensor
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config.to_dict()))
    save_file(raw, source / "model.safetensors", metadata={"format": "pt"})
    baseline, baseline_info = DeepseekV4ForCausalLM.from_pretrained(source, local_files_only=True, output_loading_info=True)
    assert not any(baseline_info[name] for name in ("missing_keys", "unexpected_keys", "mismatched_keys"))
    for key, tensor in baseline.state_dict().items():
        assert torch.equal(tensor, state[key]), key
    prune_checkpoint(source, output, {"main_1": [3, 1]}, hash_policy="preserve", trust_remote_code=True)
    restored, loading = AutoModelForCausalLM.from_pretrained(
        output, local_files_only=True, trust_remote_code=True, output_loading_info=True)
    assert get_model_conversion_mapping(restored, add_legacy=False)
    assert not any(loading[name] for name in ("missing_keys", "unexpected_keys", "mismatched_keys"))
    for key, tensor in restored.state_dict().items():
        expected = state[key]
        if key.startswith("model.layers.1.mlp.") and (".experts." in key or ".gate." in key):
            expected = expected[[3, 1]]
        assert torch.equal(tensor, expected), key


def test_generation_topk_is_not_an_expert_alias(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "hy_v3")
    path = source / "config.json"
    config = json.loads(path.read_text())
    config["top_k"] = 17
    path.write_text(json.dumps(config))
    prune_checkpoint(source, output, {"main_0": [0, 2]}, target_top_k=1)
    actual = json.loads((output / "config.json").read_text())
    assert actual["top_k"] == 17 and actual["num_experts_per_tok"] == 1


def test_public_budget_and_profile_contract(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "glm_moe_dsa")
    meta = inspect_checkpoint(source)
    assert isinstance(meta, dict) and meta["topk_group"] == 1
    with pytest.raises(ValueError, match="target_experts disagrees"):
        prune_checkpoint(source, output, {"main_0": [0, 2]}, target_experts=3)
    assert not output.exists()
    result = prune_checkpoint(source, output, {"main_0": [0, 2]}, target_experts=2)
    assert result["storage_profile"] == "safetensors_raw_v1"
    assert json.loads((output / "pruning_info.json").read_text())["storage_profile"] == "safetensors_raw_v1"


def test_budget_preflight_without_weights():
    from razor.checkpoint import validate_budget

    metadata = {"family": "glm", "num_experts": 12, "top_k": 4,
                "n_group": 3, "topk_group": 2, "hash_layers": []}
    assert validate_budget(metadata, 6) == 4
    with pytest.raises(ValueError, match="divisible"):
        validate_budget(metadata, 5, 2)
    with pytest.raises(ValueError, match="cannot supply"):
        validate_budget(metadata, 3, 3)
    with pytest.raises(ValueError, match="at least two"):
        validate_budget(metadata, 3, 2)
    with pytest.raises(ValueError, match="source expert count"):
        validate_budget(metadata, 15)
    with pytest.raises(ValueError, match="original top-k"):
        validate_budget(metadata, 6, 5)
    metadata["hash_layers"] = ["main_0"]
    with pytest.raises(ValueError, match="hash routing"):
        validate_budget(metadata, 6, 2)


def test_native_hy3_save_prune_reload(tmp_path):
    from transformers import AutoModelForCausalLM
    from transformers.models.hy_v3.configuration_hy_v3 import HYV3Config
    from transformers.models.hy_v3.modeling_hy_v3 import HYV3ForCausalLM

    source, output = tmp_path / "source", tmp_path / "output"
    config = HYV3Config(vocab_size=16, hidden_size=16, intermediate_size=24,
                       num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                       head_dim=8, moe_intermediate_size=8, num_experts=4,
                       num_experts_per_tok=2, mlp_layer_types=["dense", "sparse"],
                       max_position_embeddings=32)
    model = HYV3ForCausalLM(config).eval()
    model.model.layers[1].mlp.e_score_correction_bias.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
    model.save_pretrained(source)
    assert inspect_checkpoint(source)["layers"] == ["main_1"]
    result = prune_checkpoint(source, output, {"main_1": [3, 1]}, aggregation="sum")
    assert result["aggregation"] == "sum"
    assert json.loads((output / "pruning_info.json").read_text())["aggregation"] == "sum"
    restored = AutoModelForCausalLM.from_pretrained(output, local_files_only=True).eval()
    assert restored.config.num_experts == 2
    assert torch.equal(restored.model.layers[1].mlp.e_score_correction_bias,
                       model.model.layers[1].mlp.e_score_correction_bias[[3, 1]])
    assert torch.equal(restored.model.layers[1].mlp.gate.weight,
                       model.model.layers[1].mlp.gate.weight[[3, 1]])
    with torch.no_grad():
        logits = restored(input_ids=torch.tensor([[1, 2, 3]])).logits
    assert torch.isfinite(logits).all()


def test_empty_hash_mtp_keep_is_explicitly_rejected(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "deepseek_v4", hash_layer=True, mtp=True)
    original = _hashes(source)
    with pytest.raises(ValueError, match="target_experts alone"):
        prune_checkpoint(source, output, {}, target_experts=2)
    assert not output.exists() and not list(tmp_path.glob(".razor-*"))
    assert _hashes(source) == original


@pytest.mark.parametrize("unknown", ["gemma_unrecognized", "qwen_custom", "glm_unknown", "kimi_unlisted"])
def test_unknown_model_type_is_not_inferred_from_substrings(tmp_path, unknown):
    source = tmp_path / "source"
    _checkpoint(source, "gemma4")
    path = source / "config.json"
    config = json.loads(path.read_text())
    config["model_type"] = unknown
    config["text_config"]["model_type"] = unknown
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="unrecognized checkpoint family"):
        inspect_checkpoint(source)


def _set_tensor(root, key, value):
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    name = index["weight_map"].get(key, next(iter(index["weight_map"].values())))
    path = root / name
    with safe_open(path, framework="pt") as handle:
        data = {name: handle.get_tensor(name).clone() for name in handle.keys()}
    data[key] = value
    save_file(data, path, metadata={"format": "pt"})
    index["weight_map"][key] = name
    index_path.write_text(json.dumps(index))


@pytest.mark.parametrize("dtype,scale_shape", [
    (torch.float8_e4m3fn, (4, 1)), (torch.float8_e4m3fn, (1, 1)),
    (torch.float8_e4m3fn, None), (torch.uint8, None),
    (torch.uint8, (4, 1)), (torch.float32, (4, 1)), (torch.float32, (1, 1)),
])
def test_quantized_router_rejects_before_writing(tmp_path, monkeypatch, dtype, scale_shape):
    import razor.checkpoint as checkpoint

    source, output = tmp_path / "source", tmp_path / "output"
    values, prefix, gate = _checkpoint(source, "hy_v3")
    key = f"{prefix}.{gate}"
    _set_tensor(source, key, values[key].to(dtype))
    if scale_shape is not None:
        _set_tensor(source, key.rsplit(".", 1)[0] + ".weight_scale_inv", torch.ones(scale_shape))
    original = _hashes(source)
    monkeypatch.setattr(checkpoint, "_check_publish", lambda *_: pytest.fail("router must fail before publication probe"))
    with pytest.raises(ValueError, match="router"):
        inspect_checkpoint(source)
    with pytest.raises(ValueError, match="router"):
        prune_checkpoint(source, output, {"main_0": [3, 1]})
    assert _hashes(source) == original
    assert not output.exists() and not list(tmp_path.glob(".razor-*"))


def test_preserve_then_remap_has_native_consistent_widths(tmp_path):
    from transformers import AutoConfig
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4SparseMoeBlock

    source, mixed, output = (tmp_path / name for name in ("source", "mixed", "output"))
    _checkpoint(source, "deepseek_v4", hash_layer=True)
    config = DeepseekV4Config(vocab_size=8, hidden_size=8, moe_intermediate_size=4,
                              num_hidden_layers=2, n_routed_experts=4, num_experts_per_tok=2,
                              mlp_layer_types=["hash_moe", "moe"])
    (source / "config.json").write_text(json.dumps(config.to_dict()))
    prune_checkpoint(source, mixed, {"main_1": [0, 2]}, hash_policy="preserve", trust_remote_code=True)
    assert inspect_checkpoint(mixed)["counts"] == {"main_0": 4, "main_1": 2}
    prune_checkpoint(mixed, output, {"main_1": [1, 0]}, hash_policy="remap", trust_remote_code=True)
    assert inspect_checkpoint(output)["counts"] == {"main_0": 2, "main_1": 2}
    assert verify_checkpoint(mixed, output)["ok"]
    restored = AutoConfig.from_pretrained(output, local_files_only=True, trust_remote_code=False)
    assert type(restored) is DeepseekV4Config
    assert restored.hash_n_routed_experts == restored.n_routed_experts == 2
    assert restored.architectures == ["DeepseekV4ForCausalLM"]
    with torch.device("meta"):
        block = DeepseekV4SparseMoeBlock(restored, 0)
    assert block.gate.weight.shape[0] == block.experts.gate_up_proj.shape[0] == 2
    again = tmp_path / "again"
    prune_checkpoint(output, again, {"main_1": [0, 1]})
    assert inspect_checkpoint(again)["counts"] == {"main_0": 2, "main_1": 2}


@pytest.mark.parametrize("verify", [True, False])
def test_bad_generated_config_cannot_be_published(tmp_path, monkeypatch, verify):
    import razor.checkpoint as checkpoint

    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "deepseek_v4", hash_layer=True)
    original_config = checkpoint._config

    def broken(*args):
        config = original_config(*args)
        config["hash_n_routed_experts"] = 4
        return config

    monkeypatch.setattr(checkpoint, "_config", broken)
    with pytest.raises(ValueError, match="tensor expert widths"):
        prune_checkpoint(source, output, {"main_1": [0, 2]}, verify=verify)
    assert not output.exists() and not list(tmp_path.glob(".razor-*"))


def test_verify_checks_actual_config_independently(tmp_path, monkeypatch):
    import razor.checkpoint as checkpoint

    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "deepseek_v4", hash_layer=True)
    prune_checkpoint(source, output, {"main_1": [0, 2]})
    config_path = output / "config.json"
    broken = json.loads(config_path.read_text())
    broken["hash_n_routed_experts"] = 4
    config_path.write_text(json.dumps(broken))
    monkeypatch.setattr(checkpoint, "_config", lambda *_: broken)
    with pytest.raises(ValueError, match="tensor expert widths"):
        verify_checkpoint(source, output)


@pytest.mark.parametrize("damage", ["one_tensor", "whole_shard", "missing_index", "nonstandard_index"])
def test_output_index_must_be_complete_and_standard(tmp_path, damage):
    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "hy_v3", mtp=True)
    prune_checkpoint(source, output, {"main_0": [0, 2]}, max_shard_size="50B")
    index_path = output / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    if damage == "one_tensor":
        index["weight_map"].pop(next(iter(index["weight_map"])))
        index_path.write_text(json.dumps(index))
    elif damage == "whole_shard":
        shard = next(iter(index["weight_map"].values()))
        index["weight_map"] = {key: value for key, value in index["weight_map"].items() if value != shard}
        index_path.write_text(json.dumps(index))
    elif damage == "missing_index":
        index_path.unlink()
    else:
        index_path.rename(output / "other.safetensors.index.json")
    with pytest.raises(ValueError, match="index"):
        verify_checkpoint(source, output)


def test_output_shards_require_standard_format_metadata(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "hy_v3")
    prune_checkpoint(source, output, {"main_0": [0, 2]})
    path = next(output.glob("*.safetensors"))
    with safe_open(path, framework="pt") as handle:
        data = {key: handle.get_tensor(key) for key in handle.keys()}
    save_file(data, path)
    with pytest.raises(ValueError, match="format metadata"):
        verify_checkpoint(source, output)


@pytest.mark.parametrize("indexed", [True, False])
def test_huggingface_snapshot_blob_links_are_supported(tmp_path, indexed):
    import os

    repository = tmp_path / "models--synthetic--tiny"
    source = repository / "snapshots" / ("a" * 40)
    source.parent.mkdir(parents=True)
    blobs = repository / "blobs"
    blobs.mkdir()
    values, prefix, gate = _checkpoint(source, "hy_v3", quantized=True)
    if not indexed:
        (source / "model.safetensors.index.json").unlink()
    for path in list(source.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            blob = blobs / hashlib.sha256(data).hexdigest()
            blob.write_bytes(data)
            path.unlink()
            path.symlink_to(os.path.relpath(blob, path.parent))
    original = _hashes(source)
    assert inspect_checkpoint(source)["num_experts"] == 4
    output = tmp_path / "output"
    prune_checkpoint(source, output, {"main_0": [3, 1]})
    assert verify_checkpoint(source, output)["ok"]
    assert torch.equal(_get(output, f"{prefix}.{gate}"), values[f"{prefix}.{gate}"][[3, 1]])
    assert _hashes(source) == original


@pytest.mark.parametrize("escape", ["wrong_blob_name", "absolute_link", "blob_symlink", "blob_directory_symlink"])
def test_huggingface_layout_does_not_authorize_arbitrary_escapes(tmp_path, escape):
    import os

    repository = tmp_path / "models--synthetic--tiny"
    source = repository / "snapshots" / ("a" * 40)
    source.parent.mkdir(parents=True)
    blobs = repository / "blobs"
    blobs.mkdir()
    _checkpoint(source, "hy_v3")
    path = next(source.glob("*.safetensors"))
    data = path.read_bytes()
    blob = blobs / ("arbitrary" if escape == "wrong_blob_name" else hashlib.sha256(data).hexdigest())
    blob.write_bytes(data)
    if escape == "blob_symlink":
        external = tmp_path / "external"
        blob.rename(external)
        blob.symlink_to(external)
    elif escape == "blob_directory_symlink":
        external = tmp_path / "external_blobs"
        blobs.rename(external)
        blobs.symlink_to(external, target_is_directory=True)
    path.unlink()
    path.symlink_to(blob if escape == "absolute_link" else os.path.relpath(blob, path.parent))
    with pytest.raises(ValueError, match="unsafe"):
        prune_checkpoint(source, tmp_path / "output", {"main_0": [0, 2]})
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("target", ["indexed_shard", "unindexed_shard", "config", "index"])
def test_checkpoint_rejects_noncache_external_symlinks(tmp_path, target):
    source = tmp_path / "source"
    _checkpoint(source, "hy_v3")
    if target == "config":
        path = source / "config.json"
    elif target == "index":
        path = source / "model.safetensors.index.json"
    else:
        path = next(source.glob("*.safetensors"))
        if target == "unindexed_shard":
            (source / "model.safetensors.index.json").unlink()
    external = tmp_path / "external"
    path.rename(external)
    path.symlink_to(external)
    original = external.read_bytes()
    with pytest.raises(ValueError, match="unsafe"):
        prune_checkpoint(source, tmp_path / "output", {"main_0": [0, 2]})
    assert external.read_bytes() == original
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("name", ["pruning_info.json", "kept_expert_indices.json"])
def test_verifier_rejects_external_metadata_symlinks(tmp_path, name):
    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "hy_v3")
    prune_checkpoint(source, output, {"main_0": [0, 2]})
    external = tmp_path / "external.json"
    (output / name).rename(external)
    (output / name).symlink_to(external)
    with pytest.raises(ValueError, match="unsafe"):
        verify_checkpoint(source, output)


@pytest.mark.parametrize("path_kind", ["parent", "absolute", "directory_symlink"])
def test_index_rejects_path_traversal_even_back_into_source(tmp_path, path_kind):
    source = tmp_path / "source"
    _checkpoint(source, "hy_v3")
    path = source / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    if path_kind == "directory_symlink":
        (source / "alias").symlink_to(source, target_is_directory=True)
    for key, value in index["weight_map"].items():
        index["weight_map"][key] = (f"../source/{value}" if path_kind == "parent"
                                    else str(source / value) if path_kind == "absolute" else f"alias/{value}")
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="unsafe"):
        inspect_checkpoint(source)


def _simulate_rename_error(monkeypatch, error):
    import ctypes
    from types import SimpleNamespace

    def fail(*args):
        ctypes.set_errno(error)
        return -1

    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(renameat2=fail))


@pytest.mark.parametrize("error_name", ["EINVAL", "ENOSYS", "ENOTSUP"])
def test_nfs_reserved_publication_without_unsafe_rename(tmp_path, monkeypatch, error_name):
    import errno
    import razor.checkpoint as checkpoint

    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "hy_v3")
    original = _hashes(source)
    _simulate_rename_error(monkeypatch, getattr(errno, error_name))
    monkeypatch.setattr(checkpoint.os, "rename", lambda *_: pytest.fail("unsafe rename fallback"))
    link = checkpoint.os.link
    linked = []

    def checked_link(source_path, target_path, **kwargs):
        if (isinstance(target_path, Path) and target_path.is_absolute()
                and kwargs.get("src_dir_fd") is None and kwargs.get("dst_dir_fd") is None
                and target_path.is_relative_to(output)):
            assert (output / ".razor-incomplete").is_file()
            assert not (output / "config.json").exists()
            linked.append(target_path.relative_to(output))
        link(source_path, target_path, **kwargs)

    monkeypatch.setattr(checkpoint.os, "link", checked_link)
    with pytest.warns(RuntimeWarning, match="NOT atomic"):
        prune_checkpoint(source, output, {"main_0": [0, 2]})
    assert linked[-1] == Path("config.json")
    assert output.stat().st_mode & 0o777 == 0o700
    assert not (output / ".razor-incomplete").exists()
    assert verify_checkpoint(source, output)["ok"]
    assert _hashes(source) == original and not list(tmp_path.glob(".razor-*"))


@pytest.mark.parametrize("error_name", ["EACCES", "EIO", "ENOSPC"])
def test_other_publish_errors_never_trigger_fallback(tmp_path, monkeypatch, error_name):
    import errno
    import razor.checkpoint as checkpoint

    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "hy_v3")
    _simulate_rename_error(monkeypatch, getattr(errno, error_name))
    monkeypatch.setattr(checkpoint, "_publish_reserved", lambda *_: pytest.fail("unexpected fallback"))
    monkeypatch.setattr(checkpoint, "_write_shard", lambda *_: pytest.fail("must fail at preflight"))
    with pytest.raises(OSError) as caught:
        prune_checkpoint(source, output, {"main_0": [0, 2]})
    assert caught.value.errno == getattr(errno, error_name)
    assert not output.exists() and not list(tmp_path.glob(".razor-*"))


@pytest.mark.parametrize("failure", ["io", "racing_file"])
def test_reserved_publish_failure_retains_marker_and_user_files(tmp_path, monkeypatch, failure):
    import errno
    import razor.checkpoint as checkpoint

    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "hy_v3")
    original = _hashes(source)
    _simulate_rename_error(monkeypatch, errno.EINVAL)
    link = checkpoint.os.link

    def fail(source_path, target_path, **kwargs):
        if (isinstance(target_path, Path) and target_path.is_absolute()
                and kwargs.get("src_dir_fd") is None and kwargs.get("dst_dir_fd") is None
                and target_path.is_relative_to(output) and target_path.suffix == ".safetensors"):
            (output / "user-file").write_text("do not delete")
            if failure == "io":
                raise OSError(errno.EIO, "simulated link failure")
            target_path.write_text("do not overwrite")
        link(source_path, target_path, **kwargs)

    monkeypatch.setattr(checkpoint.os, "link", fail)
    with pytest.warns(RuntimeWarning, match="NOT atomic"), pytest.raises(OSError):
        prune_checkpoint(source, output, {"main_0": [0, 2]})
    assert (output / ".razor-incomplete").is_file()
    assert not (output / "config.json").exists()
    assert (output / "user-file").read_text() == "do not delete"
    if failure == "racing_file":
        assert next(output.glob("*.safetensors")).read_text() == "do not overwrite"
    with pytest.raises(RuntimeError, match="incomplete"):
        inspect_checkpoint(output)
    assert _hashes(source) == original and not list(tmp_path.glob(".razor-*"))


def test_incomplete_marker_rejects_before_reading_headers_or_config(tmp_path, monkeypatch):
    import razor.checkpoint as checkpoint

    source = tmp_path / "source"
    _checkpoint(source, "hy_v3")
    (source / ".razor-incomplete").write_text("incomplete")
    monkeypatch.setattr(checkpoint, "_json", lambda *_: pytest.fail("must not read config"))
    for operation in (inspect_checkpoint, checkpoint._headers):
        with pytest.raises(RuntimeError, match=r"incomplete.*\.razor-incomplete"):
            operation(source)


def _remove_tolerating_absent_paths(directory):
    """Delete a tree without hiding real errors when entries vanish mid-walk."""
    import os

    def discard(path):
        try:
            path.rmdir() if path.is_dir() and not path.is_symlink() else path.unlink()
        except FileNotFoundError:
            if path.exists() or path.is_symlink():
                raise

    for parent, directories, files in os.walk(directory, topdown=False):
        for name in files + directories:
            discard(Path(parent, name))
    discard(Path(directory))
    assert not Path(directory).exists()


def test_actual_workspace_filesystem_safe_publication(tmp_path):
    import tempfile
    import warnings
    import razor.checkpoint as checkpoint

    source = tmp_path / "source"
    _checkpoint(source, "hy_v3")
    directory = Path(tempfile.mkdtemp(prefix=".razor-nfs-test-", dir=Path(__file__).resolve().parents[1]))
    try:
        output = directory / "output"
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            prune_checkpoint(source, output, {"main_0": [0, 2]})
        assert verify_checkpoint(source, output)["ok"]
        assert not (output / ".razor-incomplete").exists()
        assert all("NOT atomic" in str(item.message) for item in emitted if item.category is RuntimeWarning)
        before = _hashes(output)
        with pytest.raises(FileExistsError):
            checkpoint._publish(output, output)
        assert _hashes(output) == before
    finally:
        _remove_tolerating_absent_paths(directory)


@pytest.mark.parametrize("reserved", [False, True])
def test_racing_empty_output_is_never_overwritten(tmp_path, monkeypatch, reserved):
    import errno
    import razor.checkpoint as checkpoint

    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "hy_v3")
    original = _hashes(source)
    if reserved:
        _simulate_rename_error(monkeypatch, errno.EINVAL)
    publish = checkpoint._publish
    raced_inode = []

    def race(stage, destination):
        if destination == output:
            output.mkdir()
            raced_inode.append(output.stat().st_ino)
        return publish(stage, destination)

    monkeypatch.setattr(checkpoint, "_publish", race)
    with pytest.raises(FileExistsError):
        prune_checkpoint(source, output, {"main_0": [0, 2]})
    assert output.stat().st_ino == raced_inode[0] and not list(output.iterdir())
    assert _hashes(source) == original and not list(tmp_path.glob(".razor-*"))


@pytest.mark.parametrize("trusted", [False, True])
def test_raw_aux_copies_only_explicit_trusted_code_dependencies(tmp_path, trusted):
    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "hy_v3")
    path = source / "config.json"
    config = json.loads(path.read_text())
    config["auto_map"] = {"AutoModelForCausalLM": "modeling_custom.CustomModel"}
    path.write_text(json.dumps(config))
    (source / "modeling_custom.py").write_text("from .required_helper import VALUE\nclass CustomModel: pass\n")
    (source / "required_helper.py").write_text("VALUE = 1\n")
    (source / "unrelated.py").write_text("PRIVATE_PATH = '/private/local'\n")
    (source / "training_config.yaml").write_text("private_path: /private/local\n")
    if not trusted:
        with pytest.raises(ValueError, match="trust_remote_code"):
            prune_checkpoint(source, output, {"main_0": [0, 2]}, trust_remote_code=False)
        assert not output.exists() and not list(tmp_path.glob(".razor-*"))
        return
    prune_checkpoint(source, output, {"main_0": [0, 2]}, trust_remote_code=True)
    assert (output / "modeling_custom.py").exists()
    assert (output / "required_helper.py").exists()
    assert not (output / "unrelated.py").exists()
    assert not (output / "training_config.yaml").exists()


@pytest.mark.parametrize("name", ["configuration_deepseek_v4_mixed.py", "modeling_deepseek_v4_mixed.py"])
@pytest.mark.parametrize("damage", ["executable", "comment", "missing"])
def test_verifier_generated_templates_are_exact_bytes_without_execution(tmp_path, name, damage):
    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "deepseek_v4", hash_layer=True)
    prune_checkpoint(source, output, {"main_1": [0, 2]}, hash_policy="preserve", trust_remote_code=True)
    path = output / name
    if damage == "missing":
        path.unlink()
    else:
        prefix = b'raise RuntimeError("verification must not execute checkpoint code")\n' if damage == "executable" else b"# changed\n"
        path.write_bytes(prefix + path.read_bytes())
    with pytest.raises(ValueError, match="mixed-width"):
        verify_checkpoint(source, output)


@pytest.mark.parametrize("location", ["disk", "info", "external"])
@pytest.mark.parametrize("damage", ["missing_hash", "missing_mtp", "dropped_mtp", "wrapped_ids"])
def test_verifier_requires_complete_keep_manifests(tmp_path, location, damage):
    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "deepseek_v4", hash_layer=True, mtp=True)
    result = prune_checkpoint(source, output, {"main_1": [0, 2]},
                              mtp_policy="drop" if damage == "dropped_mtp" else "router_norm")
    keep = json.loads(json.dumps(result["keep"]))
    if damage.startswith("missing_"):
        keep.pop("main_0" if damage == "missing_hash" else "mtp_0")
    elif damage == "dropped_mtp":
        keep["mtp_0"] = [2, 3]
    else:
        keep["main_1"] = {"keep": keep["main_1"]}
    external = None
    if location == "disk":
        (output / "kept_expert_indices.json").write_text(json.dumps(keep))
        external = result["keep"]
    elif location == "info":
        path = output / "pruning_info.json"
        info = json.loads(path.read_text())
        info["keep"] = keep
        path.write_text(json.dumps(info))
    else:
        external = keep
    with pytest.raises(ValueError, match="keep manifest"):
        verify_checkpoint(source, output, external)


@pytest.mark.parametrize("location", ["disk", "info", "external"])
def test_verifier_rejects_disagreeing_keep_declarations(tmp_path, location):
    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "hy_v3")
    prune_checkpoint(source, output, {"main_0": [0, 2]})
    changed = {"main_0": [2, 0]}
    external = None
    if location == "disk":
        (output / "kept_expert_indices.json").write_text(json.dumps(changed))
        external = {"main_0": [0, 2]}
    elif location == "info":
        path = output / "pruning_info.json"
        info = json.loads(path.read_text())
        info["keep"] = changed
        path.write_text(json.dumps(info))
    else:
        external = changed
    with pytest.raises(ValueError, match="keep manifest"):
        verify_checkpoint(source, output, external)


@pytest.mark.parametrize("field,value", [
    ("kept_experts", 3), ("kept_experts", True), ("kept_experts", 2.0),
    ("original_experts", 9), ("family", "qwen"), ("experts_per_tok", 1),
    ("hash_policy", "preserve"), ("mtp_policy", "drop"),
    ("tensor_only", False), ("storage_profile", "other"),
])
def test_verifier_checks_pruning_info_declarations_even_with_overrides(tmp_path, field, value):
    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "hy_v3")
    prune_checkpoint(source, output, {"main_0": [0, 2]})
    path = output / "pruning_info.json"
    info = json.loads(path.read_text())
    info[field] = value
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="pruning_info"):
        verify_checkpoint(source, output, hash_policy="remap", mtp_policy="router_norm", target_top_k=2)


@pytest.mark.parametrize("damage", ["not_object", "empty", "missing_keep", "missing_count"])
def test_verifier_rejects_incomplete_pruning_info(tmp_path, damage):
    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "hy_v3")
    prune_checkpoint(source, output, {"main_0": [0, 2]})
    path = output / "pruning_info.json"
    info = json.loads(path.read_text())
    if damage == "not_object":
        info = []
    elif damage == "empty":
        info = {}
    else:
        info.pop("keep" if damage == "missing_keep" else "kept_experts")
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="pruning_info|keep manifest"):
        verify_checkpoint(source, output)


@pytest.mark.parametrize("hash_policy", ["remap", "preserve"])
@pytest.mark.parametrize("with_info", [False, True])
def test_verifier_optional_info_contract_and_no_automatic_selection(tmp_path, monkeypatch, hash_policy, with_info):
    import razor.checkpoint as checkpoint

    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "deepseek_v4", hash_layer=True, mtp=True)
    result = prune_checkpoint(source, output, {"main_1": [0, 2]}, hash_policy=hash_policy,
                              mtp_policy="drop", trust_remote_code=hash_policy == "preserve")
    if not with_info:
        (output / "pruning_info.json").unlink()
    monkeypatch.setattr(checkpoint, "_rank", lambda *_: pytest.fail("verification cannot fill missing keep declarations"))
    checked = verify_checkpoint(source, output, result["keep"], hash_policy=hash_policy, mtp_policy="drop")
    assert checked["ok"] and checked["checked_pruning_info"] is with_info
    assert checked["checked_generated_code"] == (2 if hash_policy == "preserve" else 0)


def test_verifier_requires_disk_keep_even_with_external_keep(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "hy_v3")
    result = prune_checkpoint(source, output, {"main_0": [0, 2]})
    (output / "kept_expert_indices.json").unlink()
    with pytest.raises(ValueError, match="keep manifest"):
        verify_checkpoint(source, output, result["keep"])


def test_verifier_explicitly_excludes_tokenizer_and_other_auxiliary_files(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "hy_v3")
    prune_checkpoint(source, output, {"main_0": [0, 2]})
    (output / "tokenizer/tokenizer.json").unlink()
    checked = verify_checkpoint(source, output)
    assert checked["ok"] and checked["auxiliary_files_verified"] is False
    assert "excludes tokenizer/other auxiliary files" in checked["scope"]


@pytest.mark.parametrize("location", ["disk", "info", "external"])
def test_verifier_only_caller_keep_accepts_a_file_path(tmp_path, location):
    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, "hy_v3")
    result = prune_checkpoint(source, output, {"main_0": [0, 2]})
    reference = tmp_path / "external_keep.json"
    reference.write_text(json.dumps(result["keep"]))
    if location == "external":
        assert verify_checkpoint(source, output, reference)["ok"]
        return
    if location == "disk":
        (output / "kept_expert_indices.json").write_text(json.dumps(str(reference)))
    else:
        path = output / "pruning_info.json"
        info = json.loads(path.read_text())
        info["keep"] = str(reference)
        path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="keep manifest"):
        verify_checkpoint(source, output)


@pytest.mark.parametrize("failure", ["already_deleted", "still_present", "dangling_symlink", "permission", "not_empty"])
def test_publish_probe_cleanup_ignores_only_an_absent_directory(tmp_path, monkeypatch, failure):
    import errno
    import razor.checkpoint as checkpoint

    remove = checkpoint.shutil.rmtree
    seen = []
    error = {"permission": errno.EACCES, "not_empty": errno.ENOTEMPTY}.get(failure, errno.ENOENT)

    def failing_cleanup(directory, *args, **kwargs):
        directory = Path(directory)
        seen.append(directory)
        if failure in ("already_deleted", "dangling_symlink"):
            try:
                remove(directory, *args, **kwargs)
            except FileNotFoundError:
                if directory.exists() or directory.is_symlink():
                    raise
            if failure == "dangling_symlink":
                directory.symlink_to(tmp_path / "missing-target", target_is_directory=True)
        raise OSError(error, "simulated probe cleanup error", str(directory))

    monkeypatch.setattr(checkpoint.shutil, "rmtree", failing_cleanup)
    if failure == "already_deleted":
        checkpoint._check_publish(tmp_path)
        assert len(seen) == 1 and not seen[0].exists() and not seen[0].is_symlink()
    else:
        with pytest.raises(OSError) as caught:
            checkpoint._check_publish(tmp_path)
        assert caught.value.errno == error
        assert len(seen) == 1 and (seen[0].exists() or seen[0].is_symlink())
