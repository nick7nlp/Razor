"""Qwen3.8 Flash Next: native Qwen4-Exp routing, PLE/HC and raw export."""
import copy
import importlib.util
import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from razor import metrics
from razor.adapters import get_adapter
from razor.checkpoint import inspect_checkpoint, prune_checkpoint, verify_checkpoint
from razor.collect import collect, verify_routing
from razor.loader import _auto_model
from razor.pipeline import _needs_checkpoint_backend


def test_qwen38_raw_export_preserves_auxiliary_state_and_mtp(tmp_path):
    from test_checkpoint import _checkpoint, _get, _hashes

    source, output = tmp_path / "source", tmp_path / "output"
    values, prefix, gate = _checkpoint(source, "qwen4_exp", mtp=True)
    (source / "LICENSE").write_text("Upstream license text\n")
    extras = {
        prefix + ".ple.ple_embedding.ngram_embeddings.0.weight": torch.arange(16).reshape(4, 4).float(),
        prefix + ".attn_hyper_connection.input_mix_weight_down.weight": torch.eye(4),
        "model.visual.patch_embed.proj.weight": torch.arange(8).float(),
    }
    save_file(extras, source / "aux.safetensors")
    original = _hashes(source)
    config = json.loads((source / "config.json").read_text())
    assert _needs_checkpoint_backend(config)
    assert get_adapter(SimpleNamespace(model_type="qwen4_exp_text", num_experts=4,
                                       num_experts_per_tok=2)).name == "qwen"
    meta = inspect_checkpoint(source)
    assert meta["counts"] == {"main_0": 4, "mtp_0": 4}
    keep = {"main_0": [3, 1], "mtp_0": [2, 0]}
    result = prune_checkpoint(source, output, keep, method="razor", aggregation="rms")
    assert result["verification"]["ok"]
    assert result["method"] == "razor" and result["aggregation"] == "rms"
    assert _hashes(source) == original
    for tag, stem in (("main_0", prefix), ("mtp_0", "mtp.layers.0")):
        for suffix in ("mlp.gate.weight", "mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
            key = stem + "." + suffix
            assert torch.equal(_get(output, key), values[key][keep[tag]])
    for key, value in extras.items():
        assert torch.equal(_get(output, key), value)
    assert (output / "LICENSE").read_bytes() == (source / "LICENSE").read_bytes()
    cfg = json.loads((output / "config.json").read_text())
    assert cfg["model_type"] == "qwen4_exp"
    assert cfg["text_config"]["num_experts"] == 2
    assert cfg["text_config"]["num_experts_per_tok"] == 2
    assert cfg["text_config"]["num_nextn_predict_layers"] == 1
    manifest = output / "kept_expert_indices.json"
    bad = dict(keep)
    bad["main_1000"] = bad.pop("mtp_0")
    manifest.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="manifest must list every retained layer"):
        verify_checkpoint(source, output)


def test_qwen38_adapter_discovers_outer_wrapper_mtp():
    def layer():
        result = torch.nn.Module()
        result.mlp = torch.nn.Module()
        result.mlp.gate = torch.nn.Linear(3, 4, bias=False)
        result.mlp.experts = torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(4)])
        return result

    config = SimpleNamespace(model_type="qwen4_exp", text_config=SimpleNamespace(
        num_experts=4, num_experts_per_tok=2))
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model.model.language_model.layers = torch.nn.ModuleList([layer()])
    model.mtp = torch.nn.Module()
    model.mtp.layers = torch.nn.ModuleList([layer()])
    assert [block.tag for block in get_adapter(config).moe_blocks(model)] == ["main_0", "mtp_0"]


@pytest.mark.parametrize("method", ["from_config", "from_pretrained"])
def test_qwen38_loader_never_downgrades_conditional_wrapper(monkeypatch, method):
    import transformers

    config = SimpleNamespace(model_type="qwen4_exp")
    sentinel = object()

    def forbidden(*args, **kwargs):
        raise AssertionError("a conditional checkpoint must not load as text-only")

    monkeypatch.setattr(transformers.AutoModelForCausalLM, method, forbidden)
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, method, lambda *a, **k: sentinel)
    assert _auto_model(method, "fixture", config=config) is sentinel

    def unsupported(*args, **kwargs):
        raise ValueError("Unrecognized configuration class Qwen4ExpConfig")

    monkeypatch.setattr(transformers.AutoModelForImageTextToText, method, unsupported)
    with pytest.raises(ValueError, match="Unrecognized configuration class"):
        _auto_model(method, "fixture", config=config)


@pytest.fixture
def native_qwen38(tmp_path):
    cfg = pytest.importorskip("transformers.models.qwen4_exp.configuration_qwen4_exp")
    device = torch.device(os.environ.get("RAZOR_QWEN38_TEST_DEVICE", "cpu"))
    if device.type == "cpu" and any(importlib.util.find_spec(name) is not None
                                    for name in ("fla", "causal_conv1d")):
        pytest.skip("native optional FLA/causal-conv kernels require CUDA; set RAZOR_QWEN38_TEST_DEVICE")
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("requested CUDA device is unavailable")
    torch.manual_seed(17)
    text = cfg.Qwen4ExpTextConfig(
        vocab_size=64, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        moe_intermediate_size=8, shared_expert_intermediate_size=8,
        num_experts=4, num_experts_per_tok=2, bos_token_id=1, eos_token_id=2,
        pad_token_id=0, layer_types=["linear_attention", "qwen_sparse_attention"],
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=2,
        hc_count=2, hc_lowrank=4, ple_layer_ids=[1], ple_embed_dim=16,
        ngram_size=3, heads_per_ngram=1, ngram_vocab_size_base=13,
        make_ngram_vocab_size_divisible_by=8, split_ngram_parts=2,
        ple_conv_kernel_size=2, indexer_n_heads=1, indexer_kv_heads=1,
        indexer_head_dim=8, indexer_budget=4, indexer_compress_ratio=2,
        rope_parameters={"rope_theta": 10000., "rope_type": "default",
                         "partial_rotary_factor": .5, "mrope_section": [1, 1, 0],
                         "mrope_interleaved": True}, output_gate_type="sigmoid")
    vision = cfg.Qwen4ExpVisionConfig(
        depth=1, hidden_size=16, intermediate_size=24, num_heads=2,
        out_hidden_size=16, patch_size=2, spatial_merge_size=1,
        temporal_patch_size=1, num_position_embeddings=16)
    config = cfg.Qwen4ExpConfig(
        text_config=text.to_dict(), vision_config=vision.to_dict(),
        image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    native = _auto_model("from_config", config=config, dtype=torch.float32).eval()
    assert type(native).__name__ == "Qwen4ExpForConditionalGeneration"
    source = tmp_path / "source"
    native.save_pretrained(source, safe_serialization=True)
    batches = [
        {"input_ids": torch.tensor([[1, 3, 4, 2, 5, 6, 0]]),
         "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0]])},
        {"input_ids": torch.tensor([[1, 9, 8, 7, 6]]),
         "attention_mask": torch.ones(1, 5, dtype=torch.long)},
    ]
    native.to(device)
    batches = [{key: value.to(device) for key, value in batch.items()} for batch in batches]
    return native, batches, source


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qwen38_native_routing_refill_and_hc_logits(native_qwen38, dtype):
    native, batches, _ = native_qwen38
    native.to(dtype)
    adapter = get_adapter(native.config)
    assert adapter.router_spec(native.model.language_model.layers[0].mlp).normalize
    assert verify_routing(native, adapter, batches[0])
    records = collect(native, adapter, batches, expert_chunk=2, verbose=False)
    assert set(records) == {"main_0", "main_1"}
    for record in records.values():
        assert record["refill_supported"]
        assert set(metrics.FIELDS["rcs-refill"]) <= set(record)
        assert torch.isfinite(metrics.score(record, "razor")).all()
        assert int(record["routed_count"].sum()) == 22
    with torch.no_grad():
        for batch in batches:
            out = native(**batch, use_cache=False).logits
            assert out.shape == (*batch["input_ids"].shape, 64)
            assert torch.isfinite(out).all()


@pytest.mark.parametrize("chunk", [None, 3])
@pytest.mark.parametrize("batch_window", [1, 2])
def test_qwen38_stream_preserves_ple_hc_and_qsa(native_qwen38, chunk, batch_window, monkeypatch):
    from razor.streaming import CheckpointWeights, collect_streaming

    native, batches, root = native_qwen38
    resident = collect(native, get_adapter(native.config), batches, expert_chunk=2, verbose=False)
    read, placement = CheckpointWeights.raw, []

    def audited_read(self, name, device="cpu"):
        if "ngram_embedding" in name:
            placement.append(str(device))
            assert torch.device(device).type == "cpu"
        return read(self, name, device)

    monkeypatch.setattr(CheckpointWeights, "raw", audited_read)
    streamed = collect_streaming(
        str(root), batches, expert_chunk=2, dtype=torch.float32, device=next(native.parameters()).device,
        attention_chunk=chunk, batch_window=batch_window,
        attn_implementation="eager", verbose=False)
    assert placement
    assert resident.keys() == streamed.keys() == {"main_0", "main_1"}
    for tag, expected in resident.items():
        assert streamed[tag].keys() == expected.keys()
        assert streamed[tag]["refill_supported"]
        for key, value in expected.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(streamed[tag][key], value, rtol=2e-5, atol=1e-8)
            else:
                assert streamed[tag][key] == value


def test_qwen38_native_export_reload_logits_and_ngram(native_qwen38, tmp_path):
    from razor.loader import load_model
    from razor.prune import prune_model

    native, batches, source = native_qwen38
    keep = {"main_0": [3, 1], "main_1": [2, 0]}
    reference = copy.deepcopy(native)
    prune_model(reference, get_adapter(reference.config), keep)
    output = tmp_path / "pruned"
    result = prune_checkpoint(source, output, keep, method="razor")
    assert result["verification"]["ok"]
    reloaded, _ = load_model(str(output), device_map=str(next(native.parameters()).device),
                             dtype=torch.float32, attn_implementation="eager", verbose=False)
    assert type(reloaded).__name__ == "Qwen4ExpForConditionalGeneration"
    with torch.no_grad():
        for batch in batches:
            expected = reference(**batch, use_cache=False).logits
            actual = reloaded(**batch, use_cache=False).logits
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert reloaded.config.text_config.hc_count == 2
    assert reloaded.config.text_config.ple_layer_ids == [1]
    assert reloaded.config.text_config.num_experts_per_tok == 2


@pytest.mark.parametrize("retain_cpu", [False, True])
def test_force_cpu_conversion_is_separate_from_target_placement(retain_cpu):
    from razor.streaming import CheckpointWeights

    class CpuConversion:
        force_cpu = True
        operations = ()

        def __init__(self):
            self.inputs = []

        def add_tensor(self, target, source, pattern, value):
            self.inputs.append(value)

        def convert(self, target, **kwargs):
            values = [value() for value in self.inputs]
            assert all(value.device.type == "cpu" for value in values)
            return {target: torch.cat(values)}

    weights = CheckpointWeights.__new__(CheckpointWeights)
    weights.model = torch.nn.Linear(2, 4, bias=False, device="meta")
    weights.model.config = SimpleNamespace()
    weights.cpu_slots = {"weight"} if retain_cpu else set()
    weights.slots = weights.model.state_dict()
    weights.groups = {"weight": [("a", "part"), ("b", "part")]}
    weights.rules = {"weight": CpuConversion()}
    weights.tensor = lambda name, device, dtype: torch.ones(2, 2, device=device, dtype=dtype)
    result = weights.converted("weight", "meta")["weight"]
    assert result.device.type == ("cpu" if retain_cpu else "meta")
    assert result.shape == (4, 2)


@pytest.mark.parametrize("family", ["qwen4_exp", "qwen4_exp_text"])
def test_qwen38_model_export_refuses_uninstantiated_mtp(tmp_path, monkeypatch, family):
    from test_checkpoint import _checkpoint, _hashes
    from razor import pipeline

    source, output = tmp_path / "source", tmp_path / "output"
    _checkpoint(source, family, mtp=True)
    original = _hashes(source)
    keep = tmp_path / "keep.json"
    keep.write_text(json.dumps({"main_0": [0, 1]}))

    def forbidden(*args, **kwargs):
        raise AssertionError("unsafe export must be rejected before loading native code or weights")

    monkeypatch.setattr(pipeline.loader, "load_model", forbidden)
    monkeypatch.setattr(pipeline.loader, "load_config", forbidden)
    with pytest.raises(ValueError, match="MTP.*checkpoint"):
        pipeline.prune(str(source), str(output), keep_indices=str(keep), export_mode="model")
    assert not output.exists()
    assert _hashes(source) == original
