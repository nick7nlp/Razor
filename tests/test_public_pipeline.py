"""Offline checks of the public pruning APIs."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from razor import pipeline
from razor.cli import main
from razor.prune import build_info, prune_model, validate_keep_indices


@pytest.fixture
def source(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM

    monkeypatch.chdir(tmp_path)
    torch.manual_seed(7)
    path = tmp_path / "source"
    config = Qwen3MoeConfig(
        vocab_size=16, hidden_size=16, intermediate_size=32,
        moe_intermediate_size=8, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        num_experts=4, num_experts_per_tok=2, max_position_embeddings=64,
        pad_token_id=1, bos_token_id=0, eos_token_id=0,
    )
    config._attn_implementation = "eager"
    Qwen3MoeForCausalLM(config).eval().save_pretrained(path)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "a": 2, "b": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(path)
    data = tmp_path / "data.json"
    data.write_text(json.dumps([{"text": "a b a b"}, {"text": "b a b a"}]))
    return path, data


def _hashes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("method", ["razor", "reap", "ean", "frequency"])
def test_public_prune_reload(source, tmp_path, method):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src, data = source
    original = _hashes(src)
    out = tmp_path / method
    pipeline.prune(str(src), str(out), method=method, ratio=0.5,
                   data=str(data), max_len=8, num_batches=1,
                   dtype=torch.float32, device_map="cpu", cache=False, verbose=False)
    model = AutoModelForCausalLM.from_pretrained(out, local_files_only=True).eval()
    assert model.config.num_experts == 2
    assert model.config.num_experts_per_tok == 2
    tokenizer = AutoTokenizer.from_pretrained(out, local_files_only=True)
    with torch.no_grad():
        logits = model(**tokenizer("a b", return_tensors="pt")).logits
    assert torch.isfinite(logits).all()
    assert _hashes(src) == original
    assert (out / "tokenizer.json").read_bytes() == (src / "tokenizer.json").read_bytes()
    for filename in ("config.json", "pruning_info.json"):
        assert str(tmp_path) not in (out / filename).read_text()


def test_external_selection_and_lower_topk(source, tmp_path):
    from transformers import AutoModelForCausalLM

    src, _ = source
    keep = tmp_path / "keep.json"
    keep.write_text(json.dumps({"main_0": [3, 1]}))
    out = tmp_path / "external"
    assert main(["prune", "--model", str(src), "--out", str(out),
                 "--keep-indices", str(keep), "--target-top-k", "1",
                 "--dtype", "float32"]) == 0
    model = AutoModelForCausalLM.from_pretrained(out, local_files_only=True).eval()
    assert model.config.num_experts_per_tok == 1
    assert json.loads((out / "kept_expert_indices.json").read_text()) == {"main_0": [3, 1]}
    from razor.verify_cmd import run
    assert run(str(src), str(out), tensors_only=True, verbose=False)
    with torch.no_grad():
        assert torch.isfinite(model(input_ids=torch.tensor([[2, 3]])).logits).all()


@pytest.mark.parametrize("bad", [[1, 1], [-1, 2], [0, 4], [True, 2], [0.5, 2], []])
def test_invalid_selections_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_keep_indices({"main_0": bad}, {"main_0": 4}, 1)


def test_invalid_later_layer_does_not_mutate_model():
    from test_end_to_end import Config, TinyMoE
    from razor.adapters import get_adapter

    model = TinyMoE("fused", Config())
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError):
        prune_model(model, get_adapter(model.config, "qwen"),
                    {"main_0": [0, 1], "main_1": [0, 1], "main_2": [0, 99]})
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_cache_identity_and_explicit_sweep_directory(source, tmp_path):
    src, data = source
    key = pipeline._cache_key(str(src), str(data), 1, 1, 8, 0)
    changed = tmp_path / "other" / data.name
    changed.parent.mkdir()
    changed.write_text(json.dumps([{"text": "a"}]))
    assert pipeline._cache_key(str(src), str(changed), 1, 1, 8, 0) != key
    data.write_text(json.dumps([{"text": "b"}]))
    assert pipeline._cache_key(str(src), str(data), 1, 1, 8, 0) != key
    outputs = pipeline.sweep(str(src), str(tmp_path / "grid"), methods=["reap", "frequency"],
                              ratios=[0.5], data=str(data), num_batches=1, max_len=8,
                              device_map="cpu", dtype=torch.float32, verbose=False,
                              out_dir=str(tmp_path / "saliency"))
    assert len(outputs) == 2 and all(Path(p, "config.json").is_file() for p in outputs)
    assert (tmp_path / "saliency" / "collection.json").is_file()


@pytest.mark.parametrize("method", ["reap", "ean", "frequency"])
def test_top1_baseline_pipeline(source, tmp_path, method):
    from transformers import AutoModelForCausalLM

    src, data = source
    config_path = src / "config.json"
    cfg = json.loads(config_path.read_text())
    cfg["num_experts_per_tok"] = 1
    config_path.write_text(json.dumps(cfg))
    out = tmp_path / method
    pipeline.prune(str(src), str(out), method=method, ratio=0.5,
                   data=str(data), max_len=8, num_batches=1,
                   dtype=torch.float32, device_map="cpu", cache=False, verbose=False)
    model = AutoModelForCausalLM.from_pretrained(out, local_files_only=True).eval()
    with torch.no_grad():
        assert torch.isfinite(model(input_ids=torch.tensor([[2, 3]])).logits).all()
    with pytest.raises(ValueError, match="top_k"):
        pipeline.prune(str(src), str(tmp_path / "unavailable"), method="razor", ratio=0.5)


def test_retained_count_can_match_a_lower_topk(source, tmp_path):
    from transformers import AutoModelForCausalLM

    src, _ = source
    keep = tmp_path / "single.json"
    keep.write_text(json.dumps({"main_0": [2]}))
    out = tmp_path / "one"
    pipeline.prune(str(src), str(out), keep_indices=str(keep), target_top_k=1,
                   dtype=torch.float32, verbose=False)
    model = AutoModelForCausalLM.from_pretrained(out, local_files_only=True).eval()
    assert model.config.num_experts == model.config.num_experts_per_tok == 1
    with torch.no_grad():
        assert torch.isfinite(model(input_ids=torch.tensor([[2, 3]])).logits).all()


def test_existing_output_is_not_overwritten(source, tmp_path):
    src, data = source
    out = tmp_path / "existing"
    out.mkdir()
    marker = out / "marker"
    marker.write_text("untouched")
    with pytest.raises(FileExistsError):
        pipeline.prune(str(src), str(out), ratio=0.5, data=str(data))
    assert marker.read_text() == "untouched"


@pytest.mark.parametrize("method", ["razor", "reap", "ean", "frequency"])
def test_grouped_glm_pipeline(source, tmp_path, method):
    import shutil
    from transformers import AutoModelForCausalLM, Glm4MoeConfig, Glm4MoeForCausalLM

    src, data = source
    grouped = tmp_path / "grouped"
    cfg = Glm4MoeConfig(
        vocab_size=16, hidden_size=16, intermediate_size=32, moe_intermediate_size=8,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, n_routed_experts=8, n_shared_experts=1,
        num_experts_per_tok=2, first_k_dense_replace=0,
        n_group=2, topk_group=1, norm_topk_prob=True, routed_scaling_factor=1.5,
        max_position_embeddings=64, pad_token_id=1, bos_token_id=0, eos_token_id=0,
    )
    cfg._attn_implementation = "eager"
    Glm4MoeForCausalLM(cfg).eval().save_pretrained(grouped)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (src / name).exists():
            shutil.copy2(src / name, grouped / name)
    out = tmp_path / ("glm-" + method)
    pipeline.prune(str(grouped), str(out), method=method, ratio=0.5,
                   data=str(data), max_len=8, num_batches=1,
                   dtype=torch.float32, device_map="cpu", cache=False, verbose=False)
    keep = json.loads((out / "kept_expert_indices.json").read_text())["main_0"]
    assert sum(i < 4 for i in keep) == sum(i >= 4 for i in keep) == 2
    model = AutoModelForCausalLM.from_pretrained(out, local_files_only=True).eval()
    assert model.config.n_routed_experts == 4
    with torch.no_grad():
        assert torch.isfinite(model(input_ids=torch.tensor([[2, 3]])).logits).all()


def test_tokenizer_metadata_does_not_copy_local_provenance(source, tmp_path):
    src, _ = source
    config = src / "tokenizer_config.json"
    value = json.loads(config.read_text())
    value["_name_or_path"] = str(tmp_path / "private-source")
    value["api_key"] = "example-not-a-real-credential"
    config.write_text(json.dumps(value))
    keep = tmp_path / "keep.json"
    keep.write_text(json.dumps({"main_0": [0, 2]}))
    out = tmp_path / "sanitized"
    pipeline.prune(str(src), str(out), keep_indices=str(keep), verbose=False)
    exported = json.loads((out / "tokenizer_config.json").read_text())
    assert "_name_or_path" not in exported and "api_key" not in exported
    assert exported["tokenizer_class"] == value["tokenizer_class"]
    assert (src / "tokenizer.json").read_bytes() == (out / "tokenizer.json").read_bytes()


def test_exported_provenance_has_no_input_locations():
    info = build_info("reap", "/private/checkpoint", 8, 4, 2, 2, 3,
                       saliency_path="/private/scores.pt", calibration="/private/data.json")
    assert "/private" not in json.dumps(info)


@pytest.mark.parametrize("filename", ["tokenizer_config.json", "processor_config.json",
                                      "preprocessor_config.json", "video_preprocessor_config.json"])
def test_nested_auxiliary_metadata_is_sanitized(source, tmp_path, filename):
    from transformers import AutoTokenizer

    src, _ = source
    config = src / filename
    value = json.loads(config.read_text()) if config.exists() else {}
    value["init_kwargs"] = {"API_Key": "synthetic-secret", "cache_dir": "/private/cache",
                            "records": [{"Password": "synthetic-secret", "path": r"C:\private\file"}],
                            "nested": {"source_model": "private/model", "size": 16}}
    if filename == "tokenizer_config.json":
        value["additional_special_tokens"] = ["/", "~/", "file://", r"C:\token"]
        value["chat_template"] = "/{{ messages[0]['content'] }}"
    config.write_text(json.dumps(value))
    original = _hashes(src)
    keep = tmp_path / "keep.json"
    keep.write_text(json.dumps({"main_0": [0, 2]}))
    out = tmp_path / "sanitized-nested"
    pipeline.prune(str(src), str(out), keep_indices=str(keep), verbose=False)
    text = (out / filename).read_text()
    exported = json.loads(text)
    assert "synthetic-secret" not in text and "private" not in text
    assert exported["init_kwargs"]["nested"]["size"] == 16
    if filename == "tokenizer_config.json":
        assert exported["additional_special_tokens"] == value["additional_special_tokens"]
        assert exported["chat_template"] == value["chat_template"]
    assert AutoTokenizer.from_pretrained(out, local_files_only=True)("a b")["input_ids"] == [2, 3]
    assert _hashes(src) == original


@pytest.mark.parametrize("filename", ["merges.txt", "vocab.txt", "tokenizer_config.json"])
def test_tokenizer_content_changes_invalidate_cache(source, filename):
    import os

    src, _ = source
    path = src / filename
    path.write_text("synthetic-a")
    before = pipeline._identity(str(src))
    stat = path.stat()
    path.write_text("synthetic-b")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert pipeline._identity(str(src)) != before


def test_sweep_accepts_explicit_none_collection_directory(source, tmp_path):
    src, data = source
    root = tmp_path / "grid-none"
    outputs = pipeline.sweep(str(src), str(root), methods=["reap"], ratios=[0.5],
                              data=str(data), num_batches=1, max_len=8,
                              device_map="cpu", dtype=torch.float32, verbose=False, out_dir=None)
    assert len(outputs) == 1 and Path(outputs[0], "config.json").is_file()
    assert (root / "saliency" / "collection.json").is_file()


@pytest.mark.parametrize("operation", ["sweep", "sweep-cache", "collect"])
def test_collection_cannot_write_inside_source(source, tmp_path, monkeypatch, operation):
    src, data = source
    before = _hashes(src)

    def unexpected(*args, **kwargs):
        pytest.fail("source path validation must run before tokenizer loading")

    monkeypatch.setattr(pipeline.loader, "load_tokenizer", unexpected)
    with pytest.raises(ValueError, match="outside the source"):
        if operation == "collect":
            pipeline.collect_saliency(str(src), str(data), out_dir=str(src / "cache"))
        else:
            root = src / "grid" if operation == "sweep" else tmp_path / "grid"
            options = {"out_dir": str(src / "cache")} if operation == "sweep-cache" else {}
            pipeline.sweep(str(src), str(root), methods=["reap"], ratios=[0.5], data=str(data), **options)
    assert _hashes(src) == before
    assert not (tmp_path / "grid").exists()


@pytest.mark.parametrize("experts,groups,top_k,ratios", [
    (8, 2, 2, [0.5, 0.75]), (8, 2, 2, [0.5, 0.375]), (12, 3, 3, [0.25, 0.5]),
])
@pytest.mark.parametrize("operation", ["prune", "sweep"])
def test_group_budget_rejected_before_collection(source, tmp_path, monkeypatch,
                                                  experts, groups, top_k, ratios, operation):
    from transformers import Glm4MoeConfig

    src, data = source
    cfg = Glm4MoeConfig(n_routed_experts=experts, num_experts_per_tok=top_k,
                        n_group=groups, topk_group=1)
    monkeypatch.setattr(pipeline.loader, "load_config", lambda *args, **kwargs: cfg)

    def unexpected(*args, **kwargs):
        pytest.fail("all group budgets must be checked before collecting or loading weights")

    monkeypatch.setattr(pipeline, "collect_saliency", unexpected)
    monkeypatch.setattr(pipeline.loader, "load_model", unexpected)
    root = tmp_path / "invalid-grid"
    with pytest.raises(ValueError, match="group"):
        if operation == "sweep":
            pipeline.sweep(str(src), str(root), methods=["reap"], ratios=ratios, data=str(data))
        else:
            pipeline.prune(str(src), str(root), method="reap", ratio=ratios[-1], data=str(data))
    assert not root.exists()


@pytest.mark.parametrize("raw", ['{"main_0":[0,1],"main_0":[2,3]}',
                                  '{"main_0":{"keep":[0,1],"keep":[2,3]}}'])
def test_duplicate_keep_json_keys_are_rejected_before_loading(source, tmp_path, monkeypatch, raw):
    src, _ = source
    keep = tmp_path / "duplicate.json"
    keep.write_text(raw)

    def unexpected(*args, **kwargs):
        pytest.fail("duplicate keys must be rejected before loading weights")

    monkeypatch.setattr(pipeline.loader, "load_model", unexpected)
    out = tmp_path / "invalid-keep"
    with pytest.raises(ValueError, match="duplicate JSON key"):
        pipeline.prune(str(src), str(out), keep_indices=str(keep))
    assert not out.exists()


def test_late_saturation_blocks_razor_after_serialization(source, tmp_path):
    from razor.adapters import get_adapter
    from razor.collect import SaliencyCollector

    src, _ = source
    config_path = src / "config.json"
    config = json.loads(config_path.read_text())
    config["norm_topk_prob"] = True
    config_path.write_text(json.dumps(config))
    model, adapter = pipeline.loader.load_model(str(src), dtype=torch.float32, device_map="cpu", verbose=False)
    block = adapter.moe_blocks(model)[0].module
    assert adapter.router_spec(block).normalize is True
    collector = SaliencyCollector(get_adapter(model.config)).install(model)
    try:
        with torch.no_grad():
            block.gate.weight.zero_()
            block.gate.weight[0, 0] = 1000.0
            hidden = torch.zeros(1, 2, model.config.hidden_size)
            hidden[:, :, 1] = 1.0
            collector.set_attention_mask(torch.ones(1, 2, dtype=torch.long))
            block(hidden)
            assert collector.finalize()["main_0"]["razor_supported"] is True
            hidden[0, 0, 0] = 1.0
            block(hidden)
        sal = tmp_path / "saturated.pt"
        torch.save(collector.finalize(), sal)
    finally:
        collector.remove()
    loaded = pipeline._load_saliency(sal)["main_0"]
    assert loaded["razor_supported"] is False and "counterfactual_delta_mean" not in loaded
    with pytest.raises((ValueError, KeyError), match="RAZOR|razor|counterfactual"):
        pipeline.prune(str(src), str(tmp_path / "razor"), ratio=0.5, saliency=str(sal))
    with pytest.raises((ValueError, KeyError), match="RAZOR|razor|counterfactual"):
        pipeline.sweep(str(src), str(tmp_path / "late-grid"), methods=["reap", "razor"],
                       ratios=[0.5], saliency=str(sal))
    assert not (tmp_path / "late-grid").exists()
    pipeline.prune(str(src), str(tmp_path / "reap"), method="reap", ratio=0.5,
                   saliency=str(sal), dtype=torch.float32, verbose=False)
    assert (tmp_path / "reap" / "config.json").is_file()


@pytest.mark.parametrize("method,total,square", [
    ("razor", "rcs_refill_sum", "rcs_refill_square_sum"),
    ("rcs-refill", "rcs_refill_sum", "rcs_refill_square_sum"),
    ("rcs-loo", "counterfactual_delta_sum", "counterfactual_delta_square_sum"),
    ("rcs", "rcs_sum", "rcs_square_sum"),
    ("reap", "weighted_ean_sum", "weighted_ean_square_sum"),
    ("ean", "ean_sum", "ean_square_sum"),
])
def test_explicit_rms_and_sum_aggregation(method, total, square):
    from razor import metrics

    rec = {"routed_count": torch.tensor([2, 2, 1, 0]),
           total: torch.tensor([4., 5., 3., 0.]),
           square: torch.tensor([16., 12.5, 9., 0.])}
    torch.testing.assert_close(metrics.score(rec, method, aggregation="rms"),
                               torch.tensor([8., 6.25, 9., 0.]).sqrt())
    assert metrics.keep_indices(rec, method, 2, aggregation="rms") == [0, 2]
    assert metrics.keep_indices(rec, method, 2, aggregation="sum") == [0, 1]
    del rec[square]
    with pytest.raises(KeyError, match="RMS"):
        metrics.score(rec, method, aggregation="rms")


def test_additive_pack_import_preserves_aggregation(tmp_path):
    from razor import metrics

    state = {"count": torch.tensor([2., 2., 1., 0.]),
             "ean_sum": torch.tensor([4., 5., 3., 0.]),
             "reap_sum": torch.tensor([4., 5., 3., 0.]),
             "d1_sum": torch.tensor([4., 5., 3., 0.]),
             "rcs_sum": torch.tensor([4., 5., 3., 0.]),
             "refill_sum": torch.tensor([4., 5., 3., 0.]),
             "ean_sq": torch.tensor([16., 12.5, 9., 0.]),
             "reap_sq": torch.tensor([16., 12.5, 9., 0.]),
             "d1_sq": torch.tensor([16., 12.5, 9., 0.]),
             "rcs_sq": torch.tensor([16., 12.5, 9., 0.]),
             "refill_sq": torch.tensor([16., 12.5, 9., 0.])}
    path = tmp_path / "statistics.pt"
    torch.save({"state": {"main_0": state}, "top_k": 2, "n_tokens": 5,
                "source": "private-source-marker"}, path)
    imported = pipeline._load_saliency(path)
    rec = imported["main_0"]
    assert "source" not in imported and "source" not in rec
    assert rec["total_tokens"] == 5
    for method in ("razor", "rcs", "rcs-loo", "rcs-refill", "reap", "ean"):
        assert metrics.keep_indices(rec, method, 2, aggregation="mean") == [1, 2]
        assert metrics.keep_indices(rec, method, 2) == [0, 2], "default must be RMS"
        assert metrics.keep_indices(rec, method, 2, aggregation="rms") == [0, 2]
        assert metrics.keep_indices(rec, method, 2, aggregation="sum") == [0, 1]
    frequency = metrics.score(rec, "frequency")
    assert frequency.tolist() == [2, 2, 1, 0] and not frequency.is_floating_point()
    state["count"][0] = 1.5
    torch.save({"state": {"main_0": state}}, path)
    with pytest.raises(ValueError, match="integers"):
        pipeline._load_saliency(path)


def test_native_chat_encoder_receives_tool_links_and_fails_closed():
    from razor.calibration import render

    record = {"messages": [{"role": "tool", "content": "result", "tool_call_id": "call_a", "name": "lookup"}],
              "tools": [{"type": "function", "function": {"name": "lookup"}}]}

    class Encoder:
        def render_record(self, incoming):
            assert incoming == record
            return "encoded record"

        def apply_chat_template(self, *args, **kwargs):
            pytest.fail("native checkpoint encoder must not fall through to chat_template")

    assert render([record], Encoder(), verbose=False) == ["encoded record"]

    class BrokenEncoder(Encoder):
        def render_record(self, incoming):
            raise ValueError("encoder failed")

    with pytest.raises(ValueError, match="encoder failed"):
        render([record], BrokenEncoder(), verbose=False)


@pytest.mark.parametrize("family,quantized", [
    ("hy_v3", False), ("deepseek_v4", True), ("glm_moe_dsa", False),
    ("glm5_next", True), ("qwen3_5_moe", False), ("qwen3_6_moe", False),
    ("gemma4", False), ("kimi_k3", True),
])
def test_public_native_storage_prune_without_loading_weights(tmp_path, monkeypatch, family, quantized):
    from test_checkpoint import _checkpoint
    from razor.checkpoint import verify_checkpoint

    src = tmp_path / "source"
    _checkpoint(src, family, quantized=quantized, mtp=True, hash_layer=family == "deepseek_v4")
    ordinary = "main_1" if family == "deepseek_v4" else "main_0"
    saliency = tmp_path / "scores.pt"
    torch.save({ordinary: {"expert_frequency": torch.ones(4, dtype=torch.long),
                           "rcs_refill_rms": torch.tensor([2., 1., 4., 3.]),
                           "total_tokens": 4}}, saliency)

    def forbidden(*args, **kwargs):
        pytest.fail("native storage export must not construct a full model or AutoConfig")

    monkeypatch.setattr(pipeline.loader, "load_model", forbidden)
    monkeypatch.setattr(pipeline.loader, "load_config", forbidden)
    out = tmp_path / "pruned"
    result = pipeline.prune(str(src), str(out), ratio=.5, saliency=str(saliency),
                             aggregation="rms", verbose=False)
    assert Path(result) == out
    assert verify_checkpoint(src, out)["ok"]
    info = json.loads((out / "pruning_info.json").read_text())
    assert info["aggregation"] == "rms"
    assert json.loads((out / "kept_expert_indices.json").read_text())[ordinary] == [2, 3]


def test_public_native_sweep_uses_one_source_pack(tmp_path, monkeypatch):
    from test_checkpoint import _checkpoint
    from razor.checkpoint import verify_checkpoint

    src = tmp_path / "source"
    _checkpoint(src, "glm5_next", quantized=True)
    sal = tmp_path / "scores.pt"
    torch.save({"state": {"main_0": {"count": torch.ones(4),
                                     "refill_sum": torch.arange(4).float(),
                                     "reap_sum": torch.arange(4).float(),
                                     "refill_sq": torch.arange(4).float().square(),
                                     "reap_sq": torch.arange(4).float().square()}}}, sal)
    monkeypatch.setattr(pipeline, "collect_saliency", lambda *a, **kw: pytest.fail("supplied pack must be reused"))
    outputs = pipeline.sweep(str(src), str(tmp_path / "grid"), saliency=str(sal),
                              methods=["razor", "reap"], ratios=[.25, .5],
                              aggregation="rms", verbose=False)
    assert len(outputs) == 4
    assert all(verify_checkpoint(src, out)["ok"] for out in outputs)


@pytest.mark.parametrize("family", ["hy_v3", "glm4_moe_lite", "glm_moe_dsa", "deepseek_v4", "qwen3_5_moe", "gemma4"])
@pytest.mark.parametrize("method", ["razor", "reap", "ean", "frequency"])
def test_public_native_stream_collect_export_reload(source, tmp_path, monkeypatch, family, method):
    import shutil
    from razor.adapters import get_adapter
    from razor.checkpoint import verify_checkpoint

    tokenizer_src, data = source
    src = tmp_path / "native"
    if family in ("hy_v3", "glm4_moe_lite"):
        from test_model_support import tiny_model
        native = tiny_model("HYV3" if family == "hy_v3" else "Glm4MoeLite", monkeypatch)
        native.save_pretrained(src)
    elif family in ("qwen3_5_moe", "gemma4"):
        from test_model_support import experiment_model
        native = experiment_model("Qwen3_5Wrapper" if family == "qwen3_5_moe" else "Gemma4")
        native.save_pretrained(src, save_original_format=False)
    else:
        from test_streaming import _tiny_native
        native, _ = _tiny_native(src, family)
        if family == "deepseek_v4":
            native.save_pretrained(src, save_original_format=False)
    original_count = get_adapter(native.config).num_experts
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (tokenizer_src / name).is_file():
            shutil.copy2(tokenizer_src / name, src / name)
    token_cfg = src / "tokenizer_config.json"
    config = json.loads(token_cfg.read_text())
    config["chat_template"] = "{% for message in messages %}{{ message.content }}{% endfor %}"
    token_cfg.write_text(json.dumps(config))
    out = tmp_path / "native-pruned"
    pipeline.prune(str(src), str(out), method=method, ratio=.5, data=str(data),
                   num_batches=1, max_len=8, expert_chunk=2, dtype=torch.float32,
                   execution="streaming", stream_device="cpu", verbose=False, cache=False)
    assert verify_checkpoint(src, out)["ok"]
    restored, adapter = pipeline.loader.load_model(str(out), dtype=torch.float32,
                                                    device_map="cpu", verbose=False)
    assert adapter.num_experts == original_count // 2
    with torch.no_grad():
        result = restored(input_ids=torch.tensor([[2, 3, 2, 3]]), use_cache=False)
    assert torch.isfinite(result.logits).all()


def test_cli_native_export_mtp_error_requires_explicit_selection(tmp_path):
    from test_checkpoint import _checkpoint
    from razor.checkpoint import verify_checkpoint

    src = tmp_path / "source"
    _checkpoint(src, "hy_v3", mtp=True)
    keep = tmp_path / "keep.json"
    keep.write_text(json.dumps({"main_0": [0, 2]}))
    out = tmp_path / "output"
    args = ["prune", "--model", str(src), "--out", str(out),
            "--keep-indices", str(keep), "--mtp-policy", "error"]
    with pytest.raises(ValueError, match="missing keep set for MTP"):
        main(args)
    assert not out.exists()
    keep.write_text(json.dumps({"main_0": [0, 2], "main_1": [1, 3]}))
    assert main(args) == 0
    assert verify_checkpoint(src, out)["ok"]


@pytest.mark.parametrize("family,mtp_tag", [("hy_v3", "main_1"), ("qwen3_5_moe", "mtp_0")])
@pytest.mark.parametrize("mtp_first", [True, False])
def test_public_mtp_drop_ignores_auxiliary_keep_budget(tmp_path, family, mtp_tag, mtp_first):
    from test_checkpoint import _checkpoint
    from razor.checkpoint import inspect_checkpoint, verify_checkpoint

    src = tmp_path / "source"
    _checkpoint(src, family, mtp=True)
    original = _hashes(src)
    entries = [(mtp_tag, {"keep": [0, 1, 2, 3]}), ("main_0", [3, 1])]
    keep = tmp_path / "keep.json"
    keep.write_text(json.dumps(dict(entries if mtp_first else reversed(entries))))
    out = tmp_path / "output"
    assert pipeline.prune(str(src), str(out), keep_indices=str(keep),
                          mtp_policy="drop", verbose=False) == str(out)
    assert verify_checkpoint(src, out)["ok"]
    meta = inspect_checkpoint(out)
    assert meta["num_experts"] == 2 and meta["counts"] == {"main_0": 2}
    assert meta["mtp_layers"] == []
    config = json.loads((out / "config.json").read_text())
    assert config.get("text_config", config)["num_nextn_predict_layers"] == 0
    assert json.loads((out / "kept_expert_indices.json").read_text()) == {"main_0": [3, 1]}
    assert _hashes(src) == original


def test_public_mtp_drop_requires_a_surviving_decoder_keep_set(tmp_path, monkeypatch):
    from test_checkpoint import _checkpoint
    from razor import checkpoint

    src = tmp_path / "source"
    _checkpoint(src, "qwen3_5_moe", mtp=True)
    keep = tmp_path / "keep.json"
    keep.write_text(json.dumps({"mtp_0": [0, 1, 2, 3]}))
    out = tmp_path / "output"
    monkeypatch.setattr(checkpoint, "prune_checkpoint",
                        lambda *a, **kw: pytest.fail("budget must fail before writing"))
    with pytest.raises(ValueError, match="ordinary decoder experts"):
        pipeline.prune(str(src), str(out), keep_indices=str(keep), mtp_policy="drop", verbose=False)
    assert not out.exists()


def test_stream_window_configuration_reaches_collector(source, tmp_path, monkeypatch):
    from razor import streaming
    from razor.cli import build_parser, _calib_kwargs

    src, data = source
    args = build_parser().parse_args(["saliency", "--model", str(src), "--out", str(tmp_path / "scores"),
                                      "--execution", "streaming", "--stream-batch-window", "2"])
    assert _calib_kwargs(args)["stream_batch_window"] == 2
    calls = []

    def collect(model, batches, **kwargs):
        calls.append(kwargs)
        return {"main_0": {"expert_frequency": torch.ones(4, dtype=torch.long)}}

    monkeypatch.setattr(streaming, "collect_streaming", collect)
    pipeline.collect_saliency(str(src), str(data), str(tmp_path / "scores"),
                              execution="streaming", stream_device="cpu", stream_batch_window=2,
                              num_batches=1, max_len=8, verbose=False)
    assert calls[0]["batch_window"] == 2
    assert pipeline._cache_key(str(src), str(data), 1, 1, 8, 0, stream_batch_window=1) != (
        pipeline._cache_key(str(src), str(data), 1, 1, 8, 0, stream_batch_window=2))


@pytest.mark.parametrize("exporter", ["checkpoint", "model"])
def test_export_filters_identity_and_checkpoint_provenance(source, tmp_path, exporter):
    from transformers import AutoModelForCausalLM

    src, _ = source
    private = {"author": "synthetic-private-author",
               "authors": [{"name": "synthetic-private-person"}],
               "checkpoint": "synthetic-private-repo/weights"}
    for name in ("config.json", "tokenizer_config.json"):
        path = src / name
        value = json.loads(path.read_text())
        value.update(private)
        value["extra_config"] = {"keep_value": 17, "records": [private]}
        path.write_text(json.dumps(value))
    (src / "LICENSE").write_text("Copyright Synthetic Authors; Apache-2.0")
    original = _hashes(src)
    keep = tmp_path / "keep.json"
    keep.write_text(json.dumps({"main_0": [3, 1]}))
    out = tmp_path / "exported"
    pipeline.prune(str(src), str(out), keep_indices=str(keep), export_mode=exporter,
                   dtype=torch.float32, device_map="cpu", verbose=False)
    for name in ("config.json", "tokenizer_config.json"):
        exported = json.loads((out / name).read_text())
        assert not set(private) & set(exported)
        assert "synthetic-private" not in json.dumps(exported)
        assert exported["extra_config"] == {"keep_value": 17, "records": [{}]}
    assert (out / "LICENSE").read_bytes() == (src / "LICENSE").read_bytes()
    restored = AutoModelForCausalLM.from_pretrained(out, local_files_only=True, dtype=torch.float32)
    with torch.no_grad():
        assert torch.isfinite(restored(input_ids=torch.tensor([[2, 3]]), use_cache=False).logits).all()
    assert _hashes(src) == original


def test_public_config_rejects_provenance_but_preserves_algorithm_and_tokens():
    from razor.prune import _public_config

    private = "synthetic-private-marker"
    value = {
        "metadata": {"url": f"https://{private}.internal/run", "email": f"{private}@example.invalid"},
        "text_config": {"training_provenance": {"path": f"../{private}/checkpoint"},
                        "credentials": {"bearer": private},
                        "num_experts": 8, "rope_scaling": {"rope_type": "yarn", "factor": 16.0},
                        "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
                        "custom_algorithm": {"score_function": "sigmoid", "normalize": False}},
        "init_kwargs": {"endpoint": f"http://{private}.internal", "location": f"../{private}/checkpoint",
                        "user": f"{private}@example.invalid", "num_tokens": 32},
        "chat_template": "{{ messages[0]['content'] }} / ../token http://host.internal token@example.invalid",
        "additional_special_tokens": ["/", "../", "token@example.invalid", "http://host.internal"],
        "added_tokens_decoder": {"0": {"content": "password", "special": True}},
        "license": "Apache-2.0", "homepage": "https://huggingface.co/public/model",
    }
    clean = _public_config(value)
    assert private not in json.dumps(clean)
    assert clean["text_config"] == {key: item for key, item in value["text_config"].items()
                                    if key not in {"training_provenance", "credentials"}}
    assert clean["init_kwargs"]["num_tokens"] == 32
    for name in ("chat_template", "additional_special_tokens", "added_tokens_decoder", "license", "homepage"):
        assert clean[name] == value[name]
    assert "metadata" in value


@pytest.mark.parametrize("remote_code", [False, True])
def test_auxiliary_copy_uses_asset_allowlist_only(tmp_path, remote_code):
    from razor.prune import copy_aux_files

    src, dst = tmp_path / "source", tmp_path / "out"
    src.mkdir()
    dst.mkdir()
    (src / "config.json").write_text(json.dumps({"model_type": "qwen3_moe"}))
    allowed = {"vocab.txt": "a\nb\n", "tokenizer.model": "synthetic-token-content",
               "LICENSE": "Copyright Synthetic Authors; Apache-2.0",
               "NOTICE": "Public source: https://example.org/model",
               "chat_templates/default.jinja": "{{ messages[0].content }}",
               "tokenizer/tokenizer.json": '{"literal":"/token/text"}',
               "tokenizer/chat_templates/default.jinja": "{{ messages[0].content }}",
               "processor/vocab.txt": "a\nb\n",
               "encoding/tokens.tiktoken": "YQ== 0\n",
               "encoding/sentencepiece.model": "synthetic-token-content"}
    rejected = {"train.py": "raise RuntimeError('synthetic-no-execution')\n",
                "training_config.yaml": "credential: synthetic-private-marker\n",
                "processor/notes.txt": "synthetic-private-marker",
                "encoding/arbitrary.json": '{"private":"synthetic-private-marker"}',
                "encoding/deep/tokens.tiktoken": "synthetic-private-marker",
                "tokenizer/receipt.json": '{"private":"synthetic-private-marker"}',
                "tokenizer/train.py": "raise RuntimeError('synthetic-private-marker')\n",
                "tokenizer/training_config.yaml": "private: synthetic-private-marker\n",
                "tokenizer/misc.bin": "synthetic-private-marker",
                "private.model": "synthetic-private-marker",
                "chat_templates/notes.txt": "synthetic-private-marker"}
    for name, content in {**allowed, **rejected}.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    original = _hashes(src)
    copy_aux_files(src, dst, remote_code=remote_code, verbose=False)
    assert set(_hashes(dst)) == set(allowed)
    assert _hashes(src) == original
    for name in allowed:
        assert (src / name).read_bytes() == (dst / name).read_bytes()


def _trusted_export_source(src, *, auto_map=True):
    sources = {
        "configuration_public_fixture.py": (
            "# Copyright Synthetic Authors, Apache-2.0\n"
            "from transformers import Qwen3MoeConfig\n"
            "class PublicFixtureConfig(Qwen3MoeConfig):\n"
            "    model_type = 'qwen3_moe'\n"),
        "modeling_public_fixture.py": (
            "# Copyright Synthetic Authors, Apache-2.0\n"
            "from transformers import Qwen3MoeForCausalLM\n"
            "from .configuration_public_fixture import PublicFixtureConfig\n"
            "from .public_helper import fixture_scale\n"
            "class PublicFixtureForCausalLM(Qwen3MoeForCausalLM):\n"
            "    config_class = PublicFixtureConfig\n"
            "    fixture_scale = fixture_scale\n"),
        "public_helper.py": "from .public_constant import SCALE\nfixture_scale = SCALE\n",
        "public_constant.py": "SCALE = 1.25\n",
        "train.py": "raise RuntimeError('unrelated synthetic training code executed')\n",
    }
    if auto_map:
        sources["modeling_public_fixture.py"] = sources["modeling_public_fixture.py"].replace(
            "from .public_helper import fixture_scale\n",
            "from .public_helper import fixture_scale\nfrom .public_constant import SCALE\n")
    for name, text in sources.items():
        (src / name).write_text(text)
    path = src / "config.json"
    config = json.loads(path.read_text())
    config["architectures"] = ["PublicFixtureForCausalLM"]
    if auto_map:
        config["auto_map"] = {
            "AutoConfig": "synthetic-private/repository--configuration_public_fixture.PublicFixtureConfig",
            "AutoModelForCausalLM": "synthetic-private/repository--modeling_public_fixture.PublicFixtureForCausalLM",
        }
    path.write_text(json.dumps(config))
    return config, {name: text for name, text in sources.items() if name != "train.py"}


@pytest.mark.parametrize("auto_map", [False, True])
def test_authorized_source_closure_exports_and_reloads(source, tmp_path, auto_map):
    import shutil
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from razor.prune import _public_config, copy_aux_files
    from razor.loader import load_checkpoint_class

    src, _ = source
    config, expected = _trusted_export_source(src, auto_map=auto_map)
    original = _hashes(src)
    out = tmp_path / "trusted-output"
    out.mkdir()
    copy_aux_files(src, out, remote_code=True, verbose=False)
    (out / "config.json").write_text(json.dumps(_public_config(config, code_root=out)))
    for weights in src.glob("*.safetensors"):
        shutil.copyfile(weights, out / weights.name)
    assert {path.name for path in out.glob("*.py")} == set(expected)
    for name, text in expected.items():
        assert (out / name).read_text() == text
    assert "synthetic-private" not in (out / "config.json").read_text()
    assert _hashes(src) == original
    if auto_map:
        restored = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True,
                                                       local_files_only=True, dtype=torch.float32).eval()
        assert type(AutoConfig.from_pretrained(out, trust_remote_code=True,
                                               local_files_only=True)).__name__ == "PublicFixtureConfig"
    else:
        cls = load_checkpoint_class(out, class_name="PublicFixtureForCausalLM", trust_remote_code=True)
        restored = cls.from_pretrained(out, local_files_only=True, dtype=torch.float32).eval()
    assert restored.fixture_scale == 1.25
    tokens = AutoTokenizer.from_pretrained(out, local_files_only=True)("a b", return_tensors="pt")
    with torch.no_grad():
        assert torch.isfinite(restored(**tokens).logits).all()


@pytest.mark.parametrize("failure", ["not-authorized", "missing-class", "missing-module", "missing-dependency"])
def test_auto_map_is_never_localized_without_usable_authorized_source(source, tmp_path, failure):
    from razor.prune import copy_aux_files

    src, _ = source
    _trusted_export_source(src)
    if failure == "missing-class":
        (src / "modeling_public_fixture.py").write_text("class DifferentClass: pass\n")
    elif failure == "missing-module":
        (src / "modeling_public_fixture.py").unlink()
    elif failure == "missing-dependency":
        (src / "public_constant.py").unlink()
    out = tmp_path / "invalid-output"
    out.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        copy_aux_files(src, out, remote_code=failure != "not-authorized", verbose=False)
    assert not list(out.rglob("*.py"))
    assert not (out / "config.json").exists()


def test_public_config_without_copied_code_refuses_external_auto_map():
    from razor.prune import _public_config

    with pytest.raises(ValueError, match="auto_map"):
        _public_config({"auto_map": {"AutoConfig": "synthetic-private/repository--missing.Missing"}})


@pytest.mark.parametrize("auto_map", [False, True])
def test_auxiliary_dependency_packages_are_copied_without_unrelated_files(tmp_path, auto_map):
    from razor.prune import copy_aux_files

    src, out = tmp_path / "source", tmp_path / "out"
    (src / "support").mkdir(parents=True)
    out.mkdir()
    config = {"auto_map": {"AutoConfig": "configuration_fixture.Fixture"}} if auto_map else {"architectures": ["Fixture"]}
    (src / "config.json").write_text(json.dumps(config))
    code = {"configuration_fixture.py": "from .support import SCALE\nclass Fixture: pass\n",
            "support/__init__.py": "from .constants import SCALE\n",
            "support/constants.py": "SCALE = 2\n"}
    for name, text in code.items():
        (src / name).write_text(text)
    (src / "support" / "notes.txt").write_text("synthetic-private-marker")
    (src / "support" / "unused.py").write_text("raise RuntimeError('unused')\n")
    if auto_map:
        with pytest.raises(ValueError, match="auto_map.*direct root-module"):
            copy_aux_files(src, out, remote_code=True, verbose=False)
        assert not list(out.iterdir())
    else:
        copy_aux_files(src, out, remote_code=True, verbose=False)
        assert set(_hashes(out)) == set(code)


def test_save_pruned_uses_atomic_no_overwrite_publication(source, tmp_path, monkeypatch):
    from razor import checkpoint
    from razor.prune import save_pruned
    from types import SimpleNamespace

    src, _ = source
    out = tmp_path / "racing-output"
    publish = checkpoint._publish
    calls = []

    def race(stage, destination):
        destination.mkdir()
        calls.append(destination.stat().st_ino)
        publish(stage, destination)

    def save(stage, **kwargs):
        (stage / "config.json").write_text('{"num_experts":2}')

    adapter = SimpleNamespace(set_num_experts=lambda count: None, set_top_k=lambda count: None,
                              postprocess_config=lambda config, info: None)
    monkeypatch.setattr(checkpoint, "_publish", race)
    with pytest.raises(FileExistsError):
        save_pruned(SimpleNamespace(config=object(), save_pretrained=save), adapter, out, str(src),
                    {"kept_experts": 2, "experts_per_tok": 2}, {"main_0": [0, 2]}, verbose=False)
    assert out.stat().st_ino == calls[0]
    assert not list(out.iterdir())
    assert not list(tmp_path.glob(".razor-*"))


@pytest.mark.parametrize("exporter", ["raw", "model"])
def test_real_exports_share_private_metadata_filter_and_source_closure(source, tmp_path, exporter):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from razor.adapters import get_adapter
    from razor.checkpoint import prune_checkpoint, verify_checkpoint
    from razor.prune import prune_model, save_pruned

    src, _ = source
    model = AutoModelForCausalLM.from_pretrained(src, local_files_only=True, dtype=torch.float32)
    config, expected = _trusted_export_source(src)
    config["metadata"] = {"source": "http://synthetic-private.internal", "email": "synthetic@example.invalid"}
    config["custom_algorithm"] = {"score_function": "sigmoid", "scale": 1.5}
    (src / "config.json").write_text(json.dumps(config))
    (src / "processor_config.json").write_text(json.dumps({"metadata": {"private": "synthetic-private"},
                                                            "size": {"height": 16, "width": 16}}))
    (src / "training_config.yaml").write_text("credential: synthetic-private\n")
    (src / "processor").mkdir()
    (src / "processor" / "notes.txt").write_text("synthetic-private")
    (src / "LICENSE").write_text("Copyright Synthetic Authors; Apache-2.0")
    original = _hashes(src)
    out = tmp_path / ("real-" + exporter)
    keep = {"main_0": [0, 2]}
    if exporter == "raw":
        prune_checkpoint(src, out, keep, trust_remote_code=True)
        assert verify_checkpoint(src, out)["ok"]
    else:
        model.config.auto_map = config["auto_map"]
        model.config.metadata = config["metadata"]
        model.config.custom_algorithm = config["custom_algorithm"]
        adapter = get_adapter(model.config)
        prune_model(model, adapter, keep, verbose=False)
        save_pruned(model, adapter, out, str(src), {"kept_experts": 2, "experts_per_tok": 2},
                    keep, remote_code=True, verbose=False)
    assert {path.name for path in out.glob("*.py")} == set(expected)
    for name, text in expected.items():
        assert (out / name).read_text() == text
    assert not (out / "training_config.yaml").exists()
    assert not (out / "processor").exists()
    for name in ("config.json", "processor_config.json"):
        assert "synthetic-private" not in (out / name).read_text()
    assert json.loads((out / "config.json").read_text())["custom_algorithm"] == config["custom_algorithm"]
    assert (out / "LICENSE").read_bytes() == (src / "LICENSE").read_bytes()
    assert _hashes(src) == original
    restored = AutoModelForCausalLM.from_pretrained(out, local_files_only=True,
                                                   trust_remote_code=True, dtype=torch.float32).eval()
    assert restored.config.num_experts == 2 and restored.fixture_scale == 1.25
    tokenizer = AutoTokenizer.from_pretrained(out, local_files_only=True)
    with torch.no_grad():
        assert torch.isfinite(restored(**tokenizer("a b", return_tensors="pt")).logits).all()


@pytest.mark.parametrize("remote_code", [False, True])
def test_raw_export_without_declared_code_does_not_copy_python_or_misc(source, tmp_path, remote_code):
    from razor.checkpoint import prune_checkpoint

    src, _ = source
    (src / "train.py").write_text("raise RuntimeError('synthetic-private training code')\n")
    (src / "arbitrary_config.yaml").write_text("secret: synthetic-private\n")
    (src / "processor").mkdir()
    (src / "processor" / "notes.txt").write_text("synthetic-private")
    out = tmp_path / "raw-assets"
    prune_checkpoint(src, out, {"main_0": [0, 2]}, trust_remote_code=remote_code)
    assert not list(out.rglob("*.py")) and not list(out.rglob("*.yaml"))
    assert not (out / "processor").exists()
    assert (out / "tokenizer.json").read_bytes() == (src / "tokenizer.json").read_bytes()


@pytest.mark.parametrize("filename", ["tokenizer.json", "tokenizer_config.json", "modeling_fixture.py"])
def test_auxiliary_copy_rejects_source_symlink_escape(tmp_path, filename):
    from razor.prune import copy_aux_files

    src, out = tmp_path / "source", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    private = tmp_path / "synthetic-private"
    private.write_text("class Fixture: pass\n" if filename.endswith(".py") else "{}")
    (src / filename).symlink_to(private)
    if filename.endswith(".py"):
        (src / "config.json").write_text(json.dumps({"auto_map": {"AutoModel": "modeling_fixture.Fixture"}}))
    with pytest.raises(ValueError, match="unsafe"):
        copy_aux_files(src, out, remote_code=True, verbose=False)
    assert not list(out.iterdir())


def test_auxiliary_copy_reads_hf_snapshot_blobs_but_never_writes_through_links(tmp_path):
    from razor.prune import copy_aux_files

    repo = tmp_path / "models--synthetic--fixture"
    src = repo / "snapshots" / ("a" * 40)
    blobs = repo / "blobs"
    src.mkdir(parents=True)
    blobs.mkdir()
    content = b'{"version":"1.0"}'
    blob = blobs / hashlib.sha256(content).hexdigest()
    blob.write_bytes(content)
    (src / "tokenizer.json").symlink_to("../../blobs/" + blob.name)
    out = tmp_path / "out"
    out.mkdir()
    copy_aux_files(src, out, verbose=False)
    assert (out / "tokenizer.json").read_bytes() == content
    assert not (out / "tokenizer.json").is_symlink()
    with pytest.raises(ValueError, match="output asset"):
        copy_aux_files(out, src, verbose=False)
    assert blob.read_bytes() == content


def test_nested_tokenizer_assets_reload_and_configs_are_sanitized(source, tmp_path):
    from transformers import AutoTokenizer
    from razor.prune import copy_aux_files

    src, _ = source
    nested = src / "tokenizer"
    nested.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (src / name).is_file():
            (src / name).rename(nested / name)
    path = nested / "tokenizer_config.json"
    value = json.loads(path.read_text())
    value["metadata"] = {"endpoint": "http://synthetic-private.internal", "email": "synthetic@example.invalid"}
    value["credentials"] = {"key": "synthetic-private"}
    value["chat_template"] = "{{ messages[0].content }} /literal/token"
    path.write_text(json.dumps(value))
    (src / "processor").mkdir()
    (src / "processor" / "processor_config.json").write_text(json.dumps({
        "metadata": {"location": "../synthetic-private/checkpoint"}, "size": {"height": 16, "width": 16}}))
    original = _hashes(src)
    out = tmp_path / "nested-export"
    out.mkdir()
    copy_aux_files(src, out, verbose=False)
    assert AutoTokenizer.from_pretrained(out / "tokenizer", local_files_only=True)("a b")["input_ids"] == [2, 3]
    assert (out / "tokenizer" / "tokenizer.json").read_bytes() == (nested / "tokenizer.json").read_bytes()
    assert json.loads((out / "tokenizer" / "tokenizer_config.json").read_text())["chat_template"] == value["chat_template"]
    assert json.loads((out / "processor" / "processor_config.json").read_text()) == {"size": {"height": 16, "width": 16}}
    for path in out.rglob("*_config.json"):
        assert "synthetic-private" not in path.read_text()
    assert _hashes(src) == original


def test_nested_custom_auto_map_fails_explicitly_instead_of_changing_resolution(tmp_path):
    from razor.prune import copy_aux_files

    src, out = tmp_path / "source", tmp_path / "out"
    (src / "tokenizer").mkdir(parents=True)
    out.mkdir()
    (src / "tokenizer" / "tokenizer_config.json").write_text(json.dumps({
        "auto_map": {"AutoTokenizer": ["tokenization_fixture.Fixture", None]}}))
    with pytest.raises(ValueError, match="nested auxiliary auto_map"):
        copy_aux_files(src, out, remote_code=True, verbose=False)
    assert not list(out.iterdir())


def test_declared_local_architecture_requires_authorization_without_auto_map(source, tmp_path):
    from razor.prune import copy_aux_files

    src, _ = source
    _trusted_export_source(src, auto_map=False)
    out = tmp_path / "not-authorized"
    out.mkdir()
    with pytest.raises(ValueError, match="trust_remote_code"):
        copy_aux_files(src, out, verbose=False)
    assert not list(out.iterdir())


def test_auto_map_indirect_only_dependency_layout_fails_before_copy(source, tmp_path):
    from razor.prune import copy_aux_files

    src, _ = source
    config, _ = _trusted_export_source(src, auto_map=False)
    config["auto_map"] = {"AutoModelForCausalLM": "modeling_public_fixture.PublicFixtureForCausalLM"}
    (src / "config.json").write_text(json.dumps(config))
    out = tmp_path / "unsupported-layout"
    out.mkdir()
    with pytest.raises(ValueError, match="auto_map.*direct root-module"):
        copy_aux_files(src, out, remote_code=True, verbose=False)
    assert not list(out.iterdir())


@pytest.mark.parametrize("statement", ["import public_helper", "import importlib\nimportlib.import_module('.public_helper', __package__)"])
def test_nonstatic_or_absolute_local_dependency_requires_review(tmp_path, statement):
    from razor.prune import copy_aux_files

    src, out = tmp_path / "source", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    (src / "config.json").write_text(json.dumps({"architectures": ["Fixture"]}))
    (src / "modeling_fixture.py").write_text(statement + "\nclass Fixture: pass\n")
    (src / "public_helper.py").write_text("VALUE = 1\n")
    with pytest.raises(ValueError, match="source review"):
        copy_aux_files(src, out, remote_code=True, verbose=False)
    assert not list(out.iterdir())


def test_native_encoder_export_preserves_only_authorized_dependencies(tmp_path):
    from razor.prune import copy_aux_files

    src, out = tmp_path / "source", tmp_path / "out"
    encoding = src / "encoding"
    encoding.mkdir(parents=True)
    out.mkdir()
    files = {"encoding_dsv4.py": "from .helper import render\nencode_messages = render\n",
             "helper.py": "def render(messages): return str(messages)\n"}
    for name, text in files.items():
        (encoding / name).write_text(text)
    (encoding / "train.py").write_text("raise RuntimeError('not a dependency')\n")
    (src / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="trust_remote_code"):
        copy_aux_files(src, out, verbose=False)
    assert not list(out.iterdir())
    copy_aux_files(src, out, remote_code=True, verbose=False)
    assert sorted(path.name for path in (out / "encoding").glob("*.py")) == sorted(files)
    for name, text in files.items():
        assert (out / "encoding" / name).read_text() == text


@pytest.mark.parametrize("filename", ["tokenizer_config.json", "vocab.txt", "modeling_fixture.py"])
@pytest.mark.parametrize("linked_source", [True, False])
def test_auxiliary_copy_replaces_hardlinks_without_mutating_shared_inodes(tmp_path, filename, linked_source):
    import os
    from razor.prune import copy_aux_files

    src, out = tmp_path / "source", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    content = ('{"metadata":{"private":"synthetic-private"},"bos_token":"/literal"}'
               if filename.endswith(".json") else "class Fixture: pass\n"
               if filename.endswith(".py") else "/literal\n../token\n")
    (src / filename).write_text(content)
    if filename.endswith(".py"):
        (src / "config.json").write_text('{"architectures":["Fixture"]}')
    external = tmp_path / "external"
    external.write_text("untouched external inode")
    shared = src / filename if linked_source else external
    os.link(shared, out / filename)
    original = _hashes(src)
    before = shared.read_bytes()
    copy_aux_files(src, out, remote_code=True, verbose=False)
    assert _hashes(src) == original
    assert shared.read_bytes() == before
    assert external.read_text() == "untouched external inode"
    assert not (out / filename).samefile(shared)
    if filename.endswith(".json"):
        assert json.loads((out / filename).read_text()) == {"bos_token": "/literal"}
    else:
        assert (out / filename).read_text() == content


@pytest.mark.parametrize("parent_race", [False, True])
def test_auxiliary_copy_write_race_never_follows_an_external_symlink(tmp_path, monkeypatch, parent_race):
    import importlib

    prune_module = importlib.import_module("razor.prune")
    src, out, external = tmp_path / "source", tmp_path / "out", tmp_path / "external"
    relative = Path("tokenizer/tokenizer_config.json") if parent_race else Path("tokenizer_config.json")
    for root in (src, out, external):
        (root / relative.parent).mkdir(parents=True, exist_ok=True)
    (src / relative).write_text('{"metadata":{"private":"synthetic-private"},"bos_token":"/literal"}')
    (out / relative).write_text("{}")
    victim = external / relative
    victim.write_text("untouched external file")
    original = _hashes(src)
    validate = prune_module._asset_path
    raced = []

    def swap_after_validation(root, name, *, output=False):
        path = validate(root, name, output=output)
        if output and Path(name) == relative and not raced:
            raced.append(True)
            if parent_race:
                path.parent.rename(tmp_path / "old-tokenizer")
                path.parent.symlink_to(victim.parent, target_is_directory=True)
            else:
                path.unlink()
                path.symlink_to(victim)
        return path

    monkeypatch.setattr(prune_module, "_asset_path", swap_after_validation)
    try:
        prune_module.copy_aux_files(src, out, verbose=False)
    except (ValueError, OSError):
        pass
    assert raced
    assert victim.read_text() == "untouched external file"
    assert _hashes(src) == original


def _native_encoder_sources(src, *, label="alpha", initializers=False):
    files = {"constants.py": f"LABEL = {label!r}\n",
             "encoding/encoding_dsv4.py": "from .helper import render\nencode_messages = render\n"}
    if initializers:
        files["__init__.py"] = "from .constants import LABEL\n"
        files["encoding/__init__.py"] = "SUFFIX = ':assistant'\n"
        imports = "from .. import LABEL\nfrom . import SUFFIX\n"
    else:
        imports = "from ..constants import LABEL\nSUFFIX = ':assistant'\n"
    files["encoding/helper.py"] = (
        imports + "def render(messages, thinking_mode='chat'):\n"
        "    return LABEL + ':' + '|'.join((m.get('content') or '') + "
        "m.get('tool_call_id', '') + str(m.get('tools', '')) for m in messages) + ':' + thinking_mode + SUFFIX\n")
    for name, text in files.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (src / "encoding" / "unused.py").write_text("raise AssertionError('unrelated encoder source executed')\n")
    return files


@pytest.mark.parametrize("initializers", [False, True])
@pytest.mark.parametrize("exporter", ["checkpoint", "model"])
def test_native_encoder_export_reload_renders_real_records(source, tmp_path, initializers, exporter):
    from razor.loader import load_tokenizer

    src, _ = source
    files = _native_encoder_sources(src, initializers=initializers)
    (src / "LICENSE").write_text("Copyright Synthetic Authors; Apache-2.0")
    (src / "encoding" / "tokens.tiktoken").write_text("YQ== 0\n")
    original = _hashes(src)
    record = {"messages": [{"role": "tool", "content": "result", "tool_call_id": "call_1"}],
              "tools": [{"type": "function", "function": {"name": "lookup"}}]}
    with pytest.raises(ValueError, match="trust_remote_code"):
        load_tokenizer(str(src))
    tokenizer = load_tokenizer(str(src), trust_remote_code=True, thinking_mode="reason")
    expected = tokenizer.render_record(record)
    assert "resultcall_1" in expected and "lookup" in expected
    assert tokenizer.assistant_open_str == ":reason:assistant"
    keep = tmp_path / "keep.json"
    keep.write_text('{"main_0":[0,2]}')
    out = tmp_path / "native-encoder-export"
    pipeline.prune(str(src), str(out), keep_indices=str(keep), export_mode=exporter,
                   trust_remote_code=True, dtype=torch.float32, device_map="cpu", verbose=False)
    assert not (out / "encoding" / "unused.py").exists()
    for name in (*files, "LICENSE", "encoding/tokens.tiktoken", "tokenizer.json"):
        assert (out / name).read_bytes() == (src / name).read_bytes()
    restored = load_tokenizer(str(out), trust_remote_code=True, thinking_mode="reason")
    assert restored.render_record(record) == expected
    assert restored.assistant_open_str == tokenizer.assistant_open_str
    assert _hashes(src) == original


def test_native_encoder_packages_are_isolated_and_reload_changed_source(tmp_path):
    import os
    from razor.loader import DSV4Tokenizer

    first, second = tmp_path / "first", tmp_path / "second"
    _native_encoder_sources(first, label="alpha", initializers=True)
    _native_encoder_sources(second, label="bravo", initializers=True)
    a = DSV4Tokenizer(object(), str(first), trust_remote_code=True)
    b = DSV4Tokenizer(object(), str(second), trust_remote_code=True)
    record = {"messages": [{"role": "user", "content": "hello"}]}
    assert a.render_record(record) == "alpha:hello:chat:assistant"
    assert b.render_record(record) == "bravo:hello:chat:assistant"
    path = first / "constants.py"
    stat = path.stat()
    path.write_text("LABEL = 'gamma'\n")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    changed = DSV4Tokenizer(object(), str(first), trust_remote_code=True)
    assert changed.render_record(record) == "gamma:hello:chat:assistant"
    assert a.render_record(record) == "alpha:hello:chat:assistant"
    assert b.render_record(record) == "bravo:hello:chat:assistant"
    assert not list(first.rglob("*.pyc"))


@pytest.mark.parametrize("failure", ["import", "missing-callable", "probe-exception", "probe-marker"])
def test_native_encoder_failure_cleans_package_and_preserves_exception(tmp_path, failure):
    import sys
    from razor.loader import DSV4Tokenizer

    src = tmp_path / "source"
    _native_encoder_sources(src, initializers=True)
    path = src / "encoding" / "encoding_dsv4.py"
    if failure == "import":
        path.write_text("from .helper import render\nraise LookupError('synthetic import failure')\n")
        error, match = LookupError, "synthetic import failure"
    elif failure == "missing-callable":
        path.write_text("from .helper import render\nencode_messages = None\n")
        error, match = RuntimeError, "lacks encode_messages"
    elif failure == "probe-exception":
        path.write_text("from .helper import render\ndef encode_messages(*args, **kwargs):\n"
                        "    raise LookupError('synthetic probe failure')\n")
        error, match = LookupError, "synthetic probe failure"
    else:
        path.write_text("from .helper import render\ndef encode_messages(*args, **kwargs): return 'no marker'\n")
        error, match = RuntimeError, "assistant opener"
    from razor.prune import copy_aux_files

    out = tmp_path / "exported"
    copy_aux_files(src, out, remote_code=True, verbose=False)
    before = {name for name in sys.modules if name.startswith("_razor_dsv4_")}
    finders = list(sys.meta_path)
    with pytest.raises(error, match=match):
        DSV4Tokenizer(object(), str(out), trust_remote_code=True)
    assert {name for name in sys.modules if name.startswith("_razor_dsv4_")} == before
    assert sys.meta_path == finders
    _native_encoder_sources(src, label="fixed", initializers=True)
    copy_aux_files(src, out, remote_code=True, verbose=False)
    restored = DSV4Tokenizer(object(), str(out), trust_remote_code=True)
    assert restored.render_record({"messages": [{"role": "user", "content": "hello"}]}).startswith("fixed:")


@pytest.mark.parametrize("guard", ["untrusted", "incomplete", "incomplete-symlink"])
def test_native_encoder_guards_run_before_any_checkpoint_code(tmp_path, guard):
    from razor.loader import DSV4Tokenizer

    src = tmp_path / "source"
    _native_encoder_sources(src, initializers=True)
    sentinel = tmp_path / "executed"
    (src / "encoding" / "encoding_dsv4.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')\n"
        "def encode_messages(messages, **kwargs): return messages[0]['content']\n")
    if guard == "incomplete":
        (src / ".razor-incomplete").touch()
    elif guard == "incomplete-symlink":
        (src / ".razor-incomplete").symlink_to(tmp_path / "missing")
    error = ValueError if guard == "untrusted" else RuntimeError
    with pytest.raises(error, match="trust_remote_code|incomplete"):
        DSV4Tokenizer(object(), str(src), trust_remote_code=guard != "untrusted")
    assert not sentinel.exists()


def test_auxiliary_write_requires_safe_directory_operations(tmp_path, monkeypatch):
    import os
    from razor.prune import _write_aux_asset

    out = tmp_path / "unsupported"
    monkeypatch.delattr(os, "O_NOFOLLOW")
    with pytest.raises(OSError, match="POSIX directory-descriptor"):
        _write_aux_asset(out, "tokenizer_config.json", b"{}")
    assert not out.exists()
