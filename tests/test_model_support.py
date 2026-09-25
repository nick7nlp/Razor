"""Offline native-model adapter regressions; no pretrained weights are downloaded."""
from copy import deepcopy
from dataclasses import replace
import inspect
from types import SimpleNamespace

import pytest
import torch
import transformers

from razor import loader, metrics
from razor.adapters import MoEAdapter, MoEBlock, RouterSpec, get_adapter
from razor.collect import SaliencyCollector, collect, verify_routing
from razor.prune import prune_model, select_keep_indices


FAMILIES = (
    "Qwen2Moe", "Qwen3Moe", "Qwen3Next", "Glm4Moe", "Glm4MoeLite",
    "DeepseekV2", "DeepseekV3", "HYV3",
)


def expected_methods(record):
    """Every criterion, minus refill when the routing cannot expose rank-(k+1).

    Refill needs renormalized gates and an observable losing score, so an
    unnormalized router or a block that only hands down its winners serves the
    fixed-support criteria alone.
    """
    if record.get("refill_supported", True):
        return list(metrics.CHOICES)
    return [name for name in metrics.CHOICES if metrics.canonical(name) != "rcs-refill"]


@pytest.fixture(autouse=True)
def offline_cpu(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model(family, monkeypatch, dtype=torch.float32, top_k=2, normalize=True):
    """Instantiate native random models; Qwen3-Next uses its official CPU fallback."""
    torch.manual_seed(7)
    config = dict(
        vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=64,
    )
    if family == "HYV3":
        config.update(num_experts=8, num_experts_per_tok=top_k, head_dim=8,
                      moe_intermediate_size=16, num_shared_experts=1,
                      router_scaling_factor=2.75, mlp_layer_types=["sparse", "sparse"])
    else:
        config.update(moe_intermediate_size=16, num_experts_per_tok=top_k,
                      norm_topk_prob=normalize)
        if family in ("Qwen2Moe", "Qwen3Next"):
            config.update(num_experts=8, shared_expert_intermediate_size=16)
        elif family == "Qwen3Moe":
            config.update(num_local_experts=8)
        else:
            config.update(
                n_routed_experts=8, n_shared_experts=1,
                n_group=1 if family == "DeepseekV2" else 2,
                topk_group=1, first_k_dense_replace=0, routed_scaling_factor=2.5,
            )
        if family in ("DeepseekV2", "DeepseekV3", "Glm4MoeLite"):
            config.update(q_lora_rank=16, kv_lora_rank=8,
                          qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=8)
        if family == "Qwen3Next":
            import transformers.models.qwen3_next.modeling_qwen3_next as module

            for name in ("FusedRMSNormGated", "chunk_gated_delta_rule",
                         "fused_recurrent_gated_delta_rule", "causal_conv1d_fn",
                         "causal_conv1d_update"):
                monkeypatch.setattr(module, name, None)
            config.update(
                head_dim=8, linear_key_head_dim=8, linear_value_head_dim=8,
                linear_num_key_heads=2, linear_num_value_heads=4,
                layer_types=["linear_attention", "full_attention"],
            )
    config = getattr(transformers, family + "Config")(**config)
    config._attn_implementation = "eager"
    return getattr(transformers, family + "ForCausalLM")(config).eval().to(dtype)


def batch():
    return {
        "input_ids": torch.tensor([[5, 7, 11, 13, 17, 19, 23, 29]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]),
    }


def assert_vectors_close(actual, expected):
    actual, expected = actual.float(), expected.float()
    error = torch.linalg.vector_norm(actual - expected, dim=-1)
    scale = torch.linalg.vector_norm(expected, dim=-1)
    assert torch.all(error <= 1e-7 + 2e-2 * scale)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@torch.no_grad()
def test_native_model_route_collect_prune_reload(family, dtype, monkeypatch):
    model = tiny_model(family, monkeypatch, dtype=dtype)
    adapter = get_adapter(model.config)
    adapter.validate_model(model)
    blocks = adapter.moe_blocks(model)
    assert len(blocks) == (1 if family == "Glm4MoeLite" else 2)
    block = blocks[0].module
    correction = adapter.correction_bias(block)
    if correction is not None:
        correction.copy_(torch.linspace(-0.1, 0.1, 8).to(correction))
    hidden = torch.randn(1, 7, 32, dtype=dtype)
    flat = hidden.reshape(-1, 32)
    spec = adapter.router_spec(block)
    selected, weights = adapter.route(flat, adapter.gate_module(block).weight, spec)
    native = (torch.nn.functional.linear(hidden.float(), block.gate.weight.float())
              if family == "DeepseekV2" else
              block.gate(hidden, block.e_score_correction_bias) if family == "HYV3" else
              block.gate(hidden))
    if isinstance(native, tuple):
        _, native_weights, native_selected = native
    else:
        native_selected, native_weights = block.route_tokens_to_experts(native)
    dense = torch.zeros_like(weights).scatter_(0, native_selected.t(), native_weights.t().float())
    torch.testing.assert_close(weights, dense, atol=1e-7, rtol=1e-5)
    assert torch.equal(selected.sort(1).values, native_selected.sort(1).values)
    outputs = adapter.expert_forward(block, flat, 0, 8)
    for expert in (0, 7):
        ids = torch.full((flat.shape[0], 1), expert, dtype=torch.long)
        native_expert = block.experts(flat, ids, flat.new_ones(flat.shape[0], 1))
        assert_vectors_close(outputs[expert], native_expert)
    routed = (weights.unsqueeze(-1) * outputs).sum(0)
    shared = adapter.shared_forward(block, flat)
    assert_vectors_close(routed, block.experts(flat, native_selected, native_weights))
    assert_vectors_close(routed + shared, block(hidden).reshape_as(routed))

    inputs = batch()
    assert verify_routing(model, adapter, inputs) is True
    saliency = collect(model, adapter, [inputs], expert_chunk=3, verbose=False)
    assert set(saliency) == {item.tag for item in blocks}
    for record in saliency.values():
        assert record["total_tokens"] == 6
        assert int(record["expert_frequency"].sum()) == 12
        assert record["razor_supported"] is True
        assert metrics.available(record) == expected_methods(record)
        for method in expected_methods(record):
            score = metrics.score(record, method)
            assert score.shape == (8,)
            assert torch.isfinite(score).all() and (score >= 0).all()
    if family == "DeepseekV2":
        assert all(record["counterfactual_semantics"] == "frozen_weights"
                   for record in saliency.values())

    original_shared = {name: value.clone() for name, value in model.state_dict().items()
                       if "shared" in name}
    servable = [name for name in metrics.CHOICES
                if all(name in metrics.available(record) for record in saliency.values())]
    for method in servable:
        pruned = deepcopy(model)
        pruned_adapter = get_adapter(pruned.config)
        keep = select_keep_indices(saliency, method, 4, verbose=False, n_group=spec.n_group)
        prune_model(pruned, pruned_adapter, keep, verbose=False)
        pruned_adapter.validate_model(pruned)
        assert pruned_adapter.num_experts == 4
        assert verify_routing(pruned, pruned_adapter, inputs) is True
        assert torch.isfinite(pruned(**inputs, use_cache=False).logits).all()
        for name, value in original_shared.items():
            torch.testing.assert_close(pruned.state_dict()[name], value, atol=0, rtol=0)
    assert adapter.num_experts == 8
    restored_config = type(pruned.config).from_dict(pruned.config.to_dict())
    restored_config._attn_implementation = "eager"
    restored = type(pruned)(restored_config).eval().to(dtype)
    restored.load_state_dict(pruned.state_dict(), strict=True)
    assert get_adapter(restored.config).num_experts == 4
    torch.testing.assert_close(restored(**inputs, use_cache=False).logits,
                               pruned(**inputs, use_cache=False).logits, atol=0, rtol=0)


@pytest.mark.parametrize("family", ["Qwen3Moe", "Glm4Moe"])
@pytest.mark.parametrize("normalize", [True, False])
def test_top1_keeps_three_baselines(family, normalize, monkeypatch):
    model = tiny_model(family, monkeypatch, top_k=1, normalize=normalize)
    records = collect(model, get_adapter(model.config), [batch()], verbose=False)
    for record in records.values():
        assert metrics.available(record) == ["reap", "ean", "frequency"]
        assert record["razor_supported"] is False
        assert "counterfactual_delta_sum" not in record
        assert "counterfactual_delta_mean" not in record
        assert "counterfactual_delta_square_sum" not in record
        assert "counterfactual_delta_rms" not in record
        assert "ean_rms" in record and "reap_rms" in record
        with pytest.raises((ValueError, KeyError)):
            metrics.score(record, "razor")


@pytest.mark.parametrize("saturated_is_padding", [False, True])
@torch.no_grad()
def test_saturated_gates_disable_only_razor_for_valid_tokens(saturated_is_padding, monkeypatch):
    model = tiny_model("Qwen3Moe", monkeypatch)
    adapter = get_adapter(model.config)
    block = adapter.moe_blocks(model)[0].module
    block.gate.weight.zero_()
    block.gate.weight[0, 0] = 1000.0
    hidden = torch.zeros(1, 2, 32)
    hidden[0, 0, 0] = 1.0
    hidden[0, 1, 1] = 1.0
    collector = SaliencyCollector(adapter).install(model)
    try:
        collector.set_attention_mask(torch.tensor([[int(not saturated_is_padding), 1]]))
        block(hidden)
        record = collector.finalize()["main_0"]
        assert record["razor_supported"] is saturated_is_padding
        assert ("counterfactual_delta_mean" in record) is saturated_is_padding
        assert ("counterfactual_delta_sum" in record) is saturated_is_padding
        assert ("counterfactual_delta_square_sum" in record) is saturated_is_padding
        assert ("counterfactual_delta_rms" in record) is saturated_is_padding
        if not saturated_is_padding:
            assert metrics.available(record) == ["reap", "ean", "frequency"]
        hidden.zero_()
        hidden[:, :, 1] = 1.0
        collector.set_attention_mask(torch.ones(1, 2, dtype=torch.long))
        block(hidden)
        assert collector.finalize()["main_0"]["razor_supported"] is saturated_is_padding
    finally:
        collector.remove()


@pytest.mark.parametrize("family", ["Qwen2Moe", "DeepseekV2"])
def test_non_normalized_scores_are_frozen_weight_magnitudes(family, monkeypatch):
    model = tiny_model(family, monkeypatch, normalize=False)
    records = collect(model, get_adapter(model.config), [batch()], verbose=False)
    for record in records.values():
        assert record["counterfactual_semantics"] == "frozen_weights"
        assert record["renormalized"] is False
        torch.testing.assert_close(record["counterfactual_delta_mean"], record["reap"],
                                   atol=0, rtol=0)


def test_second_layer_wrong_scale_is_rejected(monkeypatch):
    model = tiny_model("Qwen3Moe", monkeypatch)
    adapter = get_adapter(model.config)
    bad_block = adapter.moe_blocks(model)[1].module
    original = adapter.router_spec
    monkeypatch.setattr(adapter, "router_spec", lambda block:
                        replace(original(block), scaling=2.0) if block is bad_block else original(block))
    with pytest.raises(RuntimeError, match="reconstruction mismatch"):
        verify_routing(model, adapter, batch())
    with pytest.raises(RuntimeError, match="reconstruction mismatch"):
        collect(model, adapter, [batch()], verbose=False)


def test_large_shared_output_cannot_hide_wrong_routed_scale(monkeypatch):
    model = tiny_model("Qwen2Moe", monkeypatch)
    adapter = get_adapter(model.config)
    for item in adapter.moe_blocks(model):
        with torch.no_grad():
            for parameter in item.module.shared_expert.parameters():
                parameter.mul_(100)
    original = adapter.router_spec
    monkeypatch.setattr(adapter, "router_spec", lambda block: replace(original(block), scaling=2.0))
    with pytest.raises(RuntimeError, match="routed reconstruction mismatch"):
        verify_routing(model, adapter, batch())


@pytest.mark.parametrize("field", ["num_hash_layers", "n_hash_layers", "first_k_hash_replace"])
def test_hash_models_are_rejected(field):
    config = SimpleNamespace(model_type="deepseek_v3", **{field: 1})
    with pytest.raises(NotImplementedError, match="hash"):
        get_adapter(config)


@pytest.mark.parametrize("defect", ["quantization", "meta", "layout", "auxiliary", "count", "offload"])
def test_unsupported_models_fail_closed(defect, monkeypatch):
    model = tiny_model("Qwen3Moe", monkeypatch)
    adapter = get_adapter(model.config)
    block = adapter.moe_blocks(model)[0].module
    if defect == "quantization":
        model.is_quantized = True
    elif defect == "meta":
        model.to("meta")
    elif defect == "layout":
        block.experts.down_proj = torch.nn.Parameter(block.experts.down_proj.transpose(1, 2))
    elif defect == "auxiliary":
        block.experts.register_buffer("weight_scale_inv", torch.ones(8))
    elif defect == "count":
        block.experts.num_experts = 9
    else:
        block.experts._hf_hook = SimpleNamespace(offload=True)
    with pytest.raises((NotImplementedError, ValueError)):
        adapter.validate_model(model)


def test_unknown_and_unsupported_v2_modes_are_rejected():
    with pytest.raises(NotImplementedError):
        get_adapter(SimpleNamespace(model_type="unknown_moe", num_experts=8))
    with pytest.raises(NotImplementedError, match="quantized"):
        get_adapter(SimpleNamespace(model_type="qwen3_moe", quantization_config={}))
    config = transformers.DeepseekV2Config(topk_method="group_limited_greedy")
    with pytest.raises(NotImplementedError, match="only greedy"):
        get_adapter(config)


@pytest.mark.parametrize("scaling", [0.0, float("nan"), float("inf")])
def test_invalid_scaling_is_rejected(scaling):
    spec = RouterSpec(num_experts=4, top_k=2, scaling=scaling)
    with pytest.raises(ValueError, match="scaling"):
        MoEAdapter.route(torch.ones(2, 3), torch.ones(4, 3), spec)


def test_softmax_projection_bias_selection_bias_groups_and_scale():
    hidden = torch.tensor([[2.0, 1.0]])
    projection = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    bias = torch.tensor([-0.5, 2.0, 0.5, 1.0])
    spec = RouterSpec(num_experts=4, top_k=2, scaling=3.0, gate_bias=bias,
                      correction_bias=torch.tensor([0., 0., 1., 1.]),
                      n_group=2, topk_group=1)
    selected, weights = MoEAdapter.route(hidden, projection, spec)
    assert set(selected[0].tolist()) == {2, 3}
    probabilities = (hidden @ projection.t() + bias).softmax(-1)[0]
    expected = torch.zeros(4)
    expected[2:] = probabilities[2:] / probabilities[2:].sum() * 3.0
    torch.testing.assert_close(weights[:, 0], expected)


def test_invalid_group_keep_is_rejected_before_mutation(monkeypatch):
    model = tiny_model("Glm4Moe", monkeypatch)
    adapter = get_adapter(model.config)
    blocks = adapter.moe_blocks(model)
    keep = {item.tag: [0, 1, 4, 5] for item in blocks}
    keep[blocks[-1].tag] = [0, 1, 2, 4]
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(NotImplementedError, match="equal retained"):
        prune_model(model, adapter, keep, verbose=False)
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)


@pytest.mark.parametrize("shared", ["experts", "gate", "block"])
@torch.no_grad()
def test_cross_block_shared_expert_state_is_rejected_before_any_slice(shared, monkeypatch):
    """A bank or router owned by two decoder layers cannot be sliced twice."""
    model = tiny_model("Qwen3Moe", monkeypatch)
    adapter = get_adapter(model.config)
    layers = model.model.layers
    if shared == "experts":
        layers[1].mlp.experts = layers[0].mlp.experts
    elif shared == "gate":
        layers[1].mlp.gate = layers[0].mlp.gate
    else:
        layers[1].mlp = layers[0].mlp
    before = {name: value.clone() for name, value in model.state_dict().items()}
    keep = {"main_0": [0, 1, 4, 5], "main_1": [0, 1, 4, 5]}
    for call in (lambda: adapter.moe_blocks(model),
                 lambda: adapter.validate_model(model),
                 lambda: prune_model(model, adapter, keep, verbose=False)):
        with pytest.raises(NotImplementedError, match="share"):
            call()
    after = model.state_dict()
    assert set(after) == set(before)
    for name, value in before.items():
        torch.testing.assert_close(after[name], value, atol=0, rtol=0)
    assert adapter.num_experts == 8 and adapter.top_k == 2
    assert layers[0].mlp.experts.gate_up_proj.shape[0] == 8
    assert layers[0].mlp.gate.weight.shape[0] == 8


@pytest.mark.parametrize("family", ["Qwen3Moe", "Qwen2Moe", "DeepseekV2"])
@torch.no_grad()
def test_independent_banks_tied_embeddings_and_in_block_aliases_stay_supported(family, monkeypatch):
    """Per-layer banks, a same-block router alias and tied embeddings remain prunable."""
    model = tiny_model(family, monkeypatch)
    adapter = get_adapter(model.config)
    blocks = adapter.moe_blocks(model)
    assert len(blocks) == 2
    assert len({id(adapter.expert_bank(item.module)) for item in blocks}) == 2
    assert len({id(adapter.gate_module(item.module)) for item in blocks}) == 2
    model.lm_head.weight = model.model.embed_tokens.weight
    block = blocks[0].module
    block.router = adapter.gate_module(block)
    assert adapter.gate_module(block) is block.router
    if family != "Qwen3Moe":
        assert getattr(block, "shared_expert", None) is not None or \
               getattr(block, "shared_experts", None) is not None
    adapter.validate_model(model)
    keep = {item.tag: [0, 1, 4, 5] for item in blocks}
    prune_model(model, adapter, keep, verbose=False)
    adapter.validate_model(model)
    assert adapter.num_experts == 4
    assert adapter.gate_module(block).weight.shape[0] == 4
    assert torch.isfinite(model(**batch(), use_cache=False).logits).all()


def _list_expert_block(experts=None, gate=None):
    """A minimal ModuleList expert bank, independent unless an object is passed in."""
    torch.manual_seed(3)
    block = torch.nn.Module()
    block.gate = torch.nn.Linear(8, 4, bias=False) if gate is None else gate
    block.experts = (torch.nn.ModuleList([torch.nn.Linear(8, 8, bias=False) for _ in range(4)])
                     if experts is None else experts)
    return block


@pytest.mark.parametrize("shared", ["experts", "gate"])
def test_module_list_banks_are_shared_only_when_the_bank_object_is_shared(shared):
    config = SimpleNamespace(model_type="qwen3_moe", num_experts=4, num_experts_per_tok=2,
                             norm_topk_prob=True, hidden_size=8)
    adapter = get_adapter(config)
    first, second = _list_expert_block(), _list_expert_block()
    blocks = [MoEBlock("main_0", 0, "main", first), MoEBlock("main_1", 1, "main", second)]
    adapter.validate_blocks(blocks)
    second.experts[0] = first.experts[0]
    adapter.validate_blocks(blocks)
    reused = _list_expert_block(**{shared: getattr(first, shared)})
    with pytest.raises(NotImplementedError, match="share"):
        adapter.validate_blocks([blocks[0], MoEBlock("main_1", 1, "main", reused)])
    adapter.prune_block(first, torch.tensor([0, 2]))
    assert len(first.experts) == 2 and len(second.experts) == 4
    assert first.gate.weight.shape[0] == 2 and second.gate.weight.shape[0] == 4


def test_loader_binds_loaded_model_configuration(tmp_path, monkeypatch):
    source = tiny_model("Qwen3Moe", monkeypatch)
    source.save_pretrained(tmp_path / "source")
    loaded, adapter = loader.load_model(str(tmp_path / "source"), device_map="cpu",
                                        dtype=torch.float32, verbose=False)
    assert adapter.config is loaded.config
    assert adapter.text_config is loaded.config
    keep = {item.tag: [0, 2, 4, 6] for item in adapter.moe_blocks(loaded)}
    prune_model(loaded, adapter, keep, verbose=False)
    assert loaded.config.num_experts == 4
    assert source.config.num_experts == 8
    loaded.save_pretrained(tmp_path / "pruned")
    restored = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path / "pruned", local_files_only=True).eval()
    assert restored.config.num_experts == 4
    with torch.no_grad():
        assert torch.isfinite(restored(**batch(), use_cache=False).logits).all()


def test_loader_requires_explicit_remote_code_opt_in(monkeypatch):
    for function in (loader.load_config, loader.load_model, loader.load_tokenizer):
        assert inspect.signature(function).parameters["trust_remote_code"].default is False
    calls = []
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        lambda *args, **kwargs: calls.append(kwargs))
    loader.load_config("offline-test")
    loader.load_config("offline-test", trust_remote_code=True)
    assert [call["trust_remote_code"] for call in calls] == [False, True]


def experiment_model(family, dtype=torch.float32, hash_layers=False):
    """Small installed Transformers models, with no pretrained or remote code."""
    torch.manual_seed(17)
    common = dict(vocab_size=64, hidden_size=32, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=64)
    if family.startswith("Qwen3_5"):
        text = transformers.Qwen3_5MoeTextConfig(
            **common, head_dim=8, num_experts=8, num_experts_per_tok=2,
            moe_intermediate_size=16, shared_expert_intermediate_size=16,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4,
            layer_types=["full_attention", "full_attention"])
        if family.endswith("Wrapper"):
            config = transformers.Qwen3_5MoeConfig(
                text_config=text,
                vision_config=dict(depth=1, hidden_size=16, intermediate_size=32,
                                   num_heads=2, out_hidden_size=32, patch_size=2,
                                   spatial_merge_size=1, temporal_patch_size=1,
                                   num_position_embeddings=16))
            cls = transformers.Qwen3_5MoeForConditionalGeneration
        else:
            config, cls = text, transformers.Qwen3_5MoeForCausalLM
    elif family == "GlmMoeDsa":
        common["num_key_value_heads"] = common["num_attention_heads"]
        config = transformers.GlmMoeDsaConfig(
            **common, intermediate_size=48, moe_intermediate_size=16,
            n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
            n_group=2, topk_group=1, routed_scaling_factor=2.5,
            q_lora_rank=16, kv_lora_rank=8, qk_rope_head_dim=4,
            qk_nope_head_dim=4, v_head_dim=8, index_head_dim=8,
            index_n_heads=2, index_topk=4, mlp_layer_types=["sparse", "sparse"],
            indexer_types=["full", "shared"])
        cls = transformers.GlmMoeDsaForCausalLM
    elif family == "DeepseekV4":
        common["num_key_value_heads"] = 1
        config = transformers.DeepseekV4Config(
            **common, head_dim=8, moe_intermediate_size=16,
            n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
            routed_scaling_factor=1.75, q_lora_rank=16,
            o_groups=2, o_lora_rank=8, index_head_dim=8,
            index_n_heads=2, index_topk=4, hc_mult=2,
            partial_rotary_factor=0.5, swiglu_limit=0.03,
            layer_types=["sliding_attention", "sliding_attention"],
            mlp_layer_types=["hash_moe" if hash_layers else "moe", "moe"])
        cls = transformers.DeepseekV4ForCausalLM
    elif family == "Gemma4":
        config = transformers.Gemma4TextConfig(
            **common, intermediate_size=48, head_dim=8, global_head_dim=8,
            num_global_key_value_heads=2, hidden_size_per_layer_input=0,
            enable_moe_block=True, num_experts=8, top_k_experts=2,
            moe_intermediate_size=16,
            layer_types=["sliding_attention", "full_attention"])
        cls = transformers.Gemma4ForCausalLM
    else:
        raise AssertionError(family)
    config._attn_implementation = "eager"
    if getattr(config, "text_config", None) is not None:
        config.text_config._attn_implementation = "eager"
    model = cls(config).eval().to(dtype)
    adapter = get_adapter(model.config)
    with torch.no_grad():
        for item in adapter.moe_blocks(model):
            block = item.module
            bias = adapter.correction_bias(block)
            if bias is not None:
                bias.copy_(torch.linspace(-0.05, 0.05, 8).to(bias))
            if adapter.is_hash(block):
                table = block.gate.tid2eid
                table.copy_((torch.arange(table.shape[0])[:, None] + torch.arange(2)) % 8)
            if family == "Gemma4":
                block.router.per_expert_scale.copy_(torch.linspace(0.7, 1.3, 8).to(dtype))
    return model


@pytest.mark.parametrize("family", ["Qwen3_5Text", "Qwen3_5Wrapper", "GlmMoeDsa", "DeepseekV4", "Gemma4"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@torch.no_grad()
def test_experiment_native_collect_prune_and_state_rebuild(family, dtype):
    model = experiment_model(family, dtype)
    adapter = get_adapter(model.config)
    blocks = adapter.moe_blocks(model)
    assert [item.tag for item in blocks] == ["main_0", "main_1"]
    adapter.validate_model(model)
    block = blocks[0].module
    captured = {}
    def observe(module, args, output):
        captured["expert_args"] = args
        captured["expert_output"] = output
    handle = block.experts.register_forward_hook(observe)
    try:
        model(**batch(), use_cache=False)
    finally:
        handle.remove()
    hidden, indices, weights = captured["expert_args"]
    if family == "Gemma4":
        context = adapter.context(block, captured["expert_args"], captured["expert_output"])
        expected_weights = weights / block.router.per_expert_scale[indices]
        selected, dense = context.selected, context.weights
    else:
        selected, dense = adapter.route(hidden, block.gate.weight, adapter.router_spec(block))
        expected_weights = weights
    assert torch.equal(selected.sort(1).values, indices.sort(1).values)
    torch.testing.assert_close(dense, adapter.dense_weights(indices, expected_weights, 8),
                               atol=0, rtol=0)
    acts = adapter.expert_forward(block, hidden, 0, 8)
    gain = block.router.per_expert_scale if family == "Gemma4" else torch.ones(8, dtype=dtype)
    for expert in (0, 7):
        single_ids = torch.full((hidden.shape[0], 1), expert, dtype=torch.long)
        native = block.experts(hidden, single_ids, hidden.new_ones(hidden.shape[0], 1))
        assert_vectors_close(acts[expert], native.float() * gain[expert].float())
    assert verify_routing(model, adapter, batch())
    records = collect(model, adapter, [batch()], expert_chunk=3, verbose=False)
    for record in records.values():
        assert record["total_tokens"] == 6
        assert record["expert_frequency"].sum() == 12
        assert metrics.available(record) == expected_methods(record)
        assert record["pruning_policy"] == "score"
    keep = {item.tag: [0, 1, 4, 5] for item in blocks}
    preserved = {name: value.clone() for name, value in model.state_dict().items()
                 if "shared" in name}
    prune_model(model, adapter, keep, verbose=False)
    adapter.validate_model(model)
    assert verify_routing(model, adapter, batch())
    config = type(model.config).from_dict(model.config.to_dict())
    config._attn_implementation = "eager"
    restored = type(model)(config).eval().to(dtype)
    restored.load_state_dict(model.state_dict(), strict=True)
    for name, value in preserved.items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
    torch.testing.assert_close(restored(**batch(), use_cache=False).logits,
                               model(**batch(), use_cache=False).logits, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@torch.no_grad()
def test_deepseek_v4_hash_preserves_native_ids_and_is_not_razor(dtype):
    model = experiment_model("DeepseekV4", dtype, hash_layers=True)
    adapter = get_adapter(model.config)
    block = adapter.moe_blocks(model)[0].module
    hidden = torch.randn(1, 8, 32, dtype=dtype)
    context = adapter.context(block, (hidden,), kwargs={"input_ids": batch()["input_ids"]})
    assert torch.equal(context.selected, block.gate.tid2eid[batch()["input_ids"].reshape(-1)])
    assert verify_routing(model, adapter, batch())
    records = collect(model, adapter, [batch()], verbose=False)
    assert records["main_0"]["pruning_policy"] == "preserve"
    assert records["main_0"]["razor_supported"] is False
    assert metrics.available(records["main_0"]) == ["reap", "ean", "frequency"]
    assert records["main_1"]["razor_supported"] is True
    with pytest.raises(NotImplementedError, match="preserving"):
        adapter.validate_keep(block, torch.tensor([0, 1, 2, 3]))
    adapter.validate_keep(block, torch.arange(8))


@pytest.mark.parametrize("model_type", ["hunyuan_v1_moe", "unknown_synthetic_moe"])
def test_removed_and_unknown_families_are_not_registered(model_type):
    with pytest.raises(NotImplementedError):
        get_adapter(SimpleNamespace(model_type=model_type))


def _native_fused_classes(family):
    """Use standalone native classes when their Transformers release is supplied locally."""
    import ast
    import os
    from pathlib import Path
    from transformers.activations import ACT2FN
    from transformers.integrations import use_experts_implementation
    if family != "glm5_next":
        raise AssertionError(f"no native source mapping for {family!r}")
    variable = "RAZOR_NATIVE_GLM_NEXT_SOURCE"
    names = {"Glm5NextTextMLP", "Glm5NextTextExperts", "Glm5NextTextTopkRouter", "Glm5NextTextMoE"}
    block_name = "Glm5NextTextMoE"
    source = os.environ.get(variable)
    if not source:
        pytest.skip(f"set {variable} to the trusted native modeling file")
    tree = ast.parse(Path(source).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert {node.name for node in nodes} == names
    namespace = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, ACT2FN=ACT2FN,
                     use_experts_implementation=use_experts_implementation,
                     Glm5NextConfig=object)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "native_fused_moe", "exec"), namespace)
    return namespace[block_name]


@pytest.mark.parametrize("family", ["glm5_next"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("normalize", [True, False])
@torch.no_grad()
def test_native_next_gate_collect_prune_rebuild(family, dtype, normalize):
    native = _native_fused_classes(family)
    torch.manual_seed(23)
    config = SimpleNamespace(
        model_type=family + "_text", hidden_size=32, intermediate_size=48,
        moe_intermediate_size=16, shared_expert_intermediate_size=16,
        num_experts=8, n_routed_experts=8, num_local_experts=8,
        num_experts_per_tok=2, n_shared_experts=1, n_group=2, topk_group=1,
        routed_scaling_factor=2.5, norm_topk_prob=normalize, hidden_act="silu",
        swiglu_limit=0.04, _experts_implementation="eager")
    block = native(config).eval().to(dtype)
    for parameter in block.parameters():
        parameter.normal_(0, 0.12)
    adapter = get_adapter(SimpleNamespace(model_type=family, text_config=config))
    bias = adapter.correction_bias(block)
    if bias is not None:
        bias.copy_(torch.linspace(-0.1, 0.1, 8).to(dtype))
    layer = torch.nn.Module()
    layer.mlp = block
    hidden = torch.randn(1, 8, 32, dtype=dtype)
    context = adapter.context(block, (hidden,))
    _, native_weights, native_ids = block.gate(hidden)
    assert torch.equal(context.selected.sort(1).values, native_ids.sort(1).values)
    torch.testing.assert_close(context.weights, adapter.dense_weights(native_ids, native_weights, 8),
                               atol=0, rtol=0)
    acts = adapter.expert_forward(block, context.hidden, 0, 8)
    for expert in (0, 7):
        ids = torch.full((8, 1), expert, dtype=torch.long)
        expected = block.experts(context.hidden, ids, hidden.new_ones(8, 1))
        assert_vectors_close(acts[expert], expected)
    if family == "glm5_next":
        projected = torch.nn.functional.linear(context.hidden, block.experts.gate_up_proj[0])
        gate, up = projected.chunk(2, -1)
        assert (gate > config.swiglu_limit).any() or (up.abs() > config.swiglu_limit).any()
        assert not torch.equal(block.experts._apply_gate(projected), torch.nn.functional.silu(gate) * up)
    collector = SaliencyCollector(adapter, expert_chunk=3)
    with collector.layer_context("main_5", layer):
        collector.set_batch(batch()["attention_mask"], batch()["input_ids"])
        block(hidden)
    record = collector.finalize()["main_5"]
    assert record["total_tokens"] == 6 and record["expert_frequency"].sum() == 12
    assert record["renormalized"] is normalize
    assert metrics.available(record) == expected_methods(record)
    for field in ("counterfactual_delta_rms", "ean_rms", "reap_rms"):
        assert torch.isfinite(record[field]).all()
    shared = {name: value.clone() for name, value in block.state_dict().items() if "shared" in name}
    adapter.prune_block(block, torch.tensor([0, 1, 4, 5]))
    adapter.set_num_experts(4)
    restored = native(deepcopy(config)).eval().to(dtype)
    restored.load_state_dict(block.state_dict(), strict=True)
    torch.testing.assert_close(restored(hidden), block(hidden), atol=0, rtol=0)
    for name, value in shared.items():
        torch.testing.assert_close(block.state_dict()[name], value, atol=0, rtol=0)
    with SaliencyCollector(adapter).layer_context("main_5", layer):
        block(hidden)


def _native_kimi_classes():
    """Load only standalone MoE definitions from an explicitly supplied native source."""
    import ast
    import math
    import os
    from pathlib import Path
    from transformers.activations import ACT2FN
    source = os.environ.get("RAZOR_NATIVE_KIMI_SOURCE")
    if not source:
        pytest.skip("set RAZOR_NATIVE_KIMI_SOURCE to the trusted native modeling file")
    names = {"SituAndMul", "_get_situ_activation_params", "KimiRMSNorm",
             "KimiBlockSparseMLP", "KimiMLP", "KimiMoEGate", "KimiSparseMoeBlock"}
    tree = ast.parse(Path(source).read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in names]
    assert {node.name for node in nodes} == names
    namespace = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, math=math,
                     ACT2FN=ACT2FN, KimiLinearConfig=object)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "native_kimi_moe", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("latent_norm", [False, True])
@torch.no_grad()
def test_native_kimi_latent_gate_situ_collect_and_rebuild(dtype, latent_norm):
    native = _native_kimi_classes()
    config = SimpleNamespace(
        model_type="kimi_k3", hidden_size=32, intermediate_size=48,
        moe_intermediate_size=16, num_experts=8, num_experts_per_token=2,
        routed_scaling_factor=2.25, moe_router_activation_func="sigmoid",
        num_expert_group=2, topk_group=1, moe_renormalize=True,
        routed_expert_hidden_size=16, latent_moe_use_norm=latent_norm,
        num_shared_experts=1, rms_norm_eps=1e-6, hidden_act="situ",
        activation_situ_beta=1.5, activation_situ_linear_beta=0.75)
    block = native["KimiSparseMoeBlock"](config).eval().to(dtype)
    block.gate.e_score_correction_bias.copy_(torch.linspace(-0.1, 0.1, 8).to(dtype))
    layer = torch.nn.Module()
    layer.block_sparse_moe = block
    hidden = torch.randn(1, 8, 32, dtype=dtype)
    adapter = get_adapter(config)
    context = adapter.context(block, (hidden,))
    ids, weights = block.gate(hidden)
    assert torch.equal(context.selected.sort(1).values, ids.sort(1).values)
    torch.testing.assert_close(context.weights,
                               adapter.dense_weights(ids, weights, 8), atol=0, rtol=0)
    torch.testing.assert_close(context.hidden, block.routed_expert_down_proj(hidden.reshape(-1, 32)))
    collector = SaliencyCollector(adapter, expert_chunk=3)
    with collector.layer_context("main_3", layer):
        collector.set_batch(batch()["attention_mask"], batch()["input_ids"])
        block(hidden)
    record = collector.finalize()["main_3"]
    assert record["score_space"] == "latent"
    assert record["total_tokens"] == 6
    assert metrics.available(record) == expected_methods(record)
    keep = torch.tensor([0, 1, 4, 5])
    adapter.prune_block(block, keep)
    adapter.set_num_experts(4)
    restored = native["KimiSparseMoeBlock"](deepcopy(config)).eval().to(dtype)
    restored.load_state_dict(block.state_dict(), strict=True)
    torch.testing.assert_close(restored(hidden), block(hidden), atol=0, rtol=0)
    assert block.experts_per_rank == 4


@torch.no_grad()
def test_layer_context_matches_resident_collection_and_rejects_meta(monkeypatch):
    model = tiny_model("HYV3", monkeypatch)
    adapter = get_adapter(model.config)
    block = adapter.moe_blocks(model)[0].module
    hidden = torch.randn(1, 8, 32)
    resident = SaliencyCollector(adapter).install(model)
    resident.set_attention_mask(batch()["attention_mask"])
    try:
        block(hidden)
    finally:
        resident.remove()
    streamed = SaliencyCollector(adapter)
    with streamed.layer_context("main_0", model.model.layers[0]):
        streamed.set_batch(batch()["attention_mask"], batch()["input_ids"])
        block(hidden)
    for key, expected in resident.finalize()["main_0"].items():
        actual = streamed.finalize()["main_0"][key]
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        else:
            assert actual == expected
    block.to("meta")
    with pytest.raises(NotImplementedError, match="meta"):
        with streamed.layer_context("main_0", model.model.layers[0]):
            raise AssertionError("unmaterialized layer reached collection")


@pytest.mark.parametrize("family,normalize", [("Qwen2Moe", True), ("Qwen2Moe", False), ("HYV3", True)])
@torch.no_grad()
def test_second_moments_match_token_deletions_and_batch_splitting(family, normalize, monkeypatch):
    model = tiny_model(family, monkeypatch, normalize=normalize)
    adapter = get_adapter(model.config)
    block = adapter.moe_blocks(model)[0].module
    hidden = torch.randn(1, 8, 32)
    mask = batch()["attention_mask"]
    flat = hidden.reshape(-1, 32)
    spec = adapter.router_spec(block)
    selected, weights = adapter.route(flat, adapter.gate_module(block).weight, spec)
    outputs = adapter.expert_forward(block, flat, 0, 8)
    ean, wean, delta = (torch.zeros(8, 8, dtype=torch.float64) for _ in range(3))
    for token in range(8):
        if not mask[0, token]:
            continue
        mixture = (weights[:, token, None] * outputs[:, token]).sum(0)
        for expert in selected[token].tolist():
            ean[expert, token] = outputs[expert, token].norm()
            wean[expert, token] = (outputs[expert, token].norm() * weights[expert, token]).double()
            surviving = weights[:, token].clone()
            surviving[expert] = 0
            if spec.normalize:
                surviving = surviving / surviving.sum() * spec.scaling
            removed = (surviving[:, None] * outputs[:, token]).sum(0)
            delta[expert, token] = (mixture - removed).norm()
    records = []
    for slices in ((slice(None),), (slice(0, 4), slice(4, 8))):
        collector = SaliencyCollector(adapter, expert_chunk=3).install(model)
        try:
            for chunk in slices:
                collector.set_attention_mask(mask[:, chunk])
                block(hidden[:, chunk])
        finally:
            collector.remove()
        records.append(collector.finalize()["main_0"])
    record = records[0]
    count = record["routed_count"].clamp_min(1)
    for prefix, rms_name, values in (("ean", "ean_rms", ean),
                                     ("weighted_ean", "reap_rms", wean),
                                     ("counterfactual_delta", "counterfactual_delta_rms", delta)):
        square_sum = values.square().sum(1)
        torch.testing.assert_close(record[prefix + "_square_sum"], square_sum, rtol=2e-5, atol=1e-9)
        torch.testing.assert_close(record[rms_name], (square_sum / count).sqrt().float(),
                                   rtol=2e-5, atol=1e-7)
        torch.testing.assert_close(records[1][prefix + "_square_sum"], record[prefix + "_square_sum"],
                                   rtol=2e-5, atol=1e-9)
        torch.testing.assert_close(records[1][rms_name], record[rms_name], rtol=2e-5, atol=1e-7)
    assert not torch.allclose(record["ean_mean"], record["ean_rms"])
    if not normalize:
        torch.testing.assert_close(record["counterfactual_delta_rms"], record["reap_rms"], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("ordered_keep", [[0, 2, 4, 6], [4, 0, 6, 2]], ids=["sorted", "reordered"])
@torch.no_grad()
def test_deepseek_v4_explicit_hash_remap_is_collision_free_and_native(dtype, ordered_keep):
    from razor.checkpoint import remap_hash_table
    model = experiment_model("DeepseekV4", dtype, hash_layers=True)
    adapter = get_adapter(model.config)
    blocks = adapter.moe_blocks(model)
    block = blocks[0].module
    block.gate.weight.zero_()
    rows = torch.tensor([[7, 0], [7, 7], [0, 0], [1, 3]])
    block.gate.tid2eid.copy_(rows.repeat(16, 1))
    keep = torch.tensor(ordered_keep)
    expected = remap_hash_table(block.gate.tid2eid, block.gate.weight, keep.tolist())
    before = {name: value.clone() for name, value in block.state_dict().items()}
    with pytest.raises(NotImplementedError, match="preserving"):
        adapter.prune_block(block, keep)
    with pytest.raises(NotImplementedError, match="table width"):
        adapter.prune_block(block, keep, target_top_k=1, hash_policy="remap")
    for name, value in before.items():
        torch.testing.assert_close(block.state_dict()[name], value, atol=0, rtol=0)
    adapter.prune_block(block, keep, target_top_k=2, hash_policy="remap")
    assert torch.equal(block.gate.tid2eid, expected)
    reserved = ordered_keep.index(0)
    first_unused = next(index for index in range(len(ordered_keep)) if index != reserved)
    assert block.gate.tid2eid[0].tolist() == [first_unused, reserved]
    assert bool((block.gate.tid2eid[:, 0] != block.gate.tid2eid[:, 1]).all())
    adapter.prune_block(blocks[1].module, keep)
    adapter.set_num_experts(4)
    adapter.validate_model(model)
    assert verify_routing(model, adapter, batch())
    config = type(model.config).from_dict(model.config.to_dict())
    config._attn_implementation = "eager"
    restored = type(model)(config).eval().to(dtype)
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(**batch(), use_cache=False).logits,
                               model(**batch(), use_cache=False).logits, atol=0, rtol=0)
