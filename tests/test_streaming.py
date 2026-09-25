"""Native layer streaming, conversion, quantized storage and prompt gates."""
import copy
import json

import pytest
import torch

from razor.loader import DSV4Tokenizer, _auto_model, quantization_config
from razor.streaming import (CheckpointWeights, _skeleton, dequantize_fp8,
                             stream_forward, unpack_mxfp4)


def test_mxfp4_low_nibble_first_and_e8m0():
    packed = torch.tensor([[0x21, 0x43, 0x65, 0x87] * 4], dtype=torch.uint8)
    scale = torch.tensor([[128]], dtype=torch.uint8)
    actual = unpack_mxfp4(packed, scale, torch.float32)
    expected = torch.tensor([[.5, 1., 1.5, 2., 3., 4., 6., -0.] * 4]) * 2
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(unpack_mxfp4(packed.view(torch.int8), scale, torch.float32), expected)
    with pytest.raises(ValueError, match="one E8M0"):
        unpack_mxfp4(packed, torch.ones(1, 2, dtype=torch.uint8))
    with pytest.raises(ValueError, match="E8M0 bytes"):
        unpack_mxfp4(packed, scale.float())


def test_fp8_float_scales_and_ragged_blocks():
    values = torch.ones(3, 5).to(torch.float8_e4m3fn)
    scales = torch.tensor([[2., 3., 4.], [5., 6., 7.]], dtype=torch.float32)
    original = scales.clone()
    actual = dequantize_fp8(values, scales, block_size=(2, 2), dtype=torch.float32)
    expected = torch.tensor([[2., 2., 3., 3., 4.], [2., 2., 3., 3., 4.],
                             [5., 5., 6., 6., 7.]])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert scales.dtype == torch.float32 and torch.equal(scales, original)
    with pytest.raises(ValueError, match="scale shape"):
        dequantize_fp8(values, scales, block_size=(3, 3))


def test_fp8_e8m0_is_not_mxfp4():
    weights = torch.full((2, 4), 1.5).to(torch.float8_e4m3fn)
    scale = torch.tensor([[127, 128]], dtype=torch.uint8)
    actual = dequantize_fp8(weights, scale, block_size=(2, 2), dtype=torch.float32)
    torch.testing.assert_close(actual, torch.tensor([[1.5, 1.5, 3., 3.]]).expand(2, -1))
    with pytest.raises(ValueError, match="FP8"):
        dequantize_fp8(weights.view(torch.uint8), scale, block_size=(2, 2))


def _tiny_native(tmp_path, family="qwen3_moe"):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(7)
    common = dict(vocab_size=32, hidden_size=16, moe_intermediate_size=8,
                  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                  num_experts_per_tok=2, pad_token_id=0, eos_token_id=2,
                  tie_word_embeddings=False)
    if family == "glm_moe_dsa":
        from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
        config = GlmMoeDsaConfig(**common, intermediate_size=24, n_routed_experts=4,
                                q_lora_rank=8, kv_lora_rank=4, qk_rope_head_dim=4,
                                qk_nope_head_dim=4, v_head_dim=8, index_head_dim=8,
                                index_n_heads=2, index_topk=2,
                                mlp_layer_types=["sparse", "sparse"], indexer_types=["full", "shared"])
    elif family == "deepseek_v4":
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
        config = DeepseekV4Config(**common, n_routed_experts=4, head_dim=16,
                                 partial_rotary_factor=.5, q_lora_rank=8,
                                 o_lora_rank=8, o_groups=1, index_n_heads=2,
                                 index_head_dim=16, index_topk=2, sliding_window=4,
                                 mlp_layer_types=["hash_moe", "moe"],
                                 layer_types=["sliding_attention", "compressed_sparse_attention"],
                                 compress_rates={"compressed_sparse_attention": 2,
                                                 "heavily_compressed_attention": 4})
    else:
        config = transformers.Qwen3MoeConfig(**common, intermediate_size=24,
                                             num_experts=4, head_dim=8)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    native = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.float32).eval()
    native.save_pretrained(tmp_path, safe_serialization=True,
                           save_original_format=family != "deepseek_v4")
    return native, config


def _clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()
                if torch.is_tensor(item) or isinstance(item, (dict, tuple)) or item is None}
    return value


def _equal(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, tuple):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _equal(left, right)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _equal(actual[key], expected[key])
    else:
        assert actual == expected


@pytest.mark.parametrize("family", ["qwen3_moe", "glm_moe_dsa", "deepseek_v4"])
def test_native_stream_matches_all_layer_inputs_and_hidden(tmp_path, monkeypatch, family):
    native, config = _tiny_native(tmp_path, family)
    batches = [
        {"input_ids": torch.tensor([[1, 3, 4, 5, 0]]),
         "attention_mask": torch.tensor([[1, 1, 1, 1, 0]])},
        {"input_ids": torch.tensor([[6, 7, 8]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)},
    ]
    resident, streamed = {}, {}
    current = [0]
    handles = []
    for index, layer in enumerate(native.model.layers):
        def before(module, args, kwargs, index=index):
            resident[("input", index, current[0])] = (_clone(args), _clone(kwargs))
        def after(module, args, kwargs, output, index=index):
            resident[("output", index, current[0])] = (_clone(output), _clone(kwargs))
        handles += [layer.register_forward_pre_hook(before, with_kwargs=True),
                    layer.register_forward_hook(after, with_kwargs=True)]
    with torch.no_grad():
        for batch_index, batch in enumerate(batches):
            current[0] = batch_index
            native(**batch, use_cache=False, return_dict=True)
    for handle in handles:
        handle.remove()
    def forbidden(*args, **kwargs):
        raise AssertionError("streaming must never load a full resident checkpoint")
    monkeypatch.setattr(type(native), "from_pretrained", forbidden)
    skeleton = _skeleton(str(tmp_path), config, torch.float32, False)
    weights = CheckpointWeights(tmp_path, skeleton, dtype=torch.float32)
    def audit(event, layer, batch, value, kwargs):
        streamed[(event, layer, batch)] = (_clone(value), _clone(kwargs))
        for other, module in enumerate(skeleton.model.layers):
            assert all(parameter.is_meta == (other != layer) for parameter in module.parameters())
    outputs = stream_forward(skeleton, weights, batches, device="cpu", layer_callback=audit)
    assert len(outputs) == len(batches)
    assert resident.keys() == streamed.keys()
    for key in resident:
        _equal(streamed[key], resident[key])
    assert all(parameter.is_meta for parameter in skeleton.parameters())
    assert max(weights.read_count.values()) == 1
    first, last = streamed[("output", 0, 0)][0], streamed[("output", 1, 0)][0]
    first = first[0] if isinstance(first, tuple) else first
    last = last[0] if isinstance(last, tuple) else last
    assert not torch.equal(first, last)


@pytest.mark.parametrize("family", ["qwen3_moe", "glm_moe_dsa", "deepseek_v4"])
@pytest.mark.parametrize("attention_chunk", [None, 2])
def test_streaming_saliency_matches_resident_collector(tmp_path, family, attention_chunk):
    from razor import metrics
    from razor.adapters import get_adapter
    from razor.collect import collect
    from razor.streaming import collect_streaming
    native, _ = _tiny_native(tmp_path, family)
    batch = {"input_ids": torch.tensor([[1, 3, 4, 0]]),
             "attention_mask": torch.tensor([[1, 1, 1, 0]])}
    resident = collect(native, get_adapter(native.config), [batch], expert_chunk=2, verbose=False)
    streamed = collect_streaming(str(tmp_path), [batch], dtype=torch.float32, device="cpu",
                                expert_chunk=2, attn_implementation="eager", verbose=False,
                                attention_chunk=attention_chunk)
    assert resident.keys() == streamed.keys()
    for tag in resident:
        assert resident[tag].keys() == streamed[tag].keys()
        for key, value in resident[tag].items():
            if torch.is_tensor(value):
                torch.testing.assert_close(streamed[tag][key], value, rtol=1e-5, atol=1e-7)
            else:
                assert streamed[tag][key] == value

    # Agreement alone would still hold if a regression dropped the RCS-Refill
    # moments on both paths, so require the streamed pack to serve the paper's
    # criterion wherever the routing admits it.
    refill_fields = set(metrics.FIELDS["rcs-refill"])
    scored = 0
    for tag, record in streamed.items():
        served = metrics.available(record)
        if record.get("razor_supported"):
            missing = refill_fields - set(record)
            assert not missing, f"{tag}: streaming lost refill moments {sorted(missing)}"
            assert record["refill_supported"]
            assert {"razor", "rcs-refill"} <= set(served)
            scored += 1
        else:
            # Hash routing has no router ranking, so rank-(k+1) is undefined.
            assert not refill_fields & set(record)
            assert "razor" not in served
    assert scored, f"{family}: no streamed layer exercised the refill path"


def test_stream_batch_window_bounds_activations_and_accumulates(tmp_path, monkeypatch):
    import razor.streaming as streaming
    from razor.adapters import get_adapter
    from razor.collect import collect
    native, _ = _tiny_native(tmp_path)
    batches = [{"input_ids": torch.tensor([[1, 2, 3]] + [[4, 5, 6]] * index)}
               for index in range(3)]
    expected = collect(native, get_adapter(native.config), batches, verbose=False)
    original = streaming.stream_forward
    sizes, seen = [], []
    def forward(model, weights, window, **kwargs):
        sizes.append(len(window))
        return original(model, weights, window, **kwargs)
    monkeypatch.setattr(streaming, "stream_forward", forward)
    def audit(event, layer, batch, value, kwargs):
        if event == "input" and layer == 0:
            seen.append(batch)
    for options, wanted in (({}, [1, 1, 1]), ({"batch_window": 2}, [2, 1])):
        sizes.clear()
        seen.clear()
        actual = streaming.collect_streaming(str(tmp_path), iter(batches), dtype=torch.float32,
                                              device="cpu", attn_implementation="eager", verbose=False,
                                              layer_callback=audit, **options)
        assert sizes == wanted and seen == [0, 1, 2]
        _equal(actual, expected)
    with pytest.raises(ValueError, match="batch_window"):
        streaming.collect_streaming(str(tmp_path), batches, batch_window=0)


def test_stream_max_layers_does_not_read_later_weights(tmp_path):
    _, config = _tiny_native(tmp_path)
    model = _skeleton(str(tmp_path), config, torch.float32, False)
    weights = CheckpointWeights(tmp_path, model, dtype=torch.float32)
    seen = []
    stream_forward(model, weights, [{"input_ids": torch.tensor([[1, 2]])}], max_layers=1,
                   layer_callback=lambda event, layer, batch, value, kwargs: seen.append(layer))
    assert set(seen) == {0}
    assert not any("layers.1." in name for name in weights.read_count)
    assert all(value.is_meta for value in model.parameters())


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scale_storage", ["float", "e8m0"])
def test_quantized_resident_and_streaming_use_clean_runtime_config(
        tmp_path, source_dtype, compute_dtype, scale_storage):
    from safetensors.torch import load_file, save_file
    from razor.loader import load_model
    from razor.collect import collect
    from razor.streaming import collect_streaming
    _tiny_native(tmp_path)
    path = tmp_path / "model.safetensors"
    tensors = {name: value.to(source_dtype) if value.is_floating_point() else value
               for name, value in load_file(str(path)).items()}
    key = "model.layers.0.self_attn.q_proj.weight"
    tensors[key] = tensors[key].to(torch.float8_e4m3fn)
    tensors[key + "_scale_inv"] = (torch.ones(8, 8, dtype=torch.float32) if scale_storage == "float"
                                  else torch.full((8, 8), 127, dtype=torch.uint8))
    save_file(tensors, str(path))
    path = tmp_path / "config.json"
    raw = json.loads(path.read_text())
    raw["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [2, 2]}
    path.write_text(json.dumps(raw))
    model, adapter = load_model(str(tmp_path), dtype=compute_dtype, device_map="cpu",
                                attn_implementation="eager", verbose=False)
    assert getattr(model.config, "quantization_config", None) is None
    assert model._razor_source_quantization["quant_method"] == "fp8"
    assert all(value.dtype == compute_dtype for value in model.parameters())
    batch = {"input_ids": torch.tensor([[1, 2, 3]])}
    with torch.no_grad():
        logits = model(**batch, use_cache=False).logits
    assert logits.dtype == compute_dtype and torch.isfinite(logits).all()
    resident = collect(model, adapter, [batch], verbose=False)
    streamed = collect_streaming(str(tmp_path), [batch], dtype=compute_dtype, device="cpu",
                                attn_implementation="eager", verbose=False)
    _equal(streamed, resident)


def test_resolve_checkpoint_downloads_only_selected_artifacts(monkeypatch, tmp_path):
    import huggingface_hub
    from razor.loader import resolve_checkpoint
    captured = {}
    def snapshot(**kwargs):
        captured.update(kwargs)
        return str(tmp_path)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    assert resolve_checkpoint("org/model") == str(tmp_path)
    assert "*.safetensors" in captured["allow_patterns"]
    assert "*.bin" not in captured["allow_patterns"]
    resolve_checkpoint("org/model", include_weights=False)
    assert "*.safetensors" not in captured["allow_patterns"]
    assert "encoding/**" in captured["allow_patterns"]
    assert resolve_checkpoint(str(tmp_path)) == str(tmp_path.resolve())


@pytest.mark.parametrize("entry", ["resolve", "config", "model", "weights", "skeleton"])
def test_incomplete_checkpoint_rejected_before_any_load(tmp_path, monkeypatch, entry):
    import safetensors
    import transformers
    import razor.streaming as streaming
    from razor.loader import load_config, load_model, resolve_checkpoint
    _, config = _tiny_native(tmp_path)
    model = _skeleton(str(tmp_path), config, torch.float32, False)
    marker = tmp_path / ".razor-incomplete"
    marker.touch()
    def forbidden(*args, **kwargs):
        raise AssertionError("incomplete checkpoint must not read config or weights")
    calls = {
        "resolve": lambda: resolve_checkpoint(str(tmp_path)),
        "config": lambda: load_config(str(tmp_path)),
        "model": lambda: load_model(str(tmp_path), dtype=torch.float32, device_map="cpu", verbose=False),
        "weights": lambda: CheckpointWeights(tmp_path, model, dtype=torch.float32),
        "skeleton": lambda: _skeleton(str(tmp_path), config, torch.float32, False),
    }
    with monkeypatch.context() as patch:
        patch.setattr(safetensors, "safe_open", forbidden)
        patch.setattr(transformers.AutoConfig, "from_pretrained", forbidden)
        patch.setattr(streaming, "_auto_model", forbidden)
        with pytest.raises(RuntimeError, match=r"incomplete.*\.razor-incomplete"):
            calls[entry]()
    marker.unlink()
    assert calls[entry]() is not None


@pytest.mark.parametrize("kind,base", [("configuration", "PretrainedConfig"),
                                       ("modeling", "PreTrainedModel")])
def test_incomplete_checkpoint_rejects_trusted_class_read(tmp_path, monkeypatch, kind, base):
    from razor.loader import load_checkpoint_class
    (tmp_path / f"{kind}_fixture.py").write_text(
        f"from transformers import {base}\nclass Fixture({base}):\n    pass\n")
    marker = tmp_path / ".razor-incomplete"
    marker.touch()
    def forbidden(*args, **kwargs):
        raise AssertionError("incomplete checkpoint must not read Python source")
    with monkeypatch.context() as patch:
        patch.setattr(type(tmp_path), "read_text", forbidden)
        with pytest.raises(RuntimeError, match=r"incomplete.*\.razor-incomplete"):
            load_checkpoint_class(tmp_path, class_name="Fixture", kind=kind, trust_remote_code=True)
    marker.unlink()
    cls = load_checkpoint_class(tmp_path, class_name="Fixture", kind=kind, trust_remote_code=True)
    assert cls.__name__ == "Fixture"


@pytest.mark.parametrize("include_weights", [False, True])
def test_resolved_snapshot_rejects_incomplete_marker(tmp_path, monkeypatch, include_weights):
    import huggingface_hub
    from razor.loader import resolve_checkpoint
    marker = tmp_path / ".razor-incomplete"
    marker.touch()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(tmp_path))
    with pytest.raises(RuntimeError, match=r"incomplete.*\.razor-incomplete"):
        resolve_checkpoint("org/model", include_weights=include_weights)
    marker.unlink()
    assert resolve_checkpoint("org/model", include_weights=include_weights) == str(tmp_path)


def test_hf_snapshot_blob_symlinks_are_readable(tmp_path):
    snapshot = tmp_path / "snapshot"
    _, config = _tiny_native(snapshot)
    shard = snapshot / "model.safetensors"
    blob = tmp_path / "weights.blob"
    shard.rename(blob)
    shard.symlink_to(blob)
    model = _skeleton(str(snapshot), config, torch.float32, False)
    weights = CheckpointWeights(snapshot, model, dtype=torch.float32)
    stream_forward(model, weights, [{"input_ids": torch.tensor([[1, 2]])}], max_layers=1)
    assert weights.read_count


def test_native_stream_failure_cleans_hooks_and_layers(tmp_path):
    _, config = _tiny_native(tmp_path)
    skeleton = _skeleton(str(tmp_path), config, torch.float32, False)
    weights = CheckpointWeights(tmp_path, skeleton, dtype=torch.float32)
    batch = {"input_ids": torch.tensor([[1, 2]])}
    def fail(event, layer, batch, value, kwargs):
        if event == "output":
            raise RuntimeError("audit failure")
    expected = stream_forward(skeleton, weights, [batch, batch])
    for _ in range(2):
        with pytest.raises(RuntimeError, match="audit failure"):
            stream_forward(skeleton, weights, [batch, batch], device="cpu", layer_callback=fail)
        _assert_stream_released(skeleton)
        actual = stream_forward(skeleton, weights, [batch, batch])
        for left, right in zip(actual, expected):
            _equal(left, right)
        _assert_stream_released(skeleton)


def _assert_stream_released(model):
    assert all(parameter.is_meta for parameter in model.parameters())
    assert all(not layer._forward_hooks and not layer._forward_pre_hooks for layer in model.model.layers)


@pytest.mark.parametrize("direct", [False, True])
def test_missing_prelude_rolls_back_embedding_and_retries(tmp_path, direct):
    from safetensors.torch import load_file, save_file
    _, config = _tiny_native(tmp_path)
    path = tmp_path / "model.safetensors"
    tensors = load_file(str(path))
    norm = tensors.pop("model.norm.weight")
    save_file(tensors, str(path))
    model = _skeleton(str(tmp_path), config, torch.float32, False)
    weights = CheckpointWeights(tmp_path, model, dtype=torch.float32)
    batch = {"input_ids": torch.tensor([[1, 2]])}
    with pytest.raises(RuntimeError, match="missing native weights.*model.norm.weight"):
        if direct:
            weights.materialize("model.", "cpu", exclude=("model.layers.",))
        else:
            stream_forward(model, weights, [batch, batch])
    assert weights.read_count["model.embed_tokens.weight"] == 1
    _assert_stream_released(model)
    tensors["model.norm.weight"] = norm
    save_file(tensors, str(path))
    weights = CheckpointWeights(tmp_path, model, dtype=torch.float32)
    for _ in range(2):
        stream_forward(model, weights, [batch, batch])
        _assert_stream_released(model)


@pytest.mark.parametrize("prefix", ["model.", "model.layers.0."])
@pytest.mark.parametrize("direct", [False, True])
def test_partial_conversion_rolls_back_loaded_weights(tmp_path, monkeypatch, prefix, direct):
    _, config = _tiny_native(tmp_path)
    model = _skeleton(str(tmp_path), config, torch.float32, False)
    weights = CheckpointWeights(tmp_path, model, dtype=torch.float32)
    exclude = ("model.layers.",) if prefix == "model." else ()
    targets = [name for name in weights.groups if name.startswith(prefix)
               and not any(name.startswith(item) for item in exclude)]
    original = weights.converted
    observed = []
    def fail(target, device):
        if target == targets[1]:
            assert not model.get_parameter(targets[0]).is_meta
            observed.append(target)
            raise RuntimeError("conversion failure after first weight")
        return original(target, device)
    batch = {"input_ids": torch.tensor([[1, 2]])}
    with monkeypatch.context() as patch:
        patch.setattr(weights, "converted", fail)
        with pytest.raises(RuntimeError, match="conversion failure after first weight"):
            if direct:
                weights.materialize(prefix, "cpu", exclude=exclude)
            else:
                stream_forward(model, weights, [batch, batch])
    assert observed == [targets[1]]
    _assert_stream_released(model)
    for _ in range(2):
        stream_forward(model, weights, [batch, batch])
        _assert_stream_released(model)


@pytest.mark.parametrize("phase", ["buffer", "worker", "pre_hook", "post_hook", "collector", "forward"])
def test_stream_setup_failure_releases_weights_and_hooks(tmp_path, monkeypatch, phase):
    import razor.streaming as streaming
    from razor.adapters import get_adapter
    from razor.collect import SaliencyCollector
    _, config = _tiny_native(tmp_path)
    model = _skeleton(str(tmp_path), config, torch.float32, False)
    weights = CheckpointWeights(tmp_path, model, dtype=torch.float32)
    collector = None
    def fail(*args, **kwargs):
        assert not model.model.embed_tokens.weight.is_meta
        raise RuntimeError("injected setup failure")
    with monkeypatch.context() as patch:
        if phase == "buffer":
            model.register_buffer("fixture_buffer", torch.ones(1), persistent=False)
            assign = streaming._assign
            def assign_buffer(owner, name, value):
                if name == "fixture_buffer":
                    fail()
                return assign(owner, name, value)
            patch.setattr(streaming, "_assign", assign_buffer)
        elif phase == "worker":
            patch.setattr(streaming, "_NativeWorker", fail)
        elif phase == "pre_hook":
            patch.setattr(model.model.layers[1], "register_forward_pre_hook", fail)
        elif phase == "post_hook":
            patch.setattr(model.model.layers[0], "register_forward_hook", fail)
        elif phase == "collector":
            collector = SaliencyCollector(get_adapter(config))
            install = collector.install_blocks
            def install_then_fail(blocks):
                install(blocks)
                fail()
            patch.setattr(collector, "install_blocks", install_then_fail)
        else:
            patch.setattr(model.model.layers[0], "forward", fail)
        with pytest.raises(RuntimeError, match="injected setup failure"):
            stream_forward(model, weights, [{"input_ids": torch.tensor([[1, 2]])}] * 2,
                           collector=collector)
    _assert_stream_released(model)
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
    for _ in range(2):
        stream_forward(model, weights, [{"input_ids": torch.tensor([[1, 2]])}] * 2)
        _assert_stream_released(model)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_last_layer_shared_nonfinite_rejected_after_checked_batch(tmp_path, invalid):
    from razor.adapters import get_adapter
    from razor.collect import SaliencyCollector
    _, config = _tiny_native(tmp_path, "glm_moe_dsa")
    model = _skeleton(str(tmp_path), config, torch.float32, False)
    weights = CheckpointWeights(tmp_path, model, dtype=torch.float32)
    collector = SaliencyCollector(get_adapter(config))
    corrupt = [False]
    shared = model.model.layers[-1].mlp.shared_experts
    handle = shared.register_forward_hook(
        lambda module, args, output: torch.full_like(output, invalid) if corrupt[0] else output)
    def audit(event, layer, batch, value, kwargs):
        if event == "input" and layer == 1 and batch == 1:
            assert "main_1" in collector._gate_checked
            corrupt[0] = True
    batches = [{"input_ids": torch.tensor([[1, 2, 3]])}] * 2
    try:
        with pytest.raises(RuntimeError, match="layer 1.*batch 1.*non-finite"):
            stream_forward(model, weights, batches, collector=collector, layer_callback=audit)
    finally:
        handle.remove()
    _assert_stream_released(model)
    for _ in range(2):
        outputs = stream_forward(model, weights, batches, collector=collector)
        assert all(torch.isfinite(output[0]).all() for output in outputs)
        _assert_stream_released(model)


def test_decoded_resident_native_tied_weights(tmp_path):
    from razor.streaming import load_decoded_model
    native, config = _tiny_native(tmp_path)
    native.config.tie_word_embeddings = True
    native.tie_weights()
    native.save_pretrained(tmp_path, safe_serialization=True)
    loaded = load_decoded_model(str(tmp_path), config=native.config, dtype=torch.float32, device_map="cpu")
    for name, value in native.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)


def _tied_checkpoint(tmp_path):
    """Save a checkpoint that stores the embedding only, as tying requires."""
    from safetensors.torch import load_file
    native, _ = _tiny_native(tmp_path)
    native.config.tie_word_embeddings = True
    native.tie_weights()
    native.save_pretrained(tmp_path, safe_serialization=True)
    stored = load_file(str(tmp_path / "model.safetensors"))
    assert "model.embed_tokens.weight" in stored and "lm_head.weight" not in stored
    assert "lm_head.weight" in native.state_dict()
    return native


@pytest.mark.parametrize("device_map", ["cpu", {"": "cpu"}, {"model": "cpu", "lm_head": "cpu"},
                                        {"lm_head": "cpu", "model": "cpu"}],
                         ids=["string", "root", "split", "split_reversed"])
def test_decoded_tied_head_does_not_depend_on_device_map_order(tmp_path, device_map):
    from razor.streaming import load_decoded_model
    native = _tied_checkpoint(tmp_path)
    inputs = {"input_ids": torch.tensor([[1, 3, 4, 5]])}
    loaded = load_decoded_model(str(tmp_path), config=copy.deepcopy(native.config),
                                dtype=torch.float32, device_map=device_map)
    expected, actual = native.state_dict(), loaded.state_dict()
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
    assert all(not value.is_meta and value.dtype == torch.float32 for value in loaded.parameters())
    with torch.no_grad():
        _equal(loaded(**inputs, use_cache=False).logits, native(**inputs, use_cache=False).logits)


@pytest.mark.parametrize("device_map", ["cpu", {"model": "cpu", "lm_head": "cpu"}],
                         ids=["string", "split"])
@pytest.mark.parametrize("tie", [False, True])
def test_decoded_model_still_rejects_a_genuinely_missing_head(tmp_path, device_map, tie):
    from safetensors.torch import load_file, save_file
    from razor.streaming import load_decoded_model
    native, _ = _tiny_native(tmp_path)
    native.config.tie_word_embeddings = tie
    native.tie_weights()
    native.save_pretrained(tmp_path, safe_serialization=True)
    path = tmp_path / "model.safetensors"
    tensors = load_file(str(path))
    del tensors["model.embed_tokens.weight" if tie else "lm_head.weight"]
    save_file(tensors, str(path))
    with pytest.raises(RuntimeError, match="missing native weights"):
        load_decoded_model(str(tmp_path), config=copy.deepcopy(native.config),
                           dtype=torch.float32, device_map=device_map)


def test_tied_alias_materializes_only_the_requested_slots(tmp_path):
    native = _tied_checkpoint(tmp_path)
    model = _skeleton(str(tmp_path), copy.deepcopy(native.config), torch.float32, False)
    weights = CheckpointWeights(tmp_path, model, dtype=torch.float32)
    assert weights.aliases["lm_head.weight"] == weights.aliases["model.embed_tokens.weight"]
    loaded = weights.materialize("lm_head.", "cpu")
    assert loaded == {"lm_head.weight"}
    head = model.get_parameter("lm_head.weight")
    assert not head.is_meta and head.dtype == torch.float32
    torch.testing.assert_close(head, native.get_parameter("model.embed_tokens.weight"), rtol=0, atol=0)
    assert model.get_parameter("model.embed_tokens.weight").is_meta
    weights.release(loaded)
    assert all(value.is_meta for value in model.parameters())
    stream_forward(model, weights, [{"input_ids": torch.tensor([[1, 2]])}], max_layers=1)
    _assert_stream_released(model)


def test_checkpoint_mixed_attention_fp8_expert_mxfp4_and_integer_hash(tmp_path):
    from types import SimpleNamespace
    from safetensors.torch import save_file
    module = torch.nn.Module()
    module.config = SimpleNamespace(quantization_config={"quant_method": "fp8", "weight_block_size": [2, 2]})
    module.base_model_prefix = ""
    module.attention = torch.nn.Linear(4, 2, bias=False, device="meta", dtype=torch.bfloat16)
    module.expert = torch.nn.Linear(32, 2, bias=False, device="meta", dtype=torch.bfloat16)
    module.normal = torch.nn.Linear(4, 2, bias=False, device="meta", dtype=torch.bfloat16)
    module.stable = torch.nn.Linear(4, 2, bias=False, device="meta", dtype=torch.float32)
    module.register_buffer("correction", torch.empty(2, dtype=torch.float32, device="meta"))
    module.register_buffer("tid2eid", torch.empty(2, dtype=torch.long, device="meta"))
    tensors = {"attention.weight": torch.ones(2, 4).to(torch.float8_e4m3fn),
               "attention.scale": torch.tensor([[2., 4.]], dtype=torch.float32),
               "expert.weight": torch.full((2, 16), 0x21, dtype=torch.int8),
               "expert.scale": torch.tensor([[127], [128]], dtype=torch.uint8),
               "normal.weight": torch.full((2, 4), 1.0001, dtype=torch.float32),
               "stable.weight": torch.ones(2, 4).to(torch.float8_e4m3fn),
               "stable.scale": torch.full((1, 2), 1.0001, dtype=torch.float32),
               "correction": torch.tensor([1.0001, 2.0002], dtype=torch.float32),
               "tid2eid": torch.tensor([2**40 + 1, 2**40 + 3], dtype=torch.long)}
    save_file(tensors, str(tmp_path / "model.safetensors"))
    weights = CheckpointWeights(tmp_path, module, dtype=torch.bfloat16)
    for _ in range(2):
        loaded = weights.materialize("", "cpu")
        assert module.attention.weight.dtype == torch.bfloat16
        assert module.expert.weight.dtype == torch.bfloat16
        assert module.normal.weight.dtype == torch.bfloat16
        assert module.stable.weight.dtype == module.correction.dtype == torch.float32
        assert module.tid2eid.dtype == torch.int64
        assert torch.equal(module.tid2eid, tensors["tid2eid"])
        torch.testing.assert_close(module.stable.weight, torch.full((2, 4), 1.0001), rtol=0, atol=0)
        torch.testing.assert_close(module.correction, tensors["correction"], rtol=0, atol=0)
        assert weights.raw("attention.scale").dtype == torch.float32
        assert torch.equal(weights.raw("expert.scale"), tensors["expert.scale"])
        torch.testing.assert_close(module.attention.weight.float(), torch.tensor([[2., 2., 4., 4.]]).expand(2, -1))
        torch.testing.assert_close(module.expert.weight[0].float(), torch.tensor([.5, 1.] * 16))
        torch.testing.assert_close(module.expert.weight[1].float(), torch.tensor([1., 2.] * 16))
        assert torch.isfinite(module.attention(torch.ones(1, 4, dtype=torch.bfloat16))).all()
        weights.release(loaded)
        assert all(value.is_meta for value in module.state_dict().values())


def test_trusted_local_classes_without_auto_map(tmp_path):
    from razor.loader import load_config, load_checkpoint_class
    (tmp_path / "configuration_fixture.py").write_text(
        "from transformers import Qwen3MoeConfig\n"
        "class LocalFixtureConfig(Qwen3MoeConfig):\n"
        "    model_type = 'razor_local_fixture'\n")
    (tmp_path / "modeling_fixture.py").write_text(
        "from transformers import Qwen3MoeForCausalLM\n"
        "from .configuration_fixture import LocalFixtureConfig\n"
        "class LocalFixtureForCausalLM(Qwen3MoeForCausalLM):\n"
        "    config_class = LocalFixtureConfig\n")
    _, config = _tiny_native(tmp_path)
    raw = config.to_dict()
    raw.update(model_type="razor_local_fixture", architectures=["LocalFixtureForCausalLM"])
    raw.pop("auto_map", None)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_config(str(tmp_path), trust_remote_code=False)
    with pytest.raises(ValueError, match="trust_remote_code"):
        load_checkpoint_class(tmp_path, class_name="LocalFixtureForCausalLM")
    loaded = load_config(str(tmp_path), trust_remote_code=True)
    net = _auto_model("from_config", config=loaded, trust_remote_code=True, dtype=torch.float32).eval()
    assert type(net).__name__ == "LocalFixtureForCausalLM"
    with torch.no_grad():
        assert net(input_ids=torch.tensor([[1, 2]]), use_cache=False).logits.shape == (1, 2, 32)


@pytest.mark.parametrize("attention,kda", [("eager", False), ("flash_attention_2", False),
                                           ("flash_attention_2", True)])
def test_official_kimi_checkpoint_class_and_native_stream(tmp_path, attention, kda):
    import os
    from safetensors.torch import save_file
    from razor.loader import load_checkpoint_class
    from razor.adapters import get_adapter
    from razor.collect import SaliencyCollector, collect
    source = os.environ.get("RAZOR_TEST_KIMI_CHECKPOINT")
    if not source:
        pytest.skip("set RAZOR_TEST_KIMI_CHECKPOINT to explicitly trust official checkpoint code")
    if attention == "flash_attention_2" and not torch.cuda.is_available():
        pytest.skip("native flash attention requires CUDA")
    device = "cpu" if attention == "eager" else "cuda:0"
    dtype = torch.float32 if attention == "eager" else torch.bfloat16
    cls = load_checkpoint_class(source, model_type="kimi_linear", kind="configuration", trust_remote_code=True)
    config = cls(vocab_size=32, hidden_size=16, intermediate_size=24,
                 num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                 moe_intermediate_size=8, num_experts=4, num_experts_per_token=2,
                 num_shared_experts=1, q_lora_rank=8, kv_lora_rank=4,
                 qk_nope_head_dim=8, qk_rope_head_dim=0, v_head_dim=8,
                 mla_use_nope=True, routed_expert_hidden_size=8,
                 latent_moe_use_norm=True, attn_res_block_size=2, pad_token_id=0)
    if kda:
        config.linear_attn_config = {"kda_layers": [1], "full_attn_layers": [2],
                                     "short_conv_kernel_size": 4, "head_dim": 16, "num_heads": 2}
    config.architectures = ["KimiLinearForCausalLM"]
    config.name_or_path = source
    config._attn_implementation = attention
    torch.manual_seed(7)
    native = _auto_model("from_config", config=config, trust_remote_code=True, dtype=dtype).to(device).eval()
    with torch.no_grad():
        for name, parameter in native.named_parameters():
            if name.endswith(("dt_bias", "e_score_correction_bias")):
                parameter.zero_()
            assert torch.isfinite(parameter).all(), f"uninitialized native fixture parameter: {name}"
    tensors = {name: value.detach().cpu().contiguous() for name, value in native.state_dict().items()}
    save_file(tensors, str(tmp_path / "model.safetensors"))
    batch = {"input_ids": torch.tensor([[1, 2, 3, 0]]), "attention_mask": torch.tensor([[1, 1, 1, 0]])}
    resident = collect(native, get_adapter(native.config), [batch], expert_chunk=2, verbose=False)
    skeleton = _skeleton(source, config, dtype, True)
    weights = CheckpointWeights(tmp_path, skeleton, dtype=dtype)
    collector = SaliencyCollector(get_adapter(skeleton.config), expert_chunk=2)
    stream_forward(skeleton, weights, [batch], device=device, collector=collector)
    _equal(collector.finalize(), resident)
    assert all(value.is_meta for value in skeleton.parameters())


def test_unknown_quantization_rejected():
    from types import SimpleNamespace
    with pytest.raises(NotImplementedError, match="unverified"):
        quantization_config(SimpleNamespace(quantization_config={"quant_method": "gptq"}))


def test_conditional_wrapper_fallback_only_for_config(monkeypatch):
    import transformers
    sentinel = object()
    def unsupported(*args, **kwargs):
        raise ValueError("Unrecognized configuration class Dummy")
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_config", unsupported)
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_config", lambda *a, **k: sentinel)
    assert _auto_model("from_config", config=object(), trust_remote_code=False) is sentinel
    def corrupt(*args, **kwargs):
        raise ValueError("corrupt checkpoint")
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_config", corrupt)
    with pytest.raises(ValueError, match="corrupt"):
        _auto_model("from_config", config=object())


def test_dsv4_official_encoder_trust_and_no_double_merge(tmp_path):
    encoding = tmp_path / "encoding"
    encoding.mkdir()
    (encoding / "encoding_dsv4.py").write_text(
        "def encode_messages(messages, thinking_mode='chat'):\n"
        "    return '|'.join(str(m.get('content', '')) for m in messages) + '<assistant>'\n")
    tokenizer = object()
    with pytest.raises(ValueError, match="trust_remote_code"):
        DSV4Tokenizer(tokenizer, str(tmp_path))
    wrapped = DSV4Tokenizer(tokenizer, str(tmp_path), trust_remote_code=True)
    messages = [{"role": "user", "content": "request"}, {"role": "tool", "content": "result"}]
    original = copy.deepcopy(messages)
    assert wrapped.apply_chat_template(messages) == "request|result<assistant>"
    assert wrapped.assistant_open_str == "<assistant>"
    assert messages == original
    record = {"messages": [{"role": "assistant", "content": None,
                            "tool_calls": [{"id": "call-1", "function": {
                                "name": "lookup", "arguments": '{"key": 2}'}}]},
                           {"role": "tool", "content": "result", "tool_call_id": "call-1", "name": "lookup"}],
              "tools": [{"type": "function", "function": {"name": "lookup"}}]}
    untouched = copy.deepcopy(record)
    captured = []
    def encode(items, thinking_mode):
        captured.extend(items)
        return "official"
    wrapped._encoder.encode_messages = encode
    assert wrapped.render_record(record) == "official"
    assert captured[-1]["tool_call_id"] == "call-1" and captured[-1]["name"] == "lookup"
    assert captured[1]["tool_calls"][0]["function"]["arguments"] == {"key": 2}
    assert captured[0]["tools"] == record["tools"]
    assert record == untouched
