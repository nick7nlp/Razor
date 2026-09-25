"""Numerical agreement and input validation for chunked attention."""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from razor import chunked_attn                                # noqa: E402


def _kernel_module(name: str = "fake_modeling") -> ModuleType:
    """Create an eager attention kernel returning ``[B,Q,H,D]``."""
    mod = ModuleType(name)

    def eager_attention_forward(module, query, key, value, attention_mask,
                                scaling, dropout=0.0, **kwargs):
        scores = torch.matmul(query, key.transpose(-1, -2)) * scaling
        if attention_mask is not None:
            scores = scores + attention_mask[..., : key.shape[-2]]
        probs = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float32)
        probs = probs.to(query.dtype)
        out = torch.matmul(probs, value)          # [B, H, Q, D]
        return out.transpose(1, 2).contiguous(), probs   # -> [B, Q, H, D]

    mod.eager_attention_forward = eager_attention_forward
    return mod


class _Mod(torch.nn.Module):
    num_key_value_groups = 1


def _inputs(heads: int, dim: int, q: int, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda: torch.randn(1, heads, q, dim, generator=g) * 0.1
    neg = torch.finfo(torch.float32).min
    mask = torch.full((q, q), neg).triu(1).view(1, 1, q, q)
    return mk(), mk(), mk(), mask


def test_numerical_agreement_across_chunk_sizes():
    mod = _kernel_module()
    orig = mod.eager_attention_forward
    heads, dim, q = 4, 32, 256
    query, key, value, mask = _inputs(heads, dim, q)
    m, scaling = _Mod(), dim ** -0.5

    ref = orig(m, query, key, value, mask, scaling, 0.0)[0]
    for chunk in (32, 64, 128, 256, 512):
        got = chunked_attn._make(orig, chunk)(
            m, query, key, value, mask, scaling, 0.0)[0]
        assert torch.allclose(ref, got, atol=1e-6, rtol=1e-5), "chunk=%d differs, max|d|=%.3e" % (
            chunk, (ref - got).abs().max().item())
    print("[ok] numerical agreement across chunk sizes")


def test_numerical_agreement_with_short_tail_chunks():
    mod = _kernel_module()
    orig = mod.eager_attention_forward
    dim, q = 32, 256
    query, key, value, mask = _inputs(4, dim, q)
    m, scaling = _Mod(), dim ** -0.5
    ref = orig(m, query, key, value, mask, scaling, 0.0)[0]
    scale = ref.abs().max().item()
    for chunk in (1, 2, 7, 100, 255):
        got = chunked_attn._make(orig, chunk)(
            m, query, key, value, mask, scaling, 0.0)[0]
        rel = (ref - got).abs().max().item() / scale
        assert rel < 1e-6, "chunk=%d drifted %.3e, too large for rounding" % (
            chunk, rel)
    print("[ok] tiny chunks and short tails stay at fp32 rounding")


def test_query_axis_not_confused_with_head_dim():
    mod = _kernel_module()
    orig = mod.eager_attention_forward
    dim = 64
    query, key, value, mask = _inputs(4, dim, 128)
    m = _Mod()
    got = chunked_attn._make(orig, dim)(m, query, key, value, mask,
                                        dim ** -0.5, 0.0)[0]
    ref = orig(m, query, key, value, mask, dim ** -0.5, 0.0)[0]
    assert got.shape == ref.shape, "%s vs %s" % (got.shape, ref.shape)
    assert torch.allclose(ref, got, atol=1e-6, rtol=1e-5)
    print("[ok] query axis identified with chunk == head_dim == %d" % dim)


def test_install_is_idempotent_and_reversible():
    mod = _kernel_module()
    orig = mod.eager_attention_forward
    assert chunked_attn.install_module(mod, 64, verbose=False) is True
    # A second install must not wrap the wrapper: nested wrappers would chunk a
    # chunk and the reported chunk size would stop describing the peak.
    assert chunked_attn.install_module(mod, 64, verbose=False) is False
    assert chunked_attn.uninstall(mod) is True
    assert mod.eager_attention_forward is orig
    print("[ok] install idempotent, uninstall restores the original kernel")


def test_context_manager_restores():
    mod = _kernel_module()
    orig = mod.eager_attention_forward
    with chunked_attn.chunked(mod, 64):
        assert mod.eager_attention_forward is not orig
    assert mod.eager_attention_forward is orig
    print("[ok] context manager restores on exit")


def test_softmax_budget_arithmetic():
    one = chunked_attn.softmax_bytes(8, 128, 4096, tensors=1) / 2 ** 20
    both = chunked_attn.softmax_bytes(8, 128, 4096) / 2 ** 20
    assert abs(one - 16.0) < 1e-6, one
    assert abs(both - 32.0) < 1e-6, both
    assert chunked_attn.softmax_bytes(8, 64, 4096, tensors=1) / 2 ** 20 == 8.0

    # suggest_chunk budgets both score and probability tensors.
    assert chunked_attn.suggest_chunk(8, 4096, budget_gib=0.03125) == 128
    assert chunked_attn.suggest_chunk(8, 4096, budget_gib=0.02) == 64
    assert chunked_attn.suggest_chunk(4, 2048, budget_gib=1.0) == 1024
    assert chunked_attn.suggest_chunk(8, 4096, budget_gib=0.0001) == \
        chunked_attn.MIN_CHUNK
    print("[ok] softmax memory arithmetic and chunk suggestion")


def test_selftest_helper_passes():
    mod = _kernel_module()
    assert chunked_attn.selftest(mod, heads=4, dim=32, q=256, chunk=64)


def test_chunked_attention_rejects_unsupported_inputs():
    import pytest

    orig = _kernel_module().eager_attention_forward
    query, key, value, mask = _inputs(4, 8, 16)
    wrapped = chunked_attn._make(orig, 4)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="chunk"):
            chunked_attn._make(orig, bad)
    for bad_mask in (None, mask[0, 0], mask[..., :3, :], mask.bool(), mask[..., :4]):
        with pytest.raises(ValueError, match="mask"):
            wrapped(_Mod(), query, key, value, bad_mask, 0.5)
    for kwargs in ({"dropout": 0.1}, {"is_causal": True}, {"output_attentions": True}):
        with pytest.raises(ValueError):
            wrapped(_Mod(), query, key, value, mask, 0.5, **kwargs)
    with pytest.raises(ValueError, match="shape"):
        wrapped(_Mod(), query[0], key, value, mask, 0.5)


def test_chunked_attention_handles_both_output_axes():
    orig = _kernel_module().eager_attention_forward
    query, key, value, mask = _inputs(3, 3, 17)

    def untransposed(*args, **kwargs):
        output, weights = orig(*args, **kwargs)
        return output.transpose(1, 2), weights

    for kernel in (orig, untransposed):
        ref = kernel(_Mod(), query, key, value, mask, 0.5)[0]
        got = chunked_attn._make(kernel, 4)(_Mod(), query, key, value, mask, 0.5)[0]
        torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-5)


def test_chunked_context_restores_after_error():
    import pytest

    mod = _kernel_module()
    orig = mod.eager_attention_forward
    with pytest.raises(RuntimeError, match="test failure"):
        with chunked_attn.chunked(mod, 4):
            raise RuntimeError("test failure")
    assert mod.eager_attention_forward is orig
    assert id(mod) not in chunked_attn._installed


def test_native_chunk_context_rejects_overlapping_global_patches(monkeypatch):
    import pytest
    from razor._streaming_attention import native_chunked

    module = _kernel_module("synthetic_native_attention")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    model_type = type("SyntheticDecoder", (torch.nn.Module,), {"__module__": module.__name__})
    original = module.eager_attention_forward
    with native_chunked(model_type(), chunk=2):
        wrapper = module.eager_attention_forward
        for _ in range(2):
            with pytest.raises(RuntimeError, match="active"):
                with native_chunked(model_type(), chunk=4):
                    pass
            assert module.eager_attention_forward is wrapper
    assert module.eager_attention_forward is original
    with pytest.raises(RuntimeError, match="injected"):
        with native_chunked(model_type(), chunk=2):
            raise RuntimeError("injected")
    with native_chunked(model_type(), chunk=2):
        assert module.eager_attention_forward is not original
    assert module.eager_attention_forward is original


def test_native_glm_indexer_broadcast_mask():
    import inspect
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
    from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaIndexer
    from razor._streaming_attention import native_chunked

    torch.manual_seed(7)
    config = GlmMoeDsaConfig(hidden_size=16, q_lora_rank=8, index_n_heads=2,
                             index_head_dim=8, qk_rope_head_dim=4, index_topk=2)
    indexer = GlmMoeDsaIndexer(config, 0).eval()
    signature = inspect.signature(indexer.forward)
    for length in (4, 67):
        arguments = dict(hidden_states=torch.randn(1, length, 16), q_resid=torch.randn(1, length, 8),
                         position_embeddings=(torch.ones(1, length, 4), torch.zeros(1, length, 4)),
                         attention_mask=torch.zeros(1, 1, length))
        arguments["attention_mask"][..., -1] = float("-inf")
        if "position_ids" in signature.parameters:
            arguments["position_ids"] = torch.arange(length)[None]
            arguments["past_key_values"] = None
        else:
            arguments["use_cache"] = False
        with torch.no_grad():
            expected = indexer(**arguments)
            with native_chunked(indexer, chunk=2):
                actual = indexer(**arguments)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall chunked-attention tests passed")
