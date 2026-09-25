"""Query-chunked eager attention; floating-point results may differ slightly."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch

DEFAULT_CHUNK = 128
MIN_CHUNK = 32

_installed: dict[int, str] = {}


def softmax_bytes(heads: int, chunk: int, keys: int, tensors: int = 2) -> int:
    """Estimate fp32 softmax memory in bytes."""
    return heads * chunk * keys * 4 * tensors


def suggest_chunk(heads: int, keys: int, budget_gib: float,
                  floor: int = MIN_CHUNK, ceil: int = 1024) -> int:
    """Suggest a power-of-two chunk; the floor may exceed the memory budget."""
    import math
    for name, value in (("heads", heads), ("keys", keys), ("floor", floor), ("ceil", ceil)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("%s must be a positive integer" % name)
    if floor > ceil or not math.isfinite(budget_gib) or budget_gib <= 0:
        raise ValueError("require floor <= ceil and a finite positive budget")
    chunk = 1 << (ceil.bit_length() - 1)
    while chunk > floor and softmax_bytes(heads, chunk, keys) > budget_gib * 2 ** 30:
        chunk //= 2
    return max(floor, chunk)


def _slice_mask(mask, i: int, j: int, q_total: int):
    """Slice the query axis of an explicit four-dimensional additive mask."""
    if not torch.is_tensor(mask) or mask.ndim != 4:
        raise ValueError("chunked attention requires a four-dimensional additive mask")
    if mask.shape[-2] == 1:
        return mask
    if mask.shape[-2] != q_total:
        raise ValueError("attention mask query length does not match query")
    return mask[..., i:j, :]


def _make(orig, chunk: int):
    """Wrap inference attention with explicit masks and no attention-weight output."""
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError("chunk must be a positive integer")

    def chunked(module, query, key, value, attention_mask, scaling,
                dropout=0.0, **kwargs):
        if any(not torch.is_tensor(t) or t.ndim != 4 for t in (query, key, value)):
            raise ValueError("query, key and value must have shape [B,H,T,D]")
        batch, heads, q_total, dim = query.shape
        if any(size < 1 for t in (query, key, value) for size in t.shape):
            raise ValueError("attention dimensions must be nonzero")
        if (key.shape[:2] != value.shape[:2] or key.shape[-2] != value.shape[-2]
                or key.shape[0] != batch or key.shape[-1] != dim
                or heads % key.shape[1] != 0):
            raise ValueError("incompatible query, key and value shapes")
        if any(t.dtype != query.dtype or t.device != query.device for t in (key, value)):
            raise ValueError("query, key and value must share dtype and device")
        if q_total <= chunk:
            return orig(module, query, key, value, attention_mask, scaling,
                        dropout, **kwargs)
        if dropout != 0.0:
            raise ValueError("chunked attention requires dropout=0")
        if kwargs.get("output_attentions", False):
            raise ValueError("chunked attention does not return attention weights")
        if kwargs.get("is_causal", False):
            raise ValueError("provide an explicit causal mask instead of is_causal=True")
        _slice_mask(attention_mask, 0, 1, q_total)
        if (not attention_mask.is_floating_point()
                or attention_mask.device != query.device
                or attention_mask.shape[0] not in (1, batch)
                or attention_mask.shape[1] not in (1, heads)
                or attention_mask.shape[-1] < key.shape[-2]):
            raise ValueError("attention mask must be additive and broadcastable to [B,H,Q,K]")
        probe_n = next((n for n in range(1, q_total)
                        if n not in (heads, value.shape[-1])), None)
        if probe_n is None:
            raise ValueError("cannot identify query axis for these dimensions")
        probe = orig(module, query[:, :, :probe_n], key, value,
                     _slice_mask(attention_mask, 0, probe_n, q_total), scaling,
                     dropout, **kwargs)
        as_tuple = isinstance(probe, tuple)
        if as_tuple and len(probe) != 2:
            raise ValueError("attention kernel must return a tensor or a pair")
        probe = probe[0] if as_tuple else probe
        shapes = {1: (batch, probe_n, heads, value.shape[-1]),
                  2: (batch, heads, probe_n, value.shape[-1])}
        axes = [axis for axis, shape in shapes.items()
                if torch.is_tensor(probe) and tuple(probe.shape) == shape]
        if len(axes) != 1:
            raise ValueError("unsupported attention output shape")
        axis = axes[0]
        parts = []
        for i in range(0, q_total, chunk):
            j = min(i + chunk, q_total)
            out = orig(module, query[:, :, i:j], key, value,
                       _slice_mask(attention_mask, i, j, q_total), scaling,
                       dropout, **kwargs)
            if isinstance(out, tuple) != as_tuple or (as_tuple and len(out) != 2):
                raise ValueError("attention kernel changed its return structure")
            out = out[0] if as_tuple else out
            shape = list(shapes[axis])
            shape[axis] = j - i
            if not torch.is_tensor(out) or tuple(out.shape) != tuple(shape):
                raise ValueError("attention kernel returned an incompatible chunk shape")
            parts.append(out)
        result = torch.cat(parts, dim=axis)
        return (result, None) if as_tuple else result

    chunked._chunked_wrapper = True
    chunked._orig = orig
    return chunked


def _install(mod, tag: str, chunk: int, verbose: bool) -> bool:
    fn = getattr(mod, "eager_attention_forward", None)
    if fn is None:
        raise RuntimeError("%s has no eager_attention_forward to patch" % tag)
    if getattr(fn, "_chunked_wrapper", False):
        return False
    setattr(mod, "eager_attention_forward", _make(fn, chunk))
    _installed[id(mod)] = tag
    if verbose:
        print("[attn] %s: eager attention chunked at %d query rows "
              "(was holding [B,H,Q,K] whole)" % (tag, chunk))
    return True


def install_module(mod, chunk: int = DEFAULT_CHUNK,
                   verbose: bool = True) -> bool:
    """Install query chunking on a modeling module."""
    return _install(mod, getattr(mod, "__name__", "vendored"), chunk, verbose)


def install_transformers(model_type: str, chunk: int = DEFAULT_CHUNK,
                         verbose: bool = True) -> bool:
    """Install query chunking by ``config.model_type``."""
    import importlib
    mod = importlib.import_module(
        "transformers.models.%s.modeling_%s" % (model_type, model_type))
    return _install(mod, model_type, chunk, verbose)


def uninstall(mod) -> bool:
    """Restore the original kernel."""
    fn = getattr(mod, "eager_attention_forward", None)
    if fn is None or not getattr(fn, "_chunked_wrapper", False):
        return False
    setattr(mod, "eager_attention_forward", fn._orig)
    _installed.pop(id(mod), None)
    return True


@contextmanager
def chunked(mod, chunk: int = DEFAULT_CHUNK, verbose: bool = False):
    """Chunk ``mod``'s attention for the duration of the block."""
    installed = install_module(mod, chunk, verbose)
    try:
        yield
    finally:
        if installed:
            uninstall(mod)


def selftest(mod, heads: int = 4, dim: int = 64, q: int = 512,
             chunk: int = 64, device: str = "cpu",
             dtype: torch.dtype = torch.float32) -> bool:
    """Check numerical agreement with the unchunked attention kernel."""
    orig = getattr(mod, "eager_attention_forward")
    orig = getattr(orig, "_orig", orig)

    class Mod(torch.nn.Module):
        num_key_value_groups = 1

    m = Mod()
    g = torch.Generator(device="cpu").manual_seed(0)
    mk = lambda: (torch.randn(1, heads, q, dim, generator=g) * 0.1).to(
        device=device, dtype=dtype)
    query, key, value = mk(), mk(), mk()
    neg = torch.finfo(dtype).min
    mask = torch.full((q, q), neg, device=device, dtype=dtype).triu(1)
    mask = mask.view(1, 1, q, q)
    scaling = dim ** -0.5

    ref = orig(m, query, key, value, mask, scaling, 0.0)
    ref = ref[0] if isinstance(ref, tuple) else ref
    got = _make(orig, chunk)(m, query, key, value, mask, scaling, 0.0)
    got = got[0] if isinstance(got, tuple) else got

    atol, rtol = ((1e-6, 1e-5) if dtype in (torch.float32, torch.float64)
                  else (1e-3, 1e-2))
    same = torch.allclose(ref, got, atol=atol, rtol=rtol)
    print("[attn] selftest q=%d chunk=%d: close=%s  max|diff|=%.3e"
          % (q, chunk, same, (ref - got).abs().max().item()))
    return same
