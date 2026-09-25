"""Scoped, native-model attention/indexer memory bounds for scoring.

The query-chunked indexers follow the released model implementations. Short
native probes verify index decisions before a long sequence uses a replacement.
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import sys
import threading
import types
from contextlib import contextmanager

import torch
import torch.nn.functional as F

_PATCH_LOCK = threading.Lock()


def _glm52(module, chunk):
    def forward(self, hidden_states, q_resid, position_embeddings, attention_mask, use_cache=False):
        batch, length, _ = hidden_states.shape
        cos, sin = position_embeddings
        q = self.wq_b(q_resid).view(batch, length, self.n_heads, self.head_dim)
        q_pe, q_nope = q.split([self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], -1)
        q = torch.cat([module.apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2), q_nope], -1)
        k = self.k_norm(self.wk(hidden_states))
        k_pe, k_nope = k.split([self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], -1)
        k_pe = module.apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
        k = torch.cat([k_pe, k_nope], -1)
        if length > 1:
            self._cached_keys = None
        if use_cache:
            k = torch.cat([self._cached_keys, k], 1) if self._cached_keys is not None else k
            self._cached_keys = k
        weights = self.weights_proj(hidden_states).float() * self.n_heads ** -.5
        top_k = min(self.index_topk, k.shape[1])
        result = torch.empty((batch, length, top_k), dtype=torch.long, device=hidden_states.device)
        keys = k.float()
        for lo in range(0, length, chunk):
            hi = min(lo + chunk, length)
            scores = F.relu(torch.einsum("bshd,btd->bsht", q[:, lo:hi].float(), keys) * self.softmax_scale)
            scores = torch.einsum("bsht,bsh->bst", scores, weights[:, lo:hi])
            if attention_mask is not None:
                mask = attention_mask if attention_mask.shape[-2] == 1 else attention_mask[:, lo:hi]
                scores = scores + mask
            result[:, lo:hi] = scores.topk(top_k, -1).indices
        return result
    return forward


def _glm52_indexed(module, chunk):
    def forward(self, hidden_states, q_resid, position_embeddings, attention_mask,
                position_ids, past_key_values=None):
        batch, length, _ = hidden_states.shape
        cos, sin = position_embeddings
        q = self.wq_b(q_resid).view(batch, length, self.n_heads, self.head_dim)
        k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)
        widths = [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim]
        q_rot, q_pass = q.split(widths, -1)
        k_rot, k_pass = k.split(widths, -1)
        q_rot, k_rot = module.apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_rot, q_pass], -1)
        k = torch.cat([k_rot, k_pass], -1).squeeze(2)
        if past_key_values is not None:
            k = past_key_values.update_indexer(k, self.layer_idx)
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * self.n_heads ** -.5
        keys = k.transpose(-1, -2).float().unsqueeze(1)
        positions = torch.arange(k.shape[1], device=hidden_states.device)
        top_k = min(self.index_topk, k.shape[1])
        result = torch.empty((batch, length, top_k), dtype=torch.int32, device=hidden_states.device)
        for lo in range(0, length, chunk):
            hi = min(lo + chunk, length)
            scores = F.relu(torch.matmul(q[:, lo:hi].float(), keys) * self.softmax_scale)
            scores = torch.matmul(weights[:, lo:hi].unsqueeze(-2), scores).squeeze(-2)
            if attention_mask is not None:
                mask = attention_mask if attention_mask.shape[-2] == 1 else attention_mask[:, lo:hi]
                scores = scores + mask
            else:
                scores = scores.masked_fill(positions[None, None] > position_ids[:, lo:hi, None], float("-inf"))
            result[:, lo:hi] = scores.topk(top_k, -1).indices.to(torch.int32)
        return result
    return forward


def _glm53(module, chunk):
    def forward(self, hidden_states, q_resid, attention_mask, past_key_values):
        batch, length = hidden_states.shape[:2]
        shape = (batch, length, -1, self.head_dim)
        q = self.wq_b(q_resid).view(shape)
        keys = self.k_norm(self.wk(hidden_states)).view(shape).squeeze(2)
        gates = F.linear(hidden_states, self.index_kpool_compress_gate)
        packed = torch.cat([keys, gates, attention_mask.to(keys.dtype)[..., None]], -1)
        kv_len = current_length = length
        if past_key_values is not None:
            cache_layer = past_key_values.layers[self.layer_idx]
            packed = past_key_values.update_indexer(packed, self.layer_idx)
            kv_len = cache_layer.keys.shape[-2]
            current_length = cache_layer.get_seq_length()
        valid_keys = packed[..., -1].bool()
        visible = self.get_visible_tokens(valid_keys=valid_keys, q_length=length, current_length=current_length)
        pool_keys, pool_indices, pool_valid = self.get_pooled_states(packed_states=packed)
        key_t = pool_keys.transpose(-1, -2).float().unsqueeze(1)
        pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * self.n_heads ** -.5
        select_k = min(self.index_topk // self.index_kpool, pool_keys.shape[-2])
        width = self.index_topk + (self.index_kpool - 1 if self.index_kpool_always_select_tail else 0)
        result = torch.empty((batch, length, width), dtype=torch.int32, device=hidden_states.device)
        batch_index = torch.arange(batch, device=hidden_states.device)[:, None, None]
        for lo in range(0, length, chunk):
            hi = min(lo + chunk, length)
            scores = F.relu(torch.matmul(q[:, lo:hi].float(), key_t) * self.softmax_scale)
            scores = torch.matmul(weights[:, lo:hi].unsqueeze(-2), scores).squeeze(-2)
            candidates = visible[:, lo:hi].gather(-1, pool_end[:, None].expand(batch, hi - lo, -1)) & pool_valid[:, None]
            selected = scores.masked_fill(~candidates, torch.finfo(scores.dtype).min).topk(select_k, -1).indices
            selected_valid = candidates.gather(-1, selected)
            selected_indices = pool_indices[batch_index, selected]
            indices = selected_indices.flatten(-2).masked_fill(
                ~selected_valid[..., None].expand_as(selected_indices).flatten(-2), -1)
            if self.index_kpool_always_select_tail:
                indices = self.append_visible_tail(indices, visible[:, lo:hi], valid_keys)
            indices = F.pad(indices, (0, width - indices.shape[-1]), value=-1)[..., :width]
            result[:, lo:hi] = indices.masked_fill(~attention_mask[:, lo:hi, None], -1).to(torch.int32)
        return result
    return forward


def _dsv4(module, chunk):
    def forward(self, hidden_states, q_residual, position_ids, past_key_values, layer_idx):
        batch, length, _ = hidden_states.shape
        cache = past_key_values.layers[layer_idx] if past_key_values is not None else None
        kv, gate = self.kv_proj(hidden_states), self.gate_proj(hidden_states)
        if cache is None:
            usable = kv.shape[1] // self.compress_rate * self.compress_rate
            chunk_kv, chunk_gate, first_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_position = cache.store_compression_weights("indexer", kv, gate)
        if chunk_kv.shape[1]:
            windows, ratio = chunk_kv.shape[1] // self.compress_rate, self.compress_rate
            chunk_kv = chunk_kv.view(batch, windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)
            new_kv = chunk_kv.new_zeros((batch, windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full((batch, windows, 2 * ratio, self.head_dim), float("-inf"))
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim:]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim:]
            if windows > 1:
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, :self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, :self.head_dim]
            if cache is not None:
                prior_kv, prior_gate = cache.update_overlap_state("indexer", chunk_kv, chunk_gate, self.head_dim)
                if prior_kv is not None:
                    new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                    new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)
            compressed = self.kv_norm((new_kv * new_gate.softmax(2, dtype=torch.float32).to(new_kv.dtype)).sum(2))
            positions = (torch.arange(windows, device=compressed.device) * ratio + first_position)[None].expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = module.apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))
        compressed = compressed if cache is None else cache.update_compressor_states("indexer", compressed)
        cos, sin = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type)
        q = self.q_b_proj(q_residual).view(batch, length, -1, self.head_dim).transpose(1, 2)
        q = module.apply_rotary_pos_emb(q, cos, sin).transpose(1, 2)
        scorer = self.scorer if "scorer" in self._modules else self
        if scorer is not self and type(scorer).__name__ != "DeepseekV4IndexerScorer":
            raise RuntimeError("unsupported DeepSeek-V4 indexer scorer protocol")
        weights = scorer.weights_proj(hidden_states).float() * scorer.weights_scaling
        total, top_k = compressed.shape[1], min(self.index_topk, compressed.shape[1])
        key_t = compressed.transpose(-1, -2).float().unsqueeze(1)
        entries = torch.arange(total, device=kv.device)
        result = []
        for lo in range(0, length, chunk):
            hi = min(lo + chunk, length)
            scores = F.relu(torch.matmul(q[:, lo:hi].float(), key_t)) * scorer.softmax_scale
            scores = (scores * weights[:, lo:hi, :, None]).sum(2)
            if total:
                threshold = (position_ids[:, lo:hi] + 1) // self.compress_rate
                scores = scores.masked_fill(entries[None, None] >= threshold[..., None], float("-inf"))
                picks = scores.topk(top_k, -1).indices
                picks = torch.where(picks >= threshold[..., None], torch.full_like(picks, -1), picks)
            else:
                picks = scores.topk(top_k, -1).indices
            result.append(picks)
        return torch.cat(result, 1)
    return forward


def _probe_indexer(original, factory, module, instance, arguments):
    signature = inspect.signature(original)
    bound = signature.bind(*arguments[0], **arguments[1])
    bound.apply_defaults()
    values = dict(bound.arguments)
    hidden = values["hidden_states"]
    length = min(hidden.shape[1], 64)
    if length < 2:
        return False
    for name in ("hidden_states", "q_resid", "q_residual", "position_ids"):
        if name in values and values[name] is not None:
            values[name] = values[name][:, :length]
    if values.get("position_embeddings") is not None:
        values["position_embeddings"] = tuple(value[:, :length] for value in values["position_embeddings"])
    if values.get("attention_mask") is not None:
        mask = values["attention_mask"]
        values["attention_mask"] = mask[:, :length, :length] if mask.ndim == 3 else mask[:, :length]
    if "past_key_values" in values:
        values["past_key_values"] = None
    if "use_cache" in values:
        values["use_cache"] = False
    cached = getattr(instance, "_cached_keys", None)
    try:
        expected = original(**values)
        actual = factory(module, max(1, length // 2))(instance, **values)
        if not torch.equal(actual, expected):
            raise RuntimeError(f"{type(instance).__name__}: chunked native index decisions disagree")
    finally:
        if hasattr(instance, "_cached_keys"):
            instance._cached_keys = cached
    return True


@contextmanager
def native_chunked(model, chunk=128):
    """Temporarily bound eager/sink attention, DSA indexers and Kimi residuals."""
    from .chunked_attn import _make
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        raise ValueError("attention chunk must be a positive integer")
    changes = []
    modules = {sys.modules[type(item).__module__] for item in model.modules()
               if type(item).__module__ in sys.modules and type(item).__module__ != "torch.nn.modules.module"}
    factories = {"GlmMoeDsaIndexer": _glm52, "Glm5NextTextIndexer": _glm53,
                 "DeepseekV4Indexer": _dsv4}
    if not _PATCH_LOCK.acquire(blocking=False):
        raise RuntimeError("a native attention patch scope is already active; use separate processes")
    try:
        for module in modules:
            original = getattr(module, "eager_attention_forward", None)
            if callable(original) and not getattr(original, "_chunked_wrapper", False):
                changes.append((module, "eager_attention_forward", original, True))
                module.eager_attention_forward = _make(original, chunk)
            residual = getattr(module, "_apply_attn_res", None)
            if callable(residual):
                digest = hashlib.sha1(inspect.getsource(residual).encode()).hexdigest()[:16]
                if digest != "83b2af1167f64ef0":
                    raise RuntimeError("Kimi attention residual implementation changed; chunking requires revalidation")
                @functools.wraps(residual)
                def residual_chunked(prefix_sum, block_residual, proj, norm, original=residual):
                    return torch.cat([original(prefix_sum[lo:lo + chunk], block_residual[lo:lo + chunk], proj, norm)
                                      for lo in range(0, prefix_sum.shape[0], chunk)], 0)
                changes.append((module, "_apply_attn_res", residual, True))
                module._apply_attn_res = residual_chunked
        for instance in model.modules():
            factory = factories.get(type(instance).__name__)
            if factory is None:
                continue
            original = instance.forward
            if type(instance).__name__ == "GlmMoeDsaIndexer":
                parameters = tuple(inspect.signature(original).parameters)
                legacy = ("hidden_states", "q_resid", "position_embeddings", "attention_mask", "use_cache")
                indexed = ("hidden_states", "q_resid", "position_embeddings", "attention_mask", "position_ids", "past_key_values")
                if parameters == indexed:
                    factory = _glm52_indexed
                elif parameters != legacy:
                    raise RuntimeError(f"unsupported GLM DSA indexer signature: {parameters}")
            module = sys.modules[type(instance).__module__]
            implementation = factory(module, chunk)
            checked = [False]
            @functools.wraps(original)
            def wrapped(self, *args, original=original, factory=factory, module=module,
                        implementation=implementation, checked=checked, **kwargs):
                if not checked[0]:
                    try:
                        checked[0] = _probe_indexer(original, factory, module, self, (args, kwargs))
                    except (AttributeError, TypeError) as error:
                        raise RuntimeError(f"{type(self).__name__}: incompatible native indexer protocol; "
                                           "refusing chunked execution") from error
                    if not checked[0]:
                        return original(*args, **kwargs)
                return implementation(self, *args, **kwargs)
            changes.append((instance, "forward", original, "forward" in instance.__dict__))
            instance.forward = types.MethodType(wrapped, instance)
        yield
    finally:
        try:
            for owner, name, original, existed in reversed(changes):
                if existed:
                    setattr(owner, name, original)
                else:
                    delattr(owner, name)
        finally:
            _PATCH_LOCK.release()
