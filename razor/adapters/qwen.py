"""Qwen text/wrapper MoE adapters and Gemma decoder-level experts."""
from __future__ import annotations

import torch
from .base import MoEAdapter, MoEContext, RouterSpec, register


@register("qwen")
class QwenMoEAdapter(MoEAdapter):
    """Softmax routing with optional selected-weight normalization."""
    MODEL_TYPES = ("qwen2_moe", "qwen3_moe", "qwen3_next", "qwen3_5_moe",
                   "qwen3_5_moe_text", "qwen3_6_moe", "qwen3_6_moe_text")
    BLOCK_CLASSES = ("Qwen2MoeSparseMoeBlock", "Qwen3MoeSparseMoeBlock",
                     "Qwen3NextSparseMoeBlock", "Qwen3_5MoeSparseMoeBlock")
    CONFIG_NUM_EXPERTS = ("num_experts", "num_local_experts", "n_routed_experts")
    CONFIG_TOP_K = ("num_experts_per_tok", "top_k")

    def router_spec(self, block) -> RouterSpec:
        gate = self.gate_module(block)
        return RouterSpec(
            kind="softmax", top_k=self.top_k, num_experts=self.num_experts,
            normalize=bool(self._cfg(("norm_topk_prob",), True)),
            gate_bias=getattr(gate, "bias", None),
            router_dtype=gate.weight.dtype, weights_dtype=gate.weight.dtype,
        )


@register("gemma4")
class Gemma4MoEAdapter(MoEAdapter):
    """Sibling router/experts with separate expert pre-norm and per-expert gain."""
    MODEL_TYPES = ("gemma4", "gemma4_text")
    BLOCK_CLASSES = ("Gemma4TextDecoderLayer",)
    CONFIG_TOP_K = ("top_k_experts",)

    def is_moe_block(self, module):
        return (bool(getattr(module, "enable_moe_block", False))
                and hasattr(module, "experts") and hasattr(module, "router"))

    def gate_module(self, block):
        return block.router.proj

    def hook_module(self, block):
        return block.experts

    def router_spec(self, block):
        return RouterSpec(num_experts=self.num_experts, top_k=self.top_k, normalize=True)

    def context(self, block, args, output=None, kwargs=None):
        hidden, selected, weights = args[:3]
        hidden = hidden.reshape(-1, hidden.shape[-1])
        gain = block.router.per_expert_scale[selected]
        if not torch.isfinite(gain).all() or (gain <= 0).any():
            raise ValueError("Gemma per-expert gains must be finite and positive")
        probabilities = (weights / gain).float()
        spec = self.router_spec(block)
        # The decoder-level router hands down only its winners, so the losing
        # scores needed for refill are recovered by a verified replay.
        promoted, promoted_weight = self.replay_promoted(block, hidden, selected, spec)
        return MoEContext(hidden, hidden, selected,
                          self.dense_weights(selected, probabilities, spec.num_experts), spec,
                          promoted, promoted_weight)

    def expert_forward(self, block, hidden, lo, hi):
        output = super().expert_forward(block, hidden, lo, hi)
        return output * block.router.per_expert_scale[lo:hi].float()[:, None, None]

    def shared_forward(self, block, hidden):
        return torch.zeros_like(hidden, dtype=torch.float32)

    def prune_block(self, block, keep, top_k=None, *, target_top_k=None):
        super().prune_block(block, keep, top_k, target_top_k=target_top_k)
        gain = block.router.per_expert_scale
        block.router.per_expert_scale = torch.nn.Parameter(
            gain.detach().index_select(0, keep.to(gain.device)).clone(), requires_grad=False)
        block.router.config.num_experts = int(keep.numel())
        k = target_top_k if target_top_k is not None else top_k
        if k is not None:
            block.router.config.top_k_experts = k
