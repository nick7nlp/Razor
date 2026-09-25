"""Hunyuan MoE routing adapter."""
from __future__ import annotations

from .base import MoEAdapter, RouterSpec, register


@register("hunyuan")
class HunyuanMoEAdapter(MoEAdapter):
    """Hy3 sigmoid routing with block-owned selection bias and FP32 gates."""
    MODEL_TYPES = ("hy_v3",)
    BLOCK_CLASSES = ("HYV3MoE",)
    CONFIG_NUM_EXPERTS = ("num_experts", "n_routed_experts", "num_local_experts")
    CONFIG_TOP_K = ("num_experts_per_tok", "moe_topk", "top_k")

    @property
    def top_k(self) -> int:
        value = self._cfg(self.CONFIG_TOP_K)
        if isinstance(value, (list, tuple)):
            if not value or len(set(value)) != 1:
                raise NotImplementedError("nonuniform per-layer top_k is unsupported")
            value = value[0]
        k = int(value)
        if not 1 <= k <= self.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        return k

    def router_spec(self, block) -> RouterSpec:
        gate = self.gate_module(block)
        return RouterSpec(
            kind="sigmoid", top_k=self.top_k, num_experts=self.num_experts,
            normalize=True,
            scaling=float(self._cfg(("router_scaling_factor",), 1.0)),
            correction_bias=self.correction_bias(block),
            gate_bias=getattr(gate, "bias", None),
        )
