"""GLM-4 MoE grouped routing adapter."""
from __future__ import annotations

from .base import MoEAdapter, RouterSpec, register


@register("glm")
class Glm4MoEAdapter(MoEAdapter):
    """Sigmoid routing with selection bias and a shared expert."""
    MODEL_TYPES = ("glm4_moe", "glm4_moe_lite", "glm_moe_dsa", "glm5_next", "glm5_next_text")
    BLOCK_CLASSES = ("Glm4MoeMoE", "Glm4MoeLiteMoE", "GlmMoeDsaMoE", "Glm5NextTextMoE")
    CONFIG_NUM_EXPERTS = ("n_routed_experts", "num_experts", "num_local_experts")
    CONFIG_TOP_K = ("num_experts_per_tok", "top_k")
    MIN_EXPERTS_PER_GROUP = 2

    def router_spec(self, block) -> RouterSpec:
        gate = self.gate_module(block)
        return RouterSpec(
            kind="sigmoid", top_k=self.top_k, num_experts=self.num_experts,
            normalize=bool(self._cfg(("norm_topk_prob",), True)),
            scaling=float(self._cfg(("routed_scaling_factor",), 1.0)),
            correction_bias=self.correction_bias(block), gate_bias=getattr(gate, "bias", None),
            n_group=int(self._cfg(("n_group",), 1)),
            topk_group=int(self._cfg(("topk_group",), 1)), group_score_topk=2,
        )

    def postprocess_config(self, config, info: dict) -> None:
        for name in self.CONFIG_NUM_EXPERTS:
            if getattr(config, name, None) is not None:
                setattr(config, name, info["kept_experts"])
