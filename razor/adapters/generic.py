"""Explicit opt-in adapter for externally validated canonical MoE layouts."""
from __future__ import annotations

from .base import MoEAdapter, RouterSpec, register
from .deepseek import SCORING_FUNCS


@register("generic")
class GenericMoEAdapter(MoEAdapter):
    """Never auto-select an unknown architecture."""
    @classmethod
    def matches(cls, config) -> bool:
        return False

    def router_spec(self, block) -> RouterSpec:
        gate = self.gate_module(block)
        bias = self.correction_bias(block)
        raw = str(self._cfg(("scoring_func",), "softmax"))
        key = raw.strip().lower().replace(" ", "")
        if key not in SCORING_FUNCS:
            raise NotImplementedError(f"unsupported scoring_func={raw!r}")
        return RouterSpec(
            kind=SCORING_FUNCS[key], top_k=self.top_k, num_experts=self.num_experts,
            normalize=bool(self._cfg(("norm_topk_prob",), True)),
            scaling=float(self._cfg(("routed_scaling_factor",), 1.0)),
            correction_bias=bias, gate_bias=getattr(gate, "bias", None),
            n_group=int(self._cfg(("n_group",), 1)),
            topk_group=int(self._cfg(("topk_group",), 1)),
        )
