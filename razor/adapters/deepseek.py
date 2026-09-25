"""DeepSeek routing and latent Kimi expert adapters."""
from __future__ import annotations

import torch
from .base import MoEAdapter, MoEContext, RouterSpec, register

SCORING_FUNCS = {
    "sigmoid": "sigmoid", "softmax": "softmax", "sqrt_softplus": "sqrt_softplus",
    "sqrtsoftplus": "sqrt_softplus", "sqrt-softplus": "sqrt_softplus",
}


@register("deepseek")
class DeepSeekMoEAdapter(MoEAdapter):
    """V4 sqrt-softplus/hash routing and V2/V3 compatibility layouts."""
    MODEL_TYPES = ("deepseek_v2", "deepseek_v3", "deepseek_v4")
    BLOCK_CLASSES = ("DeepseekV2Moe", "DeepseekV3MoE", "DeepseekV4SparseMoeBlock",
                     "DeepseekV4MixedSparseMoeBlock")
    CONFIG_NUM_EXPERTS = ("n_routed_experts", "num_experts", "num_local_experts")
    CONFIG_TOP_K = ("num_experts_per_tok", "top_k")
    MIN_EXPERTS_PER_GROUP = 2

    def __init__(self, config):
        self.SUPPORTS_HASH = getattr(config, "model_type", None) == "deepseek_v4"
        super().__init__(config)
        if self.SUPPORTS_HASH:
            self.MIN_EXPERTS_PER_GROUP = 1
        if getattr(self.text_config, "model_type", None) == "deepseek_v2":
            self.MIN_EXPERTS_PER_GROUP = 1
            if self._cfg(("topk_method",), "greedy") != "greedy":
                raise NotImplementedError("DeepSeek-V2 supports only greedy routing")

    def router_spec(self, block) -> RouterSpec:
        if self.SUPPORTS_HASH:
            gate = self.gate_module(block)
            return RouterSpec(
                kind="sqrt_softplus", top_k=int(gate.top_k),
                num_experts=int(self._cfg(("hash_n_routed_experts",), self.num_experts))
                if self.is_hash(block) else self.num_experts,
                normalize=True, scaling=float(self._cfg(("routed_scaling_factor",), 1.0)),
                correction_bias=self.correction_bias(block), router_dtype=gate.weight.dtype,
                probabilities_dtype=gate.weight.dtype, score_fn=gate.score_fn,
            )
        if getattr(self.text_config, "model_type", None) == "deepseek_v2":
            return RouterSpec(
                kind="softmax", top_k=self.top_k, num_experts=self.num_experts,
                normalize=False, scaling=float(self._cfg(("routed_scaling_factor",), 1.0)),
                gate_bias=getattr(self.gate_module(block), "bias", None),
            )
        raw = str(self._cfg(("scoring_func",), "sigmoid"))
        key = raw.strip().lower().replace(" ", "")
        if key not in SCORING_FUNCS:
            raise NotImplementedError(f"unsupported scoring_func={raw!r}")
        method = self._cfg(("topk_method",), "noaux_tc")
        if method != "noaux_tc":
            raise NotImplementedError(f"unsupported grouped routing method {method!r}")
        gate = self.gate_module(block)
        return RouterSpec(
            kind=SCORING_FUNCS[key], top_k=self.top_k, num_experts=self.num_experts,
            normalize=bool(self._cfg(("norm_topk_prob",), True)),
            scaling=float(self._cfg(("routed_scaling_factor",), 1.0)),
            correction_bias=self.correction_bias(block),
            gate_bias=getattr(gate, "bias", None),
            n_group=int(self._cfg(("n_group",), 1)),
            topk_group=int(self._cfg(("topk_group",), 1)), group_score_topk=2,
        )

    def context(self, block, args, output=None, kwargs=None):
        if not self.is_hash(block):
            return super().context(block, args, output, kwargs)
        kwargs = kwargs or {}
        hidden = args[0] if args else kwargs.get("hidden_states")
        ids = args[1] if len(args) > 1 else kwargs.get("input_ids")
        if ids is None:
            raise ValueError("hash routing requires the native input_ids")
        flat = hidden.reshape(-1, hidden.shape[-1])
        _, weights, selected = block.gate(hidden, ids)
        spec = self.router_spec(block)
        return MoEContext(flat, flat, selected,
                          self.dense_weights(selected, weights, spec.num_experts), spec)

    def validate_keep(self, block, keep, top_k=None, *, target_top_k=None):
        super().validate_keep(block, keep, top_k, target_top_k=target_top_k)
        if self.is_hash(block):
            E = self.gate_module(block).weight.shape[0]
            if not torch.equal(keep.cpu(), torch.arange(E)):
                raise NotImplementedError("hash layers require preserving every expert in original order")
            k = target_top_k if target_top_k is not None else top_k
            if k is not None and k != self.router_spec(block).top_k:
                raise NotImplementedError("hash layers require preserving their native top_k")

    @staticmethod
    def _remap_hash_table(table, gate_weight, keep):
        """Build a source-only, collision-free native reference for an explicit remap."""
        if (table.ndim != 2 or table.dtype not in (torch.int32, torch.int64)
                or table.is_meta or (table < 0).any() or (table >= gate_weight.shape[0]).any()):
            raise ValueError("invalid native hash table")
        if keep.numel() < table.shape[1]:
            raise ValueError("keep-set cannot fill the native hash table")
        weight = gate_weight.detach().float().cpu()
        if not torch.isfinite(weight).all():
            raise ValueError("hash remapping requires finite router weights")
        unit = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-12)
        similarity = (unit @ unit.index_select(0, keep.cpu()).t()).tolist()
        inverse = {old: new for new, old in enumerate(keep.tolist())}
        rows = []
        for original in table.cpu().tolist():
            mapped, used = [-1] * len(original), set()
            for slot, old in enumerate(original):
                new = inverse.get(old)
                if new is not None and new not in used:
                    mapped[slot] = new
                    used.add(new)
            for slot, old in enumerate(original):
                if mapped[slot] >= 0:
                    continue
                available = (candidate for candidate in range(keep.numel()) if candidate not in used)
                new = max(available, key=lambda candidate: (similarity[old][candidate], -candidate))
                mapped[slot] = new
                used.add(new)
            rows.append(mapped)
        return torch.tensor(rows, dtype=table.dtype, device=table.device).reshape_as(table)

    def prune_block(self, block, keep, top_k=None, *, target_top_k=None, hash_policy="preserve"):
        if hash_policy not in ("preserve", "remap"):
            raise ValueError("hash_policy must be 'preserve' or 'remap'")
        if self.is_hash(block):
            if hash_policy == "preserve":
                self.validate_keep(block, keep, top_k, target_top_k=target_top_k)
                return
            MoEAdapter.validate_keep(self, block, keep, top_k, target_top_k=target_top_k)
            k = target_top_k if target_top_k is not None else top_k
            if k is not None and k != self.router_spec(block).top_k:
                raise NotImplementedError("hash remapping must preserve the native table width/top_k")
            gate = self.gate_module(block)
            if gate.tid2eid.ndim != 2 or gate.tid2eid.shape[1] != self.router_spec(block).top_k:
                raise ValueError("native hash table width disagrees with router top_k")
            remapped = self._remap_hash_table(gate.tid2eid, gate.weight, keep)
            self._slice_block(block, keep, k)
            gate.tid2eid = remapped
            return
        super().prune_block(block, keep, top_k, target_top_k=target_top_k)


@register("kimi")
class KimiMoEAdapter(MoEAdapter):
    """Native expert modules, with statistics evaluated before latent up-projection."""
    MODEL_TYPES = ("kimi_linear", "kimi_k3")
    BLOCK_CLASSES = ("KimiSparseMoeBlock",)
    CONFIG_TOP_K = ("num_experts_per_token", "num_experts_per_tok", "top_k")

    def router_spec(self, block):
        gate = self.gate_module(block)
        kind = str(getattr(gate, "moe_router_activation_func",
                           self._cfg(("moe_router_activation_func",), "sigmoid")))
        if kind not in ("sigmoid", "softmax"):
            raise NotImplementedError(f"unsupported Kimi router activation {kind!r}")
        return RouterSpec(
            kind=kind, top_k=self.top_k, num_experts=self.num_experts,
            normalize=self.top_k > 1 and bool(self._cfg(("moe_renormalize",), True)),
            scaling=float(self._cfg(("routed_scaling_factor",), 1.0)),
            correction_bias=self.correction_bias(block),
            n_group=int(self._cfg(("num_expert_group", "n_group"), 1)),
            topk_group=int(self._cfg(("topk_group",), 1)),
        )

    def context(self, block, args, output=None, kwargs=None):
        context = super().context(block, args, output, kwargs)
        if getattr(block, "use_latent_moe", False):
            context.hidden = block.routed_expert_down_proj(context.router_input)
        return context

    def native_routed_capture(self, block):
        if getattr(block, "use_latent_moe", False):
            target = (block.routed_expert_norm if getattr(block, "latent_moe_use_norm", False)
                      else block.routed_expert_up_proj)
            return target, True
        return super().native_routed_capture(block)

    def routed_to_output(self, block, routed):
        if getattr(block, "use_latent_moe", False):
            routed = routed.to(block.routed_expert_up_proj.weight.dtype)
            if getattr(block, "latent_moe_use_norm", False):
                routed = block.routed_expert_norm(routed)
            return block.routed_expert_up_proj(routed).float()
        return routed

    def validate_blocks(self, blocks):
        super().validate_blocks(blocks)
        for item in blocks:
            if getattr(item.module, "ep_size", 1) != 1:
                raise NotImplementedError("sharded Kimi expert parallelism must be gathered before scoring")

    def prune_block(self, block, keep, top_k=None, *, target_top_k=None):
        super().prune_block(block, keep, top_k, target_top_k=target_top_k)
        if hasattr(block, "experts_per_rank"):
            block.experts_per_rank = int(keep.numel())
