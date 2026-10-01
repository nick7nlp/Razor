"""MoE adapter interfaces and validated tensor operations."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

_REGISTRY: Dict[str, type] = {}


def register(name: str) -> Callable[[type], type]:
    """Register an adapter by name."""
    def _wrap(cls: type) -> type:
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return _wrap


def list_adapters() -> List[str]:
    return sorted(_REGISTRY)


def get_adapter(config, name: Optional[str] = None) -> "MoEAdapter":
    """Select an adapter; registration alone does not establish model support."""
    if name is not None:
        if name not in _REGISTRY:
            raise KeyError(f"unknown adapter {name!r}; registered: {list_adapters()}")
        return _REGISTRY[name](config)
    for key, cls in _REGISTRY.items():
        if key != "generic" and cls.matches(config):
            return cls(config)
    generic = _REGISTRY.get("generic")
    if generic is not None and generic.matches(config):
        return generic(config)
    raise NotImplementedError(
        f"unsupported model_type={getattr(config, 'model_type', '?')!r}")


@dataclass
class RouterSpec:
    """Projection and selected-weight semantics for an (E, H) router."""
    kind: str = "softmax"
    top_k: int = 8
    num_experts: int = 0
    normalize: bool = True
    scaling: float = 1.0
    correction_bias: Optional[torch.Tensor] = None
    n_group: int = 1
    topk_group: int = 1
    group_score_topk: int = 2
    gate_bias: Optional[torch.Tensor] = None
    router_dtype: Optional[torch.dtype] = torch.float32
    weights_dtype: Optional[torch.dtype] = None
    score_fn: Optional[Callable] = None
    probabilities_dtype: Optional[torch.dtype] = None


@dataclass
class MoEContext:
    """Routing weights and expert-domain inputs for one native invocation.

    ``promoted`` and ``promoted_weight`` describe the refill counterfactual of
    Proposition 1: the highest-ranked unselected expert under the router's own
    selection rule, and its pseudo-weight on the same normalized scale as
    ``weights``. They are ``None`` when an adapter cannot observe the full
    router scores, in which case only the fixed-support criteria are available.
    """
    hidden: torch.Tensor
    router_input: torch.Tensor
    selected: torch.Tensor
    weights: torch.Tensor
    spec: RouterSpec
    promoted: Optional[torch.Tensor] = None
    promoted_weight: Optional[torch.Tensor] = None


@dataclass
class MoEBlock:
    """A hookable MoE block and its decoder-layer identity."""
    tag: str
    index: int
    trunk: str
    module: nn.Module


class MoEAdapter:
    """Validated canonical fused experts or a ModuleList of expert modules."""
    name: str = "base"
    BLOCK_CLASSES: Tuple[str, ...] = ()
    MODEL_TYPES: Tuple[str, ...] = ()
    CONFIG_NUM_EXPERTS = ("num_experts", "n_routed_experts", "num_local_experts")
    CONFIG_TOP_K = ("num_experts_per_tok", "moe_topk", "top_k")
    ROUTER_ATTRS = ("gate", "router", "mlp_gate", "wg")
    BIAS_ATTRS = ("e_score_correction_bias", "expert_bias", "correction_bias")
    MIN_EXPERTS_PER_GROUP = 1
    SUPPORTS_HASH = False

    def __init__(self, config):
        self.config = config
        self.text_config = getattr(config, "text_config", None) or config
        for cfg in (self.config, self.text_config):
            if getattr(cfg, "quantization_config", None) is not None:
                raise NotImplementedError("quantized checkpoints are not supported")
            for name in ("num_hash_layers", "n_hash_layers", "first_k_hash_replace"):
                if getattr(cfg, name, 0) not in (None, 0) and not self.SUPPORTS_HASH:
                    raise NotImplementedError("hash routing is not supported, including skipped layers")

    @classmethod
    def matches(cls, config) -> bool:
        configs = (config, getattr(config, "text_config", None))
        return any(re.fullmatch(pattern, str(getattr(cfg, "model_type", "") or ""))
                   for cfg in configs if cfg is not None for pattern in cls.MODEL_TYPES)

    def _cfg(self, names: Tuple[str, ...], default=None):
        for cfg in (self.text_config, self.config):
            for name in names:
                value = getattr(cfg, name, None)
                if value is not None:
                    return value
        if default is None:
            raise AttributeError(f"config requires one of {names}")
        return default

    @property
    def num_experts(self) -> int:
        n = int(self._cfg(self.CONFIG_NUM_EXPERTS))
        if n < 1:
            raise ValueError("num_experts must be positive")
        return n

    @property
    def top_k(self) -> int:
        k = int(self._cfg(self.CONFIG_TOP_K))
        if not 1 <= k <= self.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        return k

    def set_num_experts(self, n: int) -> None:
        if n < 1:
            raise ValueError("num_experts must be positive")
        for cfg in {id(self.text_config): self.text_config, id(self.config): self.config}.values():
            for name in self.CONFIG_NUM_EXPERTS:
                if getattr(cfg, name, None) is not None:
                    setattr(cfg, name, n)

    def set_top_k(self, k: int) -> None:
        if not 1 <= k <= self.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        for cfg in {id(self.text_config): self.text_config, id(self.config): self.config}.values():
            for name in self.CONFIG_TOP_K:
                if getattr(cfg, name, None) is not None:
                    setattr(cfg, name, k)

    def _trunks(self, model) -> Iterator[Tuple[str, nn.Module]]:
        inner = getattr(model, "model", model)
        pending, holders, seen = [model, inner], [], set()
        while pending:
            holder = pending.pop(0)
            if holder is None or id(holder) in seen:
                continue
            seen.add(id(holder))
            holders.append(holder)
            pending.extend(getattr(holder, name, None) for name in ("model", "language_model", "text_model"))
        main = next((holder for holder in holders if hasattr(holder, "layers")), None)
        if main is None:
            raise AttributeError("cannot locate decoder layers")
        yield "main", main.layers
        for holder in holders:
            for name in ("mtp_layers", "mtp"):
                mtp = getattr(holder, name, None)
                if mtp is not None:
                    layers = getattr(mtp, "layers", mtp)
                    if isinstance(layers, (nn.ModuleList, list, tuple)):
                        yield "mtp", layers
                        return

    def is_moe_block(self, module: nn.Module) -> bool:
        return (type(module).__name__ in self.BLOCK_CLASSES or
                (hasattr(module, "experts") and
                 any(hasattr(module, name) for name in self.ROUTER_ATTRS)))

    def moe_blocks(self, model) -> List[MoEBlock]:
        blocks = []
        for trunk, layers in self._trunks(model):
            for index, layer in enumerate(layers):
                candidates = (layer, *(getattr(layer, name, None)
                                       for name in ("mlp", "ffn", "block_sparse_moe")))
                block = next((candidate for candidate in candidates
                              if candidate is not None and self.is_moe_block(candidate)), None)
                if block is not None:
                    blocks.append(MoEBlock(f"{trunk}_{index}", index, trunk, block))
        self.reject_shared_moe_state(blocks)
        return blocks

    def _sliced_moe_state(self, block: nn.Module) -> List[Tuple[str, nn.Module]]:
        """Return the modules that pruning rewrites in place for one block."""
        found = []
        for kind, getter in (("expert bank", self.expert_bank),
                             ("router projection", self.gate_module)):
            try:
                module = getter(block)
            except (NotImplementedError, AttributeError, KeyError):
                continue
            if isinstance(module, nn.Module):
                found.append((kind, module))
        return found

    def reject_shared_moe_state(self, blocks) -> None:
        """Reject an expert bank or router that several decoder blocks own at once.

        Pruning slices these modules in place, so one object reached through two
        blocks would be sliced twice. The refusal happens before any slice, which
        keeps a rejected model byte-identical. References inside a single block,
        tied embeddings, single-block shared experts and per-block expert lists
        are untouched, because only cross-block identity is inspected.
        """
        owners: Dict[int, Tuple[str, str, nn.Module]] = {}
        for item in blocks:
            for kind, module in self._sliced_moe_state(item.module):
                owner = owners.setdefault(id(module), (kind, item.tag, module))
                if owner[1] != item.tag:
                    raise NotImplementedError(
                        f"MoE blocks {owner[1]} and {item.tag} share one "
                        f"{type(module).__name__} {owner[0]}; cross-block expert or router "
                        "sharing is not supported, and no weights were modified")

    def hook_module(self, block):
        return block

    def is_hash(self, block) -> bool:
        return hasattr(self.gate_module(block), "tid2eid")

    def native_routed_capture(self, block):
        return (self.expert_bank(block), False) if self.is_fused(block) else (None, False)

    def routed_to_output(self, block, routed):
        return routed

    def context(self, block, args, output=None, kwargs=None) -> MoEContext:
        kwargs = kwargs or {}
        hidden = args[0] if args else kwargs.get("hidden_states")
        if not isinstance(hidden, torch.Tensor):
            raise NotImplementedError("MoE inputs must expose the hidden-state tensor")
        flat = hidden.reshape(-1, hidden.shape[-1])
        spec = self.router_spec(block)
        gate_weight = self.gate_module(block).weight
        # route() remains the overridable entry point for the routed set.
        # promoted_route() reads the same router independently, so an override
        # of route() that changes weight semantics would not be reflected in
        # the promoted pseudo-weight; such an override must also override
        # promoted_route(), or return None to disable refill for the layer.
        selected, weights = self.route(flat, gate_weight, spec)
        promoted, promoted_weight = self.promoted_route(flat, gate_weight, spec)
        return MoEContext(flat, flat, selected, weights, spec, promoted, promoted_weight)

    @staticmethod
    def dense_weights(selected, weights, num_experts):
        dense = torch.zeros(num_experts, selected.shape[0], device=weights.device, dtype=torch.float32)
        dense.scatter_add_(0, selected.t(), weights.t().float())
        return dense

    def gate_module(self, block: nn.Module) -> nn.Module:
        """Return the router's (E, H) projection module."""
        for name in self.ROUTER_ATTRS:
            outer = getattr(block, name, None)
            if outer is None:
                continue
            for gate in (outer, *(getattr(outer, key, None) for key in self.ROUTER_ATTRS)):
                if gate is not None and hasattr(gate, "weight"):
                    if not isinstance(gate.weight, torch.Tensor) or gate.weight.ndim != 2:
                        raise NotImplementedError("unsupported router projection layout")
                    return gate
        raise NotImplementedError("no supported router projection found")

    def correction_bias(self, block: nn.Module) -> Optional[torch.Tensor]:
        for holder in (self.gate_module(block), block):
            for name in self.BIAS_ATTRS:
                bias = getattr(holder, name, None)
                if bias is not None:
                    return bias
        return None

    def router_spec(self, block: nn.Module) -> RouterSpec:
        gate = self.gate_module(block)
        return RouterSpec(top_k=self.top_k, num_experts=self.num_experts,
                          normalize=bool(self._cfg(("norm_topk_prob",), True)),
                          gate_bias=getattr(gate, "bias", None))

    @staticmethod
    @torch.no_grad()
    def route(hidden: torch.Tensor, gate_weight: torch.Tensor,
              spec: RouterSpec) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return selected ids (T, k) and scaled sparse weights (E, T)."""
        return MoEAdapter._route(hidden, gate_weight, spec)[:2]

    @staticmethod
    @torch.no_grad()
    def promoted_route(hidden: torch.Tensor, gate_weight: torch.Tensor, spec: RouterSpec):
        """Return the rank-(k+1) expert (T,) and its pseudo-weight (T,).

        The promoted expert comes from the same selection scores that choose
        the routed set, so a selection-only correction bias and group masking
        apply to it exactly as they do to the routed experts. Its pseudo-weight
        is its mixture score divided by the original selected score sum, per
        Assumption 1. Tokens with no admissible promotion get weight zero,
        which reduces the refill score to its fixed-support special case.
        """
        return MoEAdapter._route(hidden, gate_weight, spec)[2:]

    @torch.no_grad()
    def replay_promoted(self, block, router_input, selected, spec):
        """Recover rank-(k+1) for adapters given an already-routed selection.

        Native blocks that hand their routing decision to the hook do not
        expose the losing scores, so the router is replayed here. The replay is
        accepted only when it reproduces the native routed set exactly;
        otherwise refill is reported as unavailable rather than scored from a
        selection the model did not make.
        """
        try:
            gate_weight = self.gate_module(block).weight
            replayed, _, promoted, promoted_weight = self._route(router_input, gate_weight, spec)
        except (ValueError, NotImplementedError, AttributeError, RuntimeError):
            return None, None
        if promoted is None or replayed.shape != selected.shape:
            return None, None
        if not torch.equal(replayed.sort(-1).values, selected.sort(-1).values.to(replayed.device)):
            return None, None
        return promoted, promoted_weight

    @staticmethod
    @torch.no_grad()
    def _route(hidden: torch.Tensor, gate_weight: torch.Tensor, spec: RouterSpec):
        """Shared routing core: selection, weights and the promoted expert."""
        if hidden.ndim != 2 or gate_weight.ndim != 2 or hidden.shape[1] != gate_weight.shape[1]:
            raise ValueError("router input and projection shapes disagree")
        E = gate_weight.shape[0]
        if E != spec.num_experts or not 1 <= spec.top_k <= E:
            raise ValueError("router expert count or top_k is invalid")
        if not math.isfinite(spec.scaling) or spec.scaling <= 0:
            raise ValueError("router scaling must be finite and positive")
        if (spec.n_group < 1 or E % spec.n_group or
                not 1 <= spec.topk_group <= spec.n_group or
                spec.top_k > (E // spec.n_group) * spec.topk_group):
            raise ValueError("invalid router grouping")
        dtype = spec.router_dtype or hidden.dtype
        bias = spec.gate_bias
        if bias is not None:
            if tuple(bias.shape) != (E,):
                raise ValueError("router projection bias shape disagrees")
            bias = bias.to(device=hidden.device, dtype=dtype)
        logits = F.linear(hidden.to(dtype), gate_weight.to(dtype), bias)
        logits = logits.to(spec.probabilities_dtype or torch.float32)
        if not torch.isfinite(logits).all():
            raise ValueError("router logits must be finite")
        if spec.score_fn is not None:
            probs = spec.score_fn(logits)
        elif spec.kind == "softmax":
            probs = F.softmax(logits, dim=-1)
        elif spec.kind == "sigmoid":
            probs = logits.sigmoid()
        elif spec.kind == "sqrt_softplus":
            probs = F.softplus(logits).sqrt()
        else:
            raise NotImplementedError(f"unsupported router kind {spec.kind!r}")
        scores = probs
        if spec.correction_bias is not None:
            if tuple(spec.correction_bias.shape) != (E,):
                raise ValueError("selection bias shape disagrees")
            scores = scores + spec.correction_bias.to(scores)
        if not torch.isfinite(scores).all():
            raise ValueError("router selection scores must be finite")
        if spec.n_group > 1:
            per_group = E // spec.n_group
            if not 1 <= spec.group_score_topk <= per_group:
                raise ValueError("invalid group_score_topk")
            grouped = scores.reshape(-1, spec.n_group, per_group)
            group_scores = grouped.topk(spec.group_score_topk, dim=-1).values.sum(-1)
            groups = group_scores.topk(spec.topk_group, dim=-1, sorted=False).indices
            mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, groups, True)
            scores = scores.masked_fill(~mask.unsqueeze(-1).expand_as(grouped).reshape_as(scores),
                                        float("-inf"))
        selected = scores.topk(spec.top_k, dim=-1, sorted=(spec.kind == "softmax")).indices
        weights = probs.gather(1, selected)
        denom = weights.sum(-1, keepdim=True)
        if spec.normalize:
            if (denom <= 0).any():
                raise ValueError("selected router weights have zero mass")
            weights = weights / denom
        promoted = promoted_weight = None
        if spec.top_k < E:
            extra = scores.topk(spec.top_k + 1, dim=-1, sorted=True)
            promoted = extra.indices[:, -1]
            admissible = torch.isfinite(extra.values[:, -1])
            promoted_weight = probs.gather(1, promoted.unsqueeze(1)).squeeze(1)
            if spec.normalize:
                promoted_weight = promoted_weight / denom.squeeze(1)
            promoted_weight = torch.where(admissible, promoted_weight,
                                          torch.zeros_like(promoted_weight))
            if (promoted_weight < 0).any():
                raise ValueError("promoted router score must be nonnegative")
            promoted_weight = (promoted_weight * spec.scaling).float()
        weights = weights * spec.scaling
        if spec.weights_dtype is not None:
            weights = weights.to(spec.weights_dtype)
        dense = torch.zeros(E, hidden.shape[0], dtype=torch.float32, device=hidden.device)
        dense.scatter_(0, selected.t(), weights.t().float())
        return selected, dense, promoted, promoted_weight

    def expert_bank(self, block: nn.Module) -> nn.Module:
        experts = getattr(block, "experts", None)
        if experts is None:
            raise NotImplementedError("MoE block has no supported expert bank")
        return experts

    def is_fused(self, block: nn.Module) -> bool:
        return hasattr(self.expert_bank(block), "gate_up_proj")

    def validate_model(self, model) -> None:
        """Reject unsupported layouts and inconsistent expert counts."""
        if getattr(model, "is_quantized", False) or getattr(model, "hf_quantizer", None) is not None:
            raise NotImplementedError("quantized models are not supported")
        if any(value.is_meta for value in model.parameters()):
            raise NotImplementedError("meta or disk-offloaded weights are not supported")
        if any(value.is_meta for value in model.buffers()):
            raise NotImplementedError("meta or disk-offloaded buffers are not supported")
        blocks = self.moe_blocks(model)
        if not blocks:
            raise RuntimeError("no supported MoE blocks found")
        self.validate_blocks(blocks)

    def validate_blocks(self, blocks) -> None:
        """Validate resident MoE tensors without inspecting other decoder layers."""
        self.reject_shared_moe_state(blocks)
        allowed_dtypes = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
        for item in blocks:
            block = item.module
            if any(value.is_meta for value in list(block.parameters()) + list(block.buffers())):
                raise NotImplementedError("meta or disk-offloaded MoE weights are not supported")
            gate = self.gate_module(block)
            experts = self.expert_bank(block)
            spec = self.router_spec(block)
            E = spec.num_experts
            if gate.weight.shape[0] != E:
                raise ValueError(f"{item.tag}: router and config expert counts disagree")
            if (not math.isfinite(spec.scaling) or spec.scaling <= 0 or
                    not 1 <= spec.top_k <= E or spec.n_group < 1 or E % spec.n_group or
                    not 1 <= spec.topk_group <= spec.n_group or
                    E // spec.n_group < self.MIN_EXPERTS_PER_GROUP or
                    spec.top_k > (E // spec.n_group) * spec.topk_group):
                raise ValueError(f"{item.tag}: invalid router specification")
            for module in block.modules():
                hook = getattr(module, "_hf_hook", None)
                hooks = getattr(hook, "hooks", (hook,))
                if any(getattr(item, "offload", False) for item in hooks):
                    raise NotImplementedError("offloaded MoE modules are not supported")
            for name, value in block.named_parameters():
                if value.dtype not in allowed_dtypes:
                    raise NotImplementedError("unsupported parameter dtype or quantization")
                if value.device != gate.weight.device:
                    raise NotImplementedError("a MoE block must reside on a single device")
            for name, _ in list(block.named_parameters()) + list(block.named_buffers()):
                if any(part in name.lower() for part in ("weight_scale", "qweight", "qzeros", "quant_state")):
                    raise NotImplementedError("quantized module layouts are not supported")
            if self.is_fused(block):
                gu, dn = experts.gate_up_proj, getattr(experts, "down_proj", None)
                H = gate.weight.shape[1]
                if (not isinstance(gu, torch.Tensor) or gu.ndim != 3 or
                        not isinstance(dn, torch.Tensor) or dn.ndim != 3 or
                        tuple(gu.shape) != (spec.num_experts, 2 * dn.shape[-1], H) or
                        tuple(dn.shape[:2]) != (spec.num_experts, H)):
                    raise NotImplementedError("unsupported fused expert tensor layout")
                if not gu.is_floating_point() or not dn.is_floating_point():
                    raise NotImplementedError("quantized expert tensors are not supported")
                extra = set(dict(experts.named_parameters())) | set(dict(experts.named_buffers()))
                if extra - {"gate_up_proj", "down_proj", "gate_up_proj_bias", "down_proj_bias"}:
                    raise NotImplementedError("unsupported fused expert auxiliary tensors")
            elif not isinstance(experts, nn.ModuleList) or len(experts) != spec.num_experts:
                raise NotImplementedError("unsupported expert bank or inconsistent expert count")
            for holder in (gate, experts, block):
                for name in self.CONFIG_NUM_EXPERTS + ("num_routed_experts",):
                    value = getattr(holder, name, None)
                    if value is not None and int(value) != spec.num_experts:
                        raise ValueError(f"{item.tag}: module expert counts disagree")

    @torch.no_grad()
    def expert_forward(self, block: nn.Module, hidden: torch.Tensor,
                       lo: int, hi: int) -> torch.Tensor:
        """Return expert outputs (hi-lo, T, H), in float32."""
        experts = self.expert_bank(block)
        if not 0 <= lo < hi <= self.gate_module(block).weight.shape[0]:
            raise ValueError("invalid expert range")
        if self.is_fused(block):
            gu, dn = experts.gate_up_proj[lo:hi], experts.down_proj[lo:hi]
            H = hidden.shape[-1]
            if (gu.ndim != 3 or dn.ndim != 3 or gu.shape[-1] != H or
                    dn.shape[1] != H or gu.shape[1] != 2 * dn.shape[-1]):
                raise NotImplementedError("unsupported fused expert tensor layout")
            if not gu.is_floating_point() or not dn.is_floating_point():
                raise NotImplementedError("quantized experts are not supported")
            x = hidden.to(gu.dtype).unsqueeze(0).expand(hi - lo, -1, -1)
            projected = torch.bmm(x, gu.transpose(1, 2))
            bias = getattr(experts, "gate_up_proj_bias", None)
            if bias is not None:
                projected = projected + bias[lo:hi].unsqueeze(1)
            apply_gate = getattr(experts, "_apply_gate", None)
            if apply_gate is not None:
                activated = apply_gate(projected)
            else:
                g, u = projected.chunk(2, dim=-1)
                act = getattr(experts, "act_fn", None)
                if act is None:
                    raise NotImplementedError("fused experts must declare their activation")
                activated = act(g) * u
            out = torch.bmm(activated, dn.transpose(1, 2))
            bias = getattr(experts, "down_proj_bias", None)
            if bias is not None:
                out = out + bias[lo:hi].unsqueeze(1)
            return out.float()
        if not isinstance(experts, nn.ModuleList):
            raise NotImplementedError("unsupported expert bank")
        outs = []
        for expert in experts[lo:hi]:
            parameter = next(expert.parameters(), None)
            x = hidden if parameter is None else hidden.to(parameter.dtype)
            outs.append(expert(x))
        return torch.stack(outs).float()

    @torch.no_grad()
    def shared_forward(self, block: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        """Return the unchanged shared contribution (T, H), in float32."""
        shared = [getattr(block, key) for key in ("shared_expert", "shared_experts", "shared_mlp")
                  if getattr(block, key, None) is not None]
        if len(shared) > 1:
            raise NotImplementedError("ambiguous shared-expert layout")
        if not shared:
            return torch.zeros_like(hidden, dtype=torch.float32)
        output = shared[0](hidden)
        gate = getattr(block, "shared_expert_gate", None)
        if gate is not None:
            output = output * torch.sigmoid(gate(hidden))
        if output.shape != hidden.shape:
            raise NotImplementedError("unsupported shared-expert output shape")
        return output.float()

    def validate_keep(self, block: nn.Module, keep: torch.Tensor,
                      top_k: Optional[int] = None, *,
                      target_top_k: Optional[int] = None) -> None:
        """Check a keep-set and optional reduced top-k without mutation."""
        if target_top_k is not None:
            if top_k is not None and top_k != target_top_k:
                raise ValueError("conflicting target top_k values")
            top_k = target_top_k
        if any(value.is_meta for value in block.parameters()) or keep.is_meta:
            raise NotImplementedError("cannot slice meta or offloaded weights")
        E = self.gate_module(block).weight.shape[0]
        if (keep.ndim != 1 or keep.dtype != torch.long or keep.numel() < 1 or
                keep.unique().numel() != keep.numel() or
                (keep < 0).any() or (keep >= E).any()):
            raise ValueError("invalid expert keep-set")
        spec = self.router_spec(block)
        k = spec.top_k if top_k is None else top_k
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= spec.top_k:
            raise ValueError("target top_k must be between 1 and the original top_k")
        if keep.numel() < k:
            raise ValueError("keep-set is smaller than top_k")
        if spec.n_group < 1 or keep.numel() // spec.n_group < self.MIN_EXPERTS_PER_GROUP:
            raise ValueError("keep-set is too small for the native group scorer")
        if spec.n_group > 1:
            counts = torch.bincount(keep.cpu() // (E // spec.n_group), minlength=spec.n_group)
            if (not torch.all(counts == counts[0]) or counts[0] < spec.group_score_topk or
                    counts[0] * spec.topk_group < k or
                    not torch.equal(keep, keep.sort().values)):
                raise NotImplementedError("grouped pruning requires equal retained counts in original group order")

    @torch.no_grad()
    def prune_block(self, block: nn.Module, keep: torch.Tensor,
                    top_k: Optional[int] = None, *,
                    target_top_k: Optional[int] = None) -> None:
        """Slice a validated keep-set and update module counts and optional top-k."""
        self.validate_keep(block, keep, top_k=top_k, target_top_k=target_top_k)
        if target_top_k is not None:
            top_k = target_top_k
        self._slice_block(block, keep, top_k)

    @torch.no_grad()
    def _slice_block(self, block, keep, top_k=None):
        """Apply tensor slicing after the caller has validated routing semantics."""
        experts = self.expert_bank(block)
        if self.is_fused(block):
            for name in ("gate_up_proj", "down_proj", "gate_up_proj_bias", "down_proj_bias"):
                value = getattr(experts, name, None)
                if value is not None:
                    sliced = value.detach().index_select(0, keep.to(value.device)).clone()
                    setattr(experts, name, nn.Parameter(sliced, requires_grad=False)
                            if isinstance(value, nn.Parameter) else sliced)
        elif isinstance(experts, nn.ModuleList):
            block.experts = nn.ModuleList([experts[index] for index in keep.tolist()])
            experts = block.experts
        else:
            raise NotImplementedError("unsupported expert bank")
        gate = self.gate_module(block)
        for name in ("weight", "bias"):
            value = getattr(gate, name, None)
            if value is not None:
                setattr(gate, name, nn.Parameter(value.detach().index_select(0, keep.to(value.device)).clone(),
                                                 requires_grad=False))
        for holder in (gate, block):
            for name in self.BIAS_ATTRS:
                value = getattr(holder, name, None)
                if value is not None:
                    sliced = value.detach().index_select(0, keep.to(value.device)).clone()
                    setattr(holder, name, nn.Parameter(sliced, requires_grad=False)
                            if isinstance(value, nn.Parameter) else sliced)
        for holder in (experts, gate, block):
            for name in self.CONFIG_NUM_EXPERTS + ("num_routed_experts",):
                if getattr(holder, name, None) is not None:
                    setattr(holder, name, int(keep.numel()))
        if isinstance(gate, nn.Linear):
            gate.out_features = int(keep.numel())
        if top_k is not None:
            for module in block.modules():
                for name in set(self.CONFIG_TOP_K) | {"top_k", "num_experts_per_tok", "moe_topk"}:
                    if isinstance(getattr(module, name, None), int):
                        setattr(module, name, top_k)

    AUX_FILES = (
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "chat_template.jinja", "chat_template.json", "generation_config.json",
        "vocab.json", "merges.txt", "added_tokens.json", "tokenizer.model",
        "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json",
    )
    REMOTE_CODE_GLOBS = ("modeling_*.py", "configuration_*.py", "tokenization_*.py", "*.jinja")

    def postprocess_config(self, config, info: dict) -> None:
        """Apply family-specific config updates before saving."""
        return None
