"""Collect routed-token saliency with native-forward reconstruction checks."""
from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .adapters import MoEAdapter
from .adapters.base import MoEBlock

SALIENCY_FILENAME = "observer_data.pt"
SALIENCY_ALIASES = ("observer_data.pt", "saliency.pt")
PARTIAL_FILENAME = "observer_data.partial.pt"

# Both deletion denominators are clamped from below by this floor. The scoring
# identities describe the unclamped quantity, so a score computed while the
# floor is active can differ from exact local damage. See docs/scoring.md.
DENOM_FLOOR = 1e-6


def find_saliency(path) -> Optional[Path]:
    """Resolve a saliency directory or direct filename."""
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        for name in SALIENCY_ALIASES:
            candidate = path / name
            if candidate.is_file():
                return candidate
    return None


def _input_device(model):
    """Use the input embedding device, with a fallback for plain modules."""
    getter = getattr(model, "get_input_embeddings", None)
    embedding = getter() if callable(getter) else None
    if embedding is not None and hasattr(embedding, "weight"):
        return embedding.weight.device
    return next(model.parameters()).device


def _check_reconstruction(tag, routed, shared, output, rtol=2e-2, atol=1e-7,
                          native_routed=None):
    """Check routed and complete output vectors, including their magnitudes."""
    ref = output[0] if isinstance(output, tuple) else output
    ref = ref.detach().reshape_as(routed).float()
    routed_ref = ref - shared if native_routed is None else native_routed.reshape_as(routed).float()
    for name, actual, expected in (("routed", routed, routed_ref),
                                    ("routed+shared", routed + shared, ref)):
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise RuntimeError(f"{tag}: non-finite {name} reconstruction")
        error = torch.linalg.vector_norm(actual - expected, dim=-1)
        scale = torch.linalg.vector_norm(expected, dim=-1)
        if (error > atol + rtol * scale).any():
            raise RuntimeError(f"{tag}: {name} reconstruction mismatch; model is unsupported")


def _check_context_reconstruction(adapter, block, context, tag, routed, output,
                                  native=None, rtol=2e-2, atol=1e-7):
    if native is not None:
        _check_reconstruction(tag, routed, torch.zeros_like(routed), native,
                              rtol=rtol, atol=atol, native_routed=native)
    actual = adapter.routed_to_output(block, routed)
    native_output = None if native is None else adapter.routed_to_output(block, native)
    shared = adapter.shared_forward(block, context.router_input)
    _check_reconstruction(tag, actual, shared, output, rtol=rtol, atol=atol,
                          native_routed=native_output)


class SaliencyCollector:
    """Accumulate routed-token statistics; expert_chunk controls peak memory."""
    def __init__(self, adapter: MoEAdapter, expert_chunk: int = 4):
        if isinstance(expert_chunk, bool) or not isinstance(expert_chunk, int) or expert_chunk < 1:
            raise ValueError("expert_chunk must be a positive integer")
        self.adapter = adapter
        self.expert_chunk = expert_chunk
        self.state: Dict[str, dict] = {}
        self._handles: List = []
        self._mask: Optional[torch.Tensor] = None
        self.blocks: List = []
        self._gate_checked: set = set()
        self._native_routed: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _new_state(E: int) -> dict:
        state = {name: torch.zeros(E, dtype=torch.float64) for name in (
            "routed_count", "ean_sum", "weighted_ean_sum", "weighted_expert_frequency_sum",
            "normalized_gate_sum", "reap_numer", "counterfactual_delta_sum",
            "ean_square_sum", "weighted_ean_square_sum", "counterfactual_delta_square_sum",
            "rcs_sum", "rcs_square_sum", "rcs_refill_sum", "rcs_refill_square_sum")}
        state.update(total_tokens=0, expert_frequency=torch.zeros(E, dtype=torch.long),
                     max_activations=torch.zeros(E, dtype=torch.float32), router_scaling=1.0,
                     renormalized=True, razor_supported=True, refill_supported=True,
                     counterfactual_semantics="fixed_selected_set")
        return state

    def _hook(self, tag: str, block):
        adapter = self.adapter

        @torch.no_grad()
        def fn(module, args, kwargs, output):
            context = adapter.context(block, args, output, kwargs)
            flat = context.hidden
            T, H = flat.shape
            spec = context.spec
            E, lam = spec.num_experts, float(spec.scaling)
            selected, weights = context.selected, context.weights
            gates = weights / lam
            renorm = bool(spec.normalize)
            supported = spec.top_k >= 2 and not adapter.is_hash(block)
            if renorm and (gates.max() > 1.0 + 1e-3 or
                           not torch.allclose(gates.sum(0), torch.ones(T, device=flat.device),
                                              atol=1e-2, rtol=0)):
                raise RuntimeError(f"{tag}: invalid normalized gate weight or weight sum")
            routed = torch.zeros(E, T, dtype=torch.bool, device=flat.device)
            routed.scatter_(0, selected.t(), True)
            if self._mask is None:
                token_mask = torch.ones(T, dtype=torch.bool, device=flat.device)
            else:
                if self._mask.numel() != T:
                    raise ValueError(f"{tag}: attention mask does not match MoE inputs")
                token_mask = self._mask.reshape(-1).to(flat.device).bool()
            routed &= token_mask.unsqueeze(0)
            if supported and renorm:
                eps = torch.finfo(torch.float32).eps
                supported = not bool((routed & ((1.0 - gates) <= eps)).any())
            supported = supported and self.state.get(tag, {}).get("razor_supported", True)
            promoted, promoted_weight = context.promoted, context.promoted_weight
            # Refill is defined for both gate semantics: with renormalized gates
            # it is Proposition 1; with frozen weights nothing is renormalized,
            # so D = 1 and the damage is lam * ||u_i f_i - u_r f_r||.
            refill = (supported and promoted is not None
                      and promoted_weight is not None
                      and self.state.get(tag, {}).get("refill_supported", True))
            if refill and (promoted.shape != (T,) or promoted_weight.shape != (T,)):
                raise RuntimeError(f"{tag}: promoted routing tensors do not match the token count")
            sel_flat = selected[token_mask].reshape(-1)
            frequency = torch.bincount(sel_flat, minlength=E)
            d = {key: torch.zeros(E, dtype=torch.float64, device=flat.device)
                 for key in ("ean", "wean", "wfreq", "ufreq", "cf", "ean2", "wean2", "cf2",
                             "rcs", "rcs2", "rfl", "rfl2")}
            peaks = torch.zeros(E, dtype=torch.float32, device=flat.device)
            y = torch.zeros(T, H, dtype=torch.float32, device=flat.device)
            promoted_output = (torch.zeros(T, H, dtype=torch.float32, device=flat.device)
                               if refill else None)
            for lo in range(0, E, self.expert_chunk):
                hi = min(lo + self.expert_chunk, E)
                act = adapter.expert_forward(block, flat, lo, hi)
                if not torch.isfinite(act).all():
                    raise RuntimeError(f"{tag}: non-finite expert outputs")
                r, w, u = routed[lo:hi], weights[lo:hi], gates[lo:hi]
                norm = torch.linalg.vector_norm(act, dim=-1)
                norm = torch.where(r, norm, 0.0)
                weighted_norm = (norm * w).double()
                d["ean"][lo:hi] = norm.double().sum(1)
                d["wean"][lo:hi] = weighted_norm.sum(1)
                d["ean2"][lo:hi] = norm.double().square().sum(1)
                d["wean2"][lo:hi] = weighted_norm.square().sum(1)
                d["wfreq"][lo:hi] = (w * r).double().sum(1)
                d["ufreq"][lo:hi] = (u * r).double().sum(1)
                peak = act.masked_fill(~r.unsqueeze(-1), float("-inf")).amax(dim=(1, 2))
                peaks[lo:hi] = torch.where(torch.isfinite(peak), peak, 0.0)
                y += (u.unsqueeze(-1) * act).sum(0)
                if refill:
                    # The promoted expert is unrouted, so its output is only
                    # available while its own chunk is resident.
                    here = (promoted >= lo) & (promoted < hi)
                    if here.any():
                        local = (promoted - lo).clamp(0, hi - lo - 1)
                        picked = act.gather(0, local.view(1, T, 1).expand(1, T, H)).squeeze(0)
                        promoted_output = torch.where(here.unsqueeze(-1), picked, promoted_output)
            if tag not in self._gate_checked:
                _check_context_reconstruction(adapter, block, context, tag, y * lam, output,
                                              self._native_routed.pop(tag, None))
                self._gate_checked.add(tag)
            if supported and renorm:
                promoted_residual = None if not refill else promoted_output - y
                promoted_gate = None if not refill else (promoted_weight.to(flat.device).float() / lam)
                for lo in range(0, E, self.expert_chunk):
                    hi = min(lo + self.expert_chunk, E)
                    act = adapter.expert_forward(block, flat, lo, hi)
                    consensus_residual = act - y.unsqueeze(0)
                    residual = torch.linalg.vector_norm(consensus_residual, dim=-1)
                    residual = torch.where(routed[lo:hi], residual, 0.0)
                    u = gates[lo:hi]
                    # RCS: contribution magnitude relative to the consensus.
                    rcs = (lam * residual * u).double()
                    d["rcs"][lo:hi] = rcs.sum(1)
                    d["rcs2"][lo:hi] = rcs.square().sum(1)
                    # RCS-LOO: Proposition 2, fixed routed support.
                    delta = (lam * residual * u / (1 - u).clamp_min(DENOM_FLOOR)).double()
                    d["cf"][lo:hi] = delta.sum(1)
                    d["cf2"][lo:hi] = delta.square().sum(1)
                    if refill:
                        # RCS-Refill: Proposition 1. The numerator is the norm
                        # of a vector difference, so the promoted residual can
                        # reinforce or cancel the removed one and the two
                        # cannot be combined after taking norms.
                        shift = (u.unsqueeze(-1) * consensus_residual
                                 - promoted_gate.view(1, T, 1) * promoted_residual.unsqueeze(0))
                        magnitude = torch.linalg.vector_norm(shift, dim=-1)
                        magnitude = torch.where(routed[lo:hi], magnitude, 0.0)
                        denominator = (1 - u + promoted_gate.unsqueeze(0)).clamp_min(DENOM_FLOOR)
                        refill_delta = (lam * magnitude / denominator).double()
                        d["rfl"][lo:hi] = refill_delta.sum(1)
                        d["rfl2"][lo:hi] = refill_delta.square().sum(1)
            elif supported:
                # Frozen weights: deleting i removes w_i f_i and nothing else,
                # so the fixed-support score is exactly the REAP magnitude.
                d["cf"] = d["wean"].clone()
                d["cf2"] = d["wean2"].clone()
                d["rcs"] = d["wean"].clone()
                d["rcs2"] = d["wean2"].clone()
                if refill:
                    promoted_gate = promoted_weight.to(flat.device).float() / lam
                    promoted_term = promoted_gate.unsqueeze(-1) * promoted_output
                    for lo in range(0, E, self.expert_chunk):
                        hi = min(lo + self.expert_chunk, E)
                        act = adapter.expert_forward(block, flat, lo, hi)
                        u = gates[lo:hi]
                        shift = u.unsqueeze(-1) * act - promoted_term.unsqueeze(0)
                        magnitude = torch.linalg.vector_norm(shift, dim=-1)
                        magnitude = torch.where(routed[lo:hi], magnitude, 0.0)
                        refill_delta = (lam * magnitude).double()
                        d["rfl"][lo:hi] = refill_delta.sum(1)
                        d["rfl2"][lo:hi] = refill_delta.square().sum(1)
            st = self.state.setdefault(tag, self._new_state(E))
            st["total_tokens"] += int(token_mask.sum())
            st["expert_frequency"] += frequency.cpu()
            st["routed_count"] += routed.sum(1).double().cpu()
            for name, key in (("ean_sum", "ean"), ("weighted_ean_sum", "wean"),
                              ("weighted_expert_frequency_sum", "wfreq"),
                              ("normalized_gate_sum", "ufreq"), ("reap_numer", "wean"),
                              ("counterfactual_delta_sum", "cf"), ("ean_square_sum", "ean2"),
                              ("weighted_ean_square_sum", "wean2"),
                              ("counterfactual_delta_square_sum", "cf2"),
                              ("rcs_sum", "rcs"), ("rcs_square_sum", "rcs2"),
                              ("rcs_refill_sum", "rfl"), ("rcs_refill_square_sum", "rfl2")):
                st[name] += d[key].cpu()
            st["max_activations"] = torch.maximum(st["max_activations"], peaks.cpu())
            st["router_scaling"] = lam
            st["renormalized"] = renorm
            st["razor_supported"] = supported
            st["refill_supported"] = bool(refill)
            st["counterfactual_semantics"] = ("fixed_selected_set" if renorm else "frozen_weights")
            st["pruning_policy"] = "preserve" if adapter.is_hash(block) else "score"
            st["score_space"] = "latent" if getattr(block, "use_latent_moe", False) else "expert_output"
        return fn

    def install(self, model) -> "SaliencyCollector":
        if self._handles:
            raise RuntimeError("collector is already installed")
        self.adapter.validate_model(model)
        return self.install_blocks(self.adapter.moe_blocks(model))

    def install_blocks(self, blocks) -> "SaliencyCollector":
        """Attach to already materialized, validated blocks (also used by streaming)."""
        if self._handles:
            raise RuntimeError("collector is already installed")
        self.blocks = list(blocks)
        self._gate_checked.clear()
        self._native_routed.clear()
        for item in self.blocks:
            target, pre = self.adapter.native_routed_capture(item.module)
            if target is not None:
                if pre:
                    def capture_input(module, args, tag=item.tag):
                        if tag not in self._gate_checked:
                            self._native_routed[tag] = args[0].detach()
                    self._handles.append(target.register_forward_pre_hook(capture_input))
                else:
                    def capture(module, args, output, tag=item.tag):
                        if tag not in self._gate_checked:
                            self._native_routed[tag] = output.detach()
                    self._handles.append(target.register_forward_hook(capture))
            target = self.adapter.hook_module(item.module)
            self._handles.append(target.register_forward_hook(
                self._hook(item.tag, item.module), with_kwargs=True))
        return self

    def set_attention_mask(self, mask) -> None:
        self._mask = mask

    def set_batch(self, attention_mask=None, input_ids=None) -> None:
        self.set_attention_mask(attention_mask)

    @contextmanager
    def layer_context(self, tag, layer):
        """Instrument a single materialized native decoder layer, then detach."""
        candidates = (layer, *(getattr(layer, name, None)
                               for name in ("mlp", "ffn", "block_sparse_moe")))
        block = next((candidate for candidate in candidates
                      if candidate is not None and self.adapter.is_moe_block(candidate)), None)
        if block is None:
            yield self
            return
        trunk, index = tag.rsplit("_", 1)
        blocks = [MoEBlock(tag, int(index), trunk, block)]
        self.adapter.validate_blocks(blocks)
        try:
            self.install_blocks(blocks)
            yield self
        finally:
            self.remove()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._native_routed.clear()

    def finalize(self) -> Dict[str, dict]:
        """Return baseline scores and applicable RAZOR statistics."""
        result = {}
        for tag, state in self.state.items():
            count = state["routed_count"].clamp_min(1.0)
            record = {key: value.clone() if isinstance(value, torch.Tensor) else value
                      for key, value in state.items() if key != "reap_numer"}
            record["ean_mean"] = (state["ean_sum"] / count).float()
            record["reap"] = (state["reap_numer"] / count).float()
            record["ean_rms"] = (state["ean_square_sum"] / count).sqrt().float()
            record["reap_rms"] = (state["weighted_ean_square_sum"] / count).sqrt().float()
            if state["razor_supported"]:
                record["counterfactual_delta_mean"] = (state["counterfactual_delta_sum"] / count).float()
                record["counterfactual_delta_rms"] = (state["counterfactual_delta_square_sum"] / count).sqrt().float()
                record["rcs_mean"] = (state["rcs_sum"] / count).float()
                record["rcs_rms"] = (state["rcs_square_sum"] / count).sqrt().float()
            else:
                for name in ("counterfactual_delta_sum", "counterfactual_delta_square_sum",
                             "rcs_sum", "rcs_square_sum"):
                    record.pop(name)
            if state["razor_supported"] and state["refill_supported"]:
                record["rcs_refill_mean"] = (state["rcs_refill_sum"] / count).float()
                record["rcs_refill_rms"] = (state["rcs_refill_square_sum"] / count).sqrt().float()
            else:
                record["refill_supported"] = False
                record.pop("rcs_refill_sum")
                record.pop("rcs_refill_square_sum")
            result[tag] = record
        return result


def verify_routing(model, adapter, batch, atol: float = 1e-7, rtol: float = 2e-2) -> bool:
    """Verify all executed MoE blocks; mismatches raise rather than returning False."""
    adapter.validate_model(model)
    blocks = adapter.moe_blocks(model)
    captured, native, handles = {}, {}, []
    for item in blocks:
        def probe(module, args, kwargs, output, tag=item.tag, block=item.module):
            captured[tag] = (adapter.context(block, args, output, kwargs), output)
        handles.append(adapter.hook_module(item.module).register_forward_hook(probe, with_kwargs=True))
        target, pre = adapter.native_routed_capture(item.module)
        if target is not None:
            if pre:
                def probe_input(module, args, tag=item.tag):
                    native[tag] = args[0].detach()
                handles.append(target.register_forward_pre_hook(probe_input))
            else:
                def probe_experts(module, args, output, tag=item.tag):
                    native[tag] = output.detach()
                handles.append(target.register_forward_hook(probe_experts))
    try:
        device = _input_device(model)
        kwargs = {"input_ids": batch["input_ids"][:1, :256].to(device), "use_cache": False}
        if batch.get("attention_mask") is not None:
            kwargs["attention_mask"] = batch["attention_mask"][:1, :256].to(device)
        with torch.no_grad():
            model(**kwargs)
    finally:
        for handle in handles:
            handle.remove()
    missing = [item.tag for item in blocks if item.trunk == "main" and item.tag not in captured]
    if not captured or missing:
        raise RuntimeError("routing verification did not execute all main MoE blocks")
    with torch.no_grad():
        for item in blocks:
            if item.tag not in captured:
                continue
            context, output = captured[item.tag]
            flat, spec, weights = context.hidden, context.spec, context.weights
            routed = torch.zeros_like(flat, dtype=torch.float32)
            for lo in range(0, spec.num_experts, 8):
                hi = min(lo + 8, spec.num_experts)
                act = adapter.expert_forward(item.module, flat, lo, hi)
                routed += (weights[lo:hi].unsqueeze(-1) * act).sum(0)
            _check_context_reconstruction(adapter, item.module, context, item.tag, routed, output,
                                          native.get(item.tag), rtol=rtol, atol=atol)
    print(f"[verify] {len(captured)} MoE blocks: routed and shared outputs match")
    return True


def collect(model, adapter: MoEAdapter, batches, out_dir: Optional[str] = None,
            expert_chunk: int = 4, checkpoint_every: int = 16,
            verbose: bool = True) -> Dict[str, dict]:
    """Collect saliency, optionally saving periodic and final results."""
    device = _input_device(model)
    collector = SaliencyCollector(adapter, expert_chunk=expert_chunk).install(model)
    out_path = Path(out_dir) if out_dir else None
    t0 = time.time()
    try:
        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)
        for i, batch in enumerate(batches):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch.get("attention_mask")
            if mask is not None:
                mask = mask.to(device, non_blocking=True)
            collector.set_attention_mask(mask)
            with torch.no_grad():
                model(input_ids=ids, attention_mask=mask, use_cache=False)
            done = i + 1
            if verbose and (done % 4 == 0 or done == len(batches)):
                elapsed = time.time() - t0
                print(f"[collect] {done}/{len(batches)} | {elapsed / 60:.1f} min elapsed")
            if out_path and checkpoint_every and done % checkpoint_every == 0:
                torch.save(collector.finalize(), out_path / PARTIAL_FILENAME)
    finally:
        collector.remove()
        gc.collect()
    saliency = collector.finalize()
    if not saliency:
        raise RuntimeError("no saliency was collected")
    if out_path:
        torch.save(saliency, out_path / SALIENCY_FILENAME)
        partial = out_path / PARTIAL_FILENAME
        if partial.exists():
            partial.unlink()
        if verbose:
            print("[collect] wrote saliency")
    return saliency
