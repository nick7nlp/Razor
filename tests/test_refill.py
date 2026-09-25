"""Numerical checks for the three consensus-residual scoring criteria.

Reference values are built from the *definition* of each counterfactual --
delete an expert, rebuild the mixture, measure the output shift -- rather than
from the closed forms the collector evaluates, so this independently checks
both the propositions and their implementation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from razor import metrics                                    # noqa: E402
from razor.adapters import get_adapter                       # noqa: E402
from razor.collect import SaliencyCollector                  # noqa: E402

H, I, E, K, T = 16, 32, 8, 3, 20


class Config:
    def __init__(self, **kw):
        self.model_type = "test_moe"
        self.num_experts = E
        self.num_experts_per_tok = K
        self.norm_topk_prob = True
        self.scoring_func = "softmax"
        for key, value in kw.items():
            setattr(self, key, value)


class Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * I, H) * 0.3)
        self.down_proj = nn.Parameter(torch.randn(E, H, I) * 0.3)
        self.num_experts = E
        self.act_fn = F.silu


class MoE(nn.Module):
    """Renormalized top-k routing with an optional routed-output scale."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(H, E, bias=False)
        self.experts = Experts()

    def expert_out(self, flat, index):
        projected = flat @ self.experts.gate_up_proj[index].t()
        gate, up = projected.chunk(2, dim=-1)
        return (F.silu(gate) * up) @ self.experts.down_proj[index].t()

    def forward(self, x):
        flat = x.reshape(-1, H)
        probs = F.softmax(self.gate(flat), dim=-1)
        selected = probs.topk(K, dim=-1).indices
        weights = probs.gather(1, selected)
        weights = weights / weights.sum(-1, keepdim=True)
        weights = weights * getattr(self.config, "routed_scaling_factor", 1.0)
        out = torch.zeros_like(flat)
        for index in range(E):
            hit = (weights * (selected == index)).sum(-1, keepdim=True)
            out += hit * self.expert_out(flat, index)
        return out.reshape(x.shape)


def _setup(seed, **kw):
    torch.manual_seed(seed)
    config = Config(**kw)
    block = MoE(config).eval().double()
    adapter = get_adapter(config, "generic")
    hidden = (torch.randn(1, T, H) * 0.8).double()
    return block, adapter, hidden


def _collect(block, adapter, hidden):
    layer = nn.Module()
    layer.mlp = block
    collector = SaliencyCollector(adapter, expert_chunk=3)
    with collector.layer_context("main_0", layer):
        block(hidden)
    return collector.finalize()["main_0"]


def _reference(block, adapter, hidden):
    """Rebuild every deletion counterfactual directly from its definition."""
    flat = hidden.reshape(-1, H)
    context = adapter.context(block, (hidden,))
    lam = float(context.spec.scaling)
    gates = (context.weights / lam).double()
    outputs = torch.stack([block.expert_out(flat, i) for i in range(E)]).double()
    promoted_gate = context.promoted_weight.double() / lam
    consensus = (gates.unsqueeze(-1) * outputs).sum(0)

    shape = (E, T)
    rcs = torch.zeros(shape, dtype=torch.float64)
    loo = torch.zeros(shape, dtype=torch.float64)
    refill = torch.zeros(shape, dtype=torch.float64)
    routed = torch.zeros(shape, dtype=torch.bool)
    for t in range(T):
        support = context.selected[t].tolist()
        promoted = int(context.promoted[t])
        assert promoted not in support, "the promoted expert must be unrouted"
        for i in support:
            routed[i, t] = True
            survivors = [j for j in support if j != i]
            survived = sum(gates[j, t] * outputs[j, t] for j in survivors)
            # Assumption 2: renormalize over the surviving routed set only.
            without = survived / (1 - gates[i, t])
            loo[i, t] = lam * torch.linalg.vector_norm(consensus[t] - without)
            # Assumption 1: promote rank-(k+1) at its pseudo-weight.
            denominator = 1 - gates[i, t] + promoted_gate[t]
            refilled = (survived + promoted_gate[t] * outputs[promoted, t]) / denominator
            refill[i, t] = lam * torch.linalg.vector_norm(consensus[t] - refilled)
            rcs[i, t] = lam * gates[i, t] * torch.linalg.vector_norm(
                outputs[i, t] - consensus[t])
    return rcs, loo, refill, routed


@pytest.mark.parametrize("scaling", [1.0, 2.5])
@torch.no_grad()
def test_criteria_match_their_definitions(scaling):
    block, adapter, hidden = _setup(7, routed_scaling_factor=scaling)
    rcs, loo, refill, routed = _reference(block, adapter, hidden)
    record = _collect(block, adapter, hidden)
    count = routed.sum(1).double().clamp_min(1)

    for values, mean_field, rms_field in (
        (rcs, "rcs_mean", "rcs_rms"),
        (loo, "counterfactual_delta_mean", "counterfactual_delta_rms"),
        (refill, "rcs_refill_mean", "rcs_refill_rms"),
    ):
        torch.testing.assert_close(record[mean_field].double(), values.sum(1) / count,
                                   rtol=1e-6, atol=1e-9)
        torch.testing.assert_close(record[rms_field].double(),
                                   (values.square().sum(1) / count).sqrt(),
                                   rtol=1e-6, atol=1e-9)


@torch.no_grad()
def test_promoted_expert_is_rank_k_plus_one():
    block, adapter, hidden = _setup(11)
    flat = hidden.reshape(-1, H)
    context = adapter.context(block, (hidden,))
    probs = F.softmax(block.gate(flat), dim=-1).double()
    ranked = probs.argsort(dim=-1, descending=True)

    torch.testing.assert_close(context.promoted, ranked[:, K])
    # The pseudo-weight is the promoted mixture score over the selected sum.
    expected = (probs.gather(1, ranked[:, K].unsqueeze(1)).squeeze(1)
                / probs.gather(1, context.selected).sum(-1))
    torch.testing.assert_close(context.promoted_weight.double(), expected)


@torch.no_grad()
def test_refill_is_unavailable_without_promoted_routing():
    """Adapters that cannot expose rank-(k+1) still serve the other criteria."""
    record = {"routed_count": torch.ones(E), "expert_frequency": torch.ones(E),
              "counterfactual_delta_rms": torch.ones(E), "rcs_rms": torch.ones(E),
              "ean_rms": torch.ones(E), "reap_rms": torch.ones(E),
              "razor_supported": True, "refill_supported": False}
    for name in ("rcs-refill", "razor"):
        with pytest.raises(KeyError, match="rank-"):
            metrics.score(record, name)
    assert metrics.score(record, "rcs-loo").numel() == E
    assert "rcs-refill" not in metrics.available(record)
    assert "rcs-loo" in metrics.available(record)


@torch.no_grad()
def test_razor_names_refill_under_rms():
    """``razor`` must name the paper's criterion, not one of its ablations."""
    block, adapter, hidden = _setup(17)
    record = _collect(block, adapter, hidden)

    assert metrics.canonical("razor") == "rcs-refill"
    assert metrics.DEFAULT_AGGREGATION == "rms"
    torch.testing.assert_close(metrics.score(record, "razor"),
                               metrics.score(record, "rcs-refill", aggregation="rms"))
    # The variants have to be genuinely different scoring rules.
    assert not torch.allclose(metrics.score(record, "rcs-refill"),
                              metrics.score(record, "rcs-loo"))
    assert not torch.allclose(metrics.score(record, "rcs-loo"),
                              metrics.score(record, "rcs"))


@torch.no_grad()
def test_frozen_weight_refill_matches_its_definition():
    """Without renormalization, D = 1: deletion drops w_i f_i, refill adds w_r f_r."""
    block, adapter, hidden = _setup(29, norm_topk_prob=False, routed_scaling_factor=1.5)
    block.forward = _unnormalized_forward.__get__(block)
    flat = hidden.reshape(-1, H)
    context = adapter.context(block, (hidden,))
    assert context.spec.normalize is False
    outputs = torch.stack([block.expert_out(flat, i) for i in range(E)]).double()
    weights = context.weights.double()
    promoted_weight = context.promoted_weight.double()

    expected = torch.zeros(E, T, dtype=torch.float64)
    routed = torch.zeros(E, T, dtype=torch.bool)
    for t in range(T):
        original = sum(weights[j, t] * outputs[j, t] for j in context.selected[t].tolist())
        for i in context.selected[t].tolist():
            routed[i, t] = True
            refilled = (original - weights[i, t] * outputs[i, t]
                        + promoted_weight[t] * outputs[int(context.promoted[t]), t])
            expected[i, t] = torch.linalg.vector_norm(original - refilled)

    record = _collect(block, adapter, hidden)
    assert record["counterfactual_semantics"] == "frozen_weights"
    count = routed.sum(1).double().clamp_min(1)
    torch.testing.assert_close(record["rcs_refill_rms"].double(),
                               (expected.square().sum(1) / count).sqrt(), rtol=1e-6, atol=1e-9)


def _unnormalized_forward(self, x):
    flat = x.reshape(-1, H)
    probs = F.softmax(self.gate(flat), dim=-1)
    selected = probs.topk(K, dim=-1).indices
    weights = probs.gather(1, selected) * self.config.routed_scaling_factor
    out = torch.zeros_like(flat)
    for index in range(E):
        out += (weights * (selected == index)).sum(-1, keepdim=True) * self.expert_out(flat, index)
    return out.reshape(x.shape)


@torch.no_grad()
def test_all_criteria_come_from_one_collection():
    block, adapter, hidden = _setup(23)
    record = _collect(block, adapter, hidden)
    for name in metrics.CHOICES:
        assert metrics.score(record, name).numel() == E
