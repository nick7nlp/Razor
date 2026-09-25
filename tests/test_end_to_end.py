"""End-to-end tests using tiny synthetic MoE models."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from razor import metrics                                    # noqa: E402
from razor.adapters import get_adapter                       # noqa: E402
from razor.collect import SaliencyCollector                  # noqa: E402
from razor.prune import (fill_unobserved_layers, prune_model,  # noqa: E402
                         resolve_target_experts, select_keep_indices)

H, I, E, K, T = 16, 32, 8, 2, 24

SCALED_LAM = 3.0


class Config:
    """Minimal stand-in for a transformers config."""

    def __init__(self, **kw):
        self.model_type = "test_moe"
        self.num_experts = E
        self.num_experts_per_tok = K
        self.norm_topk_prob = True
        for k, v in kw.items():
            setattr(self, k, v)


# fused experts + softmax router
class FusedExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * I, H) * 0.1)
        self.down_proj = nn.Parameter(torch.randn(E, H, I) * 0.1)
        self.num_experts = E
        self.act_fn = F.silu


class FusedMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(H, E, bias=False)
        self.experts = FusedExperts()

    def forward(self, x):
        flat = x.reshape(-1, H)
        # read the current expert count, so the module keeps working after
        # pruning has sliced it
        n = self.experts.gate_up_proj.shape[0]
        probs = F.softmax(self.gate(flat), dim=-1)
        sel = torch.topk(probs, min(self.config.num_experts_per_tok, n), dim=-1).indices
        w = probs.gather(1, sel)
        if self.config.norm_topk_prob:
            w = w / w.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        for e in range(n):
            gu = flat @ self.experts.gate_up_proj[e].t()
            g, u = gu.chunk(2, dim=-1)
            act = (F.silu(g) * u) @ self.experts.down_proj[e].t()
            hit = (sel == e)
            coef = (w * hit).sum(-1, keepdim=True)
            out += coef * act
        return out.reshape(x.shape)


# ModuleList experts + sigmoid router with correction bias
class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SigmoidRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(E, H) * 0.1)
        self.register_buffer("e_score_correction_bias", torch.randn(E) * 0.05)
        self.n_routed_experts = E


class ListMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = SigmoidRouter()
        self.experts = nn.ModuleList(SmallMLP() for _ in range(E))
        self.n_routed_experts = E

    def forward(self, x):
        flat = x.reshape(-1, H)
        n = len(self.experts)
        probs = torch.sigmoid(F.linear(flat, self.gate.weight))
        sel = torch.topk(probs + self.gate.e_score_correction_bias,
                         min(self.config.num_experts_per_tok, n), dim=-1).indices
        w = probs.gather(1, sel)
        if self.config.norm_topk_prob:
            w = w / w.sum(-1, keepdim=True)
        w = w * getattr(self.config, "routed_scaling_factor", 1.0)
        out = torch.zeros_like(flat)
        for e in range(n):
            act = self.experts[e](flat)
            out += (w * (sel == e)).sum(-1, keepdim=True) * act
        return out.reshape(x.shape)


# nested router + block-level bias + shared expert + leading dense layer
# (structural variations an adapter has to cope with)
class NestedRouter(nn.Module):
    """A router whose projection sits one level down, as ``router.gate``."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(H, E, bias=False)


class NestedRouterMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = NestedRouter()
        # bias lives on the block, not on the router
        self.expert_bias = nn.Parameter(torch.randn(E) * 0.05,
                                        requires_grad=False)
        self.experts = nn.ModuleList(SmallMLP() for _ in range(E))
        self.shared_mlp = SmallMLP()      # fires on every token; never pruned
        self.num_experts = E

    def forward(self, x):
        flat = x.reshape(-1, H)
        n = len(self.experts)
        logits = F.linear(flat, self.router.gate.weight)
        scoring = getattr(self.config, "scoring_func", "softmax")
        probs = (torch.sigmoid(logits) if scoring == "sigmoid"
                 else F.softmax(logits, dim=-1))
        sel = torch.topk(probs + self.expert_bias,
                         min(self.config.num_experts_per_tok, n), dim=-1).indices
        w = probs.gather(1, sel)
        if getattr(self.config, "norm_topk_prob", True):
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        w = w * getattr(self.config, "routed_scaling_factor", 1.0)
        out = self.shared_mlp(flat)
        for e in range(n):
            out = out + (w * (sel == e)).sum(-1, keepdim=True) * self.experts[e](flat)
        return out.reshape(x.shape)


class DenseMLP(nn.Module):
    """Stand-in for the leading `first_k_dense_replace` layers."""

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# assembly
class Layer(nn.Module):
    def __init__(self, moe):
        super().__init__()
        self.mlp = moe


def _make_block(kind):
    return {"fused": FusedMoE, "list": ListMoE, "nested": NestedRouterMoE}[kind]()


class Trunk(nn.Module):
    def __init__(self, kind, n_layers=3, dense_first=False):
        super().__init__()
        mlps = [_make_block(kind) for _ in range(n_layers)]
        if dense_first:
            # layer 0 is dense, mirroring first_k_dense_replace = 1
            mlps[0] = DenseMLP()
        self.layers = nn.ModuleList(Layer(m) for m in mlps)


class TinyMoE(nn.Module):
    def __init__(self, kind, config, dense_first=False):
        super().__init__()
        self.model = Trunk(kind, dense_first=dense_first)
        self.config = config
        for layer in self.model.layers:
            layer.mlp.config = config

    def forward(self, hidden):
        for layer in self.model.layers:
            hidden = hidden + layer.mlp(hidden)
        return hidden


# tests
def _run(kind, config, adapter_name, dense_first=False, scaling=1.0):
    torch.manual_seed(0)
    model = TinyMoE(kind, config, dense_first=dense_first).eval()
    adapter = get_adapter(config, name=adapter_name)

    n_moe = 2 if dense_first else 3
    tags = ["main_1", "main_2"] if dense_first else ["main_0", "main_1", "main_2"]
    blocks = adapter.moe_blocks(model)
    assert len(blocks) == n_moe, \
        f"{kind}: found {len(blocks)} blocks, want {n_moe}"
    # a leading dense layer must be skipped, and tags must carry the true
    # layer index so keep-sets line up with the checkpoint
    assert [b.tag for b in blocks] == tags, f"{kind}: tags {[b.tag for b in blocks]}"
    assert adapter.num_experts == E and adapter.top_k == K

    # -- routing replay must match the module's own routing -- #
    x = torch.randn(1, T, H)
    block = blocks[0].module
    flat = x.reshape(-1, H)
    spec = adapter.router_spec(block)
    assert abs(spec.scaling - scaling) < 1e-9, \
        f"{kind}: router scaling read as {spec.scaling}, want {scaling}"
    sel, weights = adapter.route(flat, adapter.gate_module(block).weight, spec)
    assert sel.shape == (T, K)
    assert weights.shape == (E, T)
    # the K selected weights per token renormalize to 1, then get scaled
    assert torch.allclose(weights.sum(0), torch.full((T,), scaling), atol=1e-4), \
        f"{kind}: gate weights sum to {weights.sum(0)[0]:.4f}, want {scaling}"

    # -- manual expert forward must match the module's experts -- #
    act = adapter.expert_forward(block, flat, 0, 2)
    assert act.shape == (2, T, H)
    if kind in ("list", "nested"):
        ref = block.experts[1](flat)
        assert torch.allclose(act[1], ref, atol=1e-5), \
            "ModuleList expert_forward disagrees with the module"

    with torch.no_grad():
        outputs = adapter.expert_forward(block, flat, 0, E)
        replay = (weights.unsqueeze(-1) * outputs).sum(0)
        if hasattr(block, "shared_mlp"):
            replay += block.shared_mlp(flat)
        torch.testing.assert_close(replay.reshape_as(x), block(x), atol=1e-5, rtol=1e-4)

    collector = SaliencyCollector(adapter, expert_chunk=3).install(model)
    try:
        with torch.no_grad():
            for _ in range(2):
                model(torch.randn(1, T, H))
    finally:
        collector.remove()
    assert all(not b.module._forward_hooks for b in blocks)
    saliency = collector.finalize()

    assert set(saliency) == set(tags)
    rec = saliency[tags[0]]
    assert metrics.available(rec) == list(metrics.CHOICES), \
        f"{kind}: only got {metrics.available(rec)}"
    for m in metrics.CHOICES:
        s = metrics.score(rec, m)
        assert s.shape == (E,), f"{kind}/{m}: shape {tuple(s.shape)}"
        assert torch.isfinite(s).all(), f"{kind}/{m}: non-finite score"
        assert (s >= 0).all(), f"{kind}/{m}: negative score"
    assert not torch.allclose(metrics.score(rec, "razor"),
                              metrics.score(rec, "reap"))
    # total routed slots per layer == tokens * top_k
    assert int(rec["expert_frequency"].sum()) == 2 * T * K

    # -- budget arithmetic -- #
    assert resolve_target_experts(E, None, 0.5) == E // 2
    assert resolve_target_experts(E, 3, None) == 3

    # -- pruning -- #
    keep = select_keep_indices(saliency, "razor", E // 2)
    assert all(len(v) == E // 2 for v in keep.values())
    shared_before = None
    if hasattr(blocks[0].module, "shared_mlp"):
        shared_before = blocks[0].module.shared_mlp.down_proj.weight.detach().clone()

    applied = prune_model(model, adapter, keep, verbose=False)
    assert len(applied) == n_moe

    for b in adapter.moe_blocks(model):
        g = adapter.gate_module(b.module)
        assert g.weight.shape[0] == E // 2, "router not sliced"
        bias = adapter.correction_bias(b.module)
        if bias is not None:
            assert bias.shape[0] == E // 2, "correction bias not sliced"
        if adapter.is_fused(b.module):
            assert b.module.experts.gate_up_proj.shape[0] == E // 2
            assert b.module.experts.down_proj.shape[0] == E // 2
        else:
            assert len(b.module.experts) == E // 2

    if shared_before is not None:
        after = blocks[0].module.shared_mlp.down_proj.weight
        assert after.shape == shared_before.shape and \
            torch.equal(after, shared_before), "shared expert was modified"

    # the pruned model must still run
    with torch.no_grad():
        out = model(torch.randn(1, T, H))
    assert out.shape == (1, T, H) and torch.isfinite(out).all()

    adapter.set_num_experts(E // 2)
    assert adapter.num_experts == E // 2, "config not updated"
    print(f"  {kind:<6} ok ({adapter.name})")


def test_fused_softmax():
    _run("fused", Config(), "qwen")


def test_list_sigmoid():
    cfg = Config(n_routed_experts=E, scoring_func="sigmoid",
                 routed_scaling_factor=1.0, n_group=1, topk_group=1)
    _run("list", cfg, "deepseek")


def test_hunyuan_nested_router():
    cfg = Config(
        model_type="hy_v3", scoring_func="sigmoid",
        num_experts=E,
        moe_topk=K,
        num_experts_per_tok=K,
        first_k_dense_replace=1,
    )
    # auto-detection must land on the hunyuan adapter, not the generic one
    assert get_adapter(cfg).name == "hunyuan", "did not dispatch to hunyuan"
    _run("nested", cfg, "hunyuan", dense_first=True, scaling=1.0)


def test_hunyuan_reads_per_layer_top_k():
    """``moe_topk`` may be a per-layer list; a uniform one is the top_k."""
    torch.manual_seed(0)
    block = NestedRouterMoE()

    def config(moe_topk):
        # num_experts_per_tok outranks moe_topk, so it must be absent for the
        # per-layer spelling to be exercised at all
        cfg = Config(model_type="hy_v3", num_experts=E,
                     moe_topk=moe_topk)
        del cfg.num_experts_per_tok
        return cfg

    adapter = get_adapter(config([K] * 3))
    assert adapter.top_k == K, "uniform per-layer moe_topk not reduced"
    spec = adapter.router_spec(block)
    assert spec.kind == "sigmoid" and spec.normalize is True
    assert spec.scaling == 1.0
    # the router projection is block.router.gate, one level deeper than usual
    assert adapter.gate_module(block) is block.router.gate

    # a nonuniform list cannot be expressed as a single top_k
    try:
        get_adapter(config([K, K + 1, K])).top_k
    except NotImplementedError as e:
        assert "nonuniform" in str(e)
    else:
        raise AssertionError("accepted a nonuniform per-layer top_k")
    print("  hunyuan per-layer top_k ok")


def test_generic_adapter_reads_public_routing_fields():
    """The opt-in adapter must read the documented spellings, not guess."""
    torch.manual_seed(0)
    block = NestedRouterMoE()

    spec = get_adapter(Config(
        num_experts=E, num_experts_per_tok=K, scoring_func="sigmoid",
        norm_topk_prob=False, routed_scaling_factor=SCALED_LAM,
    ), name="generic").router_spec(block)
    assert spec.kind == "sigmoid"
    assert spec.scaling == SCALED_LAM, "routed_scaling_factor not picked up"
    assert spec.normalize is False, "norm_topk_prob not picked up"
    assert spec.correction_bias is not None, "bias not found on the block"
    assert spec.correction_bias.shape == (E,)

    # a softmax-scored config must not be forced to sigmoid
    spec2 = get_adapter(Config(
        num_experts=E, num_experts_per_tok=K, scoring_func="softmax",
    ), name="generic").router_spec(block)
    assert spec2.kind == "softmax", "scoring_func=softmax ignored"
    assert spec2.scaling == 1.0 and spec2.normalize is True

    # an unknown scorer must be refused rather than silently approximated
    try:
        get_adapter(Config(num_experts=E, num_experts_per_tok=K,
                           scoring_func="mystery"),
                    name="generic").router_spec(block)
    except NotImplementedError as e:
        assert "scoring_func" in str(e)
    else:
        raise AssertionError("accepted an unknown scoring_func")
    print("  generic adapter reads public routing fields")


def test_unobserved_layer_fallback():
    torch.manual_seed(0)
    E_, K_ = E, E // 2
    # three observed layers, plus an MTP-style layer with no observations
    saliency = {
        f"main_{i}": {
            "expert_frequency": torch.randint(1, 99, (E_,)),
            "ean_mean": torch.rand(E_),
            "reap": torch.rand(E_),
            "counterfactual_delta_mean": torch.rand(E_),
        }
        for i in range(3)
    }
    keep = select_keep_indices(saliency, "rcs-loo", K_, aggregation="mean")
    tags = list(keep) + ["mtp_0"]

    filled = fill_unobserved_layers(keep, tags, saliency, "rcs-loo", K_,
                                    verbose=False, aggregation="mean")
    assert "mtp_0" in filled, "unobserved layer was skipped"
    assert len(filled["mtp_0"]) == K_
    assert filled["mtp_0"] == sorted(filled["mtp_0"])
    # observed layers must be left exactly as they were
    for t in keep:
        assert filled[t] == keep[t], f"{t} was modified by the fallback"

    # the proxy must be the mean over observed layers, not an arbitrary set
    mean = torch.stack([
        metrics.score(r, "rcs-loo", aggregation="mean").double() for r in saliency.values()
    ]).mean(dim=0)
    assert filled["mtp_0"] == sorted(torch.topk(mean, K_).indices.tolist())

    # with nothing to average, it must refuse rather than invent a keep-set
    try:
        fill_unobserved_layers({}, ["mtp_0"], None, "rcs-loo", K_, verbose=False)
    except KeyError as e:
        assert "no saliency" in str(e).lower() or "no keep-set" in str(e).lower()
    else:
        raise AssertionError("fabricated a keep-set with no saliency")
    print("  unobserved-layer fallback ok")


def test_counterfactual_matches_a_recomputed_leave_one_out():
    torch.manual_seed(0)
    cfg = Config(
        num_experts=E, num_experts_per_tok=K, scoring_func="sigmoid",
        norm_topk_prob=True, routed_scaling_factor=SCALED_LAM,
    )
    block = NestedRouterMoE().eval()
    adapter = get_adapter(cfg, name="generic")
    spec = adapter.router_spec(block)
    lam = spec.scaling
    assert lam == SCALED_LAM

    x = torch.randn(T, H)
    with torch.no_grad():
        sel, w = adapter.route(x, adapter.gate_module(block).weight, spec)
        u = w / lam
        nz = u[u > 0]
        assert float(nz.max()) < 1.0, \
            "normalized gate reached 1; u is not a probability"
        acts = torch.stack([block.experts[e](x) for e in range(E)], 0).float()
    routed = torch.zeros(E, T, dtype=torch.bool)
    routed.scatter_(0, sel.t().contiguous(), True)
    y = ((u * routed.float()).unsqueeze(-1) * acts).sum(0)

    worst = 0.0
    for t in range(T):
        picks = sel[t].tolist()
        full = lam * sum(u[j, t] * acts[j, t] for j in picks)
        for i in picks:
            ui = u[i, t]
            loo = lam * sum((u[j, t] / (1 - ui)) * acts[j, t]
                            for j in picks if j != i)
            truth = float(torch.linalg.norm(full - loo))
            pred = float(lam * ui / (1 - ui)
                         * torch.linalg.norm(acts[i, t] - y[t]))
            worst = max(worst, abs(truth - pred) / max(truth, 1e-9))
    assert worst < 1e-4, f"closed form is off by {worst:.3e} relative"
    print(f"  leave-one-out closed form ok (max rel err {worst:.1e})")


def test_scaled_router_keeps_the_score_finite_and_sane():
    torch.manual_seed(0)
    cfg = Config(
        num_experts=E, num_experts_per_tok=K, scoring_func="sigmoid",
        norm_topk_prob=True, routed_scaling_factor=SCALED_LAM,
    )
    model = TinyMoE("nested", cfg).eval()
    adapter = get_adapter(cfg, name="generic")
    collector = SaliencyCollector(adapter, expert_chunk=3).install(model)
    with torch.no_grad():
        model(torch.randn(1, T, H))
    collector.remove()
    rec = collector.finalize()["main_0"]

    razor = metrics.score(rec, "razor")
    reap = metrics.score(rec, "reap")
    assert torch.isfinite(razor).all() and (razor >= 0).all()
    assert rec["router_scaling"] == SCALED_LAM, "scale not recorded"
    ratio = float(razor.sum() / reap.sum())
    assert 0.1 < ratio < 100.0, \
        f"razor/reap magnitude ratio is {ratio:.3e}; the gate space is wrong"
    print(f"  scaled router ok (razor/reap = {ratio:.2f})")


def test_gate_above_one_is_refused():
    torch.manual_seed(0)
    cfg = Config(num_experts=E, num_experts_per_tok=K,
                 scoring_func="sigmoid", norm_topk_prob=True)
    model = TinyMoE("nested", cfg).eval()
    adapter = get_adapter(cfg, name="generic")
    assert adapter.router_spec(adapter.moe_blocks(model)[0].module).scaling == 1.0

    class Liar(type(adapter)):  # type: ignore[misc]
        def route(self, hidden, gate_weight, spec):  # noqa: D102
            sel, w = type(adapter).route(hidden, gate_weight, spec)
            return sel, w * SCALED_LAM     # gates now sum to 3, not 1

    liar = Liar(cfg)
    liar.name = adapter.name
    collector = SaliencyCollector(liar, expert_chunk=3).install(model)
    try:
        with torch.no_grad():
            model(torch.randn(1, T, H))
    except RuntimeError as e:
        assert "normalized" in str(e) and "weight" in str(e)
        print("  impossible gate weight refused")
    else:
        raise AssertionError("collected scores from gates that are not probabilities")
    finally:
        collector.remove()


def test_non_renormalizing_router_uses_the_other_closed_form():
    torch.manual_seed(0)

    worst_rel = 0.0
    for s in range(50):
        g = torch.Generator().manual_seed(s)
        u = torch.rand(2, generator=g) * 0.5 + 0.2      # not renormalized
        f = torch.randn(2, 8, generator=g)
        y = (u.unsqueeze(-1) * f).sum(0)
        truth = float(torch.linalg.norm(y - u[1] * f[1]))
        assert abs(truth - float(u[0] * torch.linalg.norm(f[0]))) < 1e-4, \
            "lam*u*||f_i|| is not the delta for a non-renormalizing router"
        wrong = float(u[0] / (1 - u[0]) * torch.linalg.norm(f[0] - y))
        worst_rel = max(worst_rel, abs(wrong - truth) / max(truth, 1e-9))
    assert worst_rel > 0.2, \
        f"premise: the two forms should differ materially (got {worst_rel:.3f})"

    # and in the collector
    scores = {}
    for norm in (True, False):
        torch.manual_seed(0)
        cfg = Config(norm_topk_prob=norm)
        model = TinyMoE("fused", cfg).eval()
        adapter = get_adapter(cfg, name="qwen")
        collector = SaliencyCollector(adapter, expert_chunk=3).install(model)
        with torch.no_grad():
            model(torch.randn(1, T, H))
        collector.remove()
        rec = collector.finalize()["main_0"]
        assert rec["renormalized"] is norm, "renormalized flag not recorded"
        scores[norm] = (metrics.score(rec, "rcs-loo"), metrics.score(rec, "reap"))
        # Refill is served under both gate semantics; its record says which.
        assert rec["refill_supported"] is True
        assert rec["counterfactual_semantics"] == ("fixed_selected_set" if norm else "frozen_weights")
        assert torch.isfinite(metrics.score(rec, "razor")).all()

    razor_no, reap_no = scores[False]
    assert torch.allclose(razor_no, reap_no), \
        "without renormalization the deletion score must reduce to REAP exactly"
    razor_yes, reap_yes = scores[True]
    assert not torch.allclose(razor_yes, reap_yes), \
        "with renormalization the two must differ, or the factor is dropped"
    print("  non-renormalizing router reduces RCS-LOO to REAP")


def test_top1_routing_preserves_baselines_without_razor():
    import pytest

    torch.manual_seed(0)
    cfg = Config(num_experts_per_tok=1)
    model = TinyMoE("fused", cfg).eval()
    adapter = get_adapter(cfg, name="qwen")
    assert adapter.top_k == model.model.layers[0].mlp.config.num_experts_per_tok == 1
    collector = SaliencyCollector(adapter, expert_chunk=3).install(model)
    try:
        with torch.no_grad():
            model(torch.randn(1, T, H))
    finally:
        collector.remove()
    saliency = collector.finalize()
    for rec in saliency.values():
        assert metrics.available(rec) == ["reap", "ean", "frequency"]
        assert int(rec["expert_frequency"].sum()) == T
        assert "counterfactual_delta_mean" not in rec
        assert "counterfactual_delta_sum" not in rec
        with pytest.raises(KeyError):
            metrics.score(rec, "razor")
        for method in ("reap", "ean", "frequency"):
            assert len(metrics.keep_indices(rec, method, E // 2)) == E // 2
    assert all(not b.module._forward_hooks for b in adapter.moe_blocks(model))


def test_constant_scores_have_deterministic_keep_sets():
    saliency = {
        f"main_{i}": {"expert_frequency": torch.full((E,), 2), "total_tokens": E}
        for i in range(3)
    }
    keep = select_keep_indices(saliency, "frequency", E // 2, verbose=False)
    assert all(indices == list(range(E // 2)) for indices in keep.values())
    rec = {"reap": torch.tensor([2.0, 3.0, 3.0, 1.0])}
    assert metrics.keep_indices(rec, "reap", 1, aggregation="mean") == [1]
    assert metrics.keep_indices(rec, "reap", 2, aggregation="mean") == [1, 2]


def test_an_all_padding_run_cannot_produce_a_prune():
    """The end-to-end version of the same guard."""
    from razor.prune import select_keep_indices

    torch.manual_seed(0)
    cfg = Config()
    model = TinyMoE("fused", cfg).eval()
    adapter = get_adapter(cfg, name="qwen")
    collector = SaliencyCollector(adapter, expert_chunk=3).install(model)
    collector.set_attention_mask(torch.zeros(1, T, dtype=torch.long))
    with torch.no_grad():
        model(torch.randn(1, T, H))
    collector.remove()
    sal = collector.finalize()

    assert sal["main_0"]["total_tokens"] == 0, "padding counted as real tokens"
    assert torch.isfinite(metrics.score(sal["main_0"], "razor")).all()
    try:
        select_keep_indices(sal, "razor", E // 2, verbose=False)
    except ValueError as e:
        assert "token" in str(e).lower()
        print("  an all-padding run cannot produce a keep-set")
    else:
        raise AssertionError("pruned from a pack that scored nothing")


def test_keep_indices_validates_the_budget():
    rec = {"counterfactual_delta_mean": torch.rand(E)}
    for bad in (-1, 0):
        try:
            metrics.keep_indices(rec, "rcs-loo", bad, aggregation="mean")
        except ValueError as e:
            assert "at least 1" in str(e)
        else:
            raise AssertionError(f"target_experts={bad} accepted")
    try:
        metrics.keep_indices(rec, "rcs-loo", E + 1, aggregation="mean")
    except ValueError as e:
        assert "exceeds" in str(e)
    else:
        raise AssertionError("target_experts above E accepted")
    assert len(metrics.keep_indices(rec, "rcs-loo", 1, aggregation="mean")) == 1
    assert len(metrics.keep_indices(rec, "rcs-loo", E, aggregation="mean")) == E


def test_documented_package_paths_are_importable():
    import razor

    assert callable(razor.prune), "the one-call API must stay callable"
    try:
        razor.prune.prune_model
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "razor.prune resolved to the module; the docs assume the function")

    from razor.prune import (build_info, fill_unobserved_layers,  # noqa: F401
                             prune_model, resolve_target_experts,
                             save_pruned, select_keep_indices)
    from razor.calibration import build_batches  # noqa: F401
    from razor.geometry import (NON_ADDITIVE_CONST,  # noqa: F401
                                NON_ADDITIVE_MAX, merge_shards, pack_rows,
                                shard_balance, suggest_num_shards)
    from razor.adapters.deepseek import SCORING_FUNCS
    assert set(SCORING_FUNCS.values()) == {"sigmoid", "softmax",
                                           "sqrt_softplus"}
    print("  documented package paths import cleanly")


def test_public_calibration_name():
    from pathlib import Path
    from razor.cli import build_parser
    from razor.pipeline import DEFAULT_DATA

    assert Path(DEFAULT_DATA).name == "RazorCal.json"
    args = build_parser().parse_args(["saliency", "--model", "sample/model", "--out", "out"])
    assert args.data == DEFAULT_DATA


def test_public_corpus_metadata_contract():
    import json
    import re
    from collections import Counter
    from pathlib import Path
    import pytest

    data_dir = Path(__file__).resolve().parents[1] / "data"
    path = data_dir / "RazorCal.json"
    if not path.exists():
        pytest.skip("calibration corpus is supplied separately from the Python package")
    if path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/"):
        pytest.skip("calibration corpus requires Git LFS")
    assert sorted(p.name for p in data_dir.glob("RazorCal*.json")) == ["RazorCal.json"]
    corpus = json.loads(path.read_text(encoding="utf-8"))
    assert set(corpus) == {"meta", "data"}
    meta = corpus["meta"]
    assert set(meta) == {"name", "description", "total_samples", "max_length_per_sample",
                         "domains", "coding_subclass_distribution", "license", "homepage",
                         "privacy_processing"}
    assert meta["name"] == "RazorCal"
    assert meta["total_samples"] == len(corpus["data"]) == 2048
    assert meta["domains"] == dict(Counter(record["domain"] for record in corpus["data"]))
    assert all("reasoning_content" not in message
               for record in corpus["data"] for message in record["messages"])
    public_labels = {
        "Dolci-Precise-IF", "Dolci-Python-Algorithms",
        "Nemotron-Agentic-tool-calling-v2", "Nemotron-OpenCode-general",
        "Nemotron-SFT-SWE-V2-agentless", "Nemotron-Science-V1-MCQ",
        "Nemotron-Science-V1-RQA", "OpenHermes-2.5",
        "competitive-programming-cpp", "competitive-programming-python",
        "multilingual-chinese-code", "multilingual-chinese-math",
        "multilingual-chinese-stem", "nemotron-cascade2-math",
        "nemotron-math-v2-high-aime-style",
    }
    assert {record["sub_source"] for record in corpus["data"]} == public_labels
    def strings(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, str):
            yield value

    for text in strings(corpus["data"]):
        assert not re.search(
            r"/home/[^\s\"']+|\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}|"
            r"192\.168\.\d{1,3}\.\d{1,3}|"
            r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b", text)
        assert not re.search(
            r"\b[A-Z0-9._%+-]+@(?!example\.com\b)[A-Z0-9.-]+\.[A-Z]{2,}\b",
            text, re.IGNORECASE)


def test_default_calibration_set_fails_clearly_when_absent():
    import tempfile
    from pathlib import Path as P

    from razor.pipeline import DEFAULT_DATA, resolve_data

    # passthrough cases
    assert resolve_data("hf:allenai/c4") == "hf:allenai/c4"
    with tempfile.TemporaryDirectory() as d:
        assert resolve_data(d) == d

        # an lfs pointer is not the data
        stub = P(d) / "RazorCal.json"
        stub.write_bytes(
            b"version https://git-lfs.github.com/spec/v1\n"
            b"oid sha256:0000000000000000000000000000000000000000000000000000"
            b"000000000000\nsize 33554432\n")
        try:
            resolve_data(str(stub))
        except FileNotFoundError as e:
            assert "git lfs pull" in str(e) and "pointer" in str(e)
        else:
            raise AssertionError("accepted an lfs pointer as calibration data")

    # the default path, when it is not there, must name the two ways out
    missing = str(P(tempfile.gettempdir()) / "definitely_not_here.json")
    try:
        resolve_data(missing)
    except FileNotFoundError as e:
        assert "missing" in str(e).lower()
        assert "git lfs pull" in str(e)
        assert missing not in str(e)
    else:
        raise AssertionError("accepted a missing calibration set")

    # and in a source checkout the default must actually resolve
    if P(DEFAULT_DATA).exists():
        assert resolve_data(None) == DEFAULT_DATA
    print("  calibration-set resolution reports both failure modes")


def test_generic_dispatch():
    import pytest

    cfg = Config(model_type="unknown_moe")
    with pytest.raises(NotImplementedError, match="unsupported model_type"):
        get_adapter(cfg)
    adapter = get_adapter(cfg, name="generic")
    assert adapter.name == "generic"
    assert adapter.num_experts == E


def test_metric_errors():
    """Bad input should fail loudly and say what was expected."""
    try:
        metrics.score({}, "nonsense")
    except ValueError as e:
        assert "unknown method" in str(e)
    else:
        raise AssertionError("unknown method silently accepted")

    try:
        metrics.score({"reap": torch.zeros(E)}, "ean")
    except KeyError as e:
        assert "ean_square_sum" in str(e)
    else:
        raise AssertionError("missing field silently accepted")

    summed = {"counterfactual_delta_sum": torch.arange(E, dtype=torch.float64),
              "expert_frequency": torch.full((E,), 2, dtype=torch.long)}
    got = metrics.score(summed, "rcs-loo", aggregation="mean")
    assert torch.allclose(got, torch.arange(E, dtype=torch.float32) / 2)

    for bad in (0, E + 1):
        try:
            resolve_target_experts(E, bad, None)
        except ValueError:
            pass
        else:
            raise AssertionError(f"target_experts={bad} silently accepted")


def test_metric_validation_and_mean_fallback():
    import pytest

    for value in (torch.tensor(1.0), torch.empty(0), torch.ones(2, 2),
                  torch.tensor([float("nan")]), torch.tensor([float("inf")]),
                  torch.tensor([-1.0]), torch.tensor([1e100], dtype=torch.float64),
                  torch.tensor([1 + 2j]), [1.0, 2.0]):
        with pytest.raises(ValueError):
            metrics.score({"reap": value}, "reap", aggregation="mean")
    rec = {"counterfactual_delta_sum": torch.tensor([6.0, 0.0]),
           "routed_count": torch.tensor([2.0, 0.0]),
           "expert_frequency": torch.tensor([3, 0])}
    assert metrics.score(rec, "rcs-loo", aggregation="mean").tolist() == [3.0, 0.0]
    del rec["expert_frequency"]
    assert metrics.score(rec, "rcs-loo", aggregation="mean").tolist() == [3.0, 0.0]
    rec["routed_count"] = torch.ones(1)
    with pytest.raises(ValueError, match="shape"):
        metrics.score(rec, "rcs-loo", aggregation="mean")
    rec["routed_count"] = torch.zeros(2)
    with pytest.raises(ValueError, match="zero routed count"):
        metrics.score(rec, "rcs-loo", aggregation="mean")
    for extra in ({"top_k": 1}, {"razor_supported": False}):
        for name in ("rcs-loo", "razor"):
            with pytest.raises(KeyError):
                metrics.score({"counterfactual_delta_mean": torch.ones(2), **extra}, name)


def test_score_pack_preserves_routing_guards(tmp_path):
    import pytest
    from razor.pipeline import _load_saliency, _select_stored

    path = tmp_path / "scores.pt"
    for wrapped in (True, False):
        for guard in ({"razor_supported": False}, {"top_k": 1}, {"pruning_policy": "preserve"}):
            rec = {"score_space": "latent", "counterfactual_semantics": "frozen_weights",
                   "reap_sum": torch.arange(4).double(), **guard}
            if wrapped:
                rec.update(count=torch.ones(4), d1_sum=torch.ones(4), d1_sq=torch.ones(4))
                content = {"state": {"main_0": rec}}
            else:
                rec.update(routed_count=torch.ones(4), counterfactual_delta_mean=torch.ones(4),
                           counterfactual_delta_sum=torch.ones(4))
                content = {"main_0": rec}
            torch.save(content, path)
            loaded = _load_saliency(path)["main_0"]
            assert all(loaded[k] == v for k, v in guard.items())
            assert loaded["score_space"] == "latent"
            assert loaded["counterfactual_semantics"] == "frozen_weights"
            with pytest.raises(KeyError):
                metrics.score(loaded, "razor")
            assert not any(key.startswith("counterfactual_delta_") for key in loaded)
    rec = {"count": torch.ones(4), "d1_sum": torch.ones(4), "top_k": 1}
    torch.save({"state": {"main_0": rec}, "top_k": 2}, path)
    with pytest.raises(ValueError, match="top_k"):
        _load_saliency(path)
    sal = {"main_0": {"routed_count": torch.ones(4), "counterfactual_delta_sum": torch.ones(4)},
           "main_1": {"routed_count": torch.zeros(4), "counterfactual_delta_sum": torch.zeros(4)}}
    meta = {"layers": ["main_0", "main_1"], "counts": {"main_0": 4, "main_1": 4},
            "hash_layers": [], "mtp_layers": [], "n_group": 1}
    with pytest.raises(ValueError, match="observed"):
        _select_stored(sal, meta, "rcs-loo", 2, "mean")


def test_baseline_only_additive_packs_do_not_require_counterfactuals(tmp_path):
    import pytest
    from razor.pipeline import _load_saliency

    path = tmp_path / "baseline.pt"
    for moments in ({}, {"reap_sum": torch.tensor([2., 3.]), "ean_sum": torch.tensor([4., 5.]),
                         "reap_sq": torch.tensor([2., 9.]), "ean_sq": torch.tensor([8., 25.])}):
        torch.save({"top_k": 1, "state": {"main_0": {"count": torch.tensor([2., 1.]), **moments}}}, path)
        rec = _load_saliency(path)["main_0"]
        frequency = metrics.score(rec, "frequency")
        assert frequency.tolist() == [2, 1] and not frequency.is_floating_point()
        with pytest.raises(KeyError):
            metrics.score(rec, "razor")
        if moments:
            for method, expected in (("reap", [1., 3.]), ("ean", [2., 5.])):
                for aggregation in ("mean", "rms"):
                    torch.testing.assert_close(metrics.score(rec, method, aggregation), torch.tensor(expected))


def test_saved_scores_cannot_bypass_count_and_moment_validation():
    import pytest

    for method, mean, total, square, rms in (
        ("rcs-loo", "counterfactual_delta_mean", "counterfactual_delta_sum",
         "counterfactual_delta_square_sum", "counterfactual_delta_rms"),
        ("rcs-refill", "rcs_refill_mean", "rcs_refill_sum",
         "rcs_refill_square_sum", "rcs_refill_rms"),
        ("rcs", "rcs_mean", "rcs_sum", "rcs_square_sum", "rcs_rms"),
        ("reap", "reap", "weighted_ean_sum", "weighted_ean_square_sum", "reap_rms"),
        ("ean", "ean_mean", "ean_sum", "ean_square_sum", "ean_rms"),
    ):
        rec = {"routed_count": torch.tensor([0., 1.]), "expert_frequency": torch.tensor([0, 1]),
               total: torch.tensor([10., 1.]), square: torch.tensor([100., 1.]),
               mean: torch.tensor([10., 1.]), rms: torch.tensor([10., 1.])}
        for aggregation in ("mean", "rms", "sum"):
            with pytest.raises(ValueError, match="zero routed count"):
                metrics.keep_indices(rec, method, 1, aggregation)
        for field in (total, square, mean, rms):
            rec[field][0] = 0.
        for aggregation in ("mean", "rms", "sum"):
            assert metrics.keep_indices(rec, method, 1, aggregation) == [1]
        rec["routed_count"] = torch.tensor([0., .5])
        with pytest.raises(ValueError, match="integer"):
            metrics.score(rec, method)
        rec["routed_count"] = torch.tensor([0., 1.])
        rec[mean][1] = 3.
        with pytest.raises(ValueError, match="disagree"):
            metrics.score(rec, method)
    for count in (torch.tensor([True, False]), torch.tensor([.5, 1.])):
        with pytest.raises(ValueError, match="integer"):
            metrics.score({"expert_frequency": count}, "frequency")


def _diagnose_probe(marker):
    """Record execution without side effects beyond a temporary marker file."""
    from pathlib import Path as _Path

    _Path(marker).write_text("executed")
    return {"main_0": {"expert_frequency": torch.tensor([1, 2, 3, 4])}}


def test_large_routing_counts_are_not_rounded_into_ties():
    boundary = 2 ** 24
    counts = torch.tensor([boundary, boundary + 1, boundary + 2], dtype=torch.int64)
    scores = metrics.score({"expert_frequency": counts}, "frequency")
    assert scores.dtype == counts.dtype
    assert scores.tolist() == counts.tolist()
    assert metrics.keep_indices({"expert_frequency": counts}, "frequency", 1) == [2]
    assert metrics.keep_indices({"expert_frequency": counts}, "frequency", 2) == [1, 2]
    tied = torch.tensor([5, 5, 1], dtype=torch.int64)
    assert metrics.keep_indices({"expert_frequency": tied}, "frequency", 1) == [0]


def test_saliency_cache_never_pairs_old_fingerprint_with_new_scores(tmp_path, monkeypatch):
    import pytest
    from razor import loader, pipeline, streaming

    source, data = tmp_path / "model", tmp_path / "data.json"
    source.mkdir()
    data.write_text("[]")
    out = tmp_path / "scores"
    first = {"main_0": {"expert_frequency": torch.tensor([4, 3, 2, 1])}}
    second = {"main_0": {"expert_frequency": torch.tensor([1, 2, 3, 4])}}
    monkeypatch.setattr(pipeline, "_raw_config", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "_local_checkpoint", lambda model: str(source))
    monkeypatch.setattr(loader, "load_tokenizer", lambda *a, **k: object())
    monkeypatch.setattr(pipeline.calibration, "build_batches", lambda *a, **k: [
        {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}])
    options = dict(execution="streaming", stream_device="cpu", verbose=False)

    def collect(scores):
        return lambda *a, **k: scores

    def run(seed=0):
        result = pipeline.collect_saliency(str(source), str(data), str(out), seed=seed, **options)
        return result["main_0"]["expert_frequency"].tolist()

    expected_first = first["main_0"]["expert_frequency"].tolist()
    monkeypatch.setattr(streaming, "collect_streaming", collect(first))
    assert run() == expected_first
    original = Path.write_text

    def refuse_manifest(self, *args, **kwargs):
        if self.name == "collection.json":
            raise PermissionError("synthetic manifest failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(streaming, "collect_streaming", collect(second))
    monkeypatch.setattr(Path, "write_text", refuse_manifest)
    with pytest.raises(PermissionError):
        pipeline.collect_saliency(str(source), str(data), str(out), seed=1, **options)
    monkeypatch.setattr(Path, "write_text", original)
    assert not (out / "collection.json").exists()

    class Recollected(Exception):
        pass

    def require_recollection(*args, **kwargs):
        raise Recollected()

    monkeypatch.setattr(streaming, "collect_streaming", require_recollection)
    with pytest.raises(Recollected):
        run()
    monkeypatch.setattr(streaming, "collect_streaming", collect(first))
    assert run() == expected_first
    monkeypatch.setattr(streaming, "collect_streaming",
                        lambda *a, **k: pytest.fail("verified cache must be reused"))
    assert run() == expected_first


def test_diagnose_reads_scores_as_data_only(tmp_path):
    import pytest
    from razor.diagnose import report

    marker = tmp_path / "executed"

    class Payload:
        def __reduce__(self):
            return _diagnose_probe, (str(marker),)

    path = tmp_path / "scores.pt"
    torch.save(Payload(), path)
    with pytest.raises(SystemExit, match="readable tensor saliency file"):
        report(str(path), keep=2)
    assert not marker.exists()
    torch.save({"main_0": ["not", "a", "record"]}, path)
    with pytest.raises(SystemExit, match="mapping of layer tags"):
        report(str(path), keep=2)
    torch.save({"main_0": {"expert_frequency": torch.tensor([1, 2, 3, 4])}}, path)
    assert report(str(path), keep=2) == {"frequency": {"main_0": [2, 3]}}


def test_calibration_hf_uses_bounded_streaming(monkeypatch):
    import sys
    import types
    import pytest
    from razor.calibration import load_records

    visited = []
    calls = []

    def records():
        for i in range(4):
            visited.append(i)
            yield {"text": str(i)}
        raise AssertionError("read beyond the requested limit")

    def load_dataset(dataset, **kwargs):
        calls.append((dataset, kwargs))
        return records()

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    assert len(load_records("hf:sample/config,en", split="validation", limit=3)) == 3
    assert visited == [0, 1, 2]
    assert calls == [("sample/config", {"split": "validation", "streaming": True, "name": "en"})]
    for limit in (None, 0, -1, True):
        with pytest.raises(ValueError, match="limit"):
            load_records("hf:sample/config", limit=limit)


def test_calibration_rejects_empty_records(tmp_path):
    import pytest
    from razor.calibration import load_records, render

    for text in ("[]", '{"data": []}', '[{}]', '{"data": {}}'):
        path = tmp_path / "empty.json"
        path.write_text(text)
        with pytest.raises(ValueError):
            load_records(str(path))
    with pytest.raises(ValueError, match="no records"):
        render([], object(), verbose=False)
    with pytest.raises(ValueError, match="nonempty"):
        render([{}], object(), verbose=False)


if __name__ == "__main__":
    print("end-to-end on synthetic MoE models")
    test_fused_softmax()
    test_list_sigmoid()
    test_hunyuan_nested_router()
    test_hunyuan_reads_per_layer_top_k()
    test_unobserved_layer_fallback()
    test_generic_dispatch()
    print("  generic dispatch ok")
    test_metric_errors()
    print("  error handling ok")
    print("all passed")
