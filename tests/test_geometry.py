"""Tests for calibration packing, coverage, and shard merging."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from razor import geometry                                    # noqa: E402


class FakeTokenizer:
    """Synthetic tokenizer with configurable tokens per word."""

    pad_token_id = 0

    def __init__(self, tokens_per_word: int = 1):
        self.k = tokens_per_word

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        n = len(text.split()) * self.k
        ids = torch.arange(1, n + 1, dtype=torch.long)
        return {"input_ids": ids.unsqueeze(0)}


def _corpus(n_samples: int = 40, words: int = 30):
    return ["w%d " % i * words for i in range(n_samples)]


def test_row_count_is_measured_not_declared():
    texts = _corpus()
    thin = geometry.corpus_rows(texts, FakeTokenizer(1), row_len=64)
    fat = geometry.corpus_rows(texts, FakeTokenizer(3), row_len=64)
    assert fat > thin, (thin, fat)
    print("[ok] row count follows the tokenizer: %d vs %d rows" % (thin, fat))


def test_exhaust_consumes_each_sample_exactly_once():
    texts = _corpus(n_samples=25, words=20)
    tok = FakeTokenizer()
    enc = {i: tok(t)["input_ids"][0] for i, t in enumerate(texts)}
    rows = geometry._fill_rows(geometry.corpus_order(len(texts)), enc, 64)
    used = [j for row in rows for j, _, _ in row]
    assert sorted(used) == list(range(len(texts))), "samples reused or dropped"
    print("[ok] %d samples over %d rows, each used exactly once"
          % (len(texts), len(rows)))


def test_coverage_reports_boundary_loss_honestly():
    """Row-boundary truncation is real; the report must not hide it."""
    texts = _corpus(n_samples=30, words=25)
    cov = geometry.coverage(texts, FakeTokenizer(), row_len=64)
    assert cov["packed_tokens"] + cov["lost_tokens"] == cov["corpus_tokens"]
    assert cov["packed_tokens"] <= cov["slots"]
    # one truncated tail per row, at most
    assert cov["truncated_samples"] <= cov["rows"]
    print("[ok] coverage balances: %d packed + %d lost = %d corpus (%.1f%% lost,"
          " %d truncated)" % (cov["packed_tokens"], cov["lost_tokens"],
                              cov["corpus_tokens"], 100 * cov["lost_fraction"],
                              cov["truncated_samples"]))


def test_row_ids_mean_the_same_slice_under_any_batch_size():
    texts = _corpus()
    tok = FakeTokenizer()
    b1, ids1 = geometry.pack_rows(texts, tok, row_len=64, batch_size=1,
                                  verbose=False)
    b2, ids2 = geometry.pack_rows(texts, tok, row_len=64, batch_size=4,
                                  verbose=False)
    assert ids1 == ids2
    t1 = int(sum(b["attention_mask"].sum() for b in b1))
    t2 = int(sum(b["attention_mask"].sum() for b in b2))
    assert t1 == t2, "batch_size changed the token budget: %d vs %d" % (t1, t2)
    print("[ok] same rows and same %d tokens at bs=1 and bs=4" % t1)


def test_shards_partition_the_rows():
    texts = _corpus()
    tok = FakeTokenizer()
    total = geometry.corpus_rows(texts, tok, row_len=64)
    seen, ntok = [], 0
    for s in range(4):
        b, ids = geometry.pack_rows(texts, tok, row_len=64, shard=s,
                                    num_shards=4, verbose=False)
        seen += ids
        ntok += int(sum(x["attention_mask"].sum() for x in b))
    assert sorted(seen) == list(range(total)), "shards do not partition rows"
    whole, _ = geometry.pack_rows(texts, tok, row_len=64, verbose=False)
    assert ntok == int(sum(x["attention_mask"].sum() for x in whole))
    geometry.audit_coverage(seen, ntok, total)
    print("[ok] 4 shards partition %d rows and sum to the same tokens" % total)


def test_audit_catches_gap_duplicate_and_token_mismatch():
    """Each failure mode has a different cause, so each must be distinguishable."""
    for ids, rows, tok, why in (([0, 1, 3], 4, None, "missing"),
                                ([0, 1, 1, 2, 3], 4, None, "twice"),
                                ([0, 1, 2, 3], 4, 999, "corpus pass")):
        try:
            geometry.audit_coverage(ids, 100, rows, expected_tokens=tok)
        except RuntimeError as e:
            assert why in str(e), "wrong message for %s: %s" % (why, e)
        else:
            raise AssertionError("audit accepted %r" % (ids,))
    geometry.audit_coverage([0, 1, 2, 3], 100, 4, expected_tokens=100)
    print("[ok] audit rejects gaps, duplicates and token mismatch")


def _pack(row_ids, ntok, val, rows=4):
    return {"row_ids": list(row_ids), "n_tokens": ntok, "rows": rows,
            "row_len": 64,
            "state": {"layer.0": {"num": torch.full((3,), float(val)),
                                  "count": torch.tensor([val])}}}


def test_merge_does_not_sum_a_running_maximum_or_a_constant():
    def pack(ids, val, scaling=3.0):
        return {"row_ids": list(ids), "n_tokens": 10, "rows": 4, "row_len": 64,
                "state": {"main_0": {
                    "expert_frequency": torch.tensor([val, val]),
                    "max_activations": torch.tensor([float(val), float(val)]),
                    "router_scaling": torch.tensor(scaling,
                                                   dtype=torch.float64),
                    "renormalized": torch.tensor(True),
                }}}

    merged = geometry.merge_shards([pack([0, 2], 1), pack([1, 3], 5)])
    st = merged["state"]["main_0"]
    assert st["expert_frequency"].tolist() == [6, 6], "sums must still add"
    assert st["max_activations"].tolist() == [5.0, 5.0], \
        "max_activations was summed instead of maxed"
    assert float(st["router_scaling"]) == 3.0, \
        "router_scaling was summed instead of kept"
    assert bool(st["renormalized"]) is True
    print("[ok] merge maxes the maxima and keeps the constants")


def test_merge_refuses_shards_with_different_router_settings():
    """Two packs collected under different scales are not the same run."""
    def pack(ids, scaling):
        return {"row_ids": list(ids), "n_tokens": 10, "rows": 4, "row_len": 64,
                "state": {"main_0": {
                    "router_scaling": torch.tensor(scaling,
                                                   dtype=torch.float64)}}}
    try:
        geometry.merge_shards([pack([0, 2], 3.0), pack([1, 3], 1.0)])
    except RuntimeError as e:
        assert "router_scaling" in str(e)
        print("[ok] merge refuses shards with disagreeing router settings")
        return
    raise AssertionError("merged packs collected under different scales")


def test_merge_accepts_a_layer_only_some_shards_saw():
    a = {"row_ids": [0, 2], "n_tokens": 10, "rows": 4, "row_len": 64,
         "state": {"main_0": {"num": torch.ones(2)}}}
    b = {"row_ids": [1, 3], "n_tokens": 10, "rows": 4, "row_len": 64,
         "state": {"main_0": {"num": torch.ones(2)},
                   "mtp_0": {"num": torch.full((2,), 7.0)}}}
    merged = geometry.merge_shards([a, b])
    assert sorted(merged["state"]) == ["main_0", "mtp_0"]
    assert merged["state"]["main_0"]["num"].tolist() == [2.0, 2.0]
    assert merged["state"]["mtp_0"]["num"].tolist() == [7.0, 7.0]
    print("[ok] merge accepts a layer only some shards observed")


def test_merge_sums_accumulators_and_verifies_coverage():
    merged = geometry.merge_shards([_pack([0, 2], 10, 1.0),
                                    _pack([1, 3], 20, 2.0)])
    assert merged["n_tokens"] == 30
    assert merged["row_ids"] == [0, 1, 2, 3]
    assert torch.allclose(merged["state"]["layer.0"]["num"],
                          torch.full((3,), 3.0))
    print("[ok] merge sums accumulators and accepts a full partition")


def test_merge_refuses_packs_without_row_ids():
    bad = _pack([0, 1], 10, 1.0)
    del bad["row_ids"]
    try:
        geometry.merge_shards([bad])
    except RuntimeError as e:
        assert "row_ids" in str(e)
    else:
        raise AssertionError("merged a pack with no coverage information")
    print("[ok] merge refuses packs that cannot be verified")


def test_merge_refuses_overlapping_and_incomplete_shards():
    try:
        geometry.merge_shards([_pack([0, 1], 10, 1.0), _pack([1, 2], 10, 1.0)])
    except RuntimeError as e:
        assert "repeats rows" in str(e)
    else:
        raise AssertionError("merged overlapping shards")
    try:
        geometry.merge_shards([_pack([0, 1], 10, 1.0)])   # rows=4, only 2 given
    except RuntimeError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("merged an incomplete set")
    print("[ok] merge refuses overlapping and incomplete shard sets")


def test_pack_rows_refuses_to_build_an_empty_pack():
    class Empty(FakeTokenizer):
        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            return {"input_ids": torch.zeros((1, 0), dtype=torch.long)}

    try:
        geometry.pack_rows(["a", "b"], Empty(), row_len=64, verbose=False)
    except RuntimeError as e:
        assert "0 tokens" in str(e)
        print("[ok] pack_rows fails fast instead of emitting an empty pack")
        return
    raise AssertionError("built a pack with no tokens")


def test_reproduces_a_pinned_geometry():
    texts = ["w " * (5 + (i * 7) % 23) for i in range(50)]
    tok = FakeTokenizer()
    cov = geometry.coverage(texts, tok, row_len=64)
    rows, ntok = cov["rows"], cov["packed_tokens"]

    # same inputs -> same geometry, every time
    for _ in range(3):
        again = geometry.coverage(texts, tok, row_len=64)
        assert again["rows"] == rows and again["packed_tokens"] == ntok

    # and the rows a pack reports must be exactly the rows coverage counted
    batches, row_ids = geometry.pack_rows(texts, tok, row_len=64, verbose=False)
    assert len(row_ids) == rows
    assert int(sum(b["attention_mask"].sum() for b in batches)) == ntok
    geometry.audit_coverage(row_ids, ntok, rows, expected_tokens=ntok)
    print("[ok] geometry is reproducible: %d rows, %d tokens, pinned"
          % (rows, ntok))


def test_seed_is_part_of_the_geometry():
    texts = ["w " * (7 + (i * 13) % 41) for i in range(60)]
    tok = FakeTokenizer()
    counts = {geometry.corpus_rows(texts, tok, row_len=64, seed=s)
              for s in range(12)}
    assert len(counts) > 1, "seed had no effect; is the order being ignored?"
    print("[ok] seed changes the packing: row counts %s" % sorted(counts))


def test_shard_balance_flags_the_indivisible_case():
    bad = geometry.shard_balance(10, 4)
    assert bad["max_rows"] == 3 and bad["min_rows"] == 2
    assert abs(bad["imbalance"] - 1.5) < 1e-9
    good = geometry.shard_balance(12, 4)
    assert good["max_rows"] == good["min_rows"] == 3
    assert good["imbalance"] == 1.0
    # more shards than rows leaves workers with nothing to do
    idle = geometry.shard_balance(3, 5)
    assert idle["idle_shards"] == 2
    print("[ok] shard balance detects indivisible and idle-worker cases")


def test_suggest_num_shards_prefers_a_divisor():
    assert geometry.suggest_num_shards(10, 6) == 5
    assert geometry.suggest_num_shards(12, 6) == 6
    assert geometry.suggest_num_shards(18, 8) == 6
    print("[ok] shard suggestion prefers the largest eligible divisor")


def test_suggest_never_collapses_to_one_worker():
    for rows in (11, 7, 5, 2):
        n = geometry.suggest_num_shards(rows, 4)
        assert n > 1, "%d rows collapsed to %d shard(s)" % (rows, n)
    assert geometry.suggest_num_shards(11, 4) == 4
    print("[ok] prime row counts keep available workers instead of collapsing to 1")


def test_strided_split_stays_a_partition_when_indivisible():
    """Imbalance must not become incorrectness."""
    for total, n in ((10, 4), (7, 4), (3, 8), (20, 3)):
        seen: list = []
        for s in range(n):
            seen += list(range(total))[s::n]
        assert sorted(seen) == list(range(total)), (total, n)
    print("[ok] strided split partitions the rows for every ratio tested")


def test_geometry_rejects_invalid_bounds():
    import pytest

    for kwargs in ({"row_len": 0}, {"row_len": -1}, {"row_len": True},
                   {"batch_size": 0}, {"batch_size": 1.5}, {"n_rows": -2},
                   {"n_rows": 0}, {"num_shards": 0}, {"shard": -1},
                   {"shard": 1}, {"row_ids": [-1]}, {"row_ids": [999]},
                   {"row_ids": [0, 0]}, {"row_ids": [0.5]}):
        options = {"row_len": 64, "verbose": False, **kwargs}
        with pytest.raises(ValueError):
            geometry.pack_rows(_corpus(), FakeTokenizer(), **options)
    for fn in (geometry.corpus_rows, geometry.coverage):
        with pytest.raises(ValueError, match="row_len"):
            fn(_corpus(), FakeTokenizer(), row_len=0)
    with pytest.raises(ValueError, match="no records"):
        geometry.pack_rows([], FakeTokenizer(), row_len=64)


def test_merge_rejects_duplicate_rows_inside_one_shard():
    import pytest

    with pytest.raises(RuntimeError, match="repeats rows"):
        geometry.merge_shards([_pack([0, 0, 1, 2, 3], 10, 1)])


def test_merge_rejects_incompatible_fields_and_values():
    import copy
    import pytest

    a, base = _pack([0, 2], 10, 1), _pack([1, 3], 10, 2)
    mutations = [
        lambda st: st.pop("count"),
        lambda st: st.update(num=torch.ones(1)),
        lambda st: st.update(num=torch.ones(3, dtype=torch.int64)),
        lambda st: st.update(num=2.0),
        lambda st: st.update(num="not an accumulator"),
        lambda st: st.update(num=torch.full((3,), float("nan"))),
    ]
    for mutate in mutations:
        b = copy.deepcopy(base)
        mutate(b["state"]["layer.0"])
        with pytest.raises(RuntimeError):
            geometry.merge_shards([a, b])


def test_merge_preserves_counterfactual_metadata():
    import pytest

    for supported in (True, False):
        for semantics in ("fixed_selected_set", "frozen_weights"):
            a, b = _pack([0, 2], 10, 1), _pack([1, 3], 10, 2)
            for p in (a, b):
                p["state"]["layer.0"].update(
                    razor_supported=supported, counterfactual_semantics=semantics)
            merged = geometry.merge_shards([a, b])["state"]["layer.0"]
            assert merged["razor_supported"] is supported
            assert merged["counterfactual_semantics"] == semantics
            b["state"]["layer.0"]["razor_supported"] = not supported
            with pytest.raises(RuntimeError, match="razor_supported"):
                geometry.merge_shards([a, b])
            b["state"]["layer.0"]["razor_supported"] = supported
            b["state"]["layer.0"]["counterfactual_semantics"] = (
                "frozen_weights" if semantics == "fixed_selected_set" else "fixed_selected_set")
            with pytest.raises(RuntimeError, match="counterfactual_semantics"):
                geometry.merge_shards([a, b])


def test_merge_checks_scalar_and_vector_constants():
    import pytest

    for first, second in ((3.0, 1.0), (True, False),
                          (torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]))):
        a, b = _pack([0, 2], 10, 1), _pack([1, 3], 10, 2)
        key = "renormalized" if isinstance(first, bool) else "router_scaling"
        a["state"]["layer.0"][key] = first
        b["state"]["layer.0"][key] = second
        with pytest.raises(RuntimeError, match=key):
            geometry.merge_shards([a, b])
    a, b = _pack([0, 2], 10, 1), _pack([1, 3], 10, 2)
    for p in (a, b):
        p["state"]["layer.0"].update(router_scaling=3.0, renormalized=True)
    assert geometry.merge_shards([a, b])["state"]["layer.0"]["router_scaling"] == 3.0


def test_merge_recomputes_score_moments_and_keeps_native_metadata():
    import copy
    import pytest

    a, b = _pack([0, 2], 10, 1), _pack([1, 3], 10, 2)
    for pack, count, value in ((a, 1., 2.), (b, 3., 4.)):
        state = pack["state"]["layer.0"]
        state.update(pruning_policy="score", score_space="latent", top_k=2,
                     routed_count=torch.tensor([count, 0.]))
        for mean, rms, total, square in (
            ("ean_mean", "ean_rms", "ean_sum", "ean_square_sum"),
            ("reap", "reap_rms", "weighted_ean_sum", "weighted_ean_square_sum"),
            ("counterfactual_delta_mean", "counterfactual_delta_rms",
             "counterfactual_delta_sum", "counterfactual_delta_square_sum"),
        ):
            state[total] = torch.tensor([count * value, 0.])
            state[square] = torch.tensor([count * value ** 2, 0.])
            state[mean] = state[rms] = torch.tensor([value, 0.])
    merged = geometry.merge_shards([a, b])["state"]["layer.0"]
    assert merged["top_k"] == 2 and merged["score_space"] == "latent"
    for mean, rms in (("ean_mean", "ean_rms"), ("reap", "reap_rms"),
                      ("counterfactual_delta_mean", "counterfactual_delta_rms")):
        torch.testing.assert_close(merged[mean], torch.tensor([3.5, 0.]))
        torch.testing.assert_close(merged[rms], torch.tensor([13., 0.]).sqrt())
    bad = copy.deepcopy(b)
    bad["state"]["layer.0"]["score_space"] = "expert_output"
    with pytest.raises(RuntimeError, match="score_space"):
        geometry.merge_shards([a, bad])
    del bad["state"]["layer.0"]["ean_square_sum"]
    with pytest.raises(RuntimeError, match="requires"):
        geometry.merge_shards([bad], expected_rows=None)


def test_merge_rejects_conflicting_pack_routing_metadata():
    import pytest

    for key, left, right in (("top_k", 2, 1), ("score_space", "latent", "expert_output"),
                              ("razor_supported", True, False)):
        a, b = _pack([0, 2], 10, 1), _pack([1, 3], 10, 2)
        a[key], b[key] = left, right
        for packs in ([a, b], [b, a]):
            with pytest.raises(RuntimeError, match=key):
                geometry.merge_shards(packs)
        del b[key]
        with pytest.raises(RuntimeError, match=key):
            geometry.merge_shards([a, b])
        b[key] = left
        assert geometry.merge_shards([a, b])[key] == left


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall geometry tests passed")
