"""Fixed-length calibration rows, coverage checks, and shard merging."""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch

#: Passed as ``n_rows`` to mean "as many rows as the corpus fills, once".
EXHAUST = -1

DEFAULT_SEED = 0


def _integer(value, name: str, minimum: int = 0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))


def corpus_order(n: int, seed: int = DEFAULT_SEED) -> List[int]:
    """Return a reproducible sample permutation."""
    _integer(n, "n")
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


def _encode(texts: Sequence[str], tokenizer,
            encoded: Optional[Dict[int, torch.Tensor]] = None):
    if encoded is not None:
        return encoded
    return {i: tokenizer(t, return_tensors="pt",
                         add_special_tokens=False)["input_ids"][0]
            for i, t in enumerate(texts)}


def _fill_rows(order: Sequence[int], enc: Dict[int, torch.Tensor], row_len: int):
    """Pack once with one-token gaps, dropping sample tails at row boundaries."""
    _integer(row_len, "row_len", 1)
    rows: List[List[Tuple[int, int, int]]] = []
    cur = 0
    while cur < len(order):
        at = 0
        row: List[Tuple[int, int, int]] = []
        while at < row_len and cur < len(order):
            j = order[cur]
            n = int(enc[j].numel())
            cur += 1
            end = min(at + n, row_len)
            row.append((j, end - at, n))
            at = end + 1
        rows.append(row)
    return rows


def corpus_rows(texts: Sequence[str], tokenizer, row_len: int,
                seed: int = DEFAULT_SEED,
                encoded: Optional[Dict[int, torch.Tensor]] = None) -> int:
    """Count packed rows for one corpus pass."""
    enc = _encode(texts, tokenizer, encoded)
    return len(_fill_rows(corpus_order(len(texts), seed), enc, row_len))


def coverage(texts: Sequence[str], tokenizer, row_len: int, seed: int = DEFAULT_SEED,
             encoded: Optional[Dict[int, torch.Tensor]] = None) -> dict:
    """Return packed and truncated token counts."""
    enc = _encode(texts, tokenizer, encoded)
    rows = _fill_rows(corpus_order(len(texts), seed), enc, row_len)
    total = sum(int(t.numel()) for t in enc.values())
    packed = sum(taken for row in rows for _, taken, _ in row)
    truncated = sum(1 for row in rows for _, taken, n in row if taken < n)
    return {
        "rows": len(rows),
        "row_len": row_len,
        "slots": len(rows) * row_len,
        "corpus_tokens": total,
        "packed_tokens": packed,
        "lost_tokens": total - packed,
        "lost_fraction": (total - packed) / total if total else 0.0,
        "truncated_samples": truncated,
    }


def shard_balance(total_rows: int, num_shards: int) -> dict:
    """Report minimum/maximum row counts and idle shards."""
    _integer(total_rows, "total_rows")
    _integer(num_shards, "num_shards", 1)
    counts = [len(range(total_rows)[s::num_shards]) for s in range(num_shards)]
    lo, hi = min(counts), max(counts)
    return {
        "rows": total_rows,
        "num_shards": num_shards,
        "min_rows": lo,
        "max_rows": hi,
        "imbalance": (hi / lo) if lo else float("inf"),
        "idle_shards": counts.count(0),
    }


def suggest_num_shards(total_rows: int, max_shards: int,
                       min_shards: int = 1,
                       max_imbalance: float = 1.0) -> int:
    """Prefer an even split; use ``max_shards`` if no eligible divisor exists."""
    _integer(total_rows, "total_rows", 1)
    _integer(max_shards, "max_shards", 1)
    _integer(min_shards, "min_shards", 1)
    if min_shards > max_shards:
        raise ValueError("min_shards must not exceed max_shards")
    lo = max(min_shards, 2)
    for n in range(min(max_shards, total_rows), lo - 1, -1):
        if total_rows % n == 0:
            return n
    return max_shards


def pack_rows(texts: Sequence[str], tokenizer, row_len: int,
              n_rows: int = EXHAUST, batch_size: int = 1, seed: int = DEFAULT_SEED,
              row_ids: Optional[Sequence[int]] = None,
              shard: int = 0, num_shards: int = 1,
              encoded: Optional[Dict[int, torch.Tensor]] = None,
              verbose: bool = True
              ) -> Tuple[List[Dict[str, torch.Tensor]], List[int]]:
    """Return packed ``[batch, row_len]`` batches and their corpus row ids."""
    _integer(row_len, "row_len", 1)
    _integer(batch_size, "batch_size", 1)
    _integer(num_shards, "num_shards", 1)
    _integer(shard, "shard")
    if shard >= num_shards:
        raise ValueError("shard must be less than num_shards")
    _integer(n_rows, "n_rows", EXHAUST)
    if n_rows == 0:
        raise ValueError("n_rows must be positive or EXHAUST")
    if not texts:
        raise ValueError("calibration corpus contains no records")
    enc = _encode(texts, tokenizer, encoded)
    order = corpus_order(len(texts), seed)
    rows = _fill_rows(order, enc, row_len)
    if n_rows != EXHAUST:
        rows = rows[:n_rows]
    total_rows = len(rows)

    want = list(row_ids) if row_ids is not None else list(range(total_rows))
    for row_id in want:
        _integer(row_id, "row_id")
        if row_id >= total_rows:
            raise ValueError("row_id %d is outside the %d packed rows" % (row_id, total_rows))
    if len(set(want)) != len(want):
        raise ValueError("row_ids must not contain duplicates")
    if num_shards > 1:
        want = want[shard::num_shards]

    pad_id = getattr(tokenizer, "pad_token_id", None) or 0
    built: List[Dict[str, torch.Tensor]] = []
    seq_rows, mask_rows = [], []
    for r in want:
        seq = torch.full((row_len,), pad_id, dtype=torch.long)
        mask = torch.zeros((row_len,), dtype=torch.long)
        at = 0
        for j, taken, _ in rows[r]:
            seq[at:at + taken] = enc[j][:taken]
            mask[at:at + taken] = 1
            at += taken + 1
        seq_rows.append(seq)
        mask_rows.append(mask)
        if len(seq_rows) == batch_size:
            built.append({"input_ids": torch.stack(seq_rows),
                          "attention_mask": torch.stack(mask_rows)})
            seq_rows, mask_rows = [], []
    if seq_rows:
        built.append({"input_ids": torch.stack(seq_rows),
                      "attention_mask": torch.stack(mask_rows)})

    ntok = int(sum(b["attention_mask"].sum() for b in built))
    if verbose:
        print("[data] %d rows of %d (of %d total), shard %d/%d: %d batches, "
              "%d tokens" % (len(want), row_len, total_rows, shard, num_shards,
                             len(built), ntok))
    if want and ntok == 0:
        raise RuntimeError(
            "packed %d rows but 0 tokens -- refusing to run a collection that "
            "would produce an empty pack" % len(want))
    return built, want


def audit_coverage(row_ids: Sequence[int], n_tokens: int, expected_rows: int,
                   expected_tokens: Optional[int] = None) -> None:
    """Check complete, duplicate-free row coverage and optional token totals."""
    _integer(expected_rows, "expected_rows")
    _integer(n_tokens, "n_tokens")
    if expected_tokens is not None:
        _integer(expected_tokens, "expected_tokens")
    ids = list(row_ids)
    for row_id in ids:
        _integer(row_id, "row_id")
    uniq = set(ids)
    if len(uniq) != len(ids):
        dup = sorted(i for i in uniq if ids.count(i) > 1)
        raise RuntimeError("rows counted twice: %s" % dup[:8])
    missing = sorted(set(range(expected_rows)) - uniq)
    if missing:
        raise RuntimeError("%d of %d rows missing, first: %s"
                           % (len(missing), expected_rows, missing[:8]))
    extra = sorted(uniq - set(range(expected_rows)))
    if extra:
        raise RuntimeError("rows outside 0..%d: %s" % (expected_rows - 1, extra[:8]))
    if expected_tokens is not None and n_tokens != expected_tokens:
        raise RuntimeError("pack has %d tokens, one corpus pass is %d"
                           % (n_tokens, expected_tokens))


NON_ADDITIVE_MAX = ("max_activations",)
NON_ADDITIVE_CONST = ("router_scaling", "renormalized", "razor_supported", "refill_supported",
                      "counterfactual_semantics", "pruning_policy", "score_space", "top_k")
_DERIVED = {
    "ean_mean": ("ean_sum", False), "reap": ("weighted_ean_sum", False),
    "counterfactual_delta_mean": ("counterfactual_delta_sum", False),
    "rcs_mean": ("rcs_sum", False), "rcs_refill_mean": ("rcs_refill_sum", False),
    "ean_rms": ("ean_square_sum", True), "reap_rms": ("weighted_ean_square_sum", True),
    "counterfactual_delta_rms": ("counterfactual_delta_square_sum", True),
    "rcs_rms": ("rcs_square_sum", True), "rcs_refill_rms": ("rcs_refill_square_sum", True),
}


def _combine(key: str, into, value):
    """Combine compatible numeric accumulators or matching constants."""
    if torch.is_tensor(into) != torch.is_tensor(value):
        raise RuntimeError("shards disagree on type of %r" % key)
    if torch.is_tensor(into):
        if into.shape != value.shape or into.dtype != value.dtype:
            raise RuntimeError("shards disagree on shape or dtype of %r" % key)
        if into.device != value.device:
            raise RuntimeError("shards disagree on device of %r" % key)
    elif type(into) is not type(value):
        raise RuntimeError("shards disagree on type of %r" % key)
    if key in NON_ADDITIVE_CONST:
        equal = torch.equal(into, value) if torch.is_tensor(into) else into == value
        if not equal:
            raise RuntimeError("shards disagree on %r" % key)
        return into
    if key in NON_ADDITIVE_MAX:
        return torch.maximum(into, value) if torch.is_tensor(into) else max(into, value)
    return into + value


def merge_shards(packs: Sequence[dict], expected_rows: Optional[int] = None
                 ) -> dict:
    """Merge numeric accumulator packs, checking schema and disjoint coverage."""
    if not packs:
        raise RuntimeError("nothing to merge")
    if expected_rows is not None:
        _integer(expected_rows, "expected_rows")
    total: Optional[dict] = None
    seen: set = set()
    ntok = 0
    derived_fields = {}
    for p in packs:
        if not isinstance(p, dict) or "row_ids" not in p:
            raise RuntimeError("pack has no row_ids; coverage cannot be verified")
        ids = list(p["row_ids"])
        for row_id in ids:
            _integer(row_id, "row_id")
        if len(ids) != len(set(ids)):
            raise RuntimeError("shard repeats rows within its row_ids")
        dup = sorted(seen.intersection(ids))
        if dup:
            raise RuntimeError("shard repeats rows already merged: %s" % dup[:8])
        seen.update(ids)
        if "n_tokens" not in p:
            raise RuntimeError("pack has no n_tokens")
        _integer(p["n_tokens"], "n_tokens")
        if ids and p["n_tokens"] == 0:
            raise RuntimeError("shard contains rows but 0 tokens")
        if not ids and p["n_tokens"] != 0:
            raise RuntimeError("shard contains tokens but no row_ids")
        ntok += p["n_tokens"]
        for key in ("row_len", "rows"):
            if key in p:
                _integer(p[key], key, 1 if key == "row_len" else 0)
        if expected_rows is not None and "rows" in p and p["rows"] != expected_rows:
            raise RuntimeError("shards disagree with expected_rows")
        if not isinstance(p.get("state"), dict):
            raise RuntimeError("pack state must be a dictionary")
        if total is None:
            total = {k: v for k, v in p.items() if k != "state"}
            total["state"] = {}
        else:
            for key in ("row_len", "rows") + NON_ADDITIVE_CONST:
                if (key in p) != (key in total):
                    raise RuntimeError("shards disagree on %s" % key)
                if key in p:
                    if key in NON_ADDITIVE_CONST:
                        _combine(key, total[key], p[key])
                    elif p[key] != total[key]:
                        raise RuntimeError("shards disagree on %s" % key)
        for tag, st in p["state"].items():
            if not isinstance(st, dict) or not st:
                raise RuntimeError("layer state must be a nonempty dictionary")
            present = set(st).intersection(_DERIVED)
            if tag in derived_fields and present != derived_fields[tag]:
                raise RuntimeError("shards disagree on derived score fields")
            derived_fields[tag] = present
            for field in present:
                total_field, _ = _DERIVED[field]
                if total_field not in st or "routed_count" not in st:
                    raise RuntimeError(f"merging {field} requires {total_field} and routed_count")
            st = {key: value for key, value in st.items() if key not in _DERIVED}
            for key, value in st.items():
                choices = {"counterfactual_semantics": ("fixed_selected_set", "frozen_weights"),
                           "pruning_policy": ("score", "preserve"),
                           "score_space": ("latent", "expert_output")}
                if key in choices:
                    if not isinstance(value, str) or value not in choices[key]:
                        raise RuntimeError(f"unsupported {key}")
                    continue
                if torch.is_tensor(value):
                    if value.is_complex() or not bool(torch.isfinite(value).all()):
                        raise RuntimeError("unsupported accumulator %r" % key)
                    if value.dtype == torch.bool and key not in NON_ADDITIVE_CONST:
                        raise RuntimeError("boolean accumulator %r is not additive" % key)
                elif type(value) not in (int, float, bool) or not bool(torch.isfinite(torch.tensor(value))):
                    raise RuntimeError("unsupported accumulator %r" % key)
                elif isinstance(value, bool) and key not in NON_ADDITIVE_CONST:
                    raise RuntimeError("boolean accumulator %r is not additive" % key)
            if tag not in total["state"]:
                total["state"][tag] = {
                    k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
                continue
            if set(st) != set(total["state"][tag]):
                raise RuntimeError("shards disagree on fields for %s" % tag)
            for k, v in st.items():
                total["state"][tag][k] = _combine(k, total["state"][tag][k], v)
    total["row_ids"] = sorted(seen)
    total["n_tokens"] = ntok
    rows = expected_rows if expected_rows is not None else total.get("rows")
    if rows is not None:
        audit_coverage(total["row_ids"], ntok, rows)
    for tag, fields in derived_fields.items():
        state = total["state"][tag]
        for field in fields:
            numerator, root = _DERIVED[field]
            count, sums = state["routed_count"], state[numerator]
            if (not torch.is_tensor(count) or not torch.is_tensor(sums) or
                    count.ndim != 1 or count.shape != sums.shape or
                    bool((count < 0).any()) or bool((sums < 0).any()) or
                    bool(((count == 0) & (sums != 0)).any())):
                raise RuntimeError("score moments and routed counts disagree")
            value = sums.double() / count.to(sums.device).double().clamp_min(1)
            state[field] = (value.sqrt() if root else value).float()
    return total
