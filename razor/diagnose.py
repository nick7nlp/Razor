"""Report routing statistics and agreement between expert keep-sets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from . import metrics


def _layer_index(tag: str) -> int:
    return int(tag.split("_")[-1])


def _zones(n: int):
    """Split depth into thirds, so the report works at any layer count."""
    a, b = n // 3, 2 * n // 3
    names = ("early", "mid", "late")

    def which(i: int) -> str:
        return names[0] if i < a else (names[1] if i < b else names[2])

    return which, names


def report(saliency_path: str, keep: Optional[int] = None,
           export: Optional[str] = None) -> Dict[str, dict]:
    """Report a saliency file or directory; optionally export keep-sets.

    Scores are read with ``weights_only=True``: a saliency file is data, never
    an executable payload.
    """
    from .collect import SALIENCY_ALIASES, find_saliency

    found = find_saliency(saliency_path)
    if found is None:
        raise SystemExit(
            f"no saliency file at {saliency_path!r}; expected a file, or a "
            f"directory containing one of {list(SALIENCY_ALIASES)}"
        )
    try:
        data = torch.load(found, map_location="cpu", weights_only=True)
    except Exception as error:
        raise SystemExit(f"{found} is not a readable tensor saliency file: {error}") from error
    if not isinstance(data, dict) or not data or not all(
            isinstance(tag, str) and isinstance(record, dict) for tag, record in data.items()):
        raise SystemExit(f"{found} must contain a mapping of layer tags to score records")
    tags = sorted(data.keys(), key=_layer_index)
    if not tags:
        raise SystemExit(f"{found} contains no layers")

    E = metrics.score(data[tags[0]], "frequency").numel()
    K = keep if keep is not None else E // 2
    if isinstance(K, bool) or not isinstance(K, int) or not 0 < K < E:
        raise SystemExit(f"--keep must be in [1, {E - 1}], got {K}")
    have = list(metrics.CHOICES)
    for tag in tags:
        if metrics.score(data[tag], "frequency").numel() != E:
            raise ValueError("diagnostics require the same expert count in every layer")
        available = metrics.available(data[tag])
        have = [method for method in have if method in available]
    zone_of, ZONES = _zones(len(tags))

    print(f"saliency : {found}")
    print(f"layers   : {len(tags)}  ({tags[0]} .. {tags[-1]})")
    print(f"experts  : {E}   budget: keep {K}  ({1 - K / E:.0%} pruned)")
    print(f"methods  : {', '.join(have)}")
    total = sum(int(data[t].get('total_tokens', 0)) for t in tags)
    if total:
        print(f"tokens   : {total / len(tags) / 1e6:.2f}M per layer")

    keeps: Dict[str, Dict[str, List[int]]] = {m: {} for m in have}
    w_by_zone: Dict[str, list] = {z: [] for z in ZONES}
    sources: set = set()

    for tag in tags:
        rec = data[tag]
        for m in have:
            keeps[m][tag] = metrics.keep_indices(rec, m, K)
        denom = rec.get("routed_count")
        if denom is None:
            denom = rec["expert_frequency"]
        denom = denom.double().clamp(min=1)

        if "normalized_gate_sum" in rec:
            u_bar = rec["normalized_gate_sum"].double() / denom
            sources.add("normalized")
        elif "weighted_expert_frequency_sum" in rec:
            lam = float(rec.get("router_scaling", 0.0) or 0.0)
            if lam > 0:
                u_bar = (rec["weighted_expert_frequency_sum"].double()
                         / denom / lam)
                sources.add("rescaled")
            else:
                sources.add("unscalable")
                continue
        else:
            continue
        w_by_zone[zone_of(_layer_index(tag))].append(u_bar)

    # --- 1. routing weight spread --- #
    if "unscalable" in sources:
        print("\nnormalized routing weight u: unavailable. This saliency file "
              "records neither 'normalized_gate_sum' nor 'router_scaling', so "
              "the stored gate sums cannot be converted to probabilities. "
              "Recollect to get this section; the keep-sets "
              "below are unaffected.")
    elif any(w_by_zone.values()):
        print("\nnormalized routing weight u, by depth "
              "(a wide spread is where RAZOR can differ from REAP)")
        if "rescaled" in sources:
            print("  (recovered from the scaled gate sums using "
                  "router_scaling)")
        print(f"  {'zone':<8}{'mean u':>10}{'p10':>10}{'p90':>10}"
              f"{'mean 1/(1-u)':>15}")
        for z in ZONES:
            if not w_by_zone[z]:
                continue
            w = torch.cat(w_by_zone[z]).float().clamp(0, 0.999)
            amp = 1.0 / (1.0 - w)
            print(f"  {z:<8}{w.mean():>10.4f}"
                  f"{w.quantile(0.1):>10.4f}{w.quantile(0.9):>10.4f}"
                  f"{amp.mean():>15.3f}")

    # --- 2. agreement between methods --- #
    if len(have) > 1:
        print(f"\ntop-{K} agreement (Jaccard, mean over layers)")
        head = "".join(f"{m:>12}" for m in have)
        print(f"  {'':<12}{head}")
        for a in have:
            row = ""
            for b in have:
                js = [
                    len(set(keeps[a][t]) & set(keeps[b][t]))
                    / len(set(keeps[a][t]) | set(keeps[b][t]))
                    for t in tags
                ]
                row += f"{sum(js) / len(js):>12.3f}"
            print(f"  {a:<12}{row}")

    # --- 3. is the keep-set just magnitude or frequency? --- #
    print("\nkeep-set composition (share drawn from the top quartile of ...)")
    print(f"  {'method':<12}{'magnitude':>12}{'frequency':>12}")
    for m in have:
        mag_hits, freq_hits = [], []
        for tag in tags:
            rec = data[tag]
            q = max(1, E // 4)
            magnitude = "ean" if "ean_mean" in rec else "frequency"
            top_mag = set(metrics.keep_indices(rec, magnitude, q))
            top_freq = set(metrics.keep_indices(rec, "frequency", q))
            k = set(keeps[m][tag])
            mag_hits.append(len(k & top_mag) / len(top_mag))
            freq_hits.append(len(k & top_freq) / len(top_freq))
        print(f"  {m:<12}{sum(mag_hits) / len(mag_hits):>12.3f}"
              f"{sum(freq_hits) / len(freq_hits):>12.3f}")
    print(f"\n  (a value near {K / E:.2f} means the method is indifferent to "
          f"that property;\n   near 1.00 means it is essentially that property.)")

    # --- 4. dead experts --- #
    dead = [(t, int((data[t]["expert_frequency"] == 0).sum())) for t in tags]
    n_dead = sum(c for _, c in dead)
    if n_dead:
        worst = max(dead, key=lambda x: x[1])
        print(f"\nnever routed: {n_dead} expert-slots across all layers "
              f"(worst: {worst[0]} with {worst[1]}/{E}) — "
              "these are free to prune under any criterion")

    if export:
        out = Path(export)
        out.mkdir(parents=True, exist_ok=True)
        for m in have:
            path = out / f"keep_indices_{m}_k{K}.json"
            with open(path, "w") as f:
                json.dump(keeps[m], f)
            print(f"[export] {m:<10} -> {path}")

    return keeps
