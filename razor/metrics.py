"""Per-expert scores and deterministic expert selection."""
from __future__ import annotations

from typing import Dict, List

import torch

#: Saliency fields per criterion, as (sum, square sum, mean, rms). One
#: calibration pass fills all of them, so the aggregation is a selection-time
#: choice rather than a property of the collected pack.
FIELDS: Dict[str, tuple] = {
    "rcs": ("rcs_sum", "rcs_square_sum", "rcs_mean", "rcs_rms"),
    "rcs-loo": ("counterfactual_delta_sum", "counterfactual_delta_square_sum",
                "counterfactual_delta_mean", "counterfactual_delta_rms"),
    "rcs-refill": ("rcs_refill_sum", "rcs_refill_square_sum",
                   "rcs_refill_mean", "rcs_refill_rms"),
    "reap": ("weighted_ean_sum", "weighted_ean_square_sum", "reap", "reap_rms"),
    "ean": ("ean_sum", "ean_square_sum", "ean_mean", "ean_rms"),
}

#: Accepted spellings. RAZOR is the RCS-Refill criterion under conditional RMS
#: aggregation, so ``razor`` and ``rcs-refill`` name the same scoring rule.
CANONICAL: Dict[str, str] = {
    "razor": "rcs-refill", "rcs-refill": "rcs-refill", "rcs_refill": "rcs-refill",
    "rcs-loo": "rcs-loo", "rcs_loo": "rcs-loo",
    "rcs": "rcs", "reap": "reap", "ean": "ean", "frequency": "frequency",
}

#: Default sweep set: the paper's method against its three baselines.
METHODS = ("razor", "reap", "ean", "frequency")
#: The scoring-component variants, for ablations; see docs/scoring.md.
VARIANTS = ("rcs", "rcs-loo", "rcs-refill")
CHOICES = METHODS + VARIANTS

#: Aggregation over the calibration tokens routed to each expert. The paper
#: scores experts by conditional RMS unless an aggregation ablation is stated.
AGGREGATIONS = ("rms", "mean", "sum")
DEFAULT_AGGREGATION = "rms"

#: one-line descriptions, used by ``razor models`` and ``--help``
DESCRIPTIONS = {
    "razor": "RCS-Refill under conditional RMS; the method of the paper",
    "rcs": "consensus-residual contribution  lam*u*||f_i - y||",
    "rcs-loo": "fixed-support deletion  lam*u/(1-u) * ||f_i - y||",
    "rcs-refill": "deletion with router refill  lam*||u_i r_i - u_r r_r|| / (1-u_i+u_r)",
    "reap": "routing-weighted output magnitude  w * ||f_i||",
    "ean": "expert output norm  ||f_i||",
    "frequency": "router selection count",
}


def canonical(method: str) -> str:
    """Normalize a user-facing method name to its canonical criterion name."""
    if method not in CANONICAL:
        raise ValueError(f"unknown method {method!r}; expected one of {list(CHOICES)}")
    return CANONICAL[method]


def resolve(metric: str, aggregation: str = DEFAULT_AGGREGATION) -> str:
    """Map a user-facing method name to its on-disk saliency field name."""
    name = CANONICAL.get(metric, metric)
    if name == "frequency":
        return "expert_frequency"
    if name not in FIELDS:
        return metric
    total, _, mean, rms = FIELDS[name]
    return {"sum": total, "mean": mean, "rms": rms}[aggregation]


def _vector(value, field: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{field!r} must be a nonempty one-dimensional tensor")
    if value.is_complex() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{field!r} must contain finite real values")
    if bool((value < 0).any()):
        raise ValueError(f"{field!r} must be nonnegative")
    return value


def _validate_moments(record, method):
    counts = {}
    for name in ("routed_count", "expert_frequency"):
        if name in record:
            value = _vector(record[name], name)
            if value.dtype == torch.bool or not torch.equal(value.double(), value.double().round()):
                raise ValueError(f"{name} must contain integer counts")
            counts[name] = value
    if not counts or method == "frequency":
        return
    count = counts.get("routed_count", counts.get("expert_frequency"))
    total, square, mean, rms = FIELDS[method]
    values = {}
    for name in (total, square, mean, rms):
        if name not in record:
            continue
        value = _vector(record[name], name).double()
        if value.shape != count.shape:
            raise ValueError(f"{name} and routed counts must have the same shape")
        if bool(((count.to(value.device) == 0) & (value != 0)).any()):
            raise ValueError(f"nonzero {name} has zero routed count")
        values[name] = value
    for numerator, derived, root in ((total, mean, False), (square, rms, True)):
        if numerator in values and derived in values:
            expected = values[numerator] / count.to(values[numerator].device).double().clamp_min(1)
            if root:
                expected = expected.sqrt()
            if not torch.allclose(values[derived].to(expected.device), expected, rtol=1e-5, atol=0):
                raise ValueError(f"{derived} and its saved moments disagree")


def score(layer_saliency: dict, method: str,
          aggregation: str = DEFAULT_AGGREGATION) -> torch.Tensor:
    """Return per-expert scores with explicit routed-token aggregation."""
    name = canonical(method)
    if aggregation not in AGGREGATIONS:
        raise ValueError("aggregation must be rms, mean or sum")
    if not isinstance(layer_saliency, dict):
        raise ValueError("layer saliency must be a dictionary")
    if name == "frequency":
        field = "expert_frequency"
    else:
        total, square_field, mean, rms = FIELDS[name]
        field = {"sum": total, "mean": mean, "rms": rms}[aggregation]
    if name.startswith("rcs") and (
        layer_saliency.get("top_k") == 1
        or layer_saliency.get("razor_supported") is False
    ):
        raise KeyError(f"{name}: the RCS criteria, including RAZOR, are unavailable "
                       "for this routing configuration")
    if name == "rcs-refill" and layer_saliency.get("refill_supported") is False:
        raise KeyError(
            "RCS-Refill needs the promoted rank-(k+1) expert, which this "
            "routing adapter does not expose; RCS-LOO remains available")
    _validate_moments(layer_saliency, name)
    if field in layer_saliency:
        result = _vector(layer_saliency[field], field)
        result = result.clone() if name == "frequency" else result.float()
    elif name == "frequency":
        raise KeyError(f"method {method!r} needs field {field!r}; "
                       f"available fields: {sorted(layer_saliency)}")
    else:
        # Derive the requested aggregation from the stored additive moments.
        source = square_field if aggregation == "rms" else total
        count_field = ("routed_count" if "routed_count" in layer_saliency
                       else "expert_frequency")
        if source not in layer_saliency or count_field not in layer_saliency:
            raise KeyError(
                f"{method} {aggregation.upper()} aggregation requires {source!r} and "
                f"routed counts; available fields: {sorted(layer_saliency)}")
        if aggregation == "sum":
            result = _vector(layer_saliency[total], total).float()
        else:
            moment = _vector(layer_saliency[source], source).double()
            count = _vector(layer_saliency[count_field], count_field).double().to(moment.device)
            if moment.shape != count.shape:
                raise ValueError(f"{source} and routed counts must have the same shape")
            if bool(((count == 0) & (moment != 0)).any()):
                raise ValueError(f"nonzero {source} has zero routed count")
            result = moment / count.clamp_min(1)
            result = (result.sqrt() if aggregation == "rms" else result).float()
    for count_field in ("routed_count", "expert_frequency"):
        if count_field in layer_saliency:
            count = _vector(layer_saliency[count_field], count_field)
            if count.shape != result.shape:
                raise ValueError(f"{field!r} and {count_field!r} must have the same shape")
    return _vector(result, field)


def available(layer_saliency: dict,
              aggregation: str = DEFAULT_AGGREGATION) -> List[str]:
    """Which methods this saliency record can actually serve."""
    out = []
    for name in CHOICES:
        try:
            score(layer_saliency, name, aggregation=aggregation)
        except KeyError:
            continue
        out.append(name)
    return out


def keep_indices(layer_saliency: dict, method: str, target_experts: int,
                 aggregation: str = DEFAULT_AGGREGATION) -> List[int]:
    """Return sorted expert ids, breaking score ties by the lowest index."""
    s = score(layer_saliency, method, aggregation=aggregation)
    if isinstance(target_experts, bool) or not isinstance(target_experts, int):
        raise ValueError("target_experts must be an integer")
    if target_experts < 1:
        raise ValueError(
            f"target_experts must be at least 1, got {target_experts}")
    if target_experts > s.numel():
        raise ValueError(
            f"target_experts={target_experts} exceeds the {s.numel()} experts "
            "present in the saliency file"
        )
    return sorted(torch.argsort(s, descending=True, stable=True)[:target_experts].tolist())
