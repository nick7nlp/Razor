"""Checkpoint tensor checks and sampled next-token log-probability checks."""
from __future__ import annotations

import math
from numbers import Integral
from typing import Dict, List, Optional, Sequence

import torch

NLL_TOL = 2e-3
OUTLIER_FRAC = 0.02
OUTLIER_ABS = 1e-2


def validate_keep(n_experts: int, keep: Sequence[int]) -> List[int]:
    """Validate expert ids without changing their order."""
    if isinstance(n_experts, bool) or not isinstance(n_experts, Integral) or n_experts < 1:
        raise ValueError("n_experts must be a positive integer")
    if not isinstance(keep, (list, tuple)) or not keep:
        raise ValueError("keep must be a nonempty list of integer expert ids")
    if any(isinstance(i, bool) or not isinstance(i, Integral) for i in keep):
        raise ValueError("keep must contain integer expert ids")
    ids = [int(i) for i in keep]
    if len(set(ids)) != len(ids) or any(i < 0 or i >= n_experts for i in ids):
        raise ValueError(f"keep contains duplicate or out-of-range ids for {n_experts} experts")
    return ids


@torch.no_grad()
def score_batches(model, batches: Sequence[Dict[str, torch.Tensor]],
                  device: Optional[str] = None) -> dict:
    """Score valid adjacent tokens; reject empty or malformed inputs."""
    dev = device or next(model.parameters()).device
    lps: List[torch.Tensor] = []
    valids: List[torch.Tensor] = []
    training = {module: module.training for module in model.modules()}
    try:
        model.eval()
        for batch in batches:
            ids = batch["input_ids"].to(dev)
            mask = batch["attention_mask"].to(dev)
            if ids.ndim != 2 or ids.shape != mask.shape or ids.shape[1] < 2:
                raise ValueError("input_ids and attention_mask must have matching (batch, length >= 2) shapes")
            if ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("input_ids must be integer token ids")
            if not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError("attention_mask must contain only 0 and 1")
            out = model(input_ids=ids, attention_mask=mask)
            if hasattr(out, "logits"):
                logits = out.logits
            elif isinstance(out, tuple) and out:
                logits = out[0]
            else:
                raise ValueError("model output must expose logits or return (logits, ...)")
            if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[:2] != ids.shape:
                raise ValueError("logits must have shape (batch, length, vocabulary)")
            logits = logits.float()
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("model returned non-finite logits")
            if bool((ids < 0).any()) or bool((ids >= logits.shape[-1]).any()):
                raise ValueError("input_ids contain tokens outside the output vocabulary")
            lp = torch.log_softmax(logits[:, :-1], dim=-1)
            targets = ids[:, 1:].to(logits.device, dtype=torch.long)
            tok_lp = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            valid = mask[:, :-1].bool() & mask[:, 1:].bool()
            lps.append(tok_lp.reshape(-1).cpu())
            valids.append(valid.reshape(-1).cpu())
    finally:
        for module, was_training in training.items():
            module.training = was_training
    if not lps:
        raise ValueError("no batches to score")
    lp, valid = torch.cat(lps), torch.cat(valids)
    n = int(valid.sum())
    if n == 0:
        raise ValueError("no valid next-token targets to score")
    return {"logprobs": lp, "valid": valid,
            "nll": float(-lp[valid].double().mean()), "n": n}


def _validated_result(result: dict):
    lp, valid = result["logprobs"], result["valid"]
    if not isinstance(lp, torch.Tensor) or not isinstance(valid, torch.Tensor):
        raise ValueError("logprobs and valid must be tensors")
    if lp.ndim != 1 or valid.shape != lp.shape or valid.dtype != torch.bool:
        raise ValueError("logprobs and boolean valid must have matching one-dimensional shapes")
    if not lp.is_floating_point() or not bool(torch.isfinite(lp).all()):
        raise ValueError("logprobs must be finite floating-point values")
    n = result["n"]
    if isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 or int(valid.sum()) != n:
        raise ValueError("invalid scored-token count")
    lp, valid = lp.detach().cpu().double(), valid.detach().cpu()
    nll = float(result["nll"])
    actual_nll = float(-lp[valid].mean())
    if not math.isfinite(nll) or not math.isclose(nll, actual_nll, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError("NLL is non-finite or inconsistent with the scored tokens")
    return lp, valid, actual_nll


def compare_logprobs(ref: dict, test: dict, tol: float = NLL_TOL,
                     verbose: bool = True) -> bool:
    """Compare sampled log-probabilities, not universal model equivalence."""
    try:
        if not math.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive")
        a, va, ref_nll = _validated_result(ref)
        b, vb, test_nll = _validated_result(test)
        if a.shape != b.shape or not torch.equal(va, vb):
            raise ValueError("scored-token shapes or validity masks differ")
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        if verbose:
            print(f"FAIL: {exc}")
        return False
    delta = (a[va] - b[vb]).abs()
    dnll = test_nll - ref_nll
    frac = float((delta > OUTLIER_ABS).double().mean())
    ok = abs(dnll) < tol and frac < OUTLIER_FRAC
    if verbose:
        print(f"[verify] dNLL={dnll:+.3e}; max token difference={float(delta.max()):.3e}; "
              f"outlier fraction={frac:.4f}")
        print(f"{'PASS' if ok else 'FAIL'}: sampled next-token log-probabilities")
    return ok


def tensors_identical(expected: torch.Tensor, actual: torch.Tensor) -> bool:
    """Compare finite tensors by shape, dtype, and stored bytes."""
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return False
    expected = expected.detach().cpu().contiguous().reshape(-1)
    actual = actual.detach().cpu().contiguous().reshape(-1)
    if not torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)):
        return False
    if expected.is_floating_point():
        for chunk in expected.split(1 << 20):
            if str(chunk.dtype).startswith("torch.float8"):
                chunk = chunk.float()
            if not bool(torch.isfinite(chunk).all()):
                return False
    return True


def compare_expert_tensors(src_state: Dict[str, torch.Tensor],
                           dst_state: Dict[str, torch.Tensor],
                           keep: Dict[str, Sequence[int]],
                           expert_axis: int = 0,
                           verbose: bool = True) -> bool:
    """Check explicitly named expert tensors in keep order."""
    failures = []
    if not keep:
        failures.append("no tensors requested")
    for name, ids in sorted(keep.items()):
        try:
            source, actual = src_state[name], dst_state[name]
            ids = validate_keep(source.shape[expert_axis], ids)
            expected = source.index_select(expert_axis, torch.tensor(ids, device=source.device))
            if not tensors_identical(expected, actual):
                failures.append(f"{name}: tensor differs from source[keep]")
        except (KeyError, ValueError, IndexError, RuntimeError) as exc:
            failures.append(f"{name}: {exc}")
    if verbose:
        print(f"{'FAIL' if failures else 'PASS'}: expert tensor comparison")
        for failure in failures[:10]:
            print(f"  {failure}")
    return not failures


def check_untouched(src_state: Dict[str, torch.Tensor],
                    dst_state: Dict[str, torch.Tensor],
                    skip: Sequence[str] = (),
                    verbose: bool = True) -> bool:
    """Check all keys and bytes outside the caller's explicit exclusions."""
    source = {name for name in src_state if not any(s in name for s in skip)}
    target = {name for name in dst_state if not any(s in name for s in skip)}
    ok = source == target and all(tensors_identical(src_state[name], dst_state[name])
                                 for name in source & target)
    if verbose:
        print(f"{'PASS' if ok else 'FAIL'}: untouched tensor comparison")
    return ok


def keep_mask(n_experts: int, keep: Sequence[int],
              device=None, dtype=torch.bool) -> torch.Tensor:
    """Return an expert-selection mask; this does not prune a model."""
    ids = validate_keep(n_experts, keep)
    mask = torch.zeros(n_experts, dtype=dtype, device=device)
    mask[torch.tensor(ids, dtype=torch.long, device=device)] = True
    return mask


def compare_layer_outputs(reference, actual, *, atol: float = 0.0,
                          rtol: float = 1e-3) -> dict:
    """Compare complete native hidden states, including mHC stream axes."""
    if any(isinstance(t, bool) or not isinstance(t, (int, float)) or
           not math.isfinite(t) or t < 0 for t in (atol, rtol)):
        raise ValueError("layer tolerances must be finite and nonnegative")
    if not isinstance(reference, torch.Tensor) or not isinstance(actual, torch.Tensor):
        raise ValueError("native layer outputs must be tensors")
    if reference.shape != actual.shape:
        return {"ok": False, "reason": "hidden-state shapes differ"}
    if reference.dtype != actual.dtype:
        return {"ok": False, "reason": "hidden-state dtypes differ"}
    if not reference.is_floating_point() or not reference.numel():
        raise ValueError("native layer outputs must be nonempty floating-point tensors")
    a = reference.detach().to(device="cpu", dtype=torch.float64)
    b = actual.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return {"ok": False, "reason": "non-finite hidden states"}
    delta = (a - b).abs()
    maximum = float(delta.max())
    scale = float(a.abs().max())
    return {"ok": bool((delta <= atol + rtol * a.abs()).all()),
            "max_abs": maximum, "max_rel": maximum / max(scale, 1e-30),
            "shape": list(reference.shape), "dtype": str(reference.dtype)}


@torch.no_grad()
def score_hidden_batches(hidden, batches, head, *, finalize=None,
                         chunk_size: int = 1024) -> dict:
    """Score streamed hidden states using the native epilogue and bounded logits.

    ``finalize`` must perform the family's actual final norm / hyper-head, not
    a guessed mean over mHC streams. The caller supplies the loaded output head;
    reusing embeddings is valid only for an explicitly tied checkpoint.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    hidden, batches = list(hidden), list(batches)
    if not batches or len(hidden) != len(batches):
        raise ValueError("hidden states must match a nonempty batch sequence")
    modules = {m for obj in (head, finalize) if isinstance(obj, torch.nn.Module)
               for m in obj.modules()}
    training = {m: m.training for m in modules}
    lps, masks = [], []
    try:
        for module in modules:
            module.training = False
        for h, batch in zip(hidden, batches):
            ids, mask = batch["input_ids"], batch["attention_mask"]
            if (ids.ndim != 2 or ids.shape != mask.shape or ids.shape[1] < 2 or
                    ids.dtype not in (torch.int32, torch.int64)):
                raise ValueError("input_ids and attention_mask require matching (batch, length >= 2) shapes")
            if not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError("attention_mask must contain only 0 and 1")
            h = finalize(h) if finalize is not None else h
            if (not isinstance(h, torch.Tensor) or h.ndim != 3 or
                    h.shape[:2] != ids.shape or not h.is_floating_point()):
                raise ValueError("native epilogue must return (batch, length, hidden) states")
            if not bool(torch.isfinite(h).all()):
                raise ValueError("native epilogue returned non-finite hidden states")
            h = h[:, :-1].reshape(-1, h.shape[-1])
            targets = ids[:, 1:].reshape(-1).to(h.device, dtype=torch.long)
            valid = (mask[:, :-1].bool() & mask[:, 1:].bool()).reshape(-1).cpu()
            lp = torch.empty(h.shape[0], dtype=torch.float32, device="cpu")
            for start in range(0, h.shape[0], chunk_size):
                stop = min(start + chunk_size, h.shape[0])
                logits = head(h[start:stop])
                if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[0] != stop - start:
                    raise ValueError("output head must return (tokens, vocabulary) logits")
                logits = logits.float()
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("output head returned non-finite logits")
                target = targets[start:stop].to(logits.device)
                if bool((target < 0).any()) or bool((target >= logits.shape[-1]).any()):
                    raise ValueError("input_ids contain tokens outside the output vocabulary")
                lp[start:stop] = torch.log_softmax(logits, -1).gather(1, target[:, None]).squeeze(1).cpu()
            lps.append(lp)
            masks.append(valid)
    finally:
        for module, was_training in training.items():
            module.training = was_training
    lp, valid = torch.cat(lps), torch.cat(masks)
    n = int(valid.sum())
    if not n:
        raise ValueError("no valid next-token targets to score")
    return {"logprobs": lp, "valid": valid, "n": n,
            "nll": float(-lp[valid].double().mean())}


@torch.no_grad()
def verify_streamed_reference(reference, target, batches, keep, *,
                              max_layers=None, atol=0.0, rtol=1e-3,
                              verbose=True) -> dict:
    """Run two native layer loaders in lockstep, holding one layer at a time.

    Each backend exposes ``layer_ids``, ``prepare(batch) -> (hidden, state)``,
    ``load_layer(index, keep=None)`` and ``step(layer, hidden, state, index)``.
    The reference loader must select the source experts in the supplied order,
    including router rows, auxiliary scales and hash remapping, before native
    forward. The target loader reads its saved weights without modification.
    Native caches / indexer carries are private to each side and each batch.

    This checks sampled layer outputs, not a routing reconstruction. It does not
    replace the independent checkpoint byte/hash-table verifier. Unexecuted MTP
    layers and a truncated suffix are explicitly outside the forward claim.
    """
    if reference is target:
        raise ValueError("reference and target require independent stream backends")
    layer_ids = list(reference.layer_ids)
    if (not layer_ids or layer_ids != list(target.layer_ids) or
            any(type(i) is not int or i < 0 for i in layer_ids) or
            len(set(layer_ids)) != len(layer_ids)):
        raise ValueError("source and target must expose the same nonempty unique layer ids")
    if max_layers is not None and (type(max_layers) is not int or max_layers < 1):
        raise ValueError("max_layers must be a positive integer")
    if not isinstance(keep, dict) or not keep:
        raise ValueError("streamed reference requires explicit keep sets")
    main_tags = {f"main_{i}" for i in layer_ids}
    if any(not isinstance(tag, str) or
           (tag not in main_tags and not (tag.startswith("mtp_") and tag[4:].isdigit()))
           for tag in keep):
        raise ValueError("keep-set names a main layer absent from the native stream")
    for tag, ids in keep.items():
        if (not isinstance(ids, (list, tuple)) or not ids or
                any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids)):
            raise ValueError(f"{tag}: invalid ordered expert selection")
    executed = layer_ids[:max_layers] if max_layers is not None else layer_ids
    selected = [i for i in executed if f"main_{i}" in keep]
    if not selected:
        raise ValueError("no selected MoE layer is exercised; reference comparison would be vacuous")
    batches = list(batches)
    if not batches:
        raise ValueError("streamed reference requires nonempty batches")
    hidden, states = {"reference": [], "target": []}, {"reference": [], "target": []}
    for side, backend in (("reference", reference), ("target", target)):
        for batch in batches:
            ids, mask = batch["input_ids"], batch["attention_mask"]
            if ids.ndim != 2 or ids.shape != mask.shape or ids.shape[1] < 2:
                raise ValueError("stream batches require matching (batch, length >= 2) shapes")
            if ids.dtype not in (torch.int32, torch.int64) or bool((ids < 0).any()):
                raise ValueError("stream input_ids must be nonnegative integer tokens")
            if not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError("attention_mask must contain only 0 and 1")
            if not bool((mask[:, :-1].bool() & mask[:, 1:].bool()).any()):
                raise ValueError("stream batch has no valid adjacent tokens")
            private_batch = {key: value.clone() if isinstance(value, torch.Tensor) else value
                             for key, value in batch.items()}
            h, state = backend.prepare(private_batch)
            hidden[side].append(h.detach().clone())
            states[side].append(state)
    results, first_bad = [], None

    def compare(tag):
        nonlocal first_bad
        comparisons = [compare_layer_outputs(a, b, atol=atol, rtol=rtol)
                       for a, b in zip(hidden["reference"], hidden["target"])]
        ok = all(item["ok"] for item in comparisons)
        if not ok and first_bad is None:
            first_bad = tag
        results.append({"layer": tag, "ok": ok, "batches": comparisons})
        if verbose:
            maximum = max((item.get("max_abs", float("inf")) for item in comparisons))
            print(f"[verify stream] {tag}: {'PASS' if ok else 'FAIL'}; max|delta|={maximum:.3e}")

    compare("prelude")
    for index in executed:
        tag = f"main_{index}"
        for side, backend in (("reference", reference), ("target", target)):
            ids = keep.get(tag) if side == "reference" else None
            layer = backend.load_layer(index, keep=ids)
            if not isinstance(layer, torch.nn.Module):
                raise TypeError("stream loader must materialize a native torch module")
            training = {module: module.training for module in layer.modules()}
            try:
                layer.eval()
                for i, h in enumerate(hidden[side]):
                    value, state = backend.step(layer, h, states[side][i], index)
                    if not isinstance(value, torch.Tensor):
                        raise ValueError("native layer step must return hidden states and carry")
                    hidden[side][i], states[side][i] = value.detach(), state
            finally:
                for module, was_training in training.items():
                    module.training = was_training
                release = getattr(backend, "release_layer", None)
                if callable(release):
                    release(layer)
                del training, module, layer
        compare(tag)
    partial = len(executed) < len(layer_ids)
    report = {"ok": first_bad is None, "scope": "partial layer outputs" if partial else "main layer outputs",
              "partial": partial, "layers_checked": len(executed),
              "sliced_layers": len(selected), "first_bad_layer": first_bad,
              "unexecuted_keep_tags": sorted(set(keep) - {f"main_{i}" for i in executed}),
              "layer_results": results}
    if verbose:
        label = "PARTIAL " if partial else ""
        print(f"{label}{'PASS' if report['ok'] else 'FAIL'}: native streamed layer outputs; "
              "not a full-logprob or MTP-forward claim")
    return report


def verify_native_streams(reference_model, reference_weights, target_model,
                          target_weights, batches, *, transform_reference,
                          device="cpu", max_layers=None, atol=0.0, rtol=1e-3,
                          verbose=True) -> dict:
    """Audit the streaming engine's original model forwards with bounded storage.

    Reference and target alternate materializing a layer. The reference layer is
    transformed in memory, executed through its native forward, and released
    before the target layer is loaded. Only the current layer's CPU activations
    are retained. Both native call stacks preserve their own architecture-specific
    state; no manually reconstructed mask, mHC prelude or cache is substituted.
    """
    import threading
    from .streaming import _decoder, stream_forward

    batches = list(batches)
    if not batches:
        raise ValueError("native streaming verification requires nonempty batches")
    ref_name, _, ref_layers = _decoder(reference_model)
    dst_name, _, dst_layers = _decoder(target_model)
    if len(ref_layers) != len(dst_layers):
        raise ValueError("source and target native decoder depths differ")
    if max_layers is not None and (type(max_layers) is not int or max_layers < 1):
        raise ValueError("max_layers must be a positive integer")
    depth = min(max_layers, len(ref_layers)) if max_layers is not None else len(ref_layers)
    compare_layer_outputs(torch.zeros(1), torch.zeros(1), atol=atol, rtol=rtol)
    prefixes = {"reference": (ref_name + "." if ref_name else "") + "layers.",
                "target": (dst_name + "." if dst_name else "") + "layers."}
    condition = threading.Condition()
    state = {"turn": ("reference", 0), "abort": False}
    pending, results, errors = {}, [], []

    class AlternatingWeights:
        def __init__(self, base, side):
            self.base, self.side, self.active = base, side, None
            self.slots = base.slots
            self.restore = None

        def materialize(self, prefix, dev, **kwargs):
            layer_index = None
            if prefix.startswith(prefixes[self.side]):
                tail = prefix[len(prefixes[self.side]):].rstrip(".")
                if tail.isdigit():
                    layer_index = int(tail)
            if layer_index is not None:
                with condition:
                    condition.wait_for(lambda: state["abort"] or state["turn"] == (self.side, layer_index))
                    if state["abort"]:
                        raise RuntimeError("paired native stream cancelled after verification failure")
                    self.active = layer_index
            loaded = self.base.materialize(prefix, dev, **kwargs)
            if layer_index is not None and self.side == "reference":
                self.restore = transform_reference(layer_index, ref_layers[layer_index])
            return loaded

        def release(self, names):
            if self.restore is not None:
                self.restore()
                self.restore = None
            self.base.release(names)
            if self.active is not None:
                with condition:
                    index, self.active = self.active, None
                    state["turn"] = (("target", index) if self.side == "reference"
                                     else ("reference", index + 1))
                    condition.notify_all()

    def callback(side):
        def observe(event, layer_index, batch_index, value, kwargs):
            if event == "input":
                hidden = value[0] if value else kwargs.get("hidden_states")
            elif event == "output":
                hidden = value[0] if isinstance(value, (tuple, list)) else value
            else:
                raise ValueError(f"unknown native stream event: {event}")
            if not isinstance(hidden, torch.Tensor):
                raise ValueError("native decoder boundary did not expose hidden states")
            key = (event, layer_index, batch_index)
            if side == "reference":
                if key in pending:
                    raise RuntimeError("duplicate native reference boundary")
                pending[key] = hidden.detach().cpu().clone()
            else:
                if key not in pending:
                    raise RuntimeError("target native boundary has no independent source reference")
                result = compare_layer_outputs(pending.pop(key), hidden, atol=atol, rtol=rtol)
                results.append(dict(result, event=event, layer=f"main_{layer_index}", batch=batch_index))
                if verbose and (not result["ok"] or (event == "output" and batch_index == len(batches) - 1)):
                    print(f"[verify stream] main_{layer_index} {event}: "
                          f"{'PASS' if result['ok'] else 'FAIL'}; max|delta|={result.get('max_abs', float('inf')):.3e}")
        return observe

    def run(side, model, weights):
        try:
            options = {"max_layers": depth} if depth != len(ref_layers) else {}
            stream_forward(model, AlternatingWeights(weights, side), batches,
                           device=device, layer_callback=callback(side), **options)
        except BaseException as exc:
            with condition:
                errors.append(exc)
                state["abort"] = True
                condition.notify_all()

    threads = [threading.Thread(target=run, args=("reference", reference_model, reference_weights)),
               threading.Thread(target=run, args=("target", target_model, target_weights))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    expected = {(event, f"main_{layer}", batch) for event in ("input", "output")
                for layer in range(depth) for batch in range(len(batches))}
    observed = {(item["event"], item["layer"], item["batch"]) for item in results}
    if pending or observed != expected or len(results) != len(expected):
        raise RuntimeError("native stream boundary coverage is incomplete")
    failed = next((item["layer"] for item in results if not item["ok"]), None)
    partial = depth != len(ref_layers)
    report = {"ok": failed is None, "partial": partial, "layers_checked": depth,
              "first_bad_layer": failed, "layer_results": results,
              "scope": "partial native layer inputs/outputs" if partial else "native main-layer inputs/outputs"}
    if verbose:
        print(f"{'PARTIAL ' if partial else ''}{'PASS' if report['ok'] else 'FAIL'}: "
              "native streamed layer inputs/outputs; not a logprob or MTP-forward claim")
    return report
