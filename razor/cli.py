'Command-line interface.'
from __future__ import annotations

import argparse
import sys

from . import __version__, geometry, metrics
from .adapters import list_adapters
from .pipeline import DEFAULT_DATA, collect_saliency, prune, sweep


def _csv(value: str):
    return [v.strip() for v in value.split(",") if v.strip()]


def _add_calibration_args(p: argparse.ArgumentParser) -> None:


    g = p.add_argument_group("calibration")
    g.add_argument("--data", default=DEFAULT_DATA,
                   help="calibration set: a .json/.jsonl file, a directory of "
                        "them, or hf:<dataset-id> (default: RazorCal)")
    g.add_argument("--num-batches", "--num_batches", type=int,
                   default=geometry.EXHAUST,
                   help="-1 (default) consumes the calibration corpus exactly "
                        "once. A positive value caps the run at that many "
                        "forward passes, which is a smoke test rather than a "
                        "token budget: its meaning depends on --batch-size")
    g.add_argument("--batch-size", "--batch_size", type=int, default=1,
                   help="rows per forward pass; affects memory and speed only")
    g.add_argument("--max-len", "--max_len", type=int, default=32768,
                   help="tokens per packed row (default 32768). This is what "
                        "must match across models being compared; the row "
                        "count follows from it and from the tokenizer")
    g.add_argument("--seed", type=int, default=geometry.DEFAULT_SEED,
                   help="sample visiting order; part of the geometry, since it "
                        "decides which sample loses its tail at a row boundary")
    g.add_argument("--expert-chunk", "--expert_chunk", type=int, default=4,
                   help="experts held in memory at once; lower this on OOM")
    g.add_argument("--no-verify", "--no_verify", action="store_true",
                   help="skip the routing reconstruction check")
    g.add_argument("--device-map", default="auto", help="model placement for collection")
    g.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    g.add_argument("--calibration-limit", type=int, default=None, help="maximum calibration records")
    g.add_argument("--no-cache", action="store_true", help="recompute saliency")
    g.add_argument("--execution", choices=["auto", "resident", "streaming"], default="auto",
                   help="resident model or one decoder layer at a time")
    g.add_argument("--stream-device", default=None,
                   help="device used for streaming layers, e.g. cuda:0 or cpu")
    g.add_argument("--stream-batch-window", type=int, default=1,
                   help="simultaneously retained batch activations; larger windows reuse layer reads")
    g.add_argument("--chunk-attn", type=int, default=None,
                   help="query rows per eager-attention chunk")


def _add_export_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--export-mode", choices=["auto", "checkpoint", "model"], default="auto",
                   help="preserve source checkpoint tensors or save an in-memory model")
    p.add_argument("--hash-policy", choices=["remap", "preserve"], default="remap",
                   help="remap hash-routed experts or preserve their original width")
    p.add_argument("--mtp-policy", choices=["router_norm", "drop", "error"], default="router_norm",
                   help="selection policy for unobserved auxiliary experts")
    p.add_argument("--aggregation", choices=list(metrics.AGGREGATIONS),
                   default=metrics.DEFAULT_AGGREGATION,
                   help="routed-token score aggregation; frequency always uses counts")


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True,
                   help="HF hub id or local checkpoint directory")
    p.add_argument("--adapter", default=None, choices=list_adapters(),
                   help="force a model adapter (default: auto-detect)")
    p.add_argument("--trust-remote-code", action="store_true",
                   help="allow execution of model repository code")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="razor",
        description="Training-free Mixture-of-Experts expert pruning.",
    )
    ap.add_argument("--version", action="version", version=f"razor {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)


    p = sub.add_parser(
        "prune", help="prune a model (collecting saliency first if needed)",
        description="Prune a MoE model to a smaller expert count.")
    _add_model_args(p)
    p.add_argument("--out", "--out-dir", "--out_dir", required=True,
                   help="output directory")
    p.add_argument("--method", "--metric", default="razor",
                   choices=list(metrics.CHOICES),
                   help="; ".join(f"{k}: {v}" for k, v in
                                  metrics.DESCRIPTIONS.items()))
    budget = p.add_mutually_exclusive_group()
    budget.add_argument("--ratio", type=float,
                        help="fraction of experts to remove, e.g. 0.5")
    budget.add_argument("--target-experts", "--target_experts", type=int,
                        help="absolute experts to keep per layer")
    p.add_argument("--saliency", default=None,
                   help="reuse an existing saliency file (or the directory "
                        "holding one) instead of collecting")
    p.add_argument("--keep-indices", "--keep_indices", default=None,
                   help="use an external keep-set JSON, bypassing scoring")
    p.add_argument("--target-top-k", "--target_top_k", type=int, default=None,
                   help="also lower num_experts_per_tok (the FLOPs axis)")
    p.add_argument("--max-shard-size", "--max_shard_size", default="5GB")
    _add_export_args(p)
    _add_calibration_args(p)


    s = sub.add_parser("saliency", help="collect saliency only",
                       description="One forward pass; emits every criterion, including the "
                                   "rcs, rcs-loo and rcs-refill variants.")
    _add_model_args(s)
    s.add_argument("--out", "--out-dir", "--out_dir", required=True,
                   help="directory for the saliency file")
    _add_calibration_args(s)


    w = sub.add_parser(
        "sweep", help="every method x every ratio from one forward pass",
        description="Prune the full comparison grid from a single collection.")
    _add_model_args(w)
    w.add_argument("--out", "--out-dir", "--out_dir", required=True,
                   help="output root directory")
    w.add_argument("--methods", "--metrics", default=",".join(metrics.METHODS),
                   help="comma-separated methods; also accepts the "
                        f"scoring variants {', '.join(metrics.VARIANTS)}")
    w.add_argument("--ratios", default="0.25,0.5,0.75",
                   help="comma-separated prune ratios")
    w.add_argument("--saliency", default=None,
                   help="reuse an existing saliency file or directory")
    _add_export_args(w)
    _add_calibration_args(w)


    d = sub.add_parser("diagnose", help="inspect a saliency file",
                       description="Report what the scores are doing, by depth.")
    d.add_argument("--saliency", required=True)
    d.add_argument("--keep", type=int, default=None,
                   help="budget to analyse (default: half the experts)")
    d.add_argument("--export", default=None,
                   help="write keep-set JSONs for every method to this dir")


    v = sub.add_parser(
        "verify", help="check a pruned checkpoint against the sliced original",
        description="Compare per-token log-probabilities of the pruned "
                    "checkpoint against the source model with the same "
                    "experts deleted in memory. They must agree to "
                    "floating-point noise.")
    v.add_argument("--src", required=True, help="source (unpruned) model")
    v.add_argument("--pruned", required=True, help="pruned checkpoint")
    v.add_argument("--keep", default=None,
                   help="keep-set JSON (default: <pruned>/kept_expert_indices.json)")
    v.add_argument("--trust-remote-code", action="store_true",
                   help="allow execution of model repository code")
    v.add_argument("--tensors-only", "--tensors_only", action="store_true",
                   help="compare written tensors only; no forward pass")
    v.add_argument("--num-batches", "--num_batches", type=int, default=4,
                   help="batches to score (default 4; this is a check, not a "
                        "benchmark)")
    v.add_argument("--data", default=DEFAULT_DATA)
    v.add_argument("--max-len", "--max_len", type=int, default=2048)
    v.add_argument("--adapter", default=None, choices=list_adapters())
    v.add_argument("--mode", choices=["model", "tensors", "stream"], default="model",
                   help="resident logprobs, stored tensors, or streamed native layer outputs")
    v.add_argument("--device", default="cpu", help="device for streamed verification")
    v.add_argument("--max-layers", type=int, default=None,
                   help="optional prefix only; omit for all decoder layers")
    v.add_argument("--atol", type=float, default=0.0)
    v.add_argument("--rtol", type=float, default=1e-3)


    sub.add_parser("models", help="list supported architectures")

    return ap


def _calib_kwargs(a) -> dict:
    import torch

    return dict(num_batches=a.num_batches, batch_size=a.batch_size,
                max_len=a.max_len, seed=a.seed, expert_chunk=a.expert_chunk,
                verify=not a.no_verify, cache=not a.no_cache,
                dtype=getattr(torch, a.dtype), device_map=a.device_map,
                calibration_limit=a.calibration_limit, trust_remote_code=a.trust_remote_code,
                execution=a.execution, stream_device=a.stream_device, chunk_attn=a.chunk_attn,
                stream_batch_window=a.stream_batch_window)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "models":
        from .adapters.base import _REGISTRY

        print("adapters (auto-detected from config.model_type):\n")
        for name in list_adapters():
            cls = _REGISTRY[name]
            doc = (cls.__doc__ or "").strip().splitlines()[0]
            print(f"  {name:<10} {doc}")
            if cls.MODEL_TYPES:
                print(f"{'':<12}model_type: {', '.join(cls.MODEL_TYPES)}")
        print("\nmethods:\n")
        for m in metrics.CHOICES:
            print(f"  {m:<12} {metrics.DESCRIPTIONS[m]}")
        return 0

    if args.command == "saliency":
        collect_saliency(args.model, data=args.data, out_dir=args.out,
                         adapter=args.adapter, **_calib_kwargs(args))
        return 0

    if args.command == "prune":
        prune(args.model, args.out, method=args.method, ratio=args.ratio,
              target_experts=args.target_experts, saliency=args.saliency,
              keep_indices=args.keep_indices, data=args.data,
              target_top_k=args.target_top_k,
              max_shard_size=args.max_shard_size, adapter=args.adapter,
              export_mode=args.export_mode, hash_policy=args.hash_policy,
              mtp_policy=args.mtp_policy, aggregation=args.aggregation,
              **_calib_kwargs(args))
        return 0

    if args.command == "sweep":
        outs = sweep(
            args.model, args.out,
            methods=_csv(args.methods),
            ratios=[float(r) for r in _csv(args.ratios)],
            data=args.data, saliency=args.saliency, adapter=args.adapter,
            export_mode=args.export_mode, hash_policy=args.hash_policy,
            mtp_policy=args.mtp_policy, aggregation=args.aggregation,
            **_calib_kwargs(args))
        print(f"\n[sweep] produced {len(outs)} models under {args.out}")
        return 0

    if args.command == "diagnose":
        from .diagnose import report

        report(args.saliency, keep=args.keep, export=args.export)
        return 0

    if args.command == "verify":
        from .verify_cmd import run

        return 0 if run(
            src=args.src, pruned=args.pruned, keep=args.keep,
            tensors_only=args.tensors_only, data=args.data,
            num_batches=args.num_batches, max_len=args.max_len,
            adapter=args.adapter, trust_remote_code=args.trust_remote_code,
            mode=args.mode, device=args.device, max_layers=args.max_layers,
            atol=args.atol, rtol=args.rtol,
        ) else 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
