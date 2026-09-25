"""Load JSON, JSONL, or Hugging Face records into packed calibration batches."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

from .geometry import DEFAULT_SEED


def _read_json_file(path: Path) -> List[dict]:
    if path.suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        return obj["data"]
    if isinstance(obj, list):
        return obj
    raise ValueError("unrecognized JSON layout in %s: expected {'data': [...]} "
                     "or a list" % path)


def load_records(source: str, split: str = "train",
                 limit: Optional[int] = None) -> List[dict]:
    """Read nonempty records; Hugging Face sources require a positive limit."""
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)
                              or limit < 1):
        raise ValueError("limit must be a positive integer")
    if source.startswith("hf:"):
        if limit is None:
            raise ValueError("Hugging Face calibration requires a positive limit")
        from itertools import islice
        from datasets import load_dataset

        parts = source[3:].split(",", 1)
        if not parts[0] or (len(parts) > 1 and not parts[1]):
            raise ValueError("expected hf:<dataset>[,<configuration>]")
        kwargs: Dict[str, Any] = {"split": split, "streaming": True}
        if len(parts) > 1:
            kwargs["name"] = parts[1]
        rows = list(islice(load_dataset(parts[0], **kwargs), limit))
    else:
        path = Path(source)
        if path.is_dir():
            rows = []
            for p in sorted(list(path.glob("*.json")) + list(path.glob("*.jsonl"))):
                rows.extend(_read_json_file(p))
        else:
            rows = _read_json_file(path)
        if limit is not None:
            rows = rows[:limit]
    if not rows:
        raise ValueError("calibration source contains no records")
    if any(not isinstance(row, dict) or not row for row in rows):
        raise ValueError("calibration records must be nonempty objects")
    return rows


def _clean_messages(messages: Iterable[dict]) -> List[dict]:
    """Normalize message fields for ``apply_chat_template``."""
    out: List[dict] = []
    for m in messages:
        m = dict(m)

        calls = m.get("tool_calls")
        if calls:
            fixed = []
            for c in calls:
                c = dict(c)
                fn = dict(c.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        pass
                c["function"] = fn
                fixed.append(c)
            m["tool_calls"] = fixed

        for k in ("tool_call_id", "name"):
            m.pop(k, None)

        m.pop("reasoning_content", None)

        content = m.get("content")
        if content is None:
            m["content"] = ""
        elif not isinstance(content, str):
            m["content"] = json.dumps(content, ensure_ascii=False)

        out.append(m)
    return out


def render(records: List[dict], tokenizer, verbose: bool = True) -> List[str]:
    """Turn records into strings, via the chat template where applicable."""
    if not records:
        raise ValueError("calibration source contains no records")
    texts: List[str] = []
    n_chat = 0
    for rec in records:
        if not isinstance(rec, dict) or not rec:
            raise ValueError("calibration records must be nonempty objects")
        msgs = rec.get("messages")
        if isinstance(msgs, list) and msgs:
            render_record = getattr(tokenizer, "render_record", None)
            if callable(render_record):
                text = render_record(rec)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("checkpoint encoder returned no renderable text")
                texts.append(text)
                n_chat += 1
                continue
            kwargs: Dict[str, Any] = dict(add_generation_prompt=False,
                                          tokenize=False,
                                          enable_thinking=False)
            tools = rec.get("tools")
            if tools:
                kwargs["tools"] = tools
            try:
                text = tokenizer.apply_chat_template(_clean_messages(msgs),
                                                     **kwargs)
                n_chat += 1
            except Exception:
                text = "\n".join(str(m.get("content", "")) for m in msgs)
            if text.strip():
                texts.append(text)
            continue

        text = rec.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text)
        else:
            texts.append(json.dumps(rec, ensure_ascii=False)[:4096])

    if not texts:
        raise ValueError("calibration records contain no renderable text")
    if verbose:
        print("[data] rendered %d samples (chat=%d)" % (len(texts), n_chat))
    return texts


def build_batches(source: str, tokenizer, num_batches: int = 128,
                  batch_size: int = 1, max_len: int = 32768,
                  seed: int = DEFAULT_SEED, limit: Optional[int] = None,
                  verbose: bool = True) -> List[Dict[str, torch.Tensor]]:
    """Return single-pass packed batches of shape ``[batch, max_len]``."""
    from . import geometry

    geometry._integer(batch_size, "batch_size", 1)
    geometry._integer(max_len, "max_len", 1)
    geometry._integer(num_batches, "num_batches", geometry.EXHAUST)
    if num_batches == geometry.EXHAUST:
        want_rows = geometry.EXHAUST
    elif num_batches < 1:
        raise ValueError(
            f"num_batches must be >= 1 or geometry.EXHAUST ({geometry.EXHAUST}),"
            f" got {num_batches}")
    else:
        want_rows = num_batches * batch_size

    if verbose:
        print(f"[data] loading {source}")
    records = load_records(source, limit=limit)
    texts = render(records, tokenizer, verbose=verbose)
    batches, row_ids = geometry.pack_rows(
        texts, tokenizer, row_len=max_len, n_rows=want_rows,
        batch_size=batch_size, seed=seed, verbose=verbose)
    if verbose and want_rows != geometry.EXHAUST and len(row_ids) < want_rows:
        print("[data] corpus fills %d rows, fewer than the %d requested "
              "(%d batches x %d); using one full pass instead of re-serving "
              "samples" % (len(row_ids), want_rows, num_batches, batch_size))
    return batches
