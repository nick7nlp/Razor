
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "data" / "RazorCal.json"


def load(path=DEFAULT):
    
    with open(path) as f:
        blob = json.load(f)
    if isinstance(blob, dict):
        return blob.get("data", []), blob.get("meta", {})
    return blob, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT))
    ap.add_argument("--show", type=int, default=1,
                    help="print this many example records")
    args = ap.parse_args()

    records, meta = load(args.data)
    print(f"{len(records)} records from {args.data}")
    if meta:
        keys = ", ".join(sorted(meta)[:8])
        print(f"meta keys: {keys}")

    print("\n=== domains ===")
    by_domain = collections.Counter(r.get("domain") for r in records)
    for dom, n in by_domain.most_common():
        print(f"  {dom:<24} {n:>5}  ({n / len(records):.1%})")

    print("\n=== structure ===")
    n_tools = sum(1 for r in records if r.get("tools"))
    n_multi = sum(1 for r in records
                  if sum(1 for m in r.get("messages", []) if m.get("role") == "user") > 1)
    n_sys = sum(1 for r in records
                if any(m.get("role") == "system" for m in r.get("messages", [])))
    turns = [len(r.get("messages", [])) for r in records]
    print(f"  with tool schemas   {n_tools:>5}  ({n_tools / len(records):.1%})")
    print(f"  multi-turn (user>1) {n_multi:>5}  ({n_multi / len(records):.1%})")
    print(f"  with system prompt  {n_sys:>5}  ({n_sys / len(records):.1%})")
    print(f"  messages per record min={min(turns)} mean={sum(turns)/len(turns):.1f} max={max(turns)}")

    print("\n=== upstream sources (top 10) ===")
    for src, n in collections.Counter(
            r.get("sub_source") for r in records).most_common(10):
        print(f"  {str(src):<52} {n:>5}")

    for r in records[:args.show]:
        print(f"\n=== example ({r.get('domain')}, from {r.get('sub_source')}) ===")
        for m in r.get("messages", [])[:4]:
            content = str(m.get("content", ""))
            body = content[:300] + ("..." if len(content) > 300 else "")
            print(f"  [{m.get('role')}] {body}")
        if r.get("tools"):
            print(f"  [tools] {len(r['tools'])} schema(s)")


if __name__ == "__main__":
    main()
