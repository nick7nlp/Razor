#!/usr/bin/env python3

import collections
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.abspath(os.path.join(HERE, "..", "RazorCal.json"))

EXPECTED_SAMPLES = 2048
EXPECTED_DOMAINS = {
    "Math": 256,
    "Science (STEM)": 256,
    "World Knowledge": 256,
    "Coding": 512,
    "Instruction Following": 256,
    "Tool Calling": 256,
    "Chinese-STEM": 256,
}
LENGTH_CAP = 32000


class Audit:
    def __init__(self):
        self.failures = []

    def check(self, ok, label, detail=""):
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label}" + (f"  ({detail})" if detail else ""))
        if not ok:
            self.failures.append(label)
        return ok


def first_user_hash(sample):
    for m in sample["messages"]:
        if m.get("role") == "user":
            return hashlib.sha256((m.get("content") or "").encode()).hexdigest()
    return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
    print(f"[load] {path}", flush=True)
    with open(path) as f:
        obj = json.load(f)
    data = obj["data"]
    a = Audit()

    print("\n=== composition ===")
    a.check(len(data) == EXPECTED_SAMPLES, "sample count",
            f"{len(data)} (expected {EXPECTED_SAMPLES})")
    dom = collections.Counter(s["domain"] for s in data)
    a.check(dict(dom) == EXPECTED_DOMAINS, "domain sizes", f"{dict(dom)}")

    print("\n=== uniqueness ===")
    whole = collections.Counter(
        json.dumps(s, sort_keys=True, ensure_ascii=False) for s in data)
    n_dup = sum(v - 1 for v in whole.values() if v > 1)
    a.check(n_dup == 0, "no exact duplicate samples", f"{n_dup} found")

    hashes = [first_user_hash(s) for s in data]
    n_nouser = sum(1 for h in hashes if h is None)
    a.check(n_nouser == 0, "every sample has a user message", f"{n_nouser} missing")
    dupes = [k for k, v in collections.Counter(h for h in hashes if h).items() if v > 1]
    a.check(not dupes, "no duplicate first-user hashes", f"{len(dupes)} collisions")

    print("\n=== message schema ===")
    n_msgs = sum(len(s["messages"]) for s in data)
    bad_content = sum(1 for s in data for m in s["messages"]
                      if not isinstance(m.get("content"), str))
    a.check(bad_content == 0, "all message content is str",
            f"{bad_content}/{n_msgs} bad")

    bad_tc_type = sum(1 for s in data for m in s["messages"]
                      if m.get("tool_calls") is not None
                      and not isinstance(m["tool_calls"], list))
    a.check(bad_tc_type == 0, "tool_calls is always a list (or absent/null)",
            f"{bad_tc_type} bad")

    bad_args = sum(1 for s in data for m in s["messages"]
                   for tc in (m.get("tool_calls") or [])
                   if not isinstance(tc, dict)
                   or not isinstance(tc.get("function", {}).get("arguments"), dict))
    a.check(bad_args == 0, "all tool_call arguments are dict", f"{bad_args} bad")

    print("\n=== length cap ===")
    lengths = [sum(len(m.get("content") or "") for m in s["messages"]) for s in data]
    over = sum(1 for x in lengths if x > LENGTH_CAP)
    a.check(over == 0, f"no sample exceeds {LENGTH_CAP} chars",
            f"{over} over, max {max(lengths)}")

    print("\n=== tool definitions ===")
    n_defs = spec = empty = malformed = 0
    for s in data:
        for t in s.get("tools") or []:
            n_defs += 1
            fn = t.get("function", {})
            p = fn.get("parameters")
            named = t.get("type") == "function" and isinstance(fn.get("name"), str) and fn["name"]
            if not named or not isinstance(p, dict):
                malformed += 1
            elif p == {}:
                empty += 1
            elif p.get("type") == "object" and isinstance(p.get("properties"), dict):
                spec += 1
            else:
                malformed += 1
    print(f"       total={n_defs} spec-form={spec} zero-arg-empty={empty} malformed={malformed}")
    a.check(malformed == 0, "no malformed tool definitions", f"{malformed} bad")
    a.check(spec + empty == n_defs, "every tool def accounted for")

    print("\n=== tool_call_id fidelity (no real id may be present, any role) ===")
    tool_msgs = [(s["domain"], m) for s in data
                 for m in s["messages"] if m.get("role") == "tool"]
    per_dom = collections.Counter(d for d, _ in tool_msgs)
    with_id = collections.Counter(d for d, m in tool_msgs
                                  if m.get("tool_call_id") is not None)
    for d in sorted(per_dom):
        print(f"       {d:<24} tool msgs={per_dom[d]:>4}  with real id={with_id[d]}")
    
    real_ids = sum(1 for s in data for m in s["messages"]
                   if m.get("tool_call_id") is not None)
    null_ids = sum(1 for s in data for m in s["messages"]
                   if "tool_call_id" in m and m["tool_call_id"] is None)
    print(f"       across all roles: real ids={real_ids}, explicit-null fields={null_ids}")
    a.check(real_ids == 0, "no synthesised tool_call_id (any role)",
            f"{real_ids} carry a real id")

    print("\n=== explicit null fields (harmless, but templates may care) ===")
    nulls = collections.Counter(k for s in data for m in s["messages"]
                                for k, v in m.items() if v is None)
    for k, v in sorted(nulls.items(), key=lambda kv: -kv[1]):
        print(f"       {k:<20} {v}")
    print(f"       total: {sum(nulls.values())}")

    print("\n" + "=" * 60)
    if a.failures:
        print(f"AUDIT FAILED — {len(a.failures)} check(s) failed:")
        for f in a.failures:
            print(f"  - {f}")
        return 1
    print(f"AUDIT PASSED — all checks green on {len(data)} samples.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
