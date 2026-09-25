#!/usr/bin/env python3

import json
import os
import sys
from collections import Counter


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(HERE, "..", "RazorCal.json"))


def load_razorcal(path=DATA_PATH):
    
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["meta"], obj["data"]


def get_domain(data, domain_name):
    
    return [s for s in data if s["domain"] == domain_name]


def render_sample(sample, tokenizer):
    
    kw = {"tokenize": False, "add_generation_prompt": False}
    if sample.get("tools"):
        kw["tools"] = sample["tools"]
    return tokenizer.apply_chat_template(sample["messages"], **kw)


def razorcal_iter(tokenizer, path=DATA_PATH, max_length=4096):
    
    _, data = load_razorcal(path)
    for s in data:
        text = render_sample(s, tokenizer)
        enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        yield enc["input_ids"][0]


def main():
    md, data = load_razorcal()
    print(f"{md['name']} — {md['description'].split('.')[0]}.")
    print(f"  total samples: {len(data)}")

    dc = Counter(s["domain"] for s in data)
    print(f"\n  per-domain counts:")
    for k, v in sorted(dc.items()):
        print(f"    {k:<28} {v}")


    print(f"\n  --- Example: 1 sample from Math ---")
    s = get_domain(data, "Math")[0]
    print(f"    domain={s['domain']}, sub_source={s.get('sub_source')}, "
          f"n_msgs={len(s['messages'])}, has_tools={bool(s.get('tools'))}")
    print(f"    user: {s['messages'][0]['content'][:150]!r}")

    print(f"\n  --- Example: 1 sample from Tool Calling ---")
    s = get_domain(data, "Tool Calling")[0]
    print(f"    domain={s['domain']}, has_tools={bool(s.get('tools'))} "
          f"({len(s.get('tools') or [])} tools)")
    if s.get("tools"):
        print(f"    first tool: {s['tools'][0]['function']['name']}")


    try:
        from transformers import AutoTokenizer
        
        tok_path = os.environ.get("RAZORCAL_TOKENIZER")
        if tok_path:
            print(f"\n  --- Rendering with tokenizer at {tok_path} ---")
            tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
            rendered = render_sample(data[0], tok)
            print(f"    rendered length (chars): {len(rendered)}")
            print(f"    first 200 chars: {rendered[:200]!r}")
        else:
            print(f"\n  (set RAZORCAL_TOKENIZER env to a HF tokenizer path/id "
                  f"to render a sample)")
    except ImportError:
        print(f"\n  (transformers not installed; skipping chat_template demo)")


if __name__ == "__main__":
    main()
