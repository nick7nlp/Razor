#!/usr/bin/env python3

import json
import os
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.abspath(os.path.join(HERE, "..", "RazorCal.json"))
FIG_DIR = os.path.abspath(os.path.join(HERE, "..", "figures"))


DOMAIN_COLORS = {
    "Math":                  "#E15759",
    "Science (STEM)":        "#4E79A7",
    "World Knowledge":       "#8CD17D",
    "Coding":                "#F28E2B",
    "Instruction Following": "#76B7B2",
    "Tool Calling":          "#EDC948",
    "Chinese-STEM":          "#E7A32C",
}

DOMAIN_ORDER = [
    "Math", "Science (STEM)", "World Knowledge", "Coding",
    "Instruction Following", "Tool Calling", "Chinese-STEM",
]


def ordered_domains(present):
    
    known = [d for d in DOMAIN_ORDER if d in present]
    extra = sorted(d for d in present if d not in DOMAIN_COLORS)
    return known + extra


def load_data(path):
    print(f"[load] {path}", flush=True)
    with open(path) as f:
        d = json.load(f)
    return d["data"]


def sample_total_chars(s):
    return sum(len(m.get("content") or "") for m in s["messages"])


def figure_domain_sizes(data, out_path):
    dc = Counter(s["domain"] for s in data)
    doms = ordered_domains(dc)
    sizes = [dc[d] for d in doms]
    colors = [DOMAIN_COLORS.get(d, "#888") for d in doms]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=140)
    bars = ax.bar(range(len(doms)), sizes, color=colors, edgecolor="#333", linewidth=0.6)
    for i, (b, s) in enumerate(zip(bars, sizes)):
        ax.text(b.get_x() + b.get_width() / 2, s + 5, str(s),
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(doms)))
    ax.set_xticklabels(doms, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("# samples", fontsize=11)
    ax.set_title(f"RazorCal domain sizes  (total = {sum(sizes)})", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(sizes) * 1.12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}", flush=True)


def figure_length_distribution(data, out_path):
    per_dom = defaultdict(list)
    for s in data:
        per_dom[s["domain"]].append(sample_total_chars(s))
    doms = ordered_domains(per_dom)
    box_data = [per_dom[d] for d in doms]
    colors = [DOMAIN_COLORS.get(d, "#888") for d in doms]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    bp = ax.boxplot(box_data, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=3, alpha=0.35, markerfacecolor="#333"),
                    medianprops=dict(color="black", linewidth=1.4),
                    whis=(5, 95))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor("#222")

    ax.set_yscale("log")
    ax.set_xticks(range(1, len(doms) + 1))
    ax.set_xticklabels(doms, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("chars per sample (log scale)", fontsize=11)
    ax.set_title("RazorCal — sample length distribution by domain\n(box: p25-p75; whiskers: p5-p95; dots: outliers)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--", which="both")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}", flush=True)


def figure_message_count(data, out_path):
    per_dom_msgs = defaultdict(list)
    for s in data:
        per_dom_msgs[s["domain"]].append(len(s["messages"]))
    doms = ordered_domains(per_dom_msgs)
    avgs = [np.mean(per_dom_msgs[d]) for d in doms]
    maxs = [np.max(per_dom_msgs[d]) for d in doms]
    colors = [DOMAIN_COLORS.get(d, "#888") for d in doms]

    x = np.arange(len(doms))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=140)
    bars_avg = ax.bar(x - width/2, avgs, width, label="mean", color=colors, edgecolor="#222", linewidth=0.6)
    bars_max = ax.bar(x + width/2, maxs, width, label="max",  color=colors, edgecolor="#222", linewidth=0.6, alpha=0.4, hatch="//")
    for b, v in zip(bars_avg, avgs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    for b, v in zip(bars_max, maxs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v}", ha="center", va="bottom", fontsize=7, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(doms, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("# messages per sample", fontsize=11)
    ax.set_title("RazorCal — message count per sample (mean vs max)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}", flush=True)


def figure_tool_usage(data, out_path):
    per_dom_tot = Counter()
    per_dom_with_tools = Counter()
    per_dom_with_tool_msgs = Counter()
    for s in data:
        dom = s["domain"]
        per_dom_tot[dom] += 1
        if s.get("tools"):
            per_dom_with_tools[dom] += 1
        if any(m.get("role") == "tool" for m in s["messages"]):
            per_dom_with_tool_msgs[dom] += 1

    doms = ordered_domains(per_dom_tot)
    n_tot = np.array([per_dom_tot[d] for d in doms])
    n_tools = np.array([per_dom_with_tools[d] for d in doms])
    n_tmsg = np.array([per_dom_with_tool_msgs[d] for d in doms])
    n_pure = n_tot - np.maximum(n_tools, n_tmsg)

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=140)
    x = np.arange(len(doms))
    ax.bar(x, n_pure,  color="#B0B0B0", label="pure chat")
    ax.bar(x, np.maximum(n_tools, n_tmsg), bottom=n_pure,
           color=[DOMAIN_COLORS.get(d, "#888") for d in doms],
           edgecolor="#222", linewidth=0.6,
           label="uses tools (has 'tools' field or 'tool' msg)")

    for i, d in enumerate(doms):
        with_t = max(per_dom_with_tools[d], per_dom_with_tool_msgs[d])
        if with_t > 0:
            ax.text(i, per_dom_tot[d] + 5, f"{with_t}/{per_dom_tot[d]}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(doms, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("# samples", fontsize=11)
    ax.set_title("RazorCal — tool-augmented vs pure-chat samples per domain", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(n_tot) * 1.15)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}", flush=True)


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA
    data = load_data(data_path)

    os.makedirs(FIG_DIR, exist_ok=True)
    figure_domain_sizes(data,        os.path.join(FIG_DIR, "domain_sizes.png"))
    figure_length_distribution(data, os.path.join(FIG_DIR, "length_distribution.png"))
    figure_message_count(data,       os.path.join(FIG_DIR, "message_count.png"))
    figure_tool_usage(data,          os.path.join(FIG_DIR, "tool_usage.png"))

    print("\n[done] all four figures generated in", FIG_DIR, flush=True)


if __name__ == "__main__":
    main()
