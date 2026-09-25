# RazorCal

RazorCal is a general-purpose calibration corpus released with RAZOR.
`RazorCal.json` contains 2,048 samples across seven domains (about 8 MB). Each
record includes input messages and source attribution. The records carry no
architecture-specific fields, so the corpus also suits FFN or layer pruning,
perplexity measurement and other calibration-based compression; RAZOR uses it
to estimate per-expert statistics. Reasoning traces are omitted because the
calibration loader does not consume them. Potential contact identifiers,
home-directory usernames, private-network examples and literal password
examples are replaced with placeholders.

## Composition

| Domain | Samples |
|---|---:|
| Coding | 512 |
| Math | 256 |
| Science (STEM) | 256 |
| Chinese-STEM | 256 |
| Instruction Following | 256 |
| Tool Calling | 256 |
| World Knowledge | 256 |
| **Total** | **2,048** |

## Format

The JSON object has `meta` and `data` fields. Each item in `data` includes
`domain`, `sub_source` and a `messages` list. Some records additionally contain
`tools` or `coding_subclass`.

```python
import json

with open("data/RazorCal.json", encoding="utf-8") as handle:
    corpus = json.load(handle)
samples = corpus["data"]
```

For the RAZOR CLI:

```bash
razor saliency --model <model-path> \
               --data data/RazorCal.json --out out/sal
```

The JSON file is stored with Git LFS and is not bundled in the Python wheel.

## Recorded sources

A source label records attribution information; it is not, by itself, proof
of a record-level match to an upstream release.

| Domain | `sub_source` labels and counts | Recorded source description |
|---|---|---|
| Math | `nemotron-cascade2-math` (160); `nemotron-math-v2-high-aime-style` (96) | Nemotron-Cascade-2-SFT-Data math slice; Nemotron-Math-V2 |
| Science (STEM) | `Nemotron-Science-V1-RQA` (176); `Nemotron-Science-V1-MCQ` (80) | Nemotron-Science-V1 |
| World Knowledge | `OpenHermes-2.5` (256) | OpenHermes-2.5 via smoltalk2 |
| Coding | `Dolci-Python-Algorithms` (205); `competitive-programming-python` (141); `competitive-programming-cpp` (115); `Nemotron-SFT-SWE-V2-agentless` (40); `Nemotron-OpenCode-general` (11) | Dolci; competitive-programming, SWE and OpenCode corpora |
| Instruction Following | `Dolci-Precise-IF` (256) | Dolci instruction-following corpus |
| Tool Calling | `Nemotron-Agentic-tool-calling-v2` (256) | Nemotron agentic/tool-calling corpus |
| Chinese-STEM | `multilingual-chinese-math` (86); `multilingual-chinese-code` (85); `multilingual-chinese-stem` (85) | Multilingual Chinese categories |

The `nemotron-cascade2-math` records were taken as a whole domain slice rather
than sampled individually, so the label records the upstream dataset but not a
per-record identifier within it. Source labels alone do not establish
provenance.
See [FIDELITY_AUDIT.md](FIDELITY_AUDIT.md) for provenance boundaries and
[LICENSE-DATA](LICENSE-DATA) for recorded license information.

## Usage limitations

RazorCal is calibration data, not an independent evaluation benchmark.
Domain coverage does not guarantee preservation of model quality. Review
possible overlap with evaluation data for the intended use.

The record array contains 587 tool messages without populated `tool_call_id`
links and 11 tool definitions with empty `parameters` objects. Tokenizer
chat templates and schema validators may require additional handling at load
time. Keep the original files unchanged when making consumer-specific conversions.

Public source material can still contain personal references or unsuitable
content; source labels do not establish that records are free of such content.
See [DATASHEET.md](DATASHEET.md) for intended use and limitations.

## Citation and licensing

Use [CITATION.cff](../CITATION.cff) for citation metadata and retain applicable
upstream attribution. Source-specific terms continue to apply; no single
replacement license is asserted for all RazorCal records.
See [LICENSE-DATA](LICENSE-DATA) and [NOTICE](../NOTICE).
