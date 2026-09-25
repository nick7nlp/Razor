# Datasheet for RazorCal

## Purpose

RazorCal is a general-purpose calibration corpus maintained as part of RAZOR,
which uses it to estimate per-expert statistics for MoE pruning. The records
carry no architecture-specific fields, so the corpus applies equally to FFN or
layer pruning, perplexity measurement and other calibration-based compression.
It is not an independent benchmark or a guarantee of downstream model quality.

## Composition and format

`RazorCal.json` contains 2,048 samples in seven domains: Coding has 512
samples; Math, Science (STEM), Chinese-STEM, Instruction Following, Tool Calling
and World Knowledge have 256 each. The file is approximately 8 MB.

The top-level JSON object contains `meta` and `data`. Records carry `domain`,
`sub_source` and chat `messages`; some also contain `tools` or `coding_subclass`.
No train/validation/test split is provided. See [README.md](README.md) for
loading and the source-label inventory.

## Provenance

The recorded sources include NVIDIA Nemotron, Dolci, and Teknium OpenHermes-2.5
via HuggingFaceTB smoltalk2. The 160 Math records labeled
`nemotron-cascade2-math` come from the math slice of NVIDIA
Nemotron-Cascade-2-SFT-Data, recorded as CC-BY-4.0. That slice was taken whole
rather than sampled per record, so the released metadata identifies the
upstream dataset but not an individual upstream record for each sample.

Source labels do not establish record-level source fidelity, absence of
synthetic content, privacy clearance or license compatibility. See
[FIDELITY_AUDIT.md](FIDELITY_AUDIT.md).

## Limitations

- Calibration coverage is limited to the released samples; it does not establish
  coverage of every language, task or context length.
- Source-corpus overlap with evaluation sets is possible and needs review for
  any benchmark comparison.
- Chat rendering and tool handling depend on the model's tokenizer template.
  There are 587 tool messages without populated `tool_call_id` links and
  11 tool definitions with empty `parameters` objects.
- Some optional message fields are null or template-specific. Consumers should
  validate the fields they use without modifying the original data files.
- Reasoning traces are omitted. Contact identifiers, home-directory usernames,
  private-network examples and literal password examples are replaced with
  placeholders.
- Automated replacement does not establish comprehensive privacy or safety
  clearance; downstream use needs an appropriate review.

## Distribution and terms

`RazorCal.json` is distributed through Git LFS rather than the Python wheel.

Applicable upstream licenses and attribution requirements remain in effect.
There is no claim that one dataset-wide license supersedes all upstream terms,
or that redistribution rights have been fully resolved for every record.
See [LICENSE-DATA](LICENSE-DATA) and [NOTICE](../NOTICE).

Issues may be reported through the RAZOR repository. Citation metadata is in
[CITATION.cff](../CITATION.cff).
