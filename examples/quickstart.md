# Quickstart

## Install

From the repository root:

```bash
pip install -e .
git lfs pull
razor models
```

Review [model compatibility](../docs/models.md) before choosing a checkpoint.
Replace `<model-path>` with its path. Checkpoint output directories must not
already exist. Hardware requirements depend on the model and calibration settings.

Model repository code is disabled by default. Use `--trust-remote-code` only
when the checkpoint requires it and you have reviewed and trust its code.
For collection, `--device-map` controls placement, `--dtype` selects
`bfloat16`, `float16` or `float32`, and `--calibration-limit` caps input records.

## Prune

```bash
razor prune --model <model-path> --method razor --ratio 0.5 \
            --out out/pruned-razor-50
```

`--ratio 0.5` removes half the experts in each pruned layer, subject to the
model's routing constraints. Use `--target-experts` instead to specify the
number to keep. Available methods are `razor`, `reap`, `ean` and `frequency`.

The same flow is available through Python:

```python
import razor
razor.prune("<model-path>", "out/pruned", ratio=0.5)
```

The target families use native layer streaming and source-format safetensors
export by default. To select execution and aggregation explicitly:

```bash
razor prune --model <model-path> --ratio 0.5 --aggregation rms \
            --execution streaming --stream-device cuda:0 \
            --export-mode checkpoint --out out/pruned-razor-50
```

`--aggregation rms` is the default, matching the paper; `mean` and `sum` are
available for aggregation ablations. RMS requires second moments from the
collector; it does not square an existing mean. `--execution resident` is
available when the entire decoded model fits memory. `--chunk-attn 128` bounds
supported native eager-attention query chunks. `--stream-batch-window` controls
activation residency: the default is one batch; larger windows reduce repeated
weight reads but consume more memory without changing calibration coverage.

Use `--hash-policy remap` for deterministic, collision-free reassignment of
removed hash experts. `--hash-policy preserve --trust-remote-code` keeps hash
layers at their original width and emits a mixed-width loader. The default
`--mtp-policy router_norm` fills unobserved auxiliary keep-sets; `drop` and
`error` are explicit alternatives. These policies are heuristics or structural
choices, not additional RAZOR measurements.

## Reuse saliency

```bash
razor saliency --model <model-path> \
               --data data/RazorCal.json --out out/sal
razor prune --model <model-path> --saliency out/sal \
            --method reap --ratio 0.5 --out out/pruned-reap-50
razor sweep --model <model-path> --saliency out/sal \
            --methods razor,reap,ean,frequency --ratios 0.25,0.5 \
            --out out/grid
```

A collection writes `out/sal/observer_data.pt`. Reuse it only with the matching
source model and configuration. Only load score packs from trusted sources.

## Calibration data

The default corpus is `data/RazorCal.json`. Custom inputs can be a JSON or
JSONL file, a directory containing such files, or an `hf:<dataset-id>` source.
Hub loading requires the optional `hub` dependencies.

```bash
razor saliency --model <model-path> --data my_data.jsonl --out out/custom-sal
```

Records may contain a `text` field or chat `messages`. Chat rendering depends
on the tokenizer's template and tool-schema compatibility. Review the
[data limitations](../data/DATASHEET.md) when using RazorCal.

`--num-batches -1` consumes the available packed corpus once; a positive value
limits the number of batches. `--batch-size` and `--max-len` also affect the
number of tokens processed. Lowering `--expert-chunk`, `--batch-size` or
`--max-len` can reduce peak memory use; record the settings used for comparisons.

## Checkpoint checks

```bash
razor verify --src <model-path> --pruned out/pruned-razor-50 --tensors-only
razor verify --src <model-path> --pruned out/pruned-razor-50
```

`--tensors-only` compares local safetensors against the source selection,
including paired quantization scales and unchanged tensors. Native storage
profiles also validate configuration updates and hash tables. Tensor checks do
not execute the model and cannot prove forward behavior.

For large checkpoints, compare native layer outputs without resident full-model
loading:

```bash
razor verify --src <model-path> --pruned out/pruned-razor-50 \
             --mode stream --device cuda:0 --data data/RazorCal.json
```

Omit `--max-layers` to check all decoder layers; using it verifies only a prefix.
This streamed check compares layer outputs, not a complete language-model
logprob distribution. The default `--mode model` compares sampled next-token
logprobs with a sliced source reference. Neither establishes downstream task
quality. See [model compatibility](../docs/models.md) for backend requirements
and grouped-routing constraints.

## Inspect scores

```bash
razor diagnose --saliency out/sal --keep 4
```

Choose a keep count compatible with the scored layer sizes. To export keep sets,
add `--export out/keeps`. Use a keep set only with its matching source model,
layer mapping and expert count.
