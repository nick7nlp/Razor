# Model compatibility

Model identity, in-memory routing, checkpoint storage and execution backend are
separate contracts. A recognized configuration does not certify every weight
release or inference runtime.

## Model families

| Family | Configuration types | Adapter | Important behavior |
|---|---|---|---|
| Hy3 | `hy_v3` | `hunyuan` | FP32 sigmoid projection; selection bias separate from weights; `router_scaling_factor`; shared experts |
| DeepSeek-V4 Flash / Pro | `deepseek_v4` | `deepseek` | Native sqrt-softplus router, bounded expert activation, token-ID hash routing, FP8 and MXFP4 storage |
| GLM-4.7-Flash | `glm4_moe_lite` | `glm` | Native router and expert layout; original-format export includes indexed auxiliary layers |
| GLM-5.2 | `glm_moe_dsa` | `glm` | DSA state is carried through the native decoder forward |
| GLM-5.3 | `glm5_next`, `glm5_next_text` | `glm` | Conditional/text wrapper, native mHC/KDA/DSA control flow, FP8 weights and scales |
| Qwen3.5 / Qwen3.6 MoE | `qwen3_5_moe`, `qwen3_5_moe_text`, `qwen3_6_moe`, `qwen3_6_moe_text` | `qwen` | Text/conditional wrappers, fused expert tensors, unchanged shared expert and vision weights |
| Qwen3.8 Flash Next | `qwen4_exp`, `qwen4_exp_text` | `qwen` | Native conditional/text classes, fused experts, HC, PLE/ngram and QSA; independent `mtp_N` keep sets |
| Gemma 4 MoE | `gemma4`, `gemma4_text` | `gemma4` | Decoder-level router, normalized expert inputs and per-expert output gain |
| Kimi K3 / Kimi Linear | `kimi_k3`, `kimi_linear` | `kimi` | Native gate and activation; latent expert inputs; packed expert weights and scales |

Checkpoint configuration and tensor keys determine the path; directory names,
model sizes, layer counts and routing scales are not hardcoded. Dense-only
checkpoints do not become expert-pruning models merely by sharing a family name.
The Hunyuan adapter targets Hy3. Additional Qwen2/3, Qwen3-Next, GLM4 and DeepSeek
V2/V3 compatibility code is not a claim that those checkpoints were used in any
particular evaluation. `generic` remains an explicit opt-in for custom layouts.

## Native code and loading

Forward collection requires the architecture's native implementation in the
installed Transformers distribution or in the checkpoint's Python files.
Missing implementations must be installed or supplied; Razor does not substitute
a different model architecture. GLM-5.3, Qwen3.8 and Kimi model classes may not be
present in the base Transformers installation. Qwen3.8 Flash Next uses
`qwen4_exp`, tested with Transformers 5.16.1. Its conditional wrapper is loaded
as a conditional model rather than silently reduced to a causal text model.
The native implementation may select installed CUDA-only FLA/causal-conv
kernels even for CPU tensors; use a compatible device/runtime. Its QSA indexer
remains native when query attention is chunked; this is not a bound on every
indexer or PLE allocation. Streaming honors native `force_cpu` conversion
rules and keeps `_no_placement_params` (including the large n-gram table) on
CPU; the native embedding moves only lookup indices and results across devices.
This still requires host RAM for the table and its conversion buffers.
Qwen3.8 checkpoints containing stored MTP layers must use `--export-mode auto`
or `checkpoint`: `model` export is rejected before loading, because the native
model class does not instantiate those MTP weights.

`tests/test_qwen38.py` exercises a native tiny model with PLE/ngram, HC, linear
attention and QSA enabled: FP32/BF16 routing and refill, two streaming windows,
chunked attention and export/reload logits. Set `RAZOR_QWEN38_TEST_DEVICE` to a
free CUDA device when the installed optional kernels require CUDA. Native tests
skip when the implementation or a compatible device is unavailable; skips are
not validation. Raw-storage/MTP tests do not require the native implementation.
These checks do not certify full-size multimodal quality or MTP speculative
execution. Qwen3.8 weights inherit their own Qwen Community License 1.0, not
the Apache-2.0 license of Qwen3.6.

`trust_remote_code=False` is the default. Review checkpoint code before enabling
`--trust-remote-code`. It permits native classes, relative imports and an official
DeepSeek encoder in `encoding/encoding_dsv4.py`. Encoder errors are not replaced
with plain-text concatenation. With a supplied saliency or keep-set file,
raw checkpoint export can operate without importing model code. Copying custom
Python also requires explicit consent and is limited to declared classes and
statically resolved local dependencies. Unsupported or missing code references
are rejected. Required Python, token text and licenses are not rewritten; review
those contents before distributing a checkpoint. Config metadata cleanup does
not certify arbitrary source code or calibration text as free of private data.

`--execution auto` uses layer streaming for the families above and for supported
quantized checkpoints. `--execution resident` loads the whole model;
`--execution streaming --stream-device cuda:0` keeps one decoder layer resident
and preserves the native model's masks, position embeddings, recurrent states
and hyper-connections. `--stream-batch-window 1` is the default: only one batch's
activations are retained while its layers execute. A larger window amortizes
layer reads across more batches at the cost of activation memory; a smaller
window does not truncate calibration coverage. `--chunk-attn` additionally chunks
supported native eager-attention kernels. Unsupported attention backends are
rejected. Native chunking patches module-global functions: overlapping chunking
contexts in one process are rejected; use separate processes for independent jobs.

FP8 tensors with block scales and MXFP4 packed experts with E8M0 scales are
separate formats. They are decoded for scoring, not for the saved checkpoint.
Unknown quantization layouts are rejected rather than approximated. Raw export
requires unquantized router matrices; a quantized router is rejected before
writing rather than exported with mismatched scale rows.

## Checkpoint export

`--export-mode auto` selects the raw safetensors writer for these families and
supported quantized checkpoints. `--export-mode checkpoint` explicitly requests
it. Expert tensors, router rows, correction biases and quantization scales are
kept in matching order; non-target tensor bytes are preserved. The source JSON
configuration is the export baseline, so unrelated RoPE and wrapper fields are
not regenerated by `save_pretrained`. Safe auxiliary-file export requires POSIX
directory-descriptor operations; unavailable platform protections cause an
explicit failure rather than a less-safe copy.

Output directories are never overwritten. Where atomic no-replace directory
rename is unavailable, publication uses an exclusively created output directory
with `.razor-incomplete`, commits `config.json` last and removes the marker only
after completion. This fallback is not atomically visible to other processes.
Razor refuses incomplete directories; do not load or distribute interrupted
exports with third-party tools that do not recognize the marker.

- **Hash layers:** participate in native forward but have no RAZOR refill score.
  `--hash-policy remap` derives missing hash keep-sets from table frequency,
  keeps surviving token assignments and fills removed slots by router-row
  similarity without repeated experts within a token. This is an explicit
  heuristic, not a measured RAZOR score.
- **Preserving hash width:** `--hash-policy preserve` retains the hash layers and
  exports mixed-width configuration/modeling code. It requires
  `--trust-remote-code`; other inference engines must support that layout.
- **MTP:** discovered from the checkpoint index, including layers absent from a
  normal causal-LM module tree. `--mtp-policy router_norm` fills unobserved
  auxiliary selections from router-row norms; `drop` removes auxiliary tensors
  and updates their configuration; `error` requires explicit keep-sets.
  Ordinary decoder layers never receive an automatic missing-score fallback.
- **Groups:** group-limited routers require equal retained quotas, original
  group order and enough experts to satisfy the native group scorer and top-k.
- **Shared experts:** unchanged. Auxiliary/MTP structure checks do not certify
  speculative-decoding behavior.

## Validation scope

Tests cover FP32/BF16 native router/expert output reconstruction, score moments,
physical pruning and config/state reconstruction. Public-API disk tests exercise
Hy3, GLM4 MoE Lite, GLM DSA, DeepSeek-V4 with hash layers, Qwen3.5 wrappers and Gemma 4 through
RAZOR and the three baselines: streaming collection, raw export, reload and
inference. `tests/test_refill.py` checks the RCS, RCS-LOO and RCS-Refill scores
against the deletion counterfactuals rebuilt directly from their definitions,
under both renormalized and frozen gate weights. Storage
tests separately exercise all profiles, paired quantization scales and MTP/hash
policies. Flash/Pro storage differences have decoding and raw-tensor tests;
these are not full-size pretrained-model benchmarks.

Native GLM-5.3 and Kimi MoE tests accept explicitly supplied
trusted modeling sources. They are skipped when these sources are unavailable;
a skipped test is not a verification result. Current tests do not redistribute
those model implementations or weights.

`razor verify --mode tensors` checks stored keys, shapes, dtypes, selections and
bytes. The raw-checkpoint verifier additionally checks completed keep manifests,
count/policy metadata and the exact generated mixed-width source templates.
It does not certify tokenizer assets, copied custom code or arbitrary auxiliary
files: test tokenization and model loading from the output directory separately.
`--mode stream` compares native layer outputs against a sliced source reference
with one layer resident; it is not a task-quality evaluation or a complete
logprob test. `--mode model` compares sampled next-token logprobs using batches
tokenized by the source tokenizer, not a source/output tokenizer comparison.
`--max-layers` intentionally limits verification to a prefix and must not be
reported as whole-model validation. See [quickstart](../examples/quickstart.md).
