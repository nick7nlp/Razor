# Scoring

## Notation

For token $x$, the router selects the $k$ experts $S(x)$. Expert $i$
produces $f_i(x)$ with normalized weight $w_i(x)\ge 0$,
$\sum_{i\in S(x)} w_i(x)=1$. The routed output, called the consensus, is

$$
c(x)=\sum_{i\in S(x)} w_i(x) f_i(x),
$$

and $r_j(x)=f_j(x)-c(x)$ is the consensus residual of expert $j$. The
routed-output scale $\lambda$ multiplies every score; it does not change
within-layer rankings.

## Three consensus-residual criteria

The three criteria differ in how much of the deletion counterfactual they
represent. They are computed together, from the same reconstructed expert
outputs, in one calibration pass.

**RCS** (`--method rcs`) is the weighted consensus residual

$$
s_i^{\mathrm{rcs}}=\lambda\, w_i\lVert r_i\rVert_2 .
$$

**RCS-LOO** (`--method rcs-loo`) is the exact output change when expert $i$
is deleted and the survivors are renormalized over the remaining routed set
(Proposition 2, fixed support):

$$
\delta_i^{\mathrm{loo}}=\lambda\,\frac{w_i}{1-w_i}\lVert r_i\rVert_2 .
$$

**RCS-Refill** (`--method rcs-refill`) additionally models router refill
(Proposition 1). Deleting $i$ promotes the highest-ranked unselected expert
$r$ under the router's own selection rule, so a selection-only correction
bias and group masking apply to it as they do to the routed experts. Its
pseudo-weight $w_r$ is its mixture score divided by the original selected
score sum. The survivors and the promoted expert are renormalized over
$D_i=1-w_i+w_r$:

$$
\delta_i=\lambda\,\frac{\lVert w_i r_i - w_r r_r\rVert_2}{1-w_i+w_r}.
$$

The numerator is the norm of a vector difference: the promoted residual can
reinforce or cancel the removed one, so the two cannot be combined after taking
norms. Setting $w_r=0$ recovers RCS-LOO.

**RAZOR** (`--method razor`) is RCS-Refill under conditional RMS aggregation.
`razor` and `rcs-refill` name the same scoring rule.

Both denominators are clamped from below by $10^{-6}$. The identities
describe the unclamped quantity; when the floor is active the computed score can
differ from exact local damage.

## Aggregation

Each criterion is aggregated over the calibration tokens routed to the expert,
with $n_i$ such tokens:

| `--aggregation` | Expert score |
|---|---|
| `rms` (default) | $\sqrt{\sum_x s_i(x)^2/n_i}$ |
| `mean` | $\sum_x s_i(x)/n_i$ |
| `sum` | $\sum_x s_i(x)$ |

The second moment is accumulated per token; it cannot be recovered by squaring
an averaged score. Disjoint calibration shards accumulate counts and moments
independently, and summing them before taking the root gives the pooled RMS.
REAP and EAN accept the same options; `frequency` always uses routed counts.

## Baselines

| `--method` | Criterion |
|---|---|
| `reap` | $\lambda w_i\lVert f_i\rVert_2$ |
| `ean` | $\lVert f_i\rVert_2$ |
| `frequency` | Number of routed tokens |

## Requirements

- The RCS criteria require top-$k\ge 2$ and $w_i<1$ for every routed pair.
  A layer with a saturated normalized gate reports `razor_supported=False`; the
  baselines remain available.
- RCS-Refill requires $k<E$ and an observable rank-$(k+1)$ score. Adapters
  whose native block hands down only the routed winners replay the router and
  accept the replay only if it reproduces the native selection exactly;
  otherwise the layer reports `refill_supported=False` and `rcs-loo` remains
  available. Tokens with no admissible promotion receive $w_r=0$.
- Routers without top-$k$ renormalization use frozen weights: deletion
  removes $\lambda w_i f_i$ and nothing is renormalized. RCS and RCS-LOO then
  reduce to the REAP magnitude, and RCS-Refill to
  $\lambda\lVert w_i f_i-w_r f_r\rVert_2$. The pack records
  `counterfactual_semantics="frozen_weights"`.
- Hash-lookup layers run during collection but do not receive these scores;
  their selection and remapping policy is separate.
- Gemma expert gain is included in each expert output. Kimi latent-MoE scores
  use the latent expert space and record `score_space="latent"`.
- A zero score for an unobserved expert is not evidence that the expert can be
  removed without affecting other inputs.

## Reading a score pack

The saliency output is `observer_data.pt`, a mapping from layer tags to
per-expert tensors and collection metadata. Load only trusted score files.

```python
from razor import metrics

scores = metrics.score(saliency["main_0"], "razor")          # RCS-Refill, RMS
ablation = metrics.score(saliency["main_0"], "rcs-loo", "mean")
keep = metrics.keep_indices(saliency["main_0"], "razor", 4)
```

The example assumes that `saliency` is an already loaded score pack and that
the layer has at least four experts.

| Criterion | Sum | Square sum | Mean | RMS |
|---|---|---|---|---|
| `rcs` | `rcs_sum` | `rcs_square_sum` | `rcs_mean` | `rcs_rms` |
| `rcs-loo` | `counterfactual_delta_sum` | `counterfactual_delta_square_sum` | `counterfactual_delta_mean` | `counterfactual_delta_rms` |
| `rcs-refill` | `rcs_refill_sum` | `rcs_refill_square_sum` | `rcs_refill_mean` | `rcs_refill_rms` |
| `reap` | `weighted_ean_sum` | `weighted_ean_square_sum` | `reap` | `reap_rms` |
| `ean` | `ean_sum` | `ean_square_sum` | `ean_mean` | `ean_rms` |

Auxiliary fields include `routed_count`, `total_tokens`, `router_scaling`,
`renormalized`, `razor_supported`, `refill_supported` and
`counterfactual_semantics`.

The public pruning API also accepts additive `state` packs with `count` and any
of `rcs_sum`/`rcs_sq`, `d1_sum`/`d1_sq` (RCS-LOO), `refill_sum`/`refill_sq`,
`reap_sum`/`reap_sq` and `ean_sum`/`ean_sq`. Only supplied moments are
imported; a pack without refill moments cannot serve `razor`.
