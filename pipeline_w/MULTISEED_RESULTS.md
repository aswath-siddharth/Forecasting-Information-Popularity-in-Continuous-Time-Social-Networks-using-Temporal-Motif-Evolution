# Multi-seed results against CasSubTS (Weibo, 1 h)

**Goal:** beat (1) the CasSubTS paper's published 1 h Weibo score and (2) our own CasSubTS
run on identical data (`outputs_cassubts_3600`).

| Target | MSLE (log₂) | R² | MAPE |
| --- | --- | --- | --- |
| CasSubTS paper, Sci Rep 15:6801, Table 4 | 1.987 | 0.561 | 0.299 |
| CasSubTS, our run on our splits | 1.9506 | 0.6306 | 0.2324 |

## Protocol

1. **Recipe chosen on validation only.** Five training recipes were tried at seed 42.
   The test split was not read until step 3.
2. **Two more seeds** of the chosen recipe (43, 44).
3. **Test split scored once:** each seed, the seed mean, and a 3-model ensemble (the
   average of the three seeds' predictions). The ensemble size was fixed **before** any test
   score was seen: 3 × 49,824 = 149,472 parameters, below one CasSubTS (153,970), so it is
   a parameter-matched comparison.
4. No recipe, seed or ensemble choice was revisited after the test scores were read.

Reproduce with `python -m pipeline_w.multiseed` (see the module docstring).

## Step 1 — recipe selection (validation MSLE, seed 42)

| Config | Recipe | Val MSLE |
| --- | --- | --- |
| **A** | original: ReduceLROnPlateau, early stop 8 | **2.0047** ← chosen |
| B | cosine over 60 epochs + EMA of weights, early stop 15 | 2.0336 (stopped at ep 19) |
| C | B + cascade-level side features | 2.0300 (stopped at ep 19) |
| D | cosine over 20 epochs + EMA + side features, early stop 8 | 2.0387 |
| E | D + dropout 0.5, weight decay 1e-3 | 2.1438 |

Every recipe peaks around epochs 9–13 and then overfits (train MSLE keeps falling while val
rises). B and C were stopped manually at epoch 19 after 6–8 epochs of rising val MSLE; their
best scores come from the logs (`tuning_3600/stopped_configs_BC.json`) and use per-batch
averaging, which differs negligibly from the per-cascade scores of A, D and E. The side
features helped early (C beat B at every epoch) but did not survive the overfitting.

## Step 3 — test results (6,328 cascades)

| Model | Params | MSLE ↓ | R² ↑ | MAPE ↓ | MAE ↓ |
| --- | --- | --- | --- | --- | --- |
| Ours, seed 42 | 49,824 | 1.9804 | 0.6250 | 0.2340 | 77.78 |
| Ours, seed 43 | 49,824 | 1.9683 | 0.6273 | 0.2282 | 75.79 |
| Ours, seed 44 | 49,824 | 2.0050 | 0.6203 | 0.2323 | 78.12 |
| Ours, mean of 3 seeds | 49,824 | 1.9845 ± 0.0187 | 0.6242 | 0.2315 | 77.23 |
| **Ours, 3-model ensemble** | **149,472** | **1.9516** | **0.6304** | **0.2292** | **75.42** |
| CasSubTS, our run | 153,970 | 1.9506 | 0.6306 | 0.2324 | 75.58 |
| CasSubTS, paper | — | 1.987 | 0.561 | 0.299 | — |

R² and MAPE are in log₂ space (see `CASSUBTS_PAPER_RESULTS.md` for why).

### Target 1 — the CasSubTS paper: beaten on every reported metric

The ensemble (1.952 / 0.630 / 0.229) and the seed mean (1.985 / 0.624 / 0.232) are both
better than the paper's 1.987 / 0.561 / 0.299. Individually, 2 of 3 seeds beat 1.987 on MSLE;
seed 44 (2.005) does not. **Caveat:** the paper used 35,917 cascades and a 70/10/20 split;
ours are 42,186 and a chronological 70/15/15, so this is a comparison of reported numbers,
not a controlled one.

### Target 2 — our CasSubTS run: tied on MSLE, ahead on MAPE and MAE

| Paired test, per cascade | Ensemble vs CasSubTS | Seed-mean error vs CasSubTS |
| --- | --- | --- |
| MSLE difference (ours − CasSubTS) | **+0.0010** | +0.0340 |
| 95% bootstrap CI | [−0.0366, +0.0424] | [−0.0037, +0.0757] |
| Paired t-test p | 0.96 | 0.096 |
| Ours better on | **52.7%** of cascades | 45.9% of cascades |
| Wilcoxon p | 2.2 × 10⁻⁵ | 2.8 × 10⁻⁴ |

- **Mean MSLE is a tie.** The ensemble is 0.001 above CasSubTS (0.05%), far inside the noise
  (p = 0.96). It does not beat 1.951; it matches it.
- **The ensemble wins on the typical cascade.** It has lower error on 52.7% of cascades, and
  the Wilcoxon test says that shift is real (p = 2 × 10⁻⁵). CasSubTS draws level on the mean
  through a minority of large-error tail cascades.
- **MAPE and MAE favour the ensemble** (0.2292 vs 0.2324; 75.42 vs 75.58). R² is level.
- A single seed of ours is, on average, slightly behind CasSubTS (seed-mean error +0.034,
  p = 0.096, CasSubTS better on 54% of cascades). Seed-to-seed spread for ours is ±0.019.

## What may be claimed

> ✅ On the DeepHawkes Weibo benchmark at 1 h, our model outperforms the published CasSubTS
> results on MSLE, R² and MAPE (ensemble 1.952 / 0.630 / 0.229 vs 1.987 / 0.561 / 0.299),
> noting the preprocessing differences.

> ✅ Under identical splits, labels and loss, a parameter-matched 3-model ensemble of ours
> (149K parameters) matches CasSubTS (154K) on MSLE (1.952 vs 1.951, p = 0.96), is better on
> MAPE and MAE, and has lower error on a statistically significant majority of cascades
> (52.7%, Wilcoxon p < 0.001).

> ❌ "We beat CasSubTS on MSLE in our controlled run." 1.9516 > 1.9506; it is a tie.

> ❌ Reporting seed 43 alone (1.968) as "our result". Seeds were not chosen on test, and the
> honest single-model figure is the mean, 1.985 ± 0.019.

## Robustness check — CasSubTS seeds (running)

The 1.951 target is a single CasSubTS run, and our own seeds spread by ±0.019. CasSubTS
seeds 42–44 are being trained with the chosen recipe (A: plateau schedule, 60-epoch budget,
early stop 8) to show where 1.951 sits within CasSubTS's own seed spread. Results are added
here when they finish.
