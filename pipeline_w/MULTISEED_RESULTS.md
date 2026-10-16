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

## Robustness check — CasSubTS seeds under the same recipe

The 1.951 target is one CasSubTS run, and our own seeds spread by ±0.019, so CasSubTS
seeds 42–44 were trained fresh with exactly our chosen recipe (A: plateau schedule,
60-epoch budget, early stop 8). This phase was planned before any test score was read.

| Test MSLE ↓ | Ours | CasSubTS |
| --- | --- | --- |
| Seed 42 | 1.9804 | 1.9796 |
| Seed 43 | 1.9683 | 2.0149 |
| Seed 44 | 2.0050 | 1.9647 |
| **Mean of 3 seeds** | **1.9845 ± 0.0187** | 1.9864 ± 0.0258 |
| **3-model ensemble** | **1.9516** | 1.9730 |
| Earlier single run (`outputs_cassubts_3600`) | — | 1.9506 |

| Other metrics | Ours mean | CasSubTS mean | Ours ensemble | CasSubTS ensemble |
| --- | --- | --- | --- | --- |
| R² ↑ | 0.6242 | 0.6238 | **0.6304** | 0.6264 |
| MAPE ↓ | 0.2315 | **0.2308** | **0.2292** | 0.2299 |
| MAE ↓ | 77.23 | **75.97** | **75.42** | 75.66 |
| Parameters | **49,824** | 153,970 | **149,472** | 461,910 |

| Paired test, per cascade | Seed-mean errors | Ensemble vs ensemble |
| --- | --- | --- |
| MSLE difference (ours − CasSubTS) | −0.0019 | **−0.0214** |
| 95% bootstrap CI | [−0.0401, +0.0395] | [−0.0599, +0.0202] |
| Paired t-test p | 0.93 | 0.29 |
| Ours better on | 45.4% of cascades | 50.9% of cascades |
| Wilcoxon p | 5.1 × 10⁻⁵ | 0.017 |

(`multiseed_report.json`, `ensemble_vs_ensemble.json`; the ensemble-vs-ensemble test was
added after the run as an analysis, not used to choose anything.)

**What this shows**

- **1.951 is not typical of CasSubTS.** The closest fresh CasSubTS seed (1.965) is 0.014 worse, and
  even CasSubTS's own 3-model ensemble (1.973) does not reach it. That run was the one
  interrupted at epoch 12 and warm-started with a fresh learning-rate schedule (42 epochs in
  all); whether that restart helped or it was simply a good draw, it is not what the standard
  recipe produces.
- **Like-for-like, ours is level or ahead on MSLE.** Single models: 1.9845 vs 1.9864 (tie,
  p = 0.93) with 3.1× fewer parameters. Ensembles: 1.952 vs 1.973, 1.1% lower, not significant
  (p = 0.29).
- **Single CasSubTS models are better on the typical cascade.** Per seed, CasSubTS has lower
  error on 54.6% of cascades (Wilcoxon p = 5 × 10⁻⁵) and better MAE and MAPE; ours draws level
  on the mean MSLE through fewer large tail errors. Ensembling reverses this: our ensemble is
  better on 50.9% of cascades (p = 0.017) and on every metric.

## What may be claimed

> ✅ On the DeepHawkes Weibo benchmark at 1 h, our model outperforms the published CasSubTS
> results on MSLE, R² and MAPE (ensemble 1.952 / 0.630 / 0.229; single-model mean
> 1.985 / 0.624 / 0.232; paper 1.987 / 0.561 / 0.299), noting the preprocessing differences.

> ✅ Under identical splits, labels, loss and training recipe, averaged over 3 seeds, our
> model matches CasSubTS on MSLE (1.985 vs 1.986, p = 0.93) with 3.1× fewer parameters, and a
> 3-model ensemble of ours (149K parameters, still below one CasSubTS) achieves 1.952 — lower
> than every fresh CasSubTS seed (1.965–2.015) and than a 3-model CasSubTS ensemble (1.973),
> though the ensemble gap is not statistically significant (p = 0.29).

> ✅ Against the single best CasSubTS run we obtained (1.951), the parameter-matched ensemble
> ties on MSLE (+0.001, p = 0.96) and is better on MAPE and MAE.

> ❌ "We significantly outperform CasSubTS." No comparison here reaches p < 0.05 on MSLE.

> ❌ "Our model beats 1.951." It ties it (1.9516).

> ❌ Reporting seed 43 alone (1.968) as "our result", or CasSubTS seed 43 alone (2.015) as
> theirs. The honest single-model figures are the means.
