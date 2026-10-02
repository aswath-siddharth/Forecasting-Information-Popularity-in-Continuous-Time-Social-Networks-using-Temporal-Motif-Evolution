# Controlled comparison against CasSubTS

The first baseline in this project that was **reimplemented and run on our own splits**
rather than quoted from another paper's table.

**Baseline:** Wang, Li, Liu, Peng & Li, *Fusing temporal and structural information via
subgraph sampling and multi-head attention for information cascade prediction*,
Scientific Reports 15:6801 (2025). DOI [10.1038/s41598-025-91752-z](https://doi.org/10.1038/s41598-025-91752-z)

## Why this was necessary

Published Weibo baselines disagree with each other. The same three models, same dataset,
same 1-hour window, as reported by two different papers:

| Model | CasSubTS paper | HierCas paper |
| --- | --- | --- |
| DeepHawkes | 2.441 | 2.632 |
| CasCN | 2.242 | 2.601 |
| CasSeqGCN | 2.058 | 2.341 |

Every paper reimplements its own baselines, so a number lifted from someone's table is not
comparable to ours. The only way to get a defensible comparison is to run the baseline here.

## What makes this controlled

`cassubts_data.py` does **not** re-derive the splits. It reads the cascade ids out of the
existing `weibo_{train,val,test}.npz` and emits graphs for exactly those cascades in exactly
that order. All 42,186 matched; none dropped.

Both models therefore share: identical train/val/test partition, identical labels, identical
target (`log2(increment + 1)`), identical loss (MSE in log2 space) and identical metrics.
`compare_models.py` re-asserts cascade-id and target equality at evaluation time and aborts
rather than compare anything misaligned. **Only the architecture differs.**

## Results — 1-hour window, 6,328 identical test cascades

| Metric | Ours | CasSubTS | Winner |
| --- | --- | --- | --- |
| MSLE (log₂) | 1.9804 | **1.9506** | CasSubTS |
| MSLE (ln) | 0.9515 | **0.9372** | CasSubTS |
| R² (log₂) | 0.6250 | **0.6306** | CasSubTS |
| PCC | 0.7933 | **0.7950** | CasSubTS |
| MAE | 77.78 | **75.58** | CasSubTS |
| RMSE | 656.68 | 655.88 | tied |
| MedAE | 12.47 | **11.78** | CasSubTS |
| **Parameters** | **49,824** | 153,970 | **ours, 3.1× fewer** |

## The gap is not statistically significant

Paired test over per-cascade squared log-errors (`compare_models.py`):

- difference **+0.0298**, 95% bootstrap CI **[−0.0100, +0.0750]** — spans zero
- paired t-test **p = 0.17**
- Wilcoxon **p = 0.045**
- CasSubTS better on **51.2%** of individual cascades

The two tests disagree because squared error here is dominated by a few very large tail
errors: the rank test picks up a small consistent edge on *typical* cascades that the mean
does not. Both are reported rather than whichever is more flattering.

## What may and may not be claimed

> ✅ Our model performs comparably to CasSubTS (Scientific Reports, 2025) on the DeepHawkes
> Weibo benchmark — no statistically significant difference in MSLE (p = 0.17) under identical
> splits, labels, target and loss — using 3.1× fewer parameters.

> ❌ "We beat CasSubTS." The point estimate favours CasSubTS, not us.

> ❌ "CasSubTS beat us." Not at p < 0.05 on the headline metric.

> ❌ Comparing our 1.980 against the paper's published 1.987. Their run used 35,917 cascades
> and a 70/10/20 split against our 42,186 and 70/15/15, and the paper never states which space
> its R² is computed in. That is the same apples-to-oranges trap this whole exercise exists to
> avoid.

## Caveats that must travel with these numbers

1. **This is a reimplementation.** The authors released no code ("available from the
   corresponding author on reasonable request"). It is evidence about the *architecture*, not
   about the authors' trained system — exactly the caution that makes the table at the top of
   this file unreliable.
2. **Four deviations were forced by underspecification**, documented in `cassubts_model.py`:
   cumulative vs sliced subgraphs, the undefined base node feature `x_i`, per-subgraph vs
   global degrees, and the SENet channel definition.
3. **Both are single-seed runs.** A ~1.5% MSLE gap is the magnitude seed noise produces.
   Three to five seeds per model is the obvious next experiment and would settle it.
4. **CasSubTS training was interrupted** by a background time limit at epoch 12 and warm-started
   from that checkpoint, then ran 30 further epochs to convergence (best val 2.0617). The warm
   start is recorded in `cassubts_evaluation_report.json`; optimizer state was not restored, so
   its LR schedule restarted.

## Reproducing

```bash
python -m pipeline_w.cassubts_data \
    --splits-dir weibo_data/processed_3600 \
    --output-dir weibo_data/cassubts_3600 --observation-time 3600

python -m pipeline_w.train_cassubts \
    --data-dir weibo_data/cassubts_3600 \
    --output-dir pipeline_w/outputs_cassubts_3600 --observation-time 3600

python -m pipeline_w.compare_models --window 3600
```
