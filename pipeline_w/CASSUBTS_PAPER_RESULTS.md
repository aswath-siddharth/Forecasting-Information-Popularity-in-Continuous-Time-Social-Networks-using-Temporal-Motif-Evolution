# Results against the CasSubTS paper (Weibo)

Our model placed into the two Weibo tables of Wang, Li, Liu, Peng & Li, *Fusing temporal
and structural information via subgraph sampling and multi-head attention for information
cascade prediction*, Scientific Reports 15:6801 (2025),
DOI [10.1038/s41598-025-91752-z](https://doi.org/10.1038/s41598-025-91752-z).

> **Update (multi-seed):** the single-run numbers below were superseded by a 3-seed
> comparison under an identical recipe — see [MULTISEED_RESULTS.md](MULTISEED_RESULTS.md).
> Fresh CasSubTS seeds average 1.986; ours average 1.985, and our 3-model ensemble reaches 1.952.

Published rows are copied verbatim from the paper and checked against the PDF.
Rows marked **(our run)** were trained and tested in this repository.

## Table 3 — MSLE (log₂) on Weibo, lower is better

| Model | 1 h | 2 h | 3 h | Source |
| --- | --- | --- | --- | --- |
| Feature-Linear | 3.501 | 3.435 | 3.324 | paper |
| Feature-Deep | 3.680 | 3.361 | 3.296 | paper |
| DeepCas | 2.958 | 2.689 | 2.647 | paper |
| DeepHawkes | 2.441 | 2.287 | 2.252 | paper |
| CasCN | 2.242 | 2.036 | 1.910 | paper |
| CasSeqGCN | 2.058 | 1.887 | 1.853 | paper |
| CasSubTS | 1.987 | 1.817 | 1.801 | paper |
| CasSubTS **(our run)** | **1.951** | — | — | `outputs_cassubts_3600/` |
| **Ours — attention-GRU (our run)** | **1.980** | — | — | `outputs_3600/` |

2 h and 3 h have not been run for either model.

## Table 4 — more metrics, Weibo 1 h

| Model | MSLE ↓ | R² ↑ | MAPE ↓ | Source |
| --- | --- | --- | --- | --- |
| Feature-Linear | 3.501 | 0.461 | 0.356 | paper |
| Feature-Deep | 3.680 | 0.465 | 0.353 | paper |
| DeepCas | 2.958 | 0.459 | 0.358 | paper |
| DeepHawkes | 2.441 | 0.482 | 0.352 | paper |
| CasCN | 2.242 | 0.468 | 0.350 | paper |
| CasSeqGCN | 2.058 | 0.554 | 0.314 | paper |
| CasSubTS | 1.987 | 0.561 | 0.299 | paper |
| CasSubTS **(our run)** | **1.951** | **0.631** | **0.232** | `outputs_cassubts_3600/table4_metrics.json` |
| **Ours (our run)** | **1.980** | **0.625** | **0.234** | `outputs_cassubts_3600/table4_metrics.json` |

Our R² and MAPE are computed in log₂ space. The paper does not state the space for either.
MAPE on raw retweet counts is 0.917 (ours) and 0.887 (CasSubTS), which is why log space is
the likely reading of the paper's ~0.3 values. MAPE excludes the 403 test cascades whose
increment is zero (5,925 of 6,328 remain).

## Head-to-head on identical data (the comparison that can be claimed)

Both models share the same 6,328 test cascades, labels, target and loss
(see [CASSUBTS_COMPARISON.md](CASSUBTS_COMPARISON.md)).

| Metric | Ours | CasSubTS (our run) |
| --- | --- | --- |
| MSLE (log₂) | 1.9804 | **1.9506** |
| R² (log₂) | 0.6250 | **0.6306** |
| MAPE (log₂) | 0.2340 | **0.2324** |
| MAE (count) | 77.78 | **75.58** |
| Parameters | **49,824** | 153,970 |

Paired test on per-cascade squared log error: difference +0.0298, 95% CI [−0.010, +0.075],
t-test p = 0.17, Wilcoxon p = 0.045 — **not significant** on the headline metric.

## Reading these tables

> ✅ Under identical splits, labels, target and loss, our model performs comparably to
> CasSubTS on Weibo at 1 h (MSLE 1.980 vs 1.951, p = 0.17) with 3.1× fewer parameters.

> ❌ "We beat the CasSubTS paper" from 1.980 vs 1.987. The paper used 35,917 cascades and a
> 70/10/20 split; ours uses 42,186 cascades and a chronological 70/15/15 split. The published
> rows give context, not a ranking.

Both our runs are single-seed, and CasSubTS is a reimplementation (no official code was
released); see the caveats in [CASSUBTS_COMPARISON.md](CASSUBTS_COMPARISON.md).

## Reproducing

```bash
python -m pipeline_w.paper_metrics --window 3600 \
    --output pipeline_w/outputs_cassubts_3600/table4_metrics.json
```
