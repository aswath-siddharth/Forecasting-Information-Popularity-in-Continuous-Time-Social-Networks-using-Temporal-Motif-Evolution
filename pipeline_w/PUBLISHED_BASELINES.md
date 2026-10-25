# Published Weibo MSLE baselines (for the comparison table)

Collected 2026-09-29 so the report cites real numbers rather than placeholders.

## Source: HierCas (arXiv 2310.13219), Table 2 — Weibo, incremental popularity

| Model | 0.5 h | 1 h |
| --- | --- | --- |
| DeepCas (WWW'17) | 2.911 | 2.743 |
| DeepHawkes (CIKM'17) | 2.891 | 2.632 |
| CasCN (ICDE'19) | 2.804 | 2.601 |
| CasGCN | 2.621 | 2.523 |
| CasSeqGCN | 2.432 | 2.341 |
| CasHAN | 2.212 | 2.006 |
| CasFlow (TKDE'21) | 2.171 | 1.979 |
| MUCas | 2.140 | 1.978 |
| HierCas (2023) | **1.817** | **1.695** |

HierCas Weibo splits: 19,472 / 4,173 / 4,172 at 0.5 h; 24,646 / 5,279 / 5,278 at 1 h.
Ours (DeepHawkes' own filters, reproduced): 23,585 total at 0.25 h; 42,186 total at 1 h.
Cascade counts differ between papers because preprocessing differs — another reason not to
read small MSLE gaps across papers as meaningful.

> ⚠️ Other papers report materially different numbers for the *same* baselines — e.g.
> DeepHawkes on Weibo at 1 h appears as 2.448, 2.632 and 2.914 in different papers.
> Baselines are re-implemented per paper, so cross-paper MSLE is not a like-for-like
> comparison. Cite which paper each number came from.

## ⚠️ The log-base trap — resolve this before putting any number in a table

MSLE is reported in **different log bases** across this literature:

- DeepHawkes' own released code trains on `np.log(y+1)/np.log(2)` — **log base 2**.
- HierCas states `MSLE = 1/M Σ (log ΔP − log ΔP̂)²` and appears to mean **natural log**.

The two differ by a constant factor: `MSLE_ln = MSLE_log2 × ln(2)² ≈ 0.4805 × MSLE_log2`.
Quoting a log2 number against an ln baseline (or vice versa) roughly *halves or doubles* the
apparent error and would make our model look far better or worse than it is.

`train_weibo.py` therefore reports **both** `MSLE_log2` and `MSLE_ln`. When building the
comparison table, match the unit to whichever paper you cite, and say in the caption which
base is used. If it cannot be established which base a cited number uses, do not put our
number beside it.

## Observation-window framing

Published Weibo settings are **0.5 h, 1 h, 2 h, 3 h**. Our 15-minute (0.25 h) setting is
*earlier than the earliest published window*, which is the honest way to frame the
contribution: not "we beat them", but "we predict from less information than any published
setting on this dataset". The 1-hour run exists so there is one genuinely like-for-like row.
