# Controlled Multi-Seed Comparison: CasMotif vs DeepHawkes (Sina Weibo)

Head-to-head controlled evaluation of **CasMotif** (`pipeline_w/`) against a faithful PyTorch
reimplementation of **DeepHawkes** (Cao, Shen, Cen, Ouyang & Cheng, *DeepHawkes: Bridging the
Gap between Prediction and Understanding of Information Cascades*, ACM CIKM 2017,
DOI [10.1145/3132847.3132973](https://doi.org/10.1145/3132847.3132973)).

---

## Why a controlled reimplementation was necessary

Published DeepHawkes MSLE scores on Sina Weibo (1 h) disagree widely across papers (`2.441`
in CasSubTS, `2.632` in HierCas, up to `2.914` elsewhere) because preprocessing, cascade
counts, split protocols, and log bases vary across studies. Furthermore, no published
DeepHawkes result exists for the ultra-early **15-minute** observation window.

Running DeepHawkes here directly on our exact chronological splits eliminates cross-paper
confounders and lets us run per-cascade paired significance tests at both **1 h** and **15 m**.

---

## Controlled Protocol (Held Identical Across Both Models)

| Protocol Dimension | Setting |
| :--- | :--- |
| **Splits** | Strictly chronological **70% / 15% / 15%** aligned 1-to-1 to `weibo_{train,val,test}.npz` (`29,530 / 6,328 / 6,328` at 1 h; `16,509 / 3,538 / 3,538` at 15 m). `deephawkes/compare.py` asserts cascade-ID and target equality before testing. |
| **Windows & Horizon** | $T_\text{obs} \in \{3600\text{s (1 h)},\; 900\text{s (15 m)}\} \to T_\text{pred} = 86,400\text{s (24 h)}$ |
| **Target** | Future increment only: $\Delta P = R_{24\text{h}} - R_{T_\text{obs}}$, transformed as $\log_2(\Delta P + 1)$ |
| **Loss & Primary Metric** | $\text{MSLE}_{\log_2}$ (MSE in $\log_2$ space); $\text{MSLE}_{\ln}$ also reported for reference |
| **Optimization Harness** | Batch size `64`, `AdamW` (`lr = 1e-3`, `weight_decay = 1e-4`), `ReduceLROnPlateau(factor=0.5, patience=3)`, grad clip `max_norm = 1.0`, early stop `patience = 8`, `max_epochs = 60`, restoring best validation MSLE checkpoint |
| **Seeds & Zero Leakage** | Seeds `42, 43, 44`; `SequenceScaler` (CasMotif) and user vocabulary (DeepHawkes) fitted strictly on the training split only |

### Architecture Fidelity & Disclosed Details
- `deephawkes/model.py` ports the official TensorFlow `CaoQi92/DeepHawkes/deep_learning/model_sparse.py` (`SDPP`): 50-D user embeddings initialized `Uniform(0, 1)` with `0.2` dropout, single-layer 64-unit GRU path encoder (`2 × 32`), 6 equal non-parametric time-decay intervals (`time_weight` init $\mathcal{N}(0, 0.01)$), additive sum pooling over paths, and `64 → 32 → 16 → 1` ReLU MLP.
- Following official `CaoQi92/DeepHawkes` (`run_sparse.py`), the sparse user embedding table uses `0.1×` LR (`1e-4`) and explicit active-batch L2 regularization (`0.01`) so the 5.5M–14.9M embedding parameters do not immediately memorize the training split.
- Unlike official `preprocess.py` (which indexes user IDs across train + val + test), our vocabulary is built strictly from training users (`min_user_freq = 2`; unseen users map to `<UNK>`). Known-token rates: 1 h (`82.5%` train / `56.3%` val / `50.5%` test), 15 m (`78.9%` train / `49.2%` val / `42.7%` test).

---

## Results — 1-Hour Observation Window (`3600s`, 6,328 identical test cascades)

| Model | Params (Total / Non-Emb) | MSLE ($\log_2$) $\downarrow$ | MSLE ($\ln$) $\downarrow$ | $R^2$ ($\log_2$) $\uparrow$ | PCC $\uparrow$ | MAPE ($\log_2$) $\downarrow$ | MAE (count) $\downarrow$ | MedAE (count) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Mean Baseline | 0 | 5.2825 | 2.5380 | -0.0003 | — | 0.3864 | 105.32 | 21.50 |
| DeepHawkes, seed 42 | 14,940,403 / 24,903 | 2.5582 | 1.2291 | 0.5156 | 0.7212 | 0.2459 | 100.66 | 12.85 |
| DeepHawkes, seed 43 | 14,940,403 / 24,903 | 2.7271 | 1.3102 | 0.4836 | 0.7017 | 0.2473 | 121.26 | 12.56 |
| DeepHawkes, seed 44 | 14,940,403 / 24,903 | 2.5647 | 1.2322 | 0.5143 | 0.7194 | 0.2491 | 93.94 | 12.58 |
| **DeepHawkes (3-seed mean)** | **14,940,403 / 24,903** | **2.6167 ± 0.0957** | **1.2572 ± 0.0460** | **0.5045 ± 0.0181** | **0.7141** | **0.2474** | **105.29 ± 14.24** | **12.66** |
| CasMotif, seed 42 | 49,824 | 1.9357 | 0.9300 | 0.6335 | 0.7977 | 0.2312 | 73.88 | 11.73 |
| CasMotif, seed 43 | 49,824 | 1.9537 | 0.9387 | 0.6300 | 0.7946 | 0.2290 | 74.70 | 11.31 |
| CasMotif, seed 44 | 49,824 | 1.9443 | 0.9342 | 0.6318 | 0.7971 | 0.2288 | 75.73 | 11.34 |
| **CasMotif (3-seed mean)** | **49,824** | **1.9446 ± 0.0090** | **0.9343 ± 0.0043** | **0.6318 ± 0.0017** | **0.7965** | **0.2297** | **74.77 ± 0.93** | **11.46** |
| DeepHawkes (3-model ensemble) | 44.8M / 74.7k | 2.5046 | 1.2033 | 0.5257 | 0.7263 | 0.2409 | 93.53 | 12.09 |
| **CasMotif (3-model ensemble)** | **149,472** | **1.9303** | **0.9274** | **0.6345** | **0.7981** | **0.2288** | **74.43** | **11.28** |

### Paired Significance Tests (1 h, $n = 6,328$, 10,000-sample bootstrap)
| Comparison | $\Delta\text{MSLE}$ (Ours − DH) | Relative Reduction | 95% Bootstrap CI | Paired $t$-test $p$ | Wilcoxon $p$ | CasMotif Better On |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **3-seed mean per-cascade error** | **−0.6721** | **25.68%** | **[−0.7905, −0.5557]** | **$2.72 \times 10^{-29}$** | **$4.72 \times 10^{-74}$** | **61.4%** |
| **3-model ensemble vs ensemble** | **−0.5743** | **22.93%** | **[−0.6924, −0.4579]** | **$4.87 \times 10^{-22}$** | **$1.58 \times 10^{-21}$** | **54.6%** |
| Seed 42 single run | −0.6225 | 24.33% | [−0.7472, −0.5026] | $3.64 \times 10^{-23}$ | $5.88 \times 10^{-36}$ | 56.9% |

---

## Results — 15-Minute Ultra-Early Window (`900s`, 3,538 identical test cascades)

| Model | Params (Total / Non-Emb) | MSLE ($\log_2$) $\downarrow$ | MSLE ($\ln$) $\downarrow$ | $R^2$ ($\log_2$) $\uparrow$ | PCC $\uparrow$ | MAPE ($\log_2$) $\downarrow$ | MAE (count) $\downarrow$ | MedAE (count) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Mean Baseline | 0 | 5.7563 | 2.7656 | -0.0004 | — | 0.2876 | 267.30 | 51.91 |
| DeepHawkes, seed 42 | 5,526,403 / 24,903 | 4.3002 | 2.0661 | 0.2526 | 0.5252 | 0.2295 | 340.74 | 37.45 |
| DeepHawkes, seed 43 | 5,526,403 / 24,903 | 4.0501 | 1.9459 | 0.2961 | 0.5777 | 0.2344 | 322.92 | 38.16 |
| DeepHawkes, seed 44 | 5,526,403 / 24,903 | 4.4709 | 2.1480 | 0.2230 | 0.5367 | 0.2458 | 39,529.10* | 39.91 |
| **DeepHawkes (3-seed mean)** | **5,526,403 / 24,903** | **4.2737 ± 0.2116** | **2.0533 ± 0.1017** | **0.2573 ± 0.0368** | **0.5465** | **0.2366** | **13,397.59*** | **38.51** |
| CasMotif, seed 42 | 49,824 | 2.4162 | 1.1609 | 0.5801 | 0.7635 | 0.1863 | 205.22 | 31.42 |
| CasMotif, seed 43 | 49,824 | 2.4824 | 1.1927 | 0.5686 | 0.7556 | 0.1870 | 206.82 | 30.75 |
| CasMotif, seed 44 | 49,824 | 2.4497 | 1.1770 | 0.5743 | 0.7579 | 0.1875 | 205.08 | 31.86 |
| **CasMotif (3-seed mean)** | **49,824** | **2.4494 ± 0.0331** | **1.1768 ± 0.0159** | **0.5743 ± 0.0057** | **0.7590** | **0.1869** | **205.71 ± 0.96** | **31.34** |
| DeepHawkes (3-model ensemble) | 16.6M / 74.7k | 3.8227 | 1.8366 | 0.3356 | 0.5883 | 0.2242 | 609.47 | 33.49 |
| **CasMotif (3-model ensemble)** | **149,472** | **2.4322** | **1.1686** | **0.5773** | **0.7606** | **0.1861** | **204.96** | **31.19** |

*\*On 15 m seed 44, DeepHawkes' unbounded additive sum pooling over large cascades produces extreme log-space overpredictions that blow up raw-count MAE after inverting $2^{\hat{y}} - 1$ (`MedAE = 39.91` reflects typical cascades). CasMotif's attention-pooled motif encoder exhibits no such instability.*

### Paired Significance Tests (15 m, $n = 3,538$, 10,000-sample bootstrap)
| Comparison | $\Delta\text{MSLE}$ (Ours − DH) | Relative Reduction | 95% Bootstrap CI | Paired $t$-test $p$ | Wilcoxon $p$ | CasMotif Better On |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **3-seed mean per-cascade error** | **−1.8243** | **42.69%** | **[−2.1563, −1.5098]** | **$1.38 \times 10^{-27}$** | **$1.21 \times 10^{-104}$** | **68.9%** |
| **3-model ensemble vs ensemble** | **−1.3905** | **36.37%** | **[−1.7071, −1.0904]** | **$3.68 \times 10^{-18}$** | **$6.42 \times 10^{-24}$** | **56.3%** |
| Seed 42 single run | −1.8840 | 43.81% | [−2.2540, −1.5267] | $1.24 \times 10^{-23}$ | $1.43 \times 10^{-39}$ | 57.7% |

---

## Key Takeaways

1. **Decisive improvement at 1 h (25.68% lower MSLE):** Under strictly identical chronological splits and optimization settings, CasMotif achieves `1.9446 ± 0.0090` vs DeepHawkes `2.6167 ± 0.0957` ($p = 2.7 \times 10^{-29}$), beating DeepHawkes on every single metric ($R^2$, PCC, MAPE, MAE, MedAE, RMSE) with **10.6× lower seed variance**.
2. **Even larger advantage at 15 m (42.69% lower MSLE):** When observation time is cut by 75%, DeepHawkes degrades sharply ($R^2 = 0.2573$, MSLE `4.2737`) because 57.3% of test tokens are unseen users (`<UNK>`) and its path sum pooling suffers severe tail variance. CasMotif is fully **inductive** (no user-ID embeddings), retaining $R^2 = 0.5743$ and MSLE `2.4494` ($p = 1.4 \times 10^{-27}$).
3. **Massive storage & parameter efficiency:** DeepHawkes requires `14.94M` parameters (`57.0 MB` checkpoint at 1 h) due to its static user-ID embedding table; CasMotif requires **`49,824` parameters (`0.20 MB` checkpoint)** — **300× fewer total parameters** (and only `2×` DeepHawkes' non-embedding core).

## Reproducing

```bash
python -m deephawkes.run_benchmark
```
