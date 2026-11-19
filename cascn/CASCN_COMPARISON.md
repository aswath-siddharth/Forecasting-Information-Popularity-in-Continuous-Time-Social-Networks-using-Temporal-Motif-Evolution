# Controlled Multi-Seed Comparison: CasMotif vs CasCN (Sina Weibo)

Head-to-head controlled evaluation of **CasMotif** (`pipeline_w/`) against a faithful PyTorch
reimplementation of **CasCN** (Chen, Zhou, Zhang, Trajcevski, Zhong & Zhang, *Information
Diffusion Prediction via Recurrent Cascades Convolution*, IEEE ICDE 2019,
DOI [10.1109/ICDE.2019.00075](https://doi.org/10.1109/ICDE.2019.00075)).

---

## Why a controlled reimplementation was necessary

Published CasCN MSLE scores on Sina Weibo (1 h) vary considerably across papers (`2.242` in
CasSubTS, `2.601` in HierCas) because different studies filter out large cascades ($>100$ nodes
in the original `ChenNed/CasCN/preprocessing/utils.py`), use different train/val/test ratios, or
evaluate on subsets of the dataset.

Running CasCN directly on our exact chronological 42,186-cascade splits eliminates cross-paper
confounders and allows per-cascade paired significance tests (`paired t-test`, `Wilcoxon
signed-rank test`, and `10,000-sample bootstrap 95% CI`) across all 6,328 test cascades.

---

## Controlled Protocol (Held Identical Across Both Models)

| Protocol Dimension | Setting |
| :--- | :--- |
| **Splits** | Strictly chronological **70% / 15% / 15%** aligned 1-to-1 to `weibo_{train,val,test}.npz` (`29,530` Train / `6,328` Val / `6,328` Test; `42,186` total). No cascades dropped in Val/Test. |
| **Windows & Horizon** | $T_\text{obs} = 3,600\text{s (1 h)} \to T_\text{pred} = 86,400\text{s (24 h)}$ |
| **Target** | Future increment only: $\Delta P = R_{24\text{h}} - R_{1\text{h}}$, transformed as $\log_2(\Delta P + 1)$ |
| **Loss & Primary Metric** | $\text{MSLE}_{\log_2} = \frac{1}{N}\sum (\log_2(\widehat{\Delta P}+1) - \log_2(\Delta P+1))^2$ (Base-2); $\text{MSLE}_{\ln}$ and $\text{MALE}_{\log_2}$ also reported |
| **Optimization Harness** | Batch size `64`, `AdamW` (`lr = 2e-3`, `weight_decay = 1e-4`), `ReduceLROnPlateau(factor=0.5, patience=3)`, grad clip `max_norm = 1.0`, early stop `patience = 8` (`max_epochs = 60`), restoring best validation MSLE checkpoint |
| **Seeds & Zero Leakage** | Seeds `42, 43, 44`; feature scalers (`SequenceScaler` / `train_scaler.pkl`) fitted strictly on the `29,530` training cascades only |

### Architecture Fidelity & Implementation Details
- `cascn/data.py` computes the exact random-walk normalized directed Laplacian $L = I - \frac{1}{2}(\Phi^{1/2}(I - P)\Phi^{-1/2} + \dots)$ with teleportation $\alpha = 0.95$ (`caslaplacian.py` from `ChenNed/CasCN`) and scales by $2 / \lambda_{\max} - I$.
- `cascn/model.py` implements the exact Chebyshev $K=2$ Graph Convolution (`cheby_conv`) inside the 4-gate Graph Conv-LSTM (`gcnRNNCell` with 32 hidden units, `max_nodes = 100`, `max_steps = 100`), followed by 6-interval learned time-decay weighting (`time_weight`) and a `32 → 32 → 16 → 1` MLP readout.

---

## Results — 1-Hour Observation Window (`3600s`, 6,328 identical test cascades)

| Model | Parameters | Val MSLE ($\log_2$) $\downarrow$ | Test MSLE ($\log_2$) $\downarrow$ | Test MSLE ($\ln$) $\downarrow$ | Test MALE ($\log_2$) $\downarrow$ | $R^2$ ($\log_2$) $\uparrow$ | PCC $\uparrow$ | MAPE ($\log_2$) $\downarrow$ | MedAE (count) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Mean Baseline | 0 | — | 5.2825 | 2.5380 | — | -0.0003 | — | 0.3864 | 21.50 |
| CasCN, seed 42 | 35,623 | 2.5503 | 2.4675 | 1.1855 | 1.0835 | 0.5327 | 0.7303 | 0.2452 | 12.20 |
| CasCN, seed 43 | 35,623 | 2.7801 | 2.6440 | 1.2703 | 1.1499 | 0.4993 | 0.7116 | 0.2530 | 12.66 |
| CasCN, seed 44 | 35,623 | 2.7552 | 2.5931 | 1.2459 | 1.1445 | 0.5090 | 0.7163 | 0.2537 | 12.85 |
| **CasCN (3-seed mean ± std)** | **35,623** | **2.6952 ± 0.1261** | **2.5682 ± 0.0909** | **1.2339 ± 0.0437** | **1.1260 ± 0.0369** | **0.5137 ± 0.0172** | **0.7194** | **0.2506 ± 0.0047** | **12.57** |
| CasMotif, seed 42 | 49,824 | 2.0494 | 1.9357 | 0.9300 | 0.9705 | 0.6335 | 0.7977 | 0.2312 | 11.73 |
| CasMotif, seed 43 | 49,824 | 2.0628 | 1.9537 | 0.9387 | 0.9723 | 0.6300 | 0.7946 | 0.2290 | 11.31 |
| CasMotif, seed 44 | 49,824 | 2.0491 | 1.9443 | 0.9342 | 0.9655 | 0.6318 | 0.7971 | 0.2288 | 11.34 |
| **CasMotif (3-seed mean ± std)** | **49,824** | **2.0538 ± 0.0078** | **1.9446 ± 0.0090** | **0.9343 ± 0.0043** | **0.9694 ± 0.0036** | **0.6318 ± 0.0017** | **0.7965** | **0.2297 ± 0.0013** | **11.46** |
| CasCN (3-model ensemble) | 106,869 | — | 2.4758 | 1.1895 | 1.1083 | 0.5312 | 0.7297 | 0.2472 | 12.29 |
| **CasMotif (3-model ensemble)** | **149,472** | — | **1.9303** | **0.9274** | **0.9652** | **0.6345** | **0.7981** | **0.2288** | **11.28** |

---

## Statistical Significance (Per-Cascade Paired Tests on 6,328 Test Cascades)

| Comparison | $\Delta\text{MSLE}_{\log_2}$ (CasMotif $-$ CasCN) | Relative MSLE Reduction | 95% Bootstrap CI | Paired $t$-stat | Paired $t$-test $p$-value | Wilcoxon $p$-value | CasMotif Better Fraction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3-Seed Mean Error** | **$-0.6236$** | **$24.28\%$** | **$[-0.7420, -0.5135]$** | $-10.63$ | **$3.57 \times 10^{-26}$** | **$1.16 \times 10^{-78}$** | **$60.5\%$** |
| **3-Model Ensemble** | **$-0.5455$** | **$22.03\%$** | **$[-0.6550, -0.4415]$** | $-10.01$ | **$1.99 \times 10^{-23}$** | **$1.81 \times 10^{-46}$** | **$57.0\%$** |
| **Seed 42 Head-to-Head** | **$-0.5318$** | **$21.55\%$** | **$[-0.6487, -0.4222]$** | $-9.19$ | **$5.10 \times 10^{-20}$** | **$1.94 \times 10^{-31}$** | **$55.9\%$** |

---

## Reproducing

```bash
# 1. Preprocess the 42,186 chronological Weibo cascades into CasCN directed Laplacians & sequences
python3 cascn/data.py

# 2. Run the 3-seed (42, 43, 44) CasCN benchmark under the controlled protocol
python3 cascn/train.py --seeds 42 43 44 --out_dir cascn/runs_3600/cascn

# 3. Verify 1-to-1 cascade alignment and run paired statistical significance tests vs CasMotif
python3 cascn/compare.py
```
