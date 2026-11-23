# CasSubTS Benchmark Results on Twitter Dataset (Daily Protocol: 2-Day Window)

Controlled multi-seed benchmark of **CasSubTS** (Wang et al., *Fusing temporal and structural information via subgraph sampling and multi-head attention for information cascade prediction*, Scientific Reports 15:6801, 2025) on the Twitter retweet dataset under the Daily Protocol with a 2-Day Observation Window.

---

## Controlled Daily Protocol

| Protocol Dimension | Setting |
| :--- | :--- |
| **Dataset Source** | `cascades/twitter.parquet` (88,440 cascades, unit: seconds) |
| **Observation Window ($T_\text{obs}$)** | $2\text{ days}$ ($172,800\text{s}$) |
| **Prediction Horizon ($T_\text{pred}$)** | $15\text{ days}$ ($1,296,000\text{s}$) |
| **Target ($\Delta P$)** | Future increment only: $\Delta P = R_{15\text{d}} - R_{2\text{d}}$, scaled as $\log_2(\Delta P + 1)$ |
| **Filtering Thresholds** | $10 \le N(T_\text{obs}) \le 1,000$ (24,596 eligible cascades; no hour filter) |
| **Partition** | Strictly chronological 70% / 15% / 15% by publish timestamp $t_0$ (Train: 17,217, Val: 3,689, Test: 3,690) |
| **Feature Scaler** | Fitted strictly on the 17,217 training cascades ONLY (zero data snooping) |
| **Optimization Harness** | Batch size 64, AdamW (`lr = 2e-3`, `wd = 1e-4`), ReduceLROnPlateau (`factor = 0.5`, `patience = 3`), Grad clip `max_norm = 1.0`, Early stop `patience = 8` (`max_epochs = 60`) |
| **Random Seeds** | 42, 43, 44 |

---

## Test Set Results (3,690 Identical Test Cascades)

| Model / Run | Best Epoch | Test MSLE ($\log_2$) $\downarrow$ | Test MSLE ($\ln$) $\downarrow$ | $R^2$ ($\log_2$) $\uparrow$ | PCC $\uparrow$ | MAPE ($\log_2$) $\downarrow$ | MedAE (count) $\downarrow$ | MAE (count) $\downarrow$ | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CasSubTS, Seed 42** | 10 | 2.7507 | 1.3216 | -0.4388 | 0.5368 | 0.7262 | 0.82 | 7.79 | 397.3s (6.62 min) |
| **CasSubTS, Seed 43** | 4 | 2.8593 | 1.3737 | -0.4956 | 0.5148 | 0.6247 | 1.17 | 6.95 | 270.8s (4.51 min) |
| **CasSubTS, Seed 44** | 4 | 3.0508 | 1.4658 | -0.5958 | 0.5231 | 0.7722 | 0.92 | 9.14 | 267.9s (4.46 min) |
| **CasSubTS (3-Seed Mean ± Std)** | — | **$2.8869 \pm 0.1241$** | **$1.3870 \pm 0.0596$** | **$-0.5101 \pm 0.0649$** | **$0.5249 \pm 0.0090$** | **$0.7077 \pm 0.0616$** | **$0.97 \pm 0.15$** | **$7.96 \pm 0.90$** | **936.0s (15.60 min total)** |

---

## Comparison: CasSubTS Across Observation Windows (1-Day vs. 2-Day)

| Metric | 1-Day Window ($T_\text{obs} = 1\text{d}$) | 2-Day Window ($T_\text{obs} = 2\text{d}$) | Improvement ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Eligible Cascades** | 19,665 | 24,596 | +4,931 cascades (+25.1%) |
| **Test Set Size** | 2,951 | 3,690 | +739 cascades |
| **Test MSLE ($\log_2$)** $\downarrow$ | $3.3076 \pm 0.1573$ | **$2.8869 \pm 0.1241$** | **-0.4207 (-12.7% error)** |
| **Test MSLE ($\ln$)** $\downarrow$ | $1.5891 \pm 0.0756$ | **$1.3870 \pm 0.0596$** | **-0.2021 (-12.7% error)** |
| **Pearson Correlation (PCC)** $\uparrow$ | $0.4964 \pm 0.0176$ | **$0.5249 \pm 0.0090$** | **+0.0285 (+5.7% higher correlation)** |
| **MedAE (count)** $\downarrow$ | $1.68 \pm 0.15$ | **$0.97 \pm 0.15$** | **-0.71 (-42.3% lower median error)** |
| **MAE (count)** $\downarrow$ | $8.31 \pm 0.70$ | **$7.96 \pm 0.90$** | **-0.35 (-4.2%)** |
| **Total Training Time** | 1,368.4s (22.81 min) | **936.0s (15.60 min)** | **-432.4s (31.6% faster convergence)** |

---

## Baseline Head-to-Head Comparison: CasSubTS vs. CasCN (Twitter 2-Day Window)

| Metric | CasCN (ICDE 2019) | CasSubTS (Sci Rep 2025) | Difference ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Architecture** | Chebyshev GCN + Bi-GRU | Multi-Head GAT + Bi-GRU + SENet | — |
| **Parameters** | 35,623 | 153,970 | +118,347 |
| **Test MSLE ($\log_2$)** $\downarrow$ | $3.2301 \pm 0.4613$ | **$2.8869 \pm 0.1241$** | **-0.3432 (-10.6% error)** |
| **Test MSLE ($\ln$)** $\downarrow$ | $1.5519 \pm 0.2216$ | **$1.3870 \pm 0.0596$** | **-0.1649 (-10.6% error)** |
| **$R^2$ ($\log_2$)** $\uparrow$ | $-0.6896 \pm 0.2413$ | **$-0.5101 \pm 0.0649$** | **+0.1795 (+26.0% improved calibration)** |
| **PCC** $\uparrow$ | $0.4292 \pm 0.0098$ | **$0.5249 \pm 0.0090$** | **+0.0957 (+22.3% higher correlation)** |
| **MAPE ($\log_2$)** $\downarrow$ | $0.7583 \pm 0.0343$ | **$0.7077 \pm 0.0616$** | **-0.0506 (-6.7% lower relative error)** |
| **MAE (count)** $\downarrow$ | $8.76 \pm 1.32$ | **$7.96 \pm 0.90$** | **-0.80 (-9.1% fewer count errors)** |
| **MedAE (count)** $\downarrow$ | $1.04 \pm 0.39$ | **$0.97 \pm 0.15$** | **-0.07 (-6.7%)** |
| **Total Training Time** | 1,733.6s (28.89 min) | **936.0s (15.60 min)** | **-797.6s (46.0% faster)** |
