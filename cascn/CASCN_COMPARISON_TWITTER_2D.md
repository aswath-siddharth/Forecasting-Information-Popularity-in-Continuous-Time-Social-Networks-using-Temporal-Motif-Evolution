# CasCN Benchmark Results on Twitter Dataset (Daily Protocol: 2-Day Window)

Controlled multi-seed benchmark of **CasCN** (Chen et al., *Information Diffusion Prediction via Recurrent Cascades Convolution*, IEEE ICDE 2019) on the Twitter retweet dataset under the Daily Protocol with a 2-Day Observation Window.

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
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CasCN, Seed 42** | 4 | 3.8711 | 1.8599 | -1.0249 | 0.4164 | 0.8022 | 1.59 | 10.56 | 339.4s (5.66 min) |
| **CasCN, Seed 43** | 16 | 3.0145 | 1.4483 | -0.5768 | 0.4311 | 0.7545 | 0.77 | 8.31 | 665.8s (11.10 min) |
| **CasCN, Seed 44** | 13 | 2.8046 | 1.3475 | -0.4670 | 0.4402 | 0.7184 | 0.75 | 7.41 | 583.3s (9.72 min) |
| **CasCN (3-Seed Mean ± Std)** | — | **$3.2301 \pm 0.4613$** | **$1.5519 \pm 0.2216$** | **$-0.6896 \pm 0.2413$** | **$0.4292 \pm 0.0098$** | **$0.7583 \pm 0.0343$** | **$1.04 \pm 0.39$** | **$8.76 \pm 1.32$** | **1,733.6s (28.89 min total)** |

---

## Comparison: 1-Day vs. 2-Day Observation Windows for CasCN

| Metric | 1-Day Window ($T_\text{obs} = 1\text{d}$) | 2-Day Window ($T_\text{obs} = 2\text{d}$) | Absolute Improvement ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Eligible Cascades** | 19,665 | 24,596 | +4,931 cascades (+25.1%) |
| **Test Set Size** | 2,951 | 3,690 | +739 cascades |
| **Test MSLE ($\log_2$)** $\downarrow$ | $3.6905 \pm 0.1751$ | **$3.2301 \pm 0.4613$** | **-0.4604 (-12.5%)** |
| **Test MSLE ($\ln$)** $\downarrow$ | $1.7731 \pm 0.0841$ | **$1.5519 \pm 0.2216$** | **-0.2212 (-12.5%)** |
| **PCC** $\uparrow$ | $0.4110 \pm 0.0081$ | **$0.4292 \pm 0.0098$** | **+0.0182 (+4.4%)** |
| **MAE (count)** $\downarrow$ | $10.78 \pm 2.31$ | **$8.76 \pm 1.32$** | **-2.02 (-18.7%)** |
| **MedAE (count)** $\downarrow$ | $1.41 \pm 0.04$ | **$1.04 \pm 0.39$** | **-0.37 (-26.2%)** |
| **Total Training Time** | 1,554.2s (25.90 min) | 1,733.6s (28.89 min) | +179.4s (+11.5%) |
