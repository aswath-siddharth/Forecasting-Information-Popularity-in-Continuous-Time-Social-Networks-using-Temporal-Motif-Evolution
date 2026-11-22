# CasCN Benchmark Results on Twitter Dataset (Daily Protocol: 1-Day Window)

Controlled multi-seed benchmark of **CasCN** (Chen et al., *Information Diffusion Prediction via Recurrent Cascades Convolution*, IEEE ICDE 2019) on the Twitter retweet dataset under the Daily Protocol.

---

## Controlled Daily Protocol

| Protocol Dimension | Setting |
| :--- | :--- |
| **Dataset Source** | `cascades/twitter.parquet` (88,440 cascades, unit: seconds) |
| **Observation Window ($T_\text{obs}$)** | $1\text{ day}$ ($86,400\text{s}$) |
| **Prediction Horizon ($T_\text{pred}$)** | $15\text{ days}$ ($1,296,000\text{s}$) |
| **Target ($\Delta P$)** | Future increment only: $\Delta P = R_{15\text{d}} - R_{1\text{d}}$, scaled as $\log_2(\Delta P + 1)$ |
| **Filtering Thresholds** | $10 \le N(T_\text{obs}) \le 1,000$ (19,665 eligible cascades; no hour filter) |
| **Partition** | Strictly chronological 70% / 15% / 15% by publish timestamp $t_0$ (Train: 13,765, Val: 2,949, Test: 2,951) |
| **Feature Scaler** | Fitted strictly on the 13,765 training cascades ONLY (zero data snooping) |
| **Optimization Harness** | Batch size 64, AdamW (`lr = 2e-3`, `wd = 1e-4`), ReduceLROnPlateau (`factor = 0.5`, `patience = 3`), Grad clip `max_norm = 1.0`, Early stop `patience = 8` (`max_epochs = 60`) |
| **Random Seeds** | 42, 43, 44 |

---

## Test Set Results (2,951 Identical Test Cascades)

| Model / Run | Best Epoch | Test MSLE ($\log_2$) $\downarrow$ | Test MSLE ($\ln$) $\downarrow$ | $R^2$ ($\log_2$) $\uparrow$ | PCC $\uparrow$ | MAPE ($\log_2$) $\downarrow$ | MedAE (count) $\downarrow$ | MAE (count) $\downarrow$ | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CasCN, Seed 42** | 9 | 3.6451 | 1.7513 | -0.5447 | 0.4074 | 0.6727 | 1.41 | 10.05 | 400.0s (6.67 min) |
| **CasCN, Seed 43** | 17 | 3.9240 | 1.8853 | -0.6629 | 0.4046 | 0.7179 | 1.36 | 13.43 | 552.8s (9.21 min) |
| **CasCN, Seed 44** | 13 | 3.5024 | 1.6827 | -0.4842 | 0.4208 | 0.6597 | 1.46 | 8.86 | 464.7s (7.75 min) |
| **CasCN (3-Seed Mean ± Std)** | — | **$3.6905 \pm 0.1751$** | **$1.7731 \pm 0.0841$** | **$-0.5639 \pm 0.0742$** | **$0.4110 \pm 0.0081$** | **$0.6835 \pm 0.0250$** | **$1.41 \pm 0.04$** | **$10.78 \pm 2.31$** | **1,554.2s (25.90 min total)** |
