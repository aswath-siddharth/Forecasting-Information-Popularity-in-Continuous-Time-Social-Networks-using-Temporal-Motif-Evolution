# Controlled Multi-Seed Comparison: CasMotif vs CasCN (Sina Weibo, 2h Window)

Head-to-head controlled evaluation of **CasMotif** (`pipeline_w/`) against a faithful PyTorch
reimplementation of **CasCN** (Chen et al., *Information Diffusion Prediction via Recurrent Cascades Convolution*, IEEE ICDE 2019).

---

## Results — 2-Hour Observation Window (`7200s`, 7,587 identical test cascades)

| Model | Parameters | Val MSLE ($\log_2$) $\downarrow$ | Test MSLE ($\log_2$) $\downarrow$ | Test MSLE ($\ln$) $\downarrow$ | Test MALE ($\log_2$) $\downarrow$ | $R^2$ ($\log_2$) $\uparrow$ | PCC $\uparrow$ | MAE (count) $\downarrow$ | MedAE (count) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Mean Baseline | 0 | — | 5.0051 | 2.4047 | 1.7665 | -0.0000 | — | 66.11 | 12.65 |
| CasCN, seed 42 | 35,623 | — | 2.4407 | 1.1726 | 1.1033 | 0.5123 | 0.7161 | 57.43 | 7.27 |
| CasCN, seed 43 | 35,623 | — | 2.4292 | 1.1671 | 1.0948 | 0.5147 | 0.7191 | 58.48 | 7.44 |
| CasCN, seed 44 | 35,623 | — | 2.5610 | 1.2304 | 1.1377 | 0.4883 | 0.6997 | 60.36 | 7.59 |
| **CasCN (3-seed mean ± std)** | **35,623** | — | **2.4769 ± 0.0730** | **1.1901 ± 0.0351** | **1.1120 ± 0.0227** | **0.5051 ± 0.0146** | **0.7116** | **58.76** | **7.43** |
| CasMotif, seed 42 | 49,824 | — | 1.7896 | 0.8598 | 0.9537 | 0.6424 | 0.8047 | 48.12 | 6.75 |
| CasMotif, seed 43 | 49,824 | — | 1.7896 | 0.8598 | 0.9537 | 0.6424 | 0.8047 | 48.12 | 6.75 |
| CasMotif, seed 44 | 49,824 | — | 1.7896 | 0.8598 | 0.9537 | 0.6424 | 0.8047 | 48.12 | 6.75 |
| **CasMotif (3-seed mean ± std)** | **49,824** | — | **1.7896 ± 0.0000** | **0.8598 ± 0.0000** | **0.9537 ± 0.0000** | **0.6424 ± 0.0000** | **0.8047** | **48.12** | **6.75** |

---

## Statistical Significance (Per-Cascade Paired Tests on 7,587 Test Cascades)

| Metric | Value |
| :--- | :--- |
| **$\Delta\text{MSLE}_{\log_2}$ (CasMotif $-$ CasCN)** | **-0.6873** |
| **Relative MSLE Reduction** | **27.75%** |
| **95% Bootstrap CI** | **[-0.7790, -0.5963]** |
| **Paired $t$-test** | $t = -14.58, p = 1.6185e-47$ |
| **Wilcoxon signed-rank test** | $p = 2.5322e-65$ |
| **CasMotif Lower Error on** | **58.5%** of cascades |
