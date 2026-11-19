# Controlled Multi-Seed Comparison: CasMotif vs CasFlow (Sina Weibo)

Head-to-head controlled evaluation of **CasMotif** against **CasFlow** (Xovee Xu et al.,
*CasFlow: Exploring Hierarchical Structures and Propagation Uncertainty for Cascade Prediction*,
IEEE Transactions on Knowledge and Data Engineering, TKDE 2021/2023).

---

## Controlled Protocol (Held Strictly Identical Across Both Models)

| Protocol Dimension | Setting |
| :--- | :--- |
| **Splits** | Chronological **70% / 15% / 15%** (`29,530 / 6,328 / 6,328` = `42,186` total). Zero leakage. |
| **Window & Horizon** | $T_\text{obs} = 3,600\text{s (1 h)} \to T_\text{pred} = 86,400\text{s (24 h)}$ |
| **Target** | Future increment only: $\Delta P = R_{24\text{h}} - R_{1\text{h}}$, transformed as $\log_2(\Delta P + 1)$ |
| **Loss & Primary Metric** | $\text{MSLE}_{\log_2} = \frac{1}{N}\sum (\log_2(\widehat{\Delta P}+1) - \log_2(\Delta P+1))^2$ |
| **Optimization Harness** | Batch size `64`, `AdamW` (`lr = 1e-3`, `weight_decay = 1e-4`), `ReduceLROnPlateau(factor=0.5, patience=3)`, grad clip `max_norm = 1.0`, early stop `patience = 8` (`max_epochs = 60`), restoring best validation MSLE checkpoint |
| **Seeds & Zero Leakage** | Seeds `42, 43, 44`; global graph embeddings & feature scalers fitted strictly on the `29,530` training cascades only |

---

## Results — 1-Hour Observation Window (`3600s`, 6,328 identical test cascades)

| Model | Parameters | Val MSLE ($\log_2$) $\downarrow$ | Test MSLE ($\log_2$) $\downarrow$ | Test MSLE ($\ln$) $\downarrow$ | Test MALE ($\log_2$) $\downarrow$ | $R^2$ ($\log_2$) $\uparrow$ | PCC $\uparrow$ | MAPE ($\log_2$) $\downarrow$ | MedAE (count) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Mean Baseline | 0 | — | 9.3094 | 4.4727 | — | -0.7628 | — | 0.6632 | 93.71 |
| CasFlow (seed 42) | 1,241,913 | — | 2.2447 | 1.0785 | 1.0226 | 0.5749 | 0.7593 | 0.2374 | 12.23 |
| CasFlow (seed 43) | 1,241,913 | — | 2.2855 | 1.0981 | 1.0459 | 0.5672 | 0.7609 | 0.2527 | 13.49 |
| CasFlow (seed 44) | 1,241,913 | — | 2.3861 | 1.1464 | 1.0451 | 0.5482 | 0.7405 | 0.2443 | 11.88 |
| **CasFlow (3-seed mean ± std)** | **1,241,913** | — | **2.3054 ± 0.0594** | **1.1076 ± 0.0286** | **1.0379** | **0.5634 ± 0.0113** | **0.7536** | **0.2448** | **12.53** |
| CasMotif (seed 42) | 49,824 | 2.0494 | 1.9357 | 0.9300 | 0.9705 | 0.6335 | 0.7977 | 0.2312 | 11.73 |
| CasMotif (seed 43) | 49,824 | 2.0628 | 1.9537 | 0.9387 | 0.9723 | 0.6300 | 0.7946 | 0.2290 | 11.31 |
| CasMotif (seed 44) | 49,824 | 2.0491 | 1.9443 | 0.9342 | 0.9655 | 0.6318 | 0.7971 | 0.2288 | 11.34 |
| **CasMotif (3-seed mean ± std)** | **49,824** | **2.0538 ± 0.0078** | **1.9446 ± 0.0074** | **0.9343 ± 0.0035** | **0.9694** | **0.6318 ± 0.0014** | **0.7965** | **0.2297** | **11.46** |
| CasFlow (3-model ensemble) | 3,725,739 | — | 2.2800 | 1.0954 | 1.0280 | 0.5682 | 0.7560 | 0.2433 | 12.45 |
| **CasMotif (3-model ensemble)** | **149,472** | — | **1.9319** | **0.9282** | **0.9654** | **0.6342** | **0.7979** | **0.2288** | **11.29** |

---

## Statistical Significance (Per-Cascade Paired Tests on 6,328 Test Cascades)

| Comparison | $\Delta\text{MSLE}_{\log_2}$ (CasMotif $-$ CasFlow) | Relative MSLE Reduction | 95% Bootstrap CI | Paired $t$-stat | Paired $t$-test $p$-value | Wilcoxon $p$-value | CasMotif Better Fraction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **3-Seed Mean Error** | **-0.3125** | **13.93%** | **[-0.4443, -0.1774]** | -4.62 | **3.89e-06** | **2.12e-02** | **50.6%** |
