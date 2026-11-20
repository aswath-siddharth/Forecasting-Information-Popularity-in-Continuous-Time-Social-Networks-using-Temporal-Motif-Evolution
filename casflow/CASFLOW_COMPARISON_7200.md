# Controlled Comparison: CasMotif vs CasFlow (Sina Weibo 2h)

Head-to-head controlled evaluation of **CasMotif** against **CasFlow** (Xovee Xu et al.,
*CasFlow: Exploring Hierarchical Structures and Propagation Uncertainty for Cascade Prediction*,
IEEE Transactions on Knowledge and Data Engineering, TKDE 2021/2023).

---

## Controlled Protocol (Held Strictly Identical Across Both Models)

| Protocol Dimension | Setting |
| :--- | :--- |
| **Splits** | Chronological **70% / 15% / 15%** (7,587 test cascades). Zero leakage. |
| **Window & Horizon** | T_obs = 7,200s (2 h) -> T_pred = 86,400s (24 h) |
| **Target** | Future increment only: Delta P = R_{24h} - R_{2h}, transformed as log2(Delta P + 1) |
| **Loss & Primary Metric** | MSLE (log2) |
| **Optimization Harness** | Batch size `64`, `AdamW` (`lr = 1e-3`, `weight_decay = 1e-4`), `ReduceLROnPlateau(factor=0.5, patience=3)`, grad clip `max_norm = 1.0`, early stop `patience = 8` (`max_epochs = 60`), restoring best validation MSLE checkpoint |
| **Zero Leakage** | Global graph embeddings & feature scalers fitted strictly on training cascades only |

---

## Results — 2-Hour Observation Window (`7200s`, 7,587 identical test cascades)

| Model | Parameters | Val MSLE (log2) | Test MSLE (log2) | Test MSLE (ln) | Test MALE (log2) | R2 (log2) | PCC | MAPE (log2) | MedAE (count) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Mean Baseline | 0 | — | 9.2464 | 4.4424 | — | -0.8474 | — | 0.8453 | 58.91 |
| CasFlow (seed 42) | 1,241,913 | — | 2.2057 | 1.0597 | 1.0468 | 0.5593 | 0.7555 | 0.3013 | 8.49 |
| CasFlow (seed 43) | 1,241,913 | — | 2.2662 | 1.0888 | 1.0503 | 0.5472 | 0.7474 | 0.3010 | 8.05 |
| CasFlow (seed 44) | 1,241,913 | — | 2.2768 | 1.0939 | 1.0368 | 0.5451 | 0.7418 | 0.2973 | 7.60 |
| **CasFlow (3-seed mean ± std)** | **1,241,913** | — | **2.2496 ± 0.0313** | **1.0808 ± 0.0151** | **1.0447** | **0.5505 ± 0.0063** | **0.7482** | **0.2999** | **8.04** |
| CasMotif (seed 42) | 49,824 | 2.0494 | 1.7896 | 0.8598 | 0.9537 | 0.6424 | 0.8047 | 0.2767 | 6.75 |
| CasMotif (seed 43) | 49,824 | 2.0494 | 1.7896 | 0.8598 | 0.9537 | 0.6424 | 0.8047 | 0.2767 | 6.75 |
| CasMotif (seed 44) | 49,824 | 2.0494 | 1.7896 | 0.8598 | 0.9537 | 0.6424 | 0.8047 | 0.2767 | 6.75 |
| **CasMotif (3-seed mean ± std)** | **49,824** | **2.0538 ± 0.0078** | **1.7896 ± 0.0000** | **0.8598 ± 0.0000** | **0.9537** | **0.6424 ± 0.0000** | **0.8047** | **0.2767** | **6.75** |
---

## Statistical Significance (Per-Cascade Paired Tests on 7,587 Test Cascades)

| Comparison | Delta MSLE (log2) | Relative MSLE Reduction | 95% Bootstrap CI | Paired t-stat | Paired t-test p-value | Wilcoxon p-value | CasMotif Better Fraction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single-Model Error** | **-0.4213** | **19.06%** | **[-0.5305, -0.3089]** | -7.39 | **1.57e-13** | **7.81e-10** | **53.1%** |
