# CasSubTS Benchmark Results on Twitter Dataset (Daily Protocol: 1-Day Window)

Controlled multi-seed benchmark of **CasSubTS** (Wang et al., *Fusing temporal and structural information via subgraph sampling and multi-head attention for information cascade prediction*, Scientific Reports 15:6801, 2025) on the Twitter retweet dataset under the Daily Protocol.

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
| **CasSubTS, Seed 42** | 12 | 3.2912 | 1.5813 | -0.3947 | 0.5005 | 0.6084 | 1.82 | 7.77 | 432.5s (7.21 min) |
| **CasSubTS, Seed 43** | 5 | 3.1236 | 1.5007 | -0.3237 | 0.4731 | 0.6144 | 1.47 | 7.86 | 286.7s (4.78 min) |
| **CasSubTS, Seed 44** | 20 | 3.5079 | 1.6854 | -0.4865 | 0.5158 | 0.6657 | 1.74 | 9.30 | 649.2s (10.82 min) |
| **CasSubTS (3-Seed Mean ± Std)** | — | **$3.3076 \pm 0.1573$** | **$1.5891 \pm 0.0756$** | **$-0.4016 \pm 0.0667$** | **$0.4964 \pm 0.0176$** | **$0.6295 \pm 0.0257$** | **$1.68 \pm 0.15$** | **$8.31 \pm 0.70$** | **1,368.4s (22.81 min total)** |

---

## Baseline Head-to-Head Comparison: CasSubTS vs. CasCN (Twitter 1-Day Window)

| Metric | CasCN (ICDE 2019) | CasSubTS (Sci Rep 2025) | Difference ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Architecture** | Chebyshev GCN + Bi-GRU | Multi-Head GAT + Bi-GRU + SENet | — |
| **Parameters** | 35,623 | 153,970 | +118,347 |
| **Test MSLE ($\log_2$)** $\downarrow$ | $3.6905 \pm 0.1751$ | **$3.3076 \pm 0.1573$** | **-0.3829 (-10.4% error)** |
| **Test MSLE ($\ln$)** $\downarrow$ | $1.7731 \pm 0.0841$ | **$1.5891 \pm 0.0756$** | **-0.1840 (-10.4% error)** |
| **$R^2$ ($\log_2$)** $\uparrow$ | $-0.5639 \pm 0.0742$ | **$-0.4016 \pm 0.0667$** | **+0.1623 (+28.8% improvement)** |
| **PCC** $\uparrow$ | $0.4110 \pm 0.0081$ | **$0.4964 \pm 0.0176$** | **+0.0854 (+20.8% higher correlation)** |
| **MAPE ($\log_2$)** $\downarrow$ | $0.6835 \pm 0.0250$ | **$0.6295 \pm 0.0257$** | **-0.0540 (-7.9% lower relative error)** |
| **MAE (count)** $\downarrow$ | $10.78 \pm 2.31$ | **$8.31 \pm 0.70$** | **-2.47 (-22.9% fewer count errors)** |
| **MedAE (count)** $\downarrow$ | **$1.41 \pm 0.04$** | $1.68 \pm 0.15$ | +0.27 |
| **Total Training Time** | 1,554.2s (25.90 min) | **1,368.4s (22.81 min)** | **-185.8s (12.0% faster)** |
