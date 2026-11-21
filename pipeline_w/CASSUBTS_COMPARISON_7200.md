# Controlled Comparison: CasMotif vs CasSubTS (Sina Weibo, 2h Window)

Head-to-head controlled evaluation of **CasMotif** against a faithful PyTorch reimplementation of **CasSubTS**
(Wang, Li, Liu, Peng & Li, *Fusing temporal and structural information via subgraph sampling and multi-head attention for information cascade prediction*, Scientific Reports 15:6801, 2025).

---

## Multi-Seed Results ($T_\text{obs} = 7200\text{s}$, 7,587 Test Cascades)

| Model | Parameters | Test MSLE ($\log_2$) $\downarrow$ | Test MSLE ($\ln$) $\downarrow$ | $R^2$ ($\log_2$) $\uparrow$ | PCC $\uparrow$ | MAE (count) $\downarrow$ | MedAE (count) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Mean Baseline | 0 | 5.0051 | 2.4047 | -0.0000 | — | 66.11 | 12.65 |
| CasSubTS, seed 42 | 153,970 | 1.8587 | 0.8930 | 0.6286 | 0.7950 | 49.33 | 7.08 |
| CasSubTS, seed 43 | 153,970 | 1.8866 | 0.9064 | 0.6231 | 0.7894 | 50.06 | 7.17 |
| CasSubTS, seed 44 | 153,970 | 1.8388 | 0.8835 | 0.6326 | 0.7965 | 49.33 | 7.23 |
| **CasSubTS (3-seed mean ± std)** | **153,970** | **1.8614 ± 0.0196** | **0.8943 ± 0.0094** | **0.6281 ± 0.0039** | **0.7936** | **49.57** | **7.16** |
| CasMotif, seed 42 | 49,824 | 1.7896 | 0.8598 | 0.6424 | 0.8047 | 48.12 | 6.75 |
| CasMotif, seed 43 | 49,824 | 1.7896 | 0.8598 | 0.6424 | 0.8047 | 48.12 | 6.75 |
| CasMotif, seed 44 | 49,824 | 1.7896 | 0.8598 | 0.6424 | 0.8047 | 48.12 | 6.75 |
| **CasMotif (3-seed mean ± std)** | **49,824** | **1.7896 ± 0.0000** | **0.8598 ± 0.0000** | **0.6424 ± 0.0000** | **0.8047** | **48.12** | **6.75** |

---

## Head-to-Head Controlled Protocol ($T_\text{obs} = 7200\text{s}$, 7,587 Test Cascades)

| Metric | CasMotif (Ours) | CasSubTS (Sci Rep '25) | Winner |
| :--- | :---: | :---: | :---: |
| **Test $\text{MSLE}_{\log_2} \downarrow$** | **1.7896** | **1.8587** | **CasMotif (3.72% lower error)** |
| **Parameters** | **49,824** | **153,970** | **CasMotif (3.1× fewer params)** |

---

## Statistical Significance (Per-Cascade Paired Tests on 7,587 Test Cascades)

| Metric | Value |
| :--- | :--- |
| **$\Delta\text{MSLE}_{\log_2}$ (CasMotif $-$ CasSubTS)** | **-0.0691** |
| **Relative MSLE Reduction** | **3.72%** |
| **95% Bootstrap CI** | **[-0.1033, -0.0363]** |
| **Paired $t$-test** | $t = -3.99, p = 6.5339e-05$ |
| **Wilcoxon signed-rank test** | $p = 2.5557e-03$ |
| **CasMotif Lower Error on** | **51.1%** of cascades |
