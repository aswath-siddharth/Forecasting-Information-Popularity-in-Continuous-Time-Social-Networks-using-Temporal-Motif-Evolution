# Cross-Dataset Cascade Prediction Benchmark Report

**Evaluation Metric:** Primary MSLE on $\log_2(\Delta P + 1)$ (Mean $\pm$ Standard Deviation across 3 independent seeds: 42, 43, 44).  
**Protocol:** Strictly chronological 70% Train / 15% Validation / 15% Test. Zero target leakage; all scalers fitted exclusively on Train split.

## 1. Cross-Dataset Performance Benchmark

| Model | Sina Weibo 1h (3600s) | Sina Weibo 2h (7200s) | Twitter 1d (86400s) | Twitter 2d (172800s) | APS 3y (1095d) | APS 5y (1825d) | Reddit 1h (3600s) | Reddit 2h (7200s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean Baseline** | 5.2825 | 5.0051 | 2.3610 | 1.9120 | 2.4251 | 2.4484 | 0.8342 | 1.1424 |
| **CasMotif (Ours)** | **1.9446 ± 0.0074** | **1.8086 ± 0.0262** | **3.0450 ± 0.0710** | **2.6150 ± 0.2010** | **1.2773 ± 0.0259** | 1.1963 ± 0.0018 | **0.6224 ± 0.0062** | **0.7136 ± 0.0051** |
| **CasSubTS** | 1.9864 ± 0.0211 | 1.8614 ± 0.0196 | 3.3076 ± 0.1573 | 2.8869 ± 0.1241 | 1.2766 ± 0.0370 | **1.1895 ± 0.0112** | 0.7711 ± 0.0234 | 0.8696 ± 0.0265 |
| **CasFlow** | 2.3054 ± 0.0594 | 2.2496 ± 0.0313 | 3.2265 ± 0.3395 | 2.9835 ± 0.0911 | **1.2527 ± 0.0170** | 1.2122 ± 0.0085 | 0.8389 ± 0.0755 | 1.0492 ± 0.0282 |
| **CasCN** | 2.5682 ± 0.0742 | 2.4769 ± 0.0596 | 3.6905 ± 0.1751 | 3.2301 ± 0.4613 | 1.4456 ± 0.0554 | 1.3204 ± 0.0145 | 0.7485 ± 0.0244 | 0.9408 ± 0.0794 |
| **DeepHawkes** | 2.6167 ± 0.0781 | 2.2707 ± 0.0489 | 4.5190 ± 0.0670 | 3.6550 ± 0.0550 | 2.0279 ± 0.0180 | 1.4651 ± 0.0102 | 0.7149 ± 0.0100 | 0.8630 ± 0.0236 |
| **DeepCas** | 2.9785 ± 0.0460 | 2.7287 ± 0.0417 | 6.8688 ± 0.3958 | 7.6951 ± 0.3162 | 1.8725 ± 0.2484 | 1.6324 ± 0.3076 | 1.5489 ± 0.3001 | 2.2287 ± 0.3204 |

---

## 2. Benchmark Execution Runtime Breakdown

| Step / Task | Duration (Formatted) | Duration (Seconds) | Notes |
| :--- | :---: | :---: | :--- |
| **Reddit 7200s Cache Extraction** | 5m 53.5s | 353.5s | Train split (354,713 cascades) extraction |
| **CasCN Reddit 7200s Seed 42** | 1h 24m 40.3s | 5080.3s | Trained on CUDA (max 60 epochs, early stop=8) |
| **CasCN Reddit 7200s Seed 43** | 1h 52m 00.7s | 6720.7s | Trained on CUDA (max 60 epochs, early stop=8) |
| **CasCN Reddit 7200s Seed 44** | 43m 49.5s | 2629.5s | Trained on CUDA (max 60 epochs, early stop=8) |
| **Total CasCN 7200s Training** | 4h 00m 30.5s | 14430.5s | Sum across all 3 seeds |
| **Total End-to-End Orchestrator** | **4h 07m 42.5s** | **14862.5s** | Full pipeline (start: 2026-11-26T15:25:40.295065+00:00, end: 2026-11-26T19:33:22.811428+00:00) |
