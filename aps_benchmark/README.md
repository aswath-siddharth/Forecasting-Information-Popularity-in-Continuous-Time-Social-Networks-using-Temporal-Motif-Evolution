# APS Citation Cascade Benchmark: CasMotif vs. DeepHawkes

Controlled, reproducible benchmark comparing **CasMotif** (our causal temporal motif attention regressor) against a faithful PyTorch reimplementation of **DeepHawkes** (*Cao et al., ACM CIKM 2017*) on the **American Physical Society (APS) Citation Dataset** (`cascades.zip` / `cascades/aps.parquet`).

---

## 1. Benchmark Protocol

### Data Splits & Zero Leakage
- **Dataset:** $616,316$ total APS papers with $7,945,286$ citations.
- **Pre-1997 Cutoff:** Filtered strictly to papers published before **January 1, 1997** (`published < '1997-01-01'`) to ensure that every evaluated paper has an un-truncated, full 20-year post-publication citation observation window without right-censoring.
- **Activity Filter:** $10 \le N(T_{\text{obs}}) \le 1,000$ early citations within $T_{\text{obs}}$.
- **Partition:** Strictly chronological by publication date $t_0$ (**70% Train / 15% Val / 15% Test**).
- **Zero Leakage:** `SequenceScaler` and DeepHawkes node vocabulary are fitted exclusively on the training split. Unseen test nodes map to `[UNK]`.

### Observation Windows & Targets
- **Observation Windows ($T_{\text{obs}}$):**
  - **3 Years ($1,095$ days):** $26,555$ papers ($18,588$ train / $3,983$ val / $3,984$ test).
  - **5 Years ($1,825$ days):** $45,860$ papers ($32,101$ train / $6,879$ val / $6,880$ test).
- **Prediction Horizon ($T_{\text{pred}}$):** **20 years post-publication ($7,300$ days)**.
- **Target ($\Delta P$):** Future increment only: $\Delta P = \max(0, R_{20\text{y}} - N(T_{\text{obs}}))$.
- **Primary Loss & Metric:** $\text{MSLE}_{\log_2} = \frac{1}{N} \sum_{i=1}^N \left(\log_2(\widehat{\Delta P}_i + 1) - \log_2(\Delta P_i + 1)\right)^2$.

### Optimization Harness
- **Batch Size:** 64
- **Optimizer:** AdamW (base LR $10^{-3}$, weight decay $10^{-4}$; DeepHawkes embedding table LR $10^{-4}$ with active-batch L2 penalty $0.01$).
- **LR Schedule:** `ReduceLROnPlateau(factor=0.5, patience=3)`.
- **Gradient Clipping:** Max norm $1.0$.
- **Early Stopping:** Patience 8 epochs (max 60 epochs), restoring lowest validation MSLE checkpoint.
- **Seeds:** Evaluated across identical random seeds `42, 43, 44`.

---

## 2. Quickstart & Running the Benchmark

### Full End-to-End Run (Both 3-Year and 5-Year Windows, 3 Seeds)
```bash
python -m aps_benchmark.run_benchmark
```

### Run 3-Year Window Only
```bash
python -m aps_benchmark.run_benchmark --windows 1095
```

### Run 5-Year Window Only
```bash
python -m aps_benchmark.run_benchmark --windows 1825
```

### Smoke Test on Single Seed (Seed 42)
```bash
python -m aps_benchmark.run_benchmark --windows 1095 --seeds 42
```
