# Twitter Cascade Benchmark: CasMotif vs. DeepHawkes

Controlled, reproducible evaluation of **CasMotif** (`pipeline_w/` architecture) against **DeepHawkes** (*Cao et al., ACM CIKM 2017*) on the canonical Twitter cascade dataset from `cascades.zip`.

---

## 1. Experimental Protocol (Strict Controlled Setup)

| Parameter | Setting | Protocol Rationale |
| :--- | :--- | :--- |
| **Dataset Source** | `cascades.zip` (`cascades/twitter.parquet`) | 88,440 raw cascades across 12.5M retweet events. |
| **Observation Windows ($T_\text{obs}$)** | **1 day** ($86,400\text{s}$) and **2 days** ($172,800\text{s}$) | Captures global social diffusion across daily cycles. |
| **Prediction Horizon ($T_\text{pred}$)** | **15 days** ($1,296,000\text{s}$) | Canonical medium-term horizon in Twitter cascade literature. |
| **Target ($\Delta P$)** | Future increment only: $\Delta P = R_{15\text{d}} - R_{T_\text{obs}}$ | Transformed as $\log_2(\Delta P + 1)$. Prevents inflating performance by counting already-seen retweets. |
| **Cascade Filtering** | $10 \le N(T_\text{obs}) \le 1,000$ | Standard benchmark bounds. Beijing daytime filter is disabled (`no_hour_filter`) due to global Twitter distribution. |
| **Data Partition** | Strictly Chronological 70% / 15% / 15% (by $t_0$) | **1-day:** 13,766 train / 2,950 val / 2,949 test (19,665 total)<br>**2-day:** 17,217 train / 3,689 val / 3,690 test (24,596 total) |
| **Optimization Harness** | Batch size 64, AdamW (`lr=1e-3`, `weight_decay=1e-4`), `ReduceLROnPlateau(factor=0.5, patience=3)` | Max norm clipping 1.0, early stopping patience 8 (max 60 epochs), restoring lowest validation MSLE checkpoint. |
| **Random Seeds** | `42, 43, 44` | Evaluated across identical seeds with paired Student's $t$-test, Wilcoxon signed-rank test, and 10,000-sample bootstrap 95% CI. |
| **Feature Scaler** | Fit on Train Split ONLY | `SequenceScaler` computes mean/variance strictly on training cascades to prevent future data snooping. |
| **Vocab Construction** | Fit on Train Split ONLY | DeepHawkes user vocabulary is built strictly on the training split (`min_user_freq = 2`), mapping unseen users to `<UNK>`. |

---

## 2. Model Architectures & Parameter Footprints

- **CasMotif (Ours):**
  - Input: 15-D causal temporal motif sequence features (single, star, chain, dyad, cycle motifs + graph state dynamics).
  - Encoder: Bidirectional GRU (hidden=64) + Learned Attention Pooling + Bottleneck MLP head.
  - Parameters: **49,824** (inductive, zero user-ID embedding table).
- **DeepHawkes (*CIKM '17*):**
  - Input: Raw user propagation paths + elapsed timestamps.
  - Encoder: 50-D user embeddings (108k+ users at 1d, 130k+ at 2d) + single-layer GRU (hidden=64) + 6 non-parametric time decay weights + sum pooling + MLP head.
  - Parameters: **5,451,203** at 1d / **6,556,103** at 2d (embedding-heavy).

---

## 3. Running the Benchmark

### Full End-to-End Run (Both Windows, All 3 Seeds, Metrics & Chart):
```bash
python -m twitter_benchmark.run_benchmark
```

### Run 1-Day Window Only:
```bash
python -m twitter_benchmark.run_benchmark --windows 86400
```

### Run 2-Day Window Only:
```bash
python -m twitter_benchmark.run_benchmark --windows 172800
```
