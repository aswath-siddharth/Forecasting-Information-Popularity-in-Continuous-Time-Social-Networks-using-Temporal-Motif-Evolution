# Ablation Study & Experimental Framework Design (`ABLATION_DESIGN.md`)

## 1. Executive Summary & Architectural Overview

This document specifies the experimental design for the seven-combination ablation study (**A, B, C, AB, AC, BC, ABC**), controlled single-factor hyperparameter tuning, experimental parameter sensitivity analysis, artifact storage hierarchy, and automated visualization suite for the repository:
`Forecasting-Information-Popularity-in-Continuous-Time-Social-Networks-using-Temporal-Motif-Evolution`.

### Data Flow & Pipeline Dependencies

```
[Raw Reddit Comment Thread JSONs (data/*.json)]
       │                                  │
       ▼                                  ▼
┌────────────────────────────┐     ┌────────────────────────────┐
│        PIPELINE A          │     │        PIPELINE B          │
│ Dynamic CSR + WCOJ Motifs  │     │ 24h Cascade Aggregation &  │
│ + Temporal + VADER (20-D)  │     │ Subreddit P95 Virality     │
└──────────────┬─────────────┘     └──────────────┬─────────────┘
               │                                  │
               ▼                                  ▼
   pipeline_a_features.csv             pipeline_b_labels.csv
   (11,434 window rows, 275 posts,     (275 posts, 9 columns:
    23,900 edges, K x 20 matrices)      C_24h, U_24h, p95_cutoff,
               │                        y_viral, y_reg)
               └────────────────┬─────────────────┘
                                │ Inner Join on `post_id`
                                ▼
                 ┌────────────────────────────┐
                 │        PIPELINE C          │
                 │ SequenceScaler (Train-fit) │
                 │ + LayerNorm + BiGRU        │
                 │ + LearnedAttentionPooling  │
                 │ + BottleneckHead + Focal   │
                 └────────────────────────────┘
```

---

## 2. Dataset Availability & Upstream Dependency Audit

Before designing the seven ablation combinations, we audited every local file and git branch in the workspace:

1. **Raw JSON Threads (`data/*.json`):**
   - Listed in `.gitignore` (`data/`, `output_features/`, `output_features1/`, `pipeline_b_labels_full.csv`) and **not present on the local filesystem**.
2. **Cached Pipeline A Feature Artifacts (Present & Verified):**
   - `pipeline_a_features.csv` (from `origin/cassubts-baseline-comparison` / local repo copies): **11,434 window rows** across **275 unique posts** (`post_0` .. `post_274`, subreddit `Barca`), containing **23,900 comment edges** with full 20-D feature vectors (`X_seq_json`), including exact relative arrival timestamps `t_from_root = e.timestamp - post_time` in Column 8.
   - `pipeline_a/sample_output.csv` (on `main`): **11,434 window rows** across the exact same **275 posts**, representing the earlier schema where Column 8 duplicated `tree_depth` (Column 5).
3. **Cached Pipeline B Label Artifact (Present & Verified):**
   - `pipeline_b_labels.csv`: **275 unique posts** (`post_0` .. `post_274`, subreddit `Barca`) with ground-truth 24-hour metrics (`total_comments_24h`, `unique_users_24h`, `peak_hourly_velocity`, `time_to_peak_seconds`, `p95_cutoff = 180.0`, `y_viral` with 15 viral / 260 non-viral posts = `5.4545%`, and `y_reg = log10(C_24h + 1)`).
4. **Exact Parametric Regeneration Capability from Lossless Edge Matrices:**
   - Because `pipeline_a_features.csv` preserves every individual edge's exact arrival timestamp (`Col 8: t_from_root`), idle duration (`Col 14: t_idle`), tree depth (`Col 5: tree_depth`), parent out-degree (`Col 6`), and sentiment scores (`Cols 9–12`) for all 23,900 edges across all 275 posts:
     - **Pipeline A parameter variations** (`window_size_seconds` $\in \{300, 600, 900, 1800, 3600\}\text{s}$, `reactivation_threshold_seconds` $\in \{60, 180, 300, 600, 1800\}\text{s}$, `chain_depth_threshold` $\in \{2, 3, 4, 5\}$, and feature group subsets) can be **regenerated deterministically and exactly** from the 23,900 stored edge records (verified with `0 / 23,900` mismatch at the default `900s` window and `300s` reactivation threshold).
   - Because `pipeline_b_labels.csv` preserves exact 24-hour comment volume `total_comments_24h` (`c_24h`) for all 275 posts:
     - **Pipeline B parameter variations** (`virality_percentile` $\in \{0.80, 0.85, 0.90, 0.925, 0.95, 0.975\}$ and `cutoff_floor` $\in \{5, 15, 30, 50, 185, 190\}$) can be **regenerated deterministically and exactly** by running `calculate_percentile_cutoffs` (`pipeline_b/run.py:223`) on the 275 post objects.
   - Any experiment that specifically demands uncached raw JSON files (`data/*.json`, e.g., changing the 24h comment collection horizon in Pipeline B) is explicitly flagged and recorded as `BLOCKED_MISSING_RAW_JSON` rather than fabricated.

---

## 3. Specification of the Seven Ablation Configurations (Part 3)

| Ablation ID | Enabled Pipelines | Disabled Pipelines (`N/A` Params) | Input Source Mode | Technical Executability & Validation | Reported Outputs & Metrics |
| :---: | :---: | :---: | :--- | :--- | :--- |
| **`A`** | **Pipeline A** (Feature Extraction Engine) | B (`EP_B* = N/A`), C (`HP_* = N/A, EP_C* = N/A`) | Runs Pipeline A feature transformation / verification on the 23,900 edge stream (`275` posts) + checks raw `data/` availability | **Executable** (regenerates feature CSV from edge stream; logs raw JSON status explicitly) | Feature extraction statistics: total posts (`275`), window rows (`11,434`), total edges (`23,900`), sequence length $K$ distribution, motif prevalence (`Single, Star, Chain, Dyad, Cycle`), VADER sentiment summary, reactivation rate, runtime & memory |
| **`B`** | **Pipeline B** (Target Virality Labelling Engine) | A (`EP_A* = N/A`), C (`HP_* = N/A, EP_C* = N/A`) | Runs Pipeline B `calculate_percentile_cutoffs` & `validate_pipeline_output` on the 275 post records | **Executable** (regenerates label CSV & runs all 4 automated validation checks from `pipeline_b/run.py:260`) | Labelling statistics: total posts (`275`), `C_24h` / `U_24h` / `V_max` distributions, computed `p95_cutoff` (`180.0`), viral count (`15`), non-viral count (`260`), positive class ratio (`5.45%`), `y_reg` stats, runtime & memory |
| **`C`** | **Pipeline C** (Classifier Only — Cached Inputs) | A (`EP_A* = N/A`), B (`EP_B* = N/A`) | Consumes **existing pre-cached** `pipeline_a_features.csv` and `pipeline_b_labels.csv` directly without executing A or B; also records strict no-cache status (`BLOCKED_IF_NO_CACHE`) | **Executable (Cached Mode)** & **Audited (Strict No-Cache Mode)** | Full 11 classification metrics (`Accuracy, Precision, Recall, F1, Macro F1, Weighted F1, ROC-AUC, PR-AUC, Balanced Accuracy, MCC, Confusion Matrix`) for `attention_gru`, `mean_pool`, and `max_pool` across seeds, plus parameter counts, best epoch, runtime & memory |
| **`AB`** | **Pipelines A + B** (Joint Feature & Label Generation, No Classifier) | C (`HP_* = N/A, EP_C* = N/A`) | Executes Pipeline A feature generation and Pipeline B label generation, then performs cross-pipeline `post_id` inner-join audit (`JoinStats`) | **Executable** | Combined A feature statistics + B label statistics + `JoinStats` verification (`11,434` feature rows joined with `275` label rows, `0` dropped rows, `718` viral window rows = `6.28%` vs `10,716` non-viral window rows) |
| **`AC`** | **Pipelines A + C** (Fresh Feature Generation + Classifier with Cached Labels) | B (`EP_B* = N/A`, uses cached `pipeline_b_labels.csv`) | Executes Pipeline A to newly generate feature artifacts (testing both `v2_t_from_root` and `v1_sample_output` / active A config), joins with cached B labels, and trains Pipeline C | **Executable** | Pipeline A generation stats + Full 11 classification metrics for Pipeline C on newly generated A features |
| **`BC`** | **Pipelines B + C** (Fresh Label Generation + Classifier with Cached Features) | A (`EP_A* = N/A`, uses cached `pipeline_a_features.csv`) | Executes Pipeline B to newly generate virality labels, joins with cached A features, and trains Pipeline C | **Executable** | Pipeline B labelling stats + Full 11 classification metrics for Pipeline C on newly generated B labels |
| **`ABC`** | **Pipelines A + B + C** (Full End-to-End Pipeline) | None (All A, B, C parameters active) | Executes Pipeline A (regenerating features), executes Pipeline B (regenerating labels), validates inner join, and trains/evaluates Pipeline C (`attention_gru`, `mean_pool`, `max_pool`) across multiple seeds | **Executable (Full End-to-End Benchmark)** | Complete end-to-end Pipeline A metrics, Pipeline B metrics, JoinStats, and multi-seed Pipeline C classification metrics + runtime & memory breakdown across all 3 stages |

---

## 4. Controlled Hyperparameter & Sensitivity Experiment Plan (Parts 4 & 5)

### Phase 1: Original Baseline Establishment
- Run the exact default configuration from `pipeline_c/train.py` across **5 random seeds (`42, 43, 44, 45, 46`)**:
  - `lr=3e-4`, `batch_size=256`, `epochs=20`, `patience=3`, `loss_type=focal`, `focal_alpha=0.75`, `focal_gamma=2.0`, `dropout=0.3`, `hidden_size=64`, `num_layers=1`, `attention_dim=64`, `weight_decay=1e-4`, `scheduler_type=plateau_cosine`, `cosine_epochs=3`, `lr_factor=0.5`, `max_grad_norm=1.0`, `init_prior_bias=True`, `use_bucket_sampler=True`.
- Evaluate all three internal architectures (`attention_gru`, `mean_pool`, `max_pool`) and record both:
  1. **Original `row_stratified` split** (matching `pipeline_c/train.py:208`), and
  2. **Leakage-free `post_grouped_stratified` split** (where all windows of a post stay exclusively in Train, Val, or Test).

### Phase 2: Controlled Single-Factor (One-at-a-Time) Hyperparameter Sensitivity (Pipeline C)
Holding all other parameters at their baseline defaults (seed `42`, plus 3-seed verification on top candidates), vary one hyperparameter at a time:
1. **Learning Rate (`--lr`):** `1e-4, 3e-4 (baseline), 5e-4, 1e-3, 2e-3`
2. **Batch Size (`--batch-size`):** `32, 64, 128, 256 (baseline), 512`
3. **Dropout Rate (`--dropout`):** `0.0, 0.1, 0.2, 0.3 (baseline), 0.5`
4. **Hidden Dimension (`--hidden-size`):** `32, 64 (baseline), 128, 256`
5. **Number of BiGRU Layers (`--num-layers`):** `1 (baseline), 2, 3`
6. **Attention Dimension (`--attention-dim`):** `32, 64 (baseline), 128`
7. **Loss Function Type (`--loss-type`):** `focal (baseline), bce_damped, bce_raw`
8. **Focal Loss Alpha (`--focal-alpha`):** `0.25, 0.50, 0.75 (baseline), 0.85, 0.95`
9. **Focal Loss Gamma (`--focal-gamma`):** `0.5, 1.0, 2.0 (baseline), 3.0, 5.0`
10. **Weight Decay (`--weight-decay`):** `0.0, 1e-5, 1e-4 (baseline), 1e-3, 1e-2`
11. **Optimizer Type:** `AdamW (baseline), Adam, RMSprop`
12. **LR Scheduler Type (`--scheduler-type`):** `plateau_cosine (baseline), plateau`
13. **Early Stopping & Plateau Patience (`--patience`):** `3 (baseline), 5, 8, 10`
14. **Max Epochs (`--epochs`):** `15, 20 (baseline), 30, 50`
15. **Gradient Clipping Norm (`max_grad_norm`):** `0.5, 1.0 (baseline), 2.0, 5.0`
16. **Prior Bias Initialization (`init_prior_bias`):** `True (baseline), False`
17. **Length-Grouped Bucket Sampler (`--use-bucket-sampler`):** `True (baseline), False`

### Phase 3: Controlled Single-Factor Experimental Parameter Sensitivity (Pipelines A, B, C)
Varying one experimental parameter at a time and regenerating only the necessary intermediate artifact:
1. **Pipeline A — Temporal Window Size (`window_size_seconds`):** `300s (5m), 600s (10m), 900s (15m, baseline), 1800s (30m), 3600s (60m)`
2. **Pipeline A — Dormant Reactivation Threshold (`reactivation_threshold_seconds`):** `60s, 180s, 300s (baseline), 600s, 1800s`
3. **Pipeline A — WCOJ Chain Depth Threshold (`chain_depth_threshold`):** `2, 3 (baseline), 4, 5`
4. **Pipeline A/C — Feature Group Ablation (`feature_subset`):**
   - `all_20d` (20-D baseline)
   - `no_sentiment_16d` (drop VADER cols 9–12)
   - `no_motifs_15d` (drop WCOJ motif cols 0–4)
   - `no_reactivation_17d` (drop reactivation cols 13–15)
   - `no_csr_16d` (drop CSR centrality cols 5–8)
   - `motifs_plus_temporal_12d` (cols 0–4, 8, 13–15, 16–19)
   - `structural_temporal_11d` (drop motifs & sentiment; cols 5–8, 13–19)
5. **Pipeline A — Artifact Version (`feature_artifact_version`):** `v2_t_from_root` (`pipeline_a_features.csv`) vs `v1_dup_depth` (`pipeline_a/sample_output.csv`)
6. **Pipeline B — Virality Percentile Threshold (`virality_percentile`):** `0.80 (P80), 0.85 (P85), 0.90 (P90), 0.925 (P92.5), 0.95 (P95, baseline), 0.975 (P97.5)`
7. **Pipeline B — Volume Cutoff Floor (`cutoff_floor`):** `15.0 (baseline, effective cutoff 180), 185.0 (11 viral posts), 190.0 (7 viral posts)`
8. **Pipeline C — Early Observation Window Cutoff (`max_window_idx`):** `None (all windows, baseline), 95 (<=24h, leakage-free), 15 (<=4h), 7 (<=2h), 3 (<=1h)`
9. **Pipeline C — Feature Scaling Mode (`scaling_mode`):** `log1p_zscore (baseline), standard_zscore, none (LayerNorm only)`
10. **Pipeline C — Classification Decision Threshold ($\tau$):** `0.30, 0.40, 0.50 (baseline), 0.60, 0.70, val_optimal_f1`

### Phase 4: Combined Hyperparameter Optimization
- Combine the top-performing individual hyperparameter settings (selected **strictly on Validation Macro-F1 / Validation Loss**, never on the Test set) and evaluate across **5 random seeds (`42, 43, 44, 45, 46`)** to test whether the combined configuration outperforms the default baseline with statistical significance.

---

## 5. Hardware Inspection, Storage & Runtime Estimates (Part 10 Pre-Execution Audit)

| Resource Dimension | Measured Local Value | Impact on Execution Plan |
| :--- | :--- | :--- |
| **CPU & Architecture** | Apple M2 (`8` cores: 4 performance + 4 efficiency) | Fast vectorized PyTorch execution (`0.61s/epoch`, `~12.3s` per 20-epoch `attention_gru` run) |
| **GPU / Accelerator** | CUDA: `Not Available` (`cpu` execution) | AMP (`--amp`) is automatically disabled on CPU; `num_workers=0` avoids multiprocessing IPC overhead on in-memory datasets |
| **System RAM** | `8.00 GB` (`8,589,934,592` bytes) | Peak RSS per single-file run is `< 350 MB`; fits comfortably in memory with zero swapping |
| **Available Disk Space** | `14.0 GB` free on `/System/Volumes/Data` | Total storage for all ~95 experiment JSONs, CSVs, regenerated feature/label CSVs, and 10 high-DPI PNG charts is **< 85 MB** (we avoid saving redundant `.pt` weights for every sensitivity sweep point, saving checkpoints only for the 7 ablation runs, baseline, and best combined configuration) |
| **Measured Benchmark Speed** | `12.27s` per 20-epoch `attention_gru` run (`~3.5s` for `mean_pool`/`max_pool`) | - **7 Ablation Configurations (3 seeds each):** ~2.5 minutes<br>- **Hyperparameter Sensitivity Sweep (~45 runs):** ~8–10 minutes<br>- **Experimental Parameter Sensitivity Sweep (~30 runs):** ~6–8 minutes<br>- **Combined Optimization (5 seeds):** ~2 minutes<br>- **Total Estimated Runtime for Complete Suite:** **~18–22 minutes** |
