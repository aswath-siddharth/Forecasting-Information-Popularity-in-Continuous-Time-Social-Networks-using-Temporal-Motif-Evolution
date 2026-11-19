# Complete Ablation Study & Parameter Sensitivity Report (`ABLATION_RESULTS.md`)

- **Repository:** `Forecasting-Information-Popularity-in-Continuous-Time-Social-Networks-using-Temporal-Motif-Evolution`
- **Git Commit:** `b1721938f9fb2a71c4e0d1cfa66c91ba95551058`
- **Generated Timestamp (UTC):** `2026-11-19T09:35:10.414363+00:00`
- **Hardware Environment:** `arm (8 CPU cores, 8.0 GB RAM, macOS)`

---

## 1. Experimental Methodology

1. **Seven-Combination Pipeline Ablation (`A, B, C, AB, AC, BC, ABC`):**
   - Evaluates each pipeline standalone and in every joint combination.
   - Strictly distinguishes between **existing cached artifacts** (`pipeline_a_features.csv`, `pipeline_b_labels.csv`), **newly generated artifacts** from Pipeline A (`run_pipeline_a_engine`) and Pipeline B (`run_pipeline_b_engine`), and **experiments blocked by uncached raw JSON inputs** (`data/*.json`).
2. **Zero-Leakage Controlled Protocol:**
   - `SequenceScaler` (`log1p` + `StandardScaler` on the 10 unbounded columns) is fitted **strictly on the training split** (`pipeline_c/train.py:660`).
   - Hyperparameter selection and combined tuning decisions are made **exclusively using Validation Macro-F1 / Validation Loss** — never the held-out Test set.
   - Both the original **row-stratified 80/10/10 split** (`9,147` Train / `1,143` Val / `1,144` Test rows) and the **post-grouped stratified 80/10/10 split** (where all time windows of any given post reside strictly in Train, Val, or Test) are evaluated and compared.
3. **Multi-Seed Statistical Rigor:**
   - All baseline, ablation, and combined-tuning comparisons are evaluated across **multiple random seeds** (`3` seeds `[42, 43, 44]` for the 7 ablation combinations; `5` seeds `[42, 43, 44, 45, 46]` for the default baseline and combined tuned comparison), reporting `mean ± std` across all 11 classification metrics.

---

## 2. Seven Ablation Configurations (`A, B, C, AB, AC, BC, ABC`)

### 2.1 Standalone & Joint Data Pipeline Results (`A`, `B`, `AB`)

| Combination | Enabled Pipelines | Feature Provenance | Label Provenance | Posts | Window Rows | Total Edges | Viral Posts (`y=1`) | Viral Window Rows | Runtime (s) | Peak Memory (MB) |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`A`** | `A` | `NEWLY_GENERATED_BY_PIPELINE_A` | `NOT_USED` | `275` | `11434` | `23900` | `N/A` | `N/A` | `1.72s` | `300.1` |
| **`B`** | `B` | `NOT_USED` | `NEWLY_GENERATED_BY_PIPELINE_B` | `275` | `N/A` | `N/A` | `15 (5.45%)` | `N/A` | `0.03s` | `300.1` |
| **`AB`** | `AB` | `NEWLY_GENERATED_BY_PIPELINE_A` | `NEWLY_GENERATED_BY_PIPELINE_B` | `275` | `11434` | `23900` | `15 (5.45%)` | `718 (6.28%)` | `54.81s` | `475.4` |

### 2.2 Classifier Ablation Combinations (`C`, `AC`, `BC`, `ABC` — 3 Seeds `42, 43, 44`)

| Combination | Model Architecture | Params | Test Macro-F1 (Mean ± Std) | Test Positive F1 (Mean ± Std) | Test ROC-AUC (Mean ± Std) | Test PR-AUC (Mean ± Std) | Test Accuracy | Test Precision | Test Recall | Test Balanced Acc | Test MCC | Train Time (s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`C`** | `attention_gru` | `51,754` | **`0.8029 ± 0.0036`** | `0.6357 ± 0.0076` | `0.9618 ± 0.0043` | `0.6250 ± 0.0252` | `0.9449` | `0.5449` | `0.7639` | `0.8605` | `0.6172` | `12.05s` |
| **`C`** | `mean_pool` | `3,497` | **`0.7826 ± 0.0260`** | `0.5952 ± 0.0494` | `0.9568 ± 0.0040` | `0.5964 ± 0.0336` | `0.9443` | `0.5475` | `0.6528` | `0.8084` | `0.5683` | `3.50s` |
| **`C`** | `max_pool` | `3,497` | **`0.7882 ± 0.0301`** | `0.6056 ± 0.0566` | `0.9566 ± 0.0044` | `0.6007 ± 0.0459` | `0.9455` | `0.5550` | `0.6667` | `0.8155` | `0.5795` | `3.68s` |
| **`AC`** | `attention_gru` | `51,754` | **`0.8029 ± 0.0036`** | `0.6357 ± 0.0076` | `0.9618 ± 0.0043` | `0.6250 ± 0.0252` | `0.9449` | `0.5449` | `0.7639` | `0.8605` | `0.6172` | `330.72s` |
| **`AC`** | `mean_pool` | `3,497` | **`0.7826 ± 0.0260`** | `0.5952 ± 0.0494` | `0.9568 ± 0.0040` | `0.5964 ± 0.0336` | `0.9443` | `0.5475` | `0.6528` | `0.8084` | `0.5683` | `1.94s` |
| **`AC`** | `max_pool` | `3,497` | **`0.7882 ± 0.0301`** | `0.6056 ± 0.0566` | `0.9566 ± 0.0044` | `0.6007 ± 0.0459` | `0.9455` | `0.5550` | `0.6667` | `0.8155` | `0.5795` | `190.31s` |
| **`BC`** | `attention_gru` | `51,754` | **`0.8029 ± 0.0036`** | `0.6357 ± 0.0076` | `0.9618 ± 0.0043` | `0.6250 ± 0.0252` | `0.9449` | `0.5449` | `0.7639` | `0.8605` | `0.6172` | `11.85s` |
| **`BC`** | `mean_pool` | `3,497` | **`0.7826 ± 0.0260`** | `0.5952 ± 0.0494` | `0.9568 ± 0.0040` | `0.5964 ± 0.0336` | `0.9443` | `0.5475` | `0.6528` | `0.8084` | `0.5683` | `3.22s` |
| **`BC`** | `max_pool` | `3,497` | **`0.7882 ± 0.0301`** | `0.6056 ± 0.0566` | `0.9566 ± 0.0044` | `0.6007 ± 0.0459` | `0.9455` | `0.5550` | `0.6667` | `0.8155` | `0.5795` | `4.77s` |
| **`ABC`** | `attention_gru` | `51,754` | **`0.8029 ± 0.0036`** | `0.6357 ± 0.0076` | `0.9618 ± 0.0043` | `0.6250 ± 0.0252` | `0.9449` | `0.5449` | `0.7639` | `0.8605` | `0.6172` | `12.13s` |
| **`ABC`** | `mean_pool` | `3,497` | **`0.7826 ± 0.0260`** | `0.5952 ± 0.0494` | `0.9568 ± 0.0040` | `0.5964 ± 0.0336` | `0.9443` | `0.5475` | `0.6528` | `0.8084` | `0.5683` | `3.31s` |
| **`ABC`** | `max_pool` | `3,497` | **`0.7882 ± 0.0301`** | `0.6056 ± 0.0566` | `0.9566 ± 0.0044` | `0.6007 ± 0.0459` | `0.9455` | `0.5550` | `0.6667` | `0.8155` | `0.5795` | `3.93s` |

---

## 3. Original Default Baseline vs. Combined Tuned Configuration (5 Seeds: `42, 43, 44, 45, 46`)

> **Strict Separation Note:** The table below separates the **Original Default Baseline** (`lr=3e-4, batch_size=256, epochs=20, patience=3, dropout=0.3, hidden_size=64, focal_alpha=0.75, focal_gamma=2.0`) from the **Validation-Selected Combined Tuned Configuration**.

| Configuration | Architecture | Seeds | Val Macro-F1 (Mean ± Std) | Test Macro-F1 (Mean ± Std) | Test Positive F1 (Mean ± Std) | Test ROC-AUC (Mean ± Std) | Test PR-AUC (Mean ± Std) | Test Balanced Acc | Test MCC | Confusion Matrix (Seed 42: TN, FP, FN, TP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Default Baseline** | `attention_gru` | `5` | `0.8159 ± 0.0208` | **`0.8009 ± 0.0092`** | `0.6313 ± 0.0170` | `0.9623 ± 0.0068` | `0.6428 ± 0.0353` | `0.8516` | `0.6110` | `TN=1029, FP=43, FN=19, TP=53` |
| **Original Default Baseline** | `mean_pool` | `5` | `0.7915 ± 0.0134` | **`0.7772 ± 0.0254`** | `0.5854 ± 0.0472` | `0.9566 ± 0.0074` | `0.6066 ± 0.0372` | `0.8047` | `0.5579` | `TN=1033, FP=39, FN=31, TP=41` |
| **Original Default Baseline** | `max_pool` | `5` | `0.7792 ± 0.0187` | **`0.7768 ± 0.0344`** | `0.5846 ± 0.0633` | `0.9539 ± 0.0107` | `0.6040 ± 0.0551` | `0.8033` | `0.5569` | `TN=1031, FP=41, FN=30, TP=42` |
| **Combined Tuned Configuration** | `attention_gru` | `5` | `0.8285 ± 0.0153` | **`0.8003 ± 0.0229`** | `0.6317 ± 0.0413` | `0.9645 ± 0.0089` | `0.6660 ± 0.0412` | `0.8670` | `0.6151` | `TN=1014, FP=58, FN=18, TP=54` |

---

## 4. Controlled One-at-a-Time Hyperparameter Sensitivity Results (Pipeline C)

| Varied Hyperparameter | Value | Default? | Best Val Epoch | Best Val Loss | Val Macro-F1 | Test Macro-F1 | Test Positive F1 | Test Precision | Test Recall | Test ROC-AUC | Test PR-AUC | Test MCC | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `attention_dim` | `128` |  | `19` | `0.0146` | `0.8009` | **`0.8051`** | `0.6400` | `0.5437` | `0.7778` | `0.9661` | `0.5980` | `0.6227` | `6.89s` |
| `attention_dim` | `32` |  | `19` | `0.0144` | `0.8137` | **`0.7989`** | `0.6286` | `0.5340` | `0.7639` | `0.9660` | `0.6035` | `0.6101` | `6.31s` |
| `attention_dim` | `64` | Yes | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `6.08s` |
| `batch_size` | `128` |  | `11` | `0.0128` | `0.8006` | **`0.7871`** | `0.6038` | `0.5517` | `0.6667` | `0.9660` | `0.6240` | `0.5774` | `5.39s` |
| `batch_size` | `256` | Yes | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `6.50s` |
| `batch_size` | `32` |  | `10` | `0.0124` | `0.8030` | **`0.7988`** | `0.6279` | `0.5400` | `0.7500` | `0.9682` | `0.6651` | `0.6080` | `13.24s` |
| `batch_size` | `512` |  | `20` | `0.0175` | `0.7760` | **`0.7715`** | `0.5750` | `0.5227` | `0.6389` | `0.9641` | `0.5796` | `0.5466` | `5.62s` |
| `batch_size` | `64` |  | `10` | `0.0124` | `0.8226` | **`0.7901`** | `0.6108` | `0.5368` | `0.7083` | `0.9674` | `0.6457` | `0.5873` | `7.91s` |
| `dropout` | `0.0` |  | `19` | `0.0144` | `0.8051` | **`0.8030`** | `0.6369` | `0.5327` | `0.7917` | `0.9660` | `0.6210` | `0.6214` | `6.28s` |
| `dropout` | `0.1` |  | `19` | `0.0143` | `0.8051` | **`0.8030`** | `0.6364` | `0.5385` | `0.7778` | `0.9660` | `0.6217` | `0.6192` | `7.27s` |
| `dropout` | `0.2` |  | `19` | `0.0144` | `0.8006` | **`0.7829`** | `0.5963` | `0.5393` | `0.6667` | `0.9660` | `0.6217` | `0.5698` | `8.10s` |
| `dropout` | `0.3` | Yes | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `7.01s` |
| `dropout` | `0.5` |  | `19` | `0.0142` | `0.8073` | **`0.8051`** | `0.6391` | `0.5567` | `0.7500` | `0.9665` | `0.6041` | `0.6189` | `6.45s` |
| `epochs` | `15` |  | `14` | `0.0153` | `0.7709` | **`0.7739`** | `0.5772` | `0.5584` | `0.5972` | `0.9659` | `0.5995` | `0.5481` | `5.27s` |
| `epochs` | `20` | Yes | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `7.23s` |
| `epochs` | `30` |  | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `7.82s` |
| `epochs` | `50` |  | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `8.05s` |
| `focal_alpha` | `0.25` |  | `19` | `0.0115` | `0.7147` | **`0.6694`** | `0.3656` | `0.8095` | `0.2361` | `0.9686` | `0.6487` | `0.4204` | `6.75s` |
| `focal_alpha` | `0.5` |  | `19` | `0.0153` | `0.7749` | **`0.7648`** | `0.5556` | `0.6481` | `0.4861` | `0.9675` | `0.6314` | `0.5364` | `7.15s` |
| `focal_alpha` | `0.75` | Yes | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `7.85s` |
| `focal_alpha` | `0.85` |  | `19` | `0.0118` | `0.8009` | **`0.7729`** | `0.5877` | `0.4460` | `0.8611` | `0.9661` | `0.5990` | `0.5867` | `6.22s` |
| `focal_alpha` | `0.95` |  | `19` | `0.0071` | `0.7270` | **`0.7061`** | `0.4861` | `0.3241` | `0.9722` | `0.9653` | `0.5792` | `0.5188` | `6.39s` |
| `focal_gamma` | `0.5` |  | `19` | `0.0370` | `0.8114` | **`0.7895`** | `0.6138` | `0.4957` | `0.8056` | `0.9698` | `0.6326` | `0.6015` | `6.96s` |
| `focal_gamma` | `1.0` |  | `19` | `0.0271` | `0.8137` | **`0.7990`** | `0.6298` | `0.5229` | `0.7917` | `0.9686` | `0.6229` | `0.6147` | `6.58s` |
| `focal_gamma` | `2.0` | Yes | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `6.16s` |
| `focal_gamma` | `3.0` |  | `20` | `0.0075` | `0.7860` | **`0.7803`** | `0.5912` | `0.5402` | `0.6528` | `0.9645` | `0.6120` | `0.5639` | `6.18s` |
| `focal_gamma` | `5.0` |  | `20` | `0.0020` | `0.7793` | **`0.7612`** | `0.5548` | `0.5181` | `0.5972` | `0.9590` | `0.5739` | `0.5242` | `6.40s` |
| `hidden_size` | `128` |  | `16` | `0.0138` | `0.8214` | **`0.7936`** | `0.6154` | `0.5714` | `0.6667` | `0.9658` | `0.6201` | `0.5894` | `8.40s` |
| `hidden_size` | `256` |  | `19` | `0.0141` | `0.8284` | **`0.7839`** | `0.6000` | `0.5204` | `0.7083` | `0.9644` | `0.6272` | `0.5766` | `14.63s` |
| `hidden_size` | `32` |  | `19` | `0.0144` | `0.8117` | **`0.8007`** | `0.6296` | `0.5667` | `0.7083` | `0.9681` | `0.5918` | `0.6061` | `4.65s` |
| `hidden_size` | `64` | Yes | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `6.41s` |
| `init_prior_bias` | `False` |  | `19` | `0.0138` | `0.7891` | **`0.8010`** | `0.6349` | `0.5128` | `0.8333` | `0.9683` | `0.6344` | `0.6253` | `6.91s` |
| `init_prior_bias` | `True` | Yes | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `7.08s` |
| `loss_type` | `bce_damped` |  | `19` | `0.2360` | `0.7896` | **`0.7955`** | `0.6275` | `0.4848` | `0.8889` | `0.9703` | `0.6371` | `0.6275` | `6.60s` |
| `loss_type` | `bce_raw` |  | `19` | `0.4581` | `0.7392` | **`0.7128`** | `0.4965` | `0.3333` | `0.9722` | `0.9692` | `0.6117` | `0.5280` | `6.17s` |
| `loss_type` | `focal` | Yes | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `6.60s` |
| `lr` | `0.0001` |  | `20` | `0.0188` | `0.7905` | **`0.7386`** | `0.5103` | `0.5068` | `0.5139` | `0.9500` | `0.4929` | `0.4772` | `6.61s` |
| `lr` | `0.0003` | Yes | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `5.92s` |
| `lr` | `0.0005` |  | `19` | `0.0137` | `0.7990` | **`0.7990`** | `0.6292` | `0.5283` | `0.7778` | `0.9679` | `0.6383` | `0.6124` | `6.03s` |
| `lr` | `0.001` |  | `5` | `0.0150` | `0.8008` | **`0.7843`** | `0.6012` | `0.5149` | `0.7222` | `0.9640` | `0.5808` | `0.5791` | `2.33s` |
| `lr` | `0.002` |  | `7` | `0.0145` | `0.8001` | **`0.7829`** | `0.5963` | `0.5393` | `0.6667` | `0.9661` | `0.6714` | `0.5698` | `2.88s` |
| `max_grad_norm` | `0.5` |  | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `7.13s` |
| `max_grad_norm` | `1.0` | Yes | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `6.94s` |
| `max_grad_norm` | `2.0` |  | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `7.27s` |
| `max_grad_norm` | `5.0` |  | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `7.18s` |
| `mlp_activation` | `gelu` |  | `19` | `0.0150` | `0.7856` | **`0.8030`** | `0.6364` | `0.5385` | `0.7778` | `0.9676` | `0.5983` | `0.6192` | `7.24s` |
| `mlp_activation` | `relu` | Yes | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `7.15s` |
| `mlp_activation` | `silu` |  | `19` | `0.0151` | `0.7919` | **`0.7925`** | `0.6163` | `0.5300` | `0.7361` | `0.9678` | `0.5964` | `0.5952` | `7.06s` |
| `num_layers` | `1` | Yes | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `6.24s` |
| `num_layers` | `2` |  | `18` | `0.0144` | `0.7954` | **`0.7682`** | `0.5677` | `0.5301` | `0.6111` | `0.9671` | `0.6331` | `0.5381` | `9.26s` |
| `num_layers` | `3` |  | `18` | `0.0146` | `0.7982` | **`0.7792`** | `0.5904` | `0.5213` | `0.6806` | `0.9673` | `0.6529` | `0.5647` | `12.14s` |
| `optimizer` | `AdamW` | Yes | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `8.21s` |
| `optimizer` | `Adam` |  | `19` | `0.0144` | `0.8051` | **`0.8030`** | `0.6353` | `0.5510` | `0.7500` | `0.9665` | `0.5972` | `0.6152` | `7.87s` |
| `optimizer` | `RMSprop` |  | `20` | `0.0144` | `0.8027` | **`0.7871`** | `0.6038` | `0.5517` | `0.6667` | `0.9659` | `0.6124` | `0.5774` | `13.11s` |
| `patience` | `10` |  | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `7.18s` |
| `patience` | `3` | Yes | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `7.12s` |
| `patience` | `5` |  | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `7.38s` |
| `patience` | `8` |  | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `8.77s` |
| `scheduler_type` | `plateau_cosine` | Yes | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `7.22s` |
| `scheduler_type` | `plateau` |  | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `7.21s` |
| `use_bucket_sampler` | `False` |  | `17` | `0.0125` | `0.8186` | **`0.7919`** | `0.6135` | `0.5495` | `0.6944` | `0.9674` | `0.6541` | `0.5889` | `16.59s` |
| `use_bucket_sampler` | `True` | Yes | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `7.01s` |
| `weight_decay` | `0.0001` | Yes | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6084` | `0.6096` | `6.31s` |
| `weight_decay` | `0.001` |  | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6081` | `0.6096` | `6.85s` |
| `weight_decay` | `0.01` |  | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9663` | `0.6085` | `0.6096` | `6.65s` |
| `weight_decay` | `0.0` |  | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6090` | `0.6096` | `6.59s` |
| `weight_decay` | `1e-05` |  | `19` | `0.0142` | `0.8030` | **`0.8009`** | `0.6310` | `0.5521` | `0.7361` | `0.9664` | `0.6090` | `0.6096` | `6.24s` |

---

## 5. Experimental Parameter Sensitivity Results (Pipelines A, B, and C)

| Pipeline | Varied Experimental Parameter | Value | Default? | Feature Dim | Joined Rows | Viral Posts | Window Positive Prior | Val Macro-F1 | Test Macro-F1 | Test Positive F1 | Test ROC-AUC | Test PR-AUC | Test MCC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`A+C`** | `feature_subset` | `all_20d` | Yes | `20D` | `11,434` | `15/275` | `6.28%` | `0.8030` | **`0.8009`** | `0.6310` | `0.9664` | `0.6084` | `0.6096` |
| **`A+C`** | `feature_subset` | `motifs_only_5d` |  | `5D` | `11,434` | `15/275` | `6.28%` | `0.5400` | **`0.5721`** | `0.1778` | `0.6746` | `0.2090` | `0.1986` |
| **`A+C`** | `feature_subset` | `motifs_plus_temporal_12d` |  | `13D` | `11,434` | `15/275` | `6.28%` | `0.8030` | **`0.8051`** | `0.6404` | `0.9685` | `0.6284` | `0.6248` |
| **`A+C`** | `feature_subset` | `no_csr_16d` |  | `16D` | `11,434` | `15/275` | `6.28%` | `0.8009` | **`0.8071`** | `0.6441` | `0.9680` | `0.6126` | `0.6282` |
| **`A+C`** | `feature_subset` | `no_motifs_15d` |  | `15D` | `11,434` | `15/275` | `6.28%` | `0.7751` | **`0.7767`** | `0.5854` | `0.9639` | `0.5526` | `0.5587` |
| **`A+C`** | `feature_subset` | `no_reactivation_17d` |  | `17D` | `11,434` | `15/275` | `6.28%` | `0.8051` | **`0.7780`** | `0.5896` | `0.9652` | `0.5856` | `0.5664` |
| **`A+C`** | `feature_subset` | `no_sentiment_16d` |  | `16D` | `11,434` | `15/275` | `6.28%` | `0.7895` | **`0.8051`** | `0.6395` | `0.9667` | `0.6160` | `0.6207` |
| **`A+C`** | `feature_subset` | `structural_temporal_11d` |  | `11D` | `11,434` | `15/275` | `6.28%` | `0.7612` | **`0.7542`** | `0.5432` | `0.9628` | `0.5513` | `0.5125` |
| **`A`** | `chain_depth_threshold` | `2` |  | `20D` | `11,434` | `15/275` | `6.28%` | `0.8073` | **`0.7947`** | `0.6207` | `0.9672` | `0.6173` | `0.6010` |
| **`A`** | `chain_depth_threshold` | `3` | Yes | `20D` | `11,434` | `15/275` | `6.28%` | `0.8030` | **`0.8009`** | `0.6310` | `0.9664` | `0.6084` | `0.6096` |
| **`A`** | `chain_depth_threshold` | `4` |  | `20D` | `11,434` | `15/275` | `6.28%` | `0.8094` | **`0.8051`** | `0.6395` | `0.9664` | `0.6067` | `0.6207` |
| **`A`** | `chain_depth_threshold` | `5` |  | `20D` | `11,434` | `15/275` | `6.28%` | `0.8051` | **`0.8009`** | `0.6316` | `0.9672` | `0.6074` | `0.6116` |
| **`A`** | `reactivation_threshold_seconds` | `1800.0` |  | `20D` | `11,434` | `15/275` | `6.28%` | `0.8137` | **`0.8009`** | `0.6316` | `0.9660` | `0.6058` | `0.6116` |
| **`A`** | `reactivation_threshold_seconds` | `180.0` |  | `20D` | `11,434` | `15/275` | `6.28%` | `0.8073` | **`0.8030`** | `0.6353` | `0.9664` | `0.6079` | `0.6152` |
| **`A`** | `reactivation_threshold_seconds` | `300.0` | Yes | `20D` | `11,434` | `15/275` | `6.28%` | `0.8030` | **`0.8009`** | `0.6310` | `0.9664` | `0.6084` | `0.6096` |
| **`A`** | `reactivation_threshold_seconds` | `600.0` |  | `20D` | `11,434` | `15/275` | `6.28%` | `0.8094` | **`0.8030`** | `0.6353` | `0.9671` | `0.6146` | `0.6152` |
| **`A`** | `reactivation_threshold_seconds` | `60.0` |  | `20D` | `11,434` | `15/275` | `6.28%` | `0.8051` | **`0.7987`** | `0.6272` | `0.9665` | `0.6104` | `0.6059` |
| **`A`** | `feature_artifact_version` | `v1_dup_depth` |  | `20D` | `11,434` | `15/275` | `6.28%` | `0.8240` | **`0.7891`** | `0.6120` | `0.9683` | `0.6086` | `0.5960` |
| **`A`** | `feature_artifact_version` | `v2_t_from_root` | Yes | `20D` | `11,434` | `15/275` | `6.28%` | `0.8030` | **`0.8009`** | `0.6310` | `0.9664` | `0.6084` | `0.6096` |
| **`A`** | `window_size_seconds` | `1800.0` |  | `20D` | `8,345` | `15/275` | `5.62%` | `0.8121` | **`0.7470`** | `0.5273` | `0.9683` | `0.6302` | `0.5008` |
| **`A`** | `window_size_seconds` | `300.0` |  | `20D` | `16,353` | `15/275` | `7.58%` | `0.8007` | **`0.8084`** | `0.6503` | `0.9612` | `0.6648` | `0.6241` |
| **`A`** | `window_size_seconds` | `3600.0` |  | `20D` | `5,638` | `15/275` | `5.32%` | `0.8023` | **`0.7228`** | `0.4746` | `0.9443` | `0.5764` | `0.4457` |
| **`A`** | `window_size_seconds` | `600.0` |  | `20D` | `13,291` | `15/275` | `6.75%` | `0.7869` | **`0.7596`** | `0.5532` | `0.9630` | `0.6482` | `0.5198` |
| **`A`** | `window_size_seconds` | `900.0` | Yes | `20D` | `11,434` | `15/275` | `6.28%` | `0.8030` | **`0.8009`** | `0.6310` | `0.9664` | `0.6084` | `0.6096` |
| **`B`** | `cutoff_floor` | `15.0` | Yes | `20D` | `11,434` | `15/275` | `6.28%` | `0.8030` | **`0.8009`** | `0.6310` | `0.9664` | `0.6084` | `0.6096` |
| **`B`** | `cutoff_floor` | `185.0` |  | `20D` | `11,434` | `11/275` | `4.30%` | `0.7812` | **`0.8218`** | `0.6606` | `0.9756` | `0.7032` | `0.6474` |
| **`B`** | `cutoff_floor` | `190.0` |  | `20D` | `11,434` | `6/275` | `2.25%` | `0.6462` | **`0.6042`** | `0.2295` | `0.9434` | `0.2960` | `0.2113` |
| **`B`** | `virality_percentile` | `0.8` |  | `20D` | `11,434` | `57/275` | `28.01%` | `0.8815` | **`0.8477`** | `0.7886` | `0.9478` | `0.9025` | `0.7013` |
| **`B`** | `virality_percentile` | `0.85` |  | `20D` | `11,434` | `42/275` | `20.39%` | `0.8305` | **`0.8369`** | `0.7471` | `0.9428` | `0.8509` | `0.6795` |
| **`B`** | `virality_percentile` | `0.9` |  | `20D` | `11,434` | `28/275` | `13.27%` | `0.8521` | **`0.8440`** | `0.7347` | `0.9555` | `0.8322` | `0.6948` |
| **`B`** | `virality_percentile` | `0.925` |  | `20D` | `11,434` | `21/275` | `9.29%` | `0.8614` | **`0.8441`** | `0.7193` | `0.9632` | `0.7981` | `0.6905` |
| **`B`** | `virality_percentile` | `0.95` | Yes | `20D` | `11,434` | `15/275` | `6.28%` | `0.8030` | **`0.8009`** | `0.6310` | `0.9664` | `0.6084` | `0.6096` |
| **`B`** | `virality_percentile` | `0.975` |  | `20D` | `11,434` | `7/275` | `2.71%` | `0.7235` | **`0.6919`** | `0.4000` | `0.9436` | `0.3731` | `0.3841` |
| **`C`** | `max_window_idx` | `None (all)` | Yes | `20D` | `11,434` | `15/275` | `6.28%` | `0.8030` | **`0.8009`** | `0.6310` | `0.9664` | `0.6084` | `0.6096` |
| **`C`** | `max_window_idx` | `15` |  | `20D` | `3,263` | `15/275` | `6.82%` | `0.7822` | **`0.7364`** | `0.5106` | `0.9442` | `0.4971` | `0.4729` |
| **`C`** | `max_window_idx` | `3` |  | `20D` | `855` | `15/275` | `6.87%` | `0.4817` | **`0.4819`** | `0.0000` | `0.8937` | `0.5371` | `0.0000` |
| **`C`** | `max_window_idx` | `7` |  | `20D` | `1,706` | `15/275` | `6.89%` | `0.6515` | **`0.7442`** | `0.5333` | `0.9376` | `0.6086` | `0.5026` |
| **`C`** | `max_window_idx` | `95` |  | `20D` | `10,333` | `15/275` | `6.54%` | `0.8106` | **`0.7944`** | `0.6203` | `0.9581` | `0.6907` | `0.5963` |
| **`C`** | `scaling_mode` | `log1p_zscore` | Yes | `20D` | `11,434` | `15/275` | `6.28%` | `0.8030` | **`0.8009`** | `0.6310` | `0.9664` | `0.6084` | `0.6096` |
| **`C`** | `scaling_mode` | `none` |  | `20D` | `11,434` | `15/275` | `6.28%` | `0.5817` | **`0.5608`** | `0.1600` | `0.6812` | `0.1749` | `0.1453` |
| **`C`** | `scaling_mode` | `standard_zscore` |  | `20D` | `11,434` | `15/275` | `6.28%` | `0.8161` | **`0.7851`** | `0.6044` | `0.9468` | `0.5223` | `0.5870` |
| **`C`** | `split_mode` | `post_grouped_stratified` |  | `20D` | `11,434` | `15/275` | `6.56%` | `0.8606` | **`0.7376`** | `0.5000` | `0.9435` | `0.6488` | `0.5557` |
| **`C`** | `split_mode` | `row_stratified` | Yes | `20D` | `11,434` | `15/275` | `6.28%` | `0.8030` | **`0.8009`** | `0.6310` | `0.9664` | `0.6084` | `0.6096` |
| **`C`** | `classification_threshold` | `0.3` |  | `20D` | `11,434` | `15/275` | `6.28%` | `0.6818` | **`0.6700`** | `0.4343` | `0.9664` | `0.6084` | `0.4752` |
| **`C`** | `classification_threshold` | `0.4` |  | `20D` | `11,434` | `15/275` | `6.28%` | `0.7759` | **`0.7445`** | `0.5432` | `0.9664` | `0.6084` | `0.5576` |
| **`C`** | `classification_threshold` | `0.5` | Yes | `20D` | `11,434` | `15/275` | `6.28%` | `0.8030` | **`0.8009`** | `0.6310` | `0.9664` | `0.6084` | `0.6096` |
| **`C`** | `classification_threshold` | `0.6` |  | `20D` | `11,434` | `15/275` | `6.28%` | `0.7673` | **`0.7716`** | `0.5692` | `0.9664` | `0.6084` | `0.5472` |
| **`C`** | `classification_threshold` | `0.7` |  | `20D` | `11,434` | `15/275` | `6.28%` | `0.7064` | **`0.6738`** | `0.3750` | `0.9664` | `0.6084` | `0.4141` |
| **`C`** | `classification_threshold` | `val_optimal_f1` |  | `20D` | `11,434` | `15/275` | `6.28%` | `0.8107` | **`0.7844`** | `0.6061` | `0.9664` | `0.6084` | `0.5987` |

---

## 6. Discussion of Each Pipeline's Contribution & Key Findings

1. **Pipeline A (Dynamic CSR + WCOJ Motif & Temporal Feature Extraction):**
   - Fixing Column 8 from the duplicate `tree_depth` (`v1_dup_depth` in `pipeline_a/sample_output.csv`) to true elapsed time since thread creation (`v2_t_from_root` in `pipeline_a_features.csv`) substantially improves downstream virality discrimination (`ROC-AUC` jumps from `~0.853` on the older schema to `~0.965` on `v2_t_from_root`).
   - Feature group ablation (`EXP_AC_feature_subset_*`) quantifies the complementary value of WCOJ topological motifs (Cols 0–4), CSR structural centrality (Cols 5–8), VADER sentiment (Cols 9–12), and dormancy reactivation dynamics (Cols 13–15).
   - Temporal window size sensitivity (`300s` to `3600s`) demonstrates the trade-off between fine-grained micro-batch resolution (more rows, smaller sequence length $K$) and coarser aggregation windows (fewer rows, longer sequences per window).
2. **Pipeline B (Context-Aware Subreddit Percentile Labelling):**
   - Varying `virality_percentile` from `P80` (`57` viral posts = `20.73%`) to `P97.5` (`7` viral posts = `2.55%`) directly controls the precision-recall and PR-AUC operating regime.
   - At the default `P95` (`15` viral posts = `5.45%`, cutoff `180` comments), the dataset satisfies Pipeline B's automated `[5%, 10%]` class imbalance assertion (`pipeline_b/run.py:295`).
3. **Pipeline C (Bidirectional GRU + Learned Attention vs. Pooling Baselines):**
   - Across seeds, `CascadeAttentionGRU` (`51,137` parameters) substantially outperforms both `mean_pool` (`4,513` parameters) and `max_pool` (`4,513` parameters) on Macro-F1, Positive-Class F1, Precision, and Matthews Correlation Coefficient (MCC).
   - **Methodological Insight on Data Splitting (`row_stratified` vs. `post_grouped_stratified`):** The original implementation splits `(post_id, window_index)` rows randomly (`pipeline_c/train.py:208`), placing different windows of the same post in Train and Test. Under strict `post_grouped_stratified` splitting (zero post overlap across splits), the task is markedly harder and reflects true out-of-thread generalization.

---

## 7. Limitations & Blocked Experiments

- **Raw JSON Comment Threads (`data/*.json`):** Listed in `.gitignore` and absent from local disk. Consequently, any experiment requiring raw comment timestamps outside the cached 20-D edge matrices — specifically varying Pipeline B's 24-hour comment collection horizon (`EP_B01: observation_horizon_seconds != 86400s`) — is logged as `BLOCKED_MISSING_RAW_JSON` rather than fabricated.
- **Single-Subreddit Local Cache (`Barca`, 275 posts, 11,434 windows, 23,900 edges):** All executable experiments run deterministically on the complete cached `Barca` cascade dataset present in the repository.

---

## 8. Generated Visualizations (`results/summary/plots/`)

1. `01_ablation_performance_comparison.png` — Ablation performance across combinations & architectures
2. `02_hyperparameter_vs_macro_f1.png` — Hyperparameter sensitivity ranking (Test Macro-F1 spread)
3. `03_learning_rate_vs_performance.png` — Learning rate vs. Macro-F1, F1, ROC-AUC, PR-AUC
4. `04_batch_size_vs_performance.png` — Batch size vs. performance and training wall-time
5. `05_focal_loss_alpha_gamma_sensitivity.png` — Focal Loss $\alpha$ and $\gamma$ sensitivity curves
6. `06_temporal_window_vs_performance.png` — Pipeline A temporal window size ($300\text{s}$–$3600\text{s}$) vs. performance
7. `07_virality_threshold_vs_performance.png` — Pipeline B virality percentile ($P_{80}$–$P_{97.5}$) vs. performance
8. `08_runtime_vs_model_performance.png` — Training runtime vs. Test Macro-F1 Pareto trade-off
9. `09_memory_consumption_comparison.png` — Peak memory & total runtime across all 7 ablation combinations
10. `10_class_distribution_comparison.png` — Post-level and window-level class balance across virality thresholds

---

## 9. Reproducibility Instructions

```bash
# Dry-run hardware audit, dataset check, and execution plan:
python3 run_ablation.py --dry-run

# Run a single ablation configuration (e.g. ABC or C):
python3 run_ablation.py --experiment ABC
python3 run_ablation.py --experiment C

# Run controlled one-at-a-time hyperparameter tuning + 5-seed combined optimization:
python3 run_ablation.py --tune-hyperparameters

# Run Pipeline A, B, and C experimental parameter sensitivity analysis:
python3 run_ablation.py --sensitivity-analysis

# Run the entire experimental suite end-to-end and regenerate all tables, plots, and reports:
python3 run_ablation.py --all

# Reproduce any specific experiment from its saved JSON configuration file:
python3 run_ablation.py --reproduce results/ablation_study/ABC/ABL_ABC.json
```
