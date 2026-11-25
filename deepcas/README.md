# Pipeline DeepCas — Continuous-Time Cascade Prediction on Twitter

This pipeline implements **DeepCas** (*"DeepCas: An End-to-end Predictor of Information Cascades"*, WWW 2017 by Cheng Li et al.) in PyTorch, evaluated on the Twitter cascade dataset (`cascades/twitter.parquet`).

---

## 1. Experimental Protocol & Non-Negotiable Rules

The experimental setup adheres strictly to the protocol defined in the benchmark reference:

| Parameter | 1-Day Window ($T_{obs} = 86,400$ s) | 2-Day Window ($T_{obs} = 172,800$ s) | Details / Rationale |
| :--- | :--- | :--- | :--- |
| **Observation Window ($T_{obs}$)** | 1 day (86,400 seconds) | 2 days (172,800 seconds) | Only events arriving at $t \le T_{obs}$ are visible. Captures diurnal diffusion. |
| **Prediction Horizon ($T_{pred}$)** | 15 days (1,296,000 seconds) | 15 days (1,296,000 seconds) | Standard medium-term popularity horizon in Twitter literature (CasFT, VaCas). |
| **Prediction Target ($\Delta P$)** | $\Delta P = R_{15d} - R_{1d}$ | $\Delta P = R_{15d} - R_{2d}$ | **Future increment only** — avoids inflating scores with already observed retweets. |
| **Hour Filter** | Disabled (`no_hour_filter`) | Disabled (`no_hour_filter`) | Twitter activity is globally distributed across time zones. |
| **Size Filter** | $10 \le N(T_{obs}) \le 1,000$ | $10 \le N(T_{obs}) \le 1,000$ | Standard benchmark cascade size filter. |
| **Split Strategy** | Strictly Chronological (by $t_0$) | Strictly Chronological (by $t_0$) | Prevents future social trend leakage into training. |
| **Split Ratio** | 70% Train / 15% Val / 15% Test | 70% Train / 15% Val / 15% Test | Canonical benchmark partition. |
| **Sample Counts** | Train: 13,765 / Val: 2,949 / Test: 2,951 (Total: 19,665) | Train: 17,217 / Val: 3,689 / Test: 3,690 (Total: 24,596) | Partitioned cleanly without record overlap. |

---

## 2. Optimization Hyperparameters

- **Batch Size:** 64
- **Optimizer:** AdamW (weight decay = $10^{-4}$)
- **Initial Learning Rate:** $10^{-3}$
- **Learning Rate Decay:** Factor 0.5 on plateau (patience = 3 epochs)
- **Gradient Clipping:** Max norm = 1.0
- **Early Stopping:** Patience = 8 epochs (max 60 epochs)
- **Model Checkpointing:** Restores weights from the lowest validation MSLE epoch before test evaluation.
- **Node Vocabulary:** Fitted strictly on the training partition only (142,734 nodes for 1d; 170,613 nodes for 2d).

---

## 3. DeepCas Architecture

- **Random Walks:** $N_w = 32$ random walks of length $L = 10$ sampled per cascade graph.
- **Node Embedding:** Dimension $d = 50$, dropout = 0.2.
- **Sequence Encoder:** Bidirectional GRU with hidden size $h = 32$ (output dimension = 64).
- **Hierarchical Attention:**
  - **Step-level Attention:** Softmax attention over the $L=10$ walk steps.
  - **Cascade-level Attention:** Softmax attention over the $N_w=32$ random walks.
- **Cascade Size Integration:** Concatenation of log-observed size $\log_2(N(T_{obs}) + 1)$.
- **Dense MLP Prediction Head:** Linear(65, 64) $\to$ ReLU $\to$ Dropout(0.2) $\to$ Linear(64, 32) $\to$ ReLU $\to$ Linear(32, 1).

---

## 4. Benchmark Results

### 1-Day Observation Window ($T_{obs} = 1$ day, $T_{pred} = 15$ days)
- **Test Samples:** 2,951

| Metric | DeepCas | Naive Mean Baseline | Improvement |
| :--- | :---: | :---: | :---: |
| **$\text{MSLE}_{\log_2}$** | **6.5209** | 9.5146 | **-31.5%** |
| **$\text{RMSLE}_{\log_2}$** | **2.5536** | 3.0846 | **-17.2%** |
| **$\text{MSLE}_{\ln}$** | **3.1330** | 4.5713 | **-31.5%** |
| **$\text{PCC}_{\log_2}$** | **0.1021** | 0.0000 | **+0.102** |
| **$\text{MAPE}$ (%)** | 874.70% | 780.31% | — |
| **$\text{MAE}_{\text{count}}$** | 15.11 | 12.09 | — |
| **$\text{MedAE}_{\text{count}}$** | **2.72** | 10.61 | **-74.4%** |

### 2-Day Observation Window ($T_{obs} = 2$ days, $T_{pred} = 15$ days)
- **Test Samples:** 3,690

| Metric | DeepCas | Naive Mean Baseline | Improvement |
| :--- | :---: | :---: | :---: |
| **$\text{MSLE}_{\log_2}$** | **7.9386** | 12.1907 | **-34.9%** |
| **$\text{RMSLE}_{\log_2}$** | **2.8176** | 3.4915 | **-19.3%** |
| **$\text{MSLE}_{\ln}$** | **3.8141** | 5.8570 | **-34.9%** |
| **$\text{PCC}_{\log_2}$** | **0.1643** | 0.0000 | **+0.164** |
| **$\text{MAPE}$ (%)** | **777.94%** | 1076.43% | **-27.7%** |
| **$\text{MAE}_{\text{count}}$** | **12.04** | 14.56 | **-17.3%** |
| **$\text{MedAE}_{\text{count}}$** | **4.62** | 13.40 | **-65.5%** |

---

## 5. Reproduction Commands

```bash
# 1. Parse Twitter cascades and generate random walks (1-day and 2-day)
python -m deepcas.deepcas_parser --input cascades/twitter.parquet --output-dir data_twitter/processed_1d --observation-time 86400 --horizon 1296000
python -m deepcas.deepcas_parser --input cascades/twitter.parquet --output-dir data_twitter/processed_2d --observation-time 172800 --horizon 1296000

# 2. Train and evaluate DeepCas
python -m deepcas.run_experiments --window both --epochs 60 --early-stop 8 --patience 3 --batch-size 64 --lr 1e-3 --weight-decay 1e-4 --seed 42
```
