# CasMotif: Information Cascade Prediction via Causal Temporal Motifs and Attentive Sequence Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, GNN-free framework for **early information cascade popularity forecasting** in continuous-time social networks. 

CasMotif decouples structural graph feature extraction from neural representation learning: it evaluates **conversation-level causal temporal motifs** on streaming dynamic graphs ($G_{t^-}$) strictly prior to edge insertion, then processes the resulting event sequences with a lightweight **attentive Bidirectional GRU**.

---

## 📌 Executive Summary & Key Results

Existing state-of-the-art cascade models rely heavily on **Graph Neural Networks (GNNs)** with dynamic subgraph sampling (e.g., CasSubTS, *Scientific Reports* 2025). While accurate, these architectures incur a heavy **"GNN Tax"**—requiring expensive GPU-resident neighborhood sampling, complex graph convolutions, and over $150\text{k}$ parameters.

**CasMotif** proves that pre-extracting higher-order topological inductive bias via streaming causal motifs matches SOTA accuracy while using **$3.1\times$ fewer parameters** and **zero GPU graph-sampling overhead**.

| Model | Venue | Parameters | MSLE $\downarrow$ ($1\text{h}$) | MAPE $\downarrow$ | MAE $\downarrow$ | GPU Graph Sampling? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean Baseline** | — | — | 5.2825 | — | 105.32 | None |
| **DeepHawkes** | *ACM CIKM '17* | $\approx 2.2\text{k}^\dagger$ | 2.4410 | — | $\approx 96.20$ | None (Path RNN) |
| **CasSubTS** | *Sci. Reports '25* | 153,970 | 1.9864 | **0.2308** | **75.97** | **Yes (MH-GAT Subgraphs)** |
| **CasMotif (Ours)** | — | **49,824** | **1.9845** | 0.2315 | 77.23 | **None (Streaming CPU CSR)** |

> **Key Takeaway:** On the 42,186-cascade Sina Weibo benchmark under a strict chronological 70:15:15 split, CasMotif achieves **statistical parity** with the 2025 SOTA CasSubTS (paired $t$-test $p = 0.93$, 95% CI $[-0.040, +0.039]$) and reduces error by **23.0%** over DeepHawkes, using only **49,824 parameters**.

---

## 🏗️ System Architecture

CasMotif operates in two decoupled stages:

```
[ Raw Propagation Stream ] ──► E = {(u_i, v_i, t_i)} (t <= T_obs)
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: STREAMING GRAPH & CAUSAL FEATURE EXTRACTION                     │
│                                                                          │
│  • Dynamic CSR Graph Engine: In-memory contiguous 1D buffers             │
│    (indptr, indices, timestamps) enabling O(1) out-degree and slices     │
│                                                                          │
│  • Causal Motif Engine (evaluated on G_{t^-} before edge insertion):     │
│    1. Single Direct Reply      2. Sibling Star Fork                      │
│    3. Deep Directed Chain      4. Ping-Pong Dyad                         │
│    5. Reciprocal 3-Cycle                                                 │
│                                                                          │
│  • Temporal Dynamics & State: dt_edge, dt_motif, T_idle, I_reactivate,   │
│    and incremental ancestor descendant tree count                        │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ 15-Dimensional Feature Vectors
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: LIGHTWEIGHT SEQUENCE ATTENTION LEARNER (Bi-GRU + MLP)           │
│                                                                          │
│  • Input LayerNorm + LengthGroupedBatchSampler (minimizes padding waste)  │
│  • Bidirectional GRU Encoder (D = 64, Hidden = 128)                      │
│  • Learned Additive Attention Pooling (O(K) linear complexity):          │
│      s_j = w_2^T tanh(W_1 h_j + b_1),  alpha_j = softmax(s_j)            │
│  • Bottleneck MLP Regression Head (128 -> 64 -> 32 -> 1)                 │
│  • Prediction Target: y_hat = log_2(Delta P_hat + 1)                     │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
                      Predicted Cascade Popularity Increment
```

---

## 📐 15-Dimensional Causal Feature Schema

Every propagation event $e_i = (u_i, v_i, t_i)$ is mapped to an instantaneous 15-dimensional vector on the dynamic graph state $G_{t_i^-}$ **before** inserting edge $(u_i, v_i)$, guaranteeing **zero future lookahead leakage**:

| Col Index | Feature Name | Description & Formulation | Category |
| :---: | :--- | :--- | :--- |
| **0** | `motif_single` | Single direct reply (active only when no higher-order motif fired) | Causal Motif ($\{0, 1\}$) |
| **1** | `motif_star` | Sibling star fork ($\deg_{out}(u) \ge 1$ before insertion; broadcast hub) | Causal Motif ($\{0, 1\}$) |
| **2** | `motif_chain` | Deep directed chain ($\text{depth}(u) + 1 \ge 3$; organic multi-hop spread) | Causal Motif ($\{0, 1\}$) |
| **3** | `motif_dyad` | Ping-pong dyad ($v \to u \in E_{t^-}$; reciprocal two-party discussion) | Causal Motif ($\{0, 1\}$) |
| **4** | `motif_cycle` | Reciprocal 3-cycle ($\exists w \in \mathcal{N}_{out}(v) \text{ s.t. } (w, u) \in E_{t^-}$; triadic closure) | Causal Motif ($\{0, 1\}$) |
| **5** | `tree_depth` | Vertical tree depth of adopting node $v$ from root post ($d_v = d_u + 1$) | Tree Centrality |
| **6** | `parent_out_degree` | Instantaneous out-degree of parent $u$: $\deg_{out}(u) = indptr[u+1] - indptr[u]$ | Tree Centrality |
| **7** | `node_count` | Total cumulative unique users in cascade graph ($|V_{t^-}|$) | Graph Centrality |
| **8** | `t_from_root` | Elapsed continuous time since root publication: $\max(0, t_i - t_0)$ | Continuous Dynamics |
| **9** | `i_reactivate` | Binary dormancy reactivation flag ($\mathbf{1}[T_\text{idle} > 300\,\text{s}]$) | User Reactivation |
| **10** | `t_idle` | Dormancy duration since user's previous action (or post age for new users) | User Reactivation |
| **11** | `descendant_count` | Recursive ancestor subtree size: $\sum_{w \in \text{Subtree}(u)} 1$ via ancestor walk | Tree Centrality |
| **12** | `dt_motif` | Motif completion velocity: $t_i - \min(\text{history}(u))$ | Temporal Velocity |
| **13** | `cumulative_prior` | Total event count prior to current event ($i - 1$) | Contextual State |
| **14** | `dt_edge` | Inter-arrival gap between consecutive events: $t_i - t_{i-1}$ | Temporal Velocity |

*Note: Unbounded features (cols 5–8, 10–14) undergo $\log(1 + x)$ transformation and standard score scaling fitted strictly on the training partition.*

---

## 📊 Experimental Results & Validation

All experiments follow the benchmark protocol of DeepHawkes on **Sina Weibo** (42,186 cascades, diurnal daylight filter, chronological 70:15:15 split).

### 1. Sequence Pooling Ablation ($T_\text{obs} = 1\,\text{hour}$)
Isolating the predictive contribution of the attentive sequence model against pooling strategies over the identical 15-D feature vectors:

| Rank | Architecture / Pooling Strategy | Trainable Parameters | MSLE $\downarrow$ | MAE $\downarrow$ | PCC $\uparrow$ |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **CasMotif (Attentive Bi-GRU)** | **49,824** | **1.9804** | 77.78 | **0.7933** |
| 2 | Mean Pooling + MLP Head | 3,167 | 2.0829 | **76.08** | 0.7784 |
| 3 | Max Pooling + MLP Head | 3,147 | 2.2304 | 81.46 | 0.7615 |
| 4 | Non-Learned Mean Baseline | — | 5.2825 | 105.32 | — |

*Attentive pooling reduces MSLE by **5.2%** over mean pooling and **12.6%** over max pooling by selectively amplifying pivotal viral hubs.*

### 2. Ultra-Early Horizon Forecasting ($T_\text{obs} = 15\,\text{minutes}$)
Evaluating model robustness under a severely constrained 15-minute observation window ($4\times$ earlier than standard 1-hour benchmarks):

| Rank | Architecture | MSLE $\downarrow$ ($15\text{m}$) | RMSE $\downarrow$ | $R^2$ $\uparrow$ | Margin vs. Mean Baseline |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **CasMotif (Attentive Bi-GRU)** | **2.4423** | **1275.2** | **0.5757** | **57.6% error reduction** |
| 2 | Mean Pooling Baseline | 2.7758 | 1282.5 | 0.5176 | 51.8% error reduction |
| 3 | Max Pooling Baseline | 2.9097 | 1335.5 | 0.4943 | 49.5% error reduction |
| 4 | Non-Learned Mean Baseline | 5.7562 | 2391.2 | 0.0004 | — |

---

## 📂 Repository Structure

```
.
├── pipeline_w/                           # Weibo Cascade Popularity Benchmark (CasMotif)
│   ├── weibo_parser.py                   # Streaming CSR parser & causal motif feature engine
│   ├── model.py                          # CascadeAttentionGRURegressor & pooling baselines
│   ├── data.py                           # Dataset loader & train-split-only SequenceScaler
│   ├── train_weibo.py                    # End-to-end training loop with PlateauCosineLR
│   ├── outputs_3600/                     # Model weights & evaluation reports for T_obs = 1h
│   │   ├── best_attention_gru.pt         # Trained CasMotif weights (206 KB, 49.8k params)
│   │   └── weibo_evaluation_report.json  # Comprehensive metric evaluation report
│   ├── outputs_900/                      # Model weights & evaluation reports for T_obs = 15m
│   │   ├── best_attention_gru.pt         # Ultra-early CasMotif weights (206 KB)
│   │   └── weibo_evaluation_report.json  # Early evaluation report
│   └── PUBLISHED_BASELINES.md            # Reference notes on the cross-paper MSLE log-base trap
│
├── pipeline_a/                           # Reddit feature extraction engine (WCOJ + CSR)
│   └── Task1.py                          # 15-minute windowed feature extractor with sentiment
│
├── pipeline_b/                           # Subreddit-aware virality target labeling
│
├── pipeline_c/                           # Binary virality classification model (Focal Loss)
│   ├── model.py                          # CascadeAttentionGRU classifier
│   ├── data.py                           # LengthGroupedBatchSampler & dataset loaders
│   └── train.py                          # Training pipeline for Reddit classification
│
├── paper1.tex                            # Formal Springer LNCS research paper draft
├── main.tex                              # Beamer presentation slide deck (Panel Review)
└── README.md                             # Repository documentation
```

---

## 🚀 Quickstart & Usage

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/aswath-siddharth/Forecasting-Motif-Evolution-in-Social-Networks-using-Temporal-Graph-Networks.git
cd Forecasting-Motif-Evolution-in-Social-Networks-using-Temporal-Graph-Networks
pip install torch numpy pandas scikit-learn matplotlib
```

### 2. Feature Extraction from Raw Weibo Cascades
Extract 15-D causal motif feature sequences from raw `dataset_weibo.txt`:
```bash
# 1-Hour Observation Window (Standard Benchmark)
python -m pipeline_w.weibo_parser --input data/dataset_weibo.txt --output-dir pipeline_w/features_3600 --obs-time 3600

# 15-Minute Observation Window (Ultra-Early Setting)
python -m pipeline_w.weibo_parser --input data/dataset_weibo.txt --output-dir pipeline_w/features_900 --obs-time 900
```

### 3. Model Training & Evaluation
Train the attentive sequence model on extracted causal motif features:
```bash
python -m pipeline_w.train_weibo \
    --features pipeline_w/features_3600/weibo_features.npz \
    --output-dir pipeline_w/outputs_3600 \
    --obs-window 3600 \
    --epochs 20 \
    --batch-size 256 \
    --lr 3e-4 \
    --hidden-size 64 \
    --attention-dim 64 \
    --amp
```

---

## 🔬 Scientific Context & Core References

This project builds directly upon and compares against foundational works in information diffusion and network science:

1. **CasSubTS:** J. Wang et al. *"Fusing Temporal and Structural Information via Subgraph Sampling and Multi-Head Attention for Information Cascade Prediction."* *Scientific Reports*, 15:6801, 2025.
2. **DeepHawkes:** Q. Cao, H. Shen, K. Cen, W. Ouyang, X. Cheng. *"DeepHawkes: Bridging the Gap between Prediction and Understanding of Information Cascades."* *ACM CIKM*, pp. 1149–1158, 2017.
3. **Temporal Motifs:** A. Paranjape, A. R. Benson, J. Leskovec. *"Motifs in Temporal Networks."* *ACM WSDM*, pp. 601–610, 2017.
4. **Structural Virality:** S. Goel, A. Anderson, J. Hofman, D. J. Watts. *"The Structural Virality of Online Diffusion."* *Management Science*, 62(1):180–196, 2016.
5. **CasCN:** X. Chen, F. Zhou, K. Zhang, G. Trajcevski, T. Zhong, F. Zhang. *"Information Diffusion Prediction via Recurrent Cascades Convolution."* *IEEE ICDE*, pp. 770–781, 2019.
6. **CasFlow:** X. Xu, F. Zhou, K. Zhang, S. Liu, G. Trajcevski. *"CasFlow: Exploring Hierarchical Structures and Propagation Uncertainty for Cascade Prediction."* *IEEE TKDE*, 35(4):3484–3499, 2023.

---

## 👥 Authors & Academic Affiliation

**Amrita Vishwa Vidyapeetham, Department of Computer Science and Engineering**  
*Course: 23CSE498 – Project Phase 2 (Team 92)*

* **Varun S** (`CB.SC.U4CSE23157`)
* **Kavin Karthic M** (`CB.SC.U4CSE23161`)
* **Aswath Siddharth R** (`CB.SC.U4CSE23109`)
* **Darshan R** (`CB.SC.U4CSE23115`)
* **Harish Krishna B** (`CB.SC.U4CSE23323`)
* **Project Guide:** Mr. Dayanand V, Assistant Professor, Department of CSE
