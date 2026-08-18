# SYSTEM ARCHITECTURE SPECIFICATION

## 1. Executive Summary

The system predicts whether a social media thread (e.g., Reddit post) will become "viral" within 24 hours (T = 24h) by observing only its initial early-stage comment stream (T in [0, 15m]).

The system is split into three decoupled, parallelized processing pipelines:

- **Pipeline A (Real-Time Feature Engine):** Converts early streaming comment events (T <= 15m) into continuous N x 20 structural-temporal feature matrices using dynamic Graph CSR representations and Worst-Case Optimal Joins (WCOJ).
- **Pipeline B (Asynchronous Ground-Truth Engine):** Analyzes the full 24-hour thread state to assign dynamic, subreddit-aware P95 virality labels (y_viral in {0.0, 1.0}).
- **Pipeline C (Hybrid Neural Learning Engine):** Merges Pipeline A and B outputs on a shared key (`post_id`), pads variable sequence matrices, and trains a Bidirectional GRU + Learnable Attentive Pooling + MLP model.

---

## 2. Detailed Component Architecture

### Pipeline A: Streaming Feature & Graph Extraction Engine
- **Observation Horizon:** T in [0, 15m] (relative to post creation time).
- **Input:** Raw comment event JSON.
- **Key Components:**
  - **Dynamic CSR Sparse Graph Representation (`csr_graph.py`):**
    Uses in-memory dynamic contiguous NumPy buffers (`indptr`, `indices`, `timestamps`) representing directed edges (u -> v).
    Allows zero-copy O(1) neighborhood indexing and fast out-degree lookups without high-overhead ORM/NetworkX structures.
  - **Worst-Case Optimal Join (WCOJ) Engine (`wcoj_engine.py`):**
    Implements Leapfrog Triejoin iterators over sorted CSR neighborhood arrays.
    Evaluates multi-way edge joins in worst-case optimal time O(|E|^rho*) to detect higher-order graph topologies (Single Replies, Sibling Star Forks, Directed Chains, Ping-Pong Dyads, Reciprocal Cycles).
  - **CSR Temporal & Reactivation Processor (`temporal.py`):**
    Scans CSR historical timestamps to calculate inter-arrival gaps (Δt_edge), motif completion velocity (Δt_M), and dormant node reactivation flags (I_reactivate where T_idle > 5m).
  - **20-Dimensional Matrix Assembly (`assembly.py`):**
    Maps each edge e_i to a 20-element vector:
    - **Cols 0–4:** One-hot motif layout (WCOJ).
    - **Cols 5–8:** CSR Structural Centrality (Tree Depth, Parent Out-Degree, CSR Node Count, Root Distance).
    - **Cols 9–12:** Sentiment Scores (Compound, Pos, Neg, Neu via VADER).
    - **Cols 13–15:** Structural Reactivation Metrics (I_reactivate, T_idle, Reactivated Descendants).
    - **Col 16:** Motif Velocity Δt_M.
    - **Cols 17–18:** Contextual State (`window_index`, `cumulative_prior_comments`).
    - **Col 19:** Inter-Arrival Gap Δt_edge.
- **Data Interchange Format:** `pipeline_a_features.csv`
```csv
post_id,X_seq
"d160579c_1625968453","[[0.0, ..., 0.8], [1.0, ..., -0.2]]"
```

---

### Pipeline B: Asynchronous Virality Target Engine
- **Observation Horizon:** T in [0, 24h] (full post lifecycle).
- **Input:** Raw dataset snapshot.
- **Key Components:**
  - **Tree DFS Traverser & Horizon Filter:**
    Flattens nested comment arrays while discarding late comments (Δt > 86,400s) to avoid target leakage.
  - **Subreddit Percentile Engine:**
    Groups posts by subreddit and computes the dynamic 95th percentile (P95) volume cutoff (Cutoff_s) to adjust for varying community sizes.
    Enforces a global minimum floor (Cutoff >= 15) for small subreddits.
  - **Virality Labeler:**
    Assigns the target binary label:
    `y_viral = 1.0 if C_24h >= Cutoff_s else 0.0`
- **Data Interchange Format:** `pipeline_b_labels.csv`
```csv
post_id,total_comments_24h,p95_cutoff,y_viral
"d160579c_1625968453",104,100.0,1.0
```

---

### Pipeline C: Deep Learning & Sequence Aggregation Model
- **Input:** `pipeline_a_features.csv` and `pipeline_b_labels.csv`.
- **Key Components:**
  - **Inner Join & DataLoader Engine:**
    Joins datasets on `post_id` in memory using `pd.merge()`.
    Deserializes JSON string matrices into PyTorch tensors.
    Batches B variable-length K_i x 20 matrices into a uniform [B, K_max, 20] 3D tensor and generates a boolean padding mask [B, K_max].
  - **Neural Network Pipeline Architecture:**

```
[ Input Batch Tensor ] ──────────────► Shape: [B, K_max, 20]
                                            │
                                            ▼
[ Bidirectional GRU Encoder ] ───────► Shape: [B, K_max, 2 * H_hidden]
                                            │
                                            ▼
[ Learnable Attentive Pooling ] ─────► Shape: [B, 2 * H_hidden]
  • Computes soft edge weights α_i            (Compressed 20-D Context Vector)
  • Masked Softmax over padding             │
                                            ▼
[ Bottleneck MLP Classifier ] ───────► Shape: [B, 1]
  • Dense(2*H -> 64) + ReLU + Dropout         (Unnormalized Logit)
  • Dense(64 -> 32) + ReLU                  │
  • Dense(32 -> 1)                          ▼
                               [ Loss: BCEWithLogitsLoss ]
```

---

## 3. Primary System Design Highlights

- **Parallel Engineering:** Pipelines A, B, and C communicate strictly through flat CSV contracts, allowing 3 developers to build, test, and validate their modules independently using mock data.
- **Sub-Quadratic Complexity:**
  - Pipeline A replaces O(N^2) NetworkX operations with O(1) CSR lookups and O(|E|^rho*) WCOJ joins.
  - Pipeline C replaces O(K^2) Self-Attention with O(K) Learnable Attentive Pooling, maintaining linear scalability over variable sequence lengths.
- **Context Leakage Prevention:** Pipeline A observes t <= 15m, while Pipeline B filters out t > 24h, guaranteeing zero temporal overlap between features and prediction targets.
