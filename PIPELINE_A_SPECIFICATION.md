# PIPELINE A SPECIFICATION OVERVIEW

- **Input:** Raw stream / micro-batches of comments from JSON threads (t in [0, 15m], [15m, 30m], etc.).
- **Output:** High-throughput `pipeline_a_features.csv` / Parquet containing `post_id`, `window_idx`, K x 20 edge matrix, and cumulative context features.
- **Core Engine:** High-performance Graph Analytics Engine relying on dynamic CSR representations and WCOJ query solvers.

---

### TASK 1: Ingestion, Stream Filtering & Dynamic CSR Graph Builder

**Goal:** Ingest nested JSON, filter observation windows, and maintain an ultra-fast dynamic Compressed Sparse Row (CSR) adjacency structure for graph traversal.

#### **SUBTASK 1.1: Root Thread Extractor & Stream Parser**
* Parse top-level post metadata (`post_id`, `post_time`) and flatten comments/replies into chronologically sorted tuples e_i = (u, v, t_i).

#### **SUBTASK 1.2: Relative Window Filter & Cumulative State Tracker**
* Compute Δt = t_comment - t_post and extract edges belonging strictly to window [T_start, T_end].
* Maintain thread-level state tracking `cumulative_prior_comments` (C_prior) and `window_index`.

#### **SUBTASK 1.3: Dynamic CSR Representation Generator (`CSRGraph`)**
* Implement an in-memory dynamic CSR wrapper (`indptr`, `indices`, `timestamps`).
* As new edges arrive in T_w, incrementally update CSR array buffers without full graph re-construction:
  * **Out-degree mapping:** O(1) lookups via `indptr[u] : indptr[u+1]`.
  * **Neighborhood retrieval:** Direct slicing into contiguous memory buffers.

---

### TASK 2: Worst-Case Optimal Join (WCOJ) Motif Execution Engine

**Goal:** Compute higher-order graph motifs (e.g., `Ping_Pong_Dyads`, `Sibling_Star_Forks`, `Reciprocal_Cycles`, 4-node motif completions) using Leapfrog Triejoin on the CSR graph structure.

#### **SUBTASK 2.1: Leapfrog Triejoin Iterator over CSR Indices**
* Implement Triejoin iterators over CSR neighborhood lists sorted by node ID.
* Provide O(1) leapfrog primitives: `at()`, `key()`, `next()`, `seek(key)`.

#### **SUBTASK 2.2: WCOJ Multi-Way Join Solvers**
* **Dyad & Cycle Query (u -> v and v -> u):** Perform 2-way join over out-edges in CSR to detect reciprocal edges.
* **Fork/Star Query (r -> u and r -> v):** Join neighborhood lists of root/parent nodes to identify star-fork formations.
* **Higher-Order Triangle/Cycle Query (u -> v and v -> w and w -> u):** Execute 3-way Leapfrog Triejoin bound by fractional edge cover count O(|E|^(3/2)).

#### **SUBTASK 2.3: Motif Type Classification & One-Hot Encoding**
Map the WCOJ join results for edge e_i to a 5-dimensional one-hot vector:
1. `Single_Direct_Reply`
2. `Sibling_Star_Fork`
3. `Deep_Directed_Chain`
4. `Ping_Pong_Dyad`
5. `Reciprocal_Cycle`

---

### TASK 3: CSR-Accelerated Temporal Dynamics & Reactivation

**Goal:** Leverage fast CSR array lookups to calculate temporal velocities and subgraph reactivation flags.

#### **SUBTASK 3.1: Inter-Arrival Gap Calculation (Δt_edge)**
* Compute Δt_edge = t_i - t_(i-1) using the streaming edge timestamp buffer.

#### **SUBTASK 3.2: WCOJ-Derived Motif Completion Speed (Δt_M)**
* Retrieve the earliest edge timestamp t_trigger among the edges participating in the WCOJ-detected motif.
* Compute speed: Δt_M = t_i - t_trigger.

#### **SUBTASK 3.3: Dormant Subgraph Reactivation Engine**
* Using CSR timestamp arrays, perform O(deg(u)) scan on the target node u's history.
* Compute T_idle = t_i - max(t_prev_edges(u)).
* Set binary reactivation flag I_reactivate = 1.0 if T_idle > T_threshold (e.g., > 5 mins), else 0.0.

---

### TASK 4: Content Analysis & 20-D Matrix Assembly

**Goal:** Process comment sentiment and aggregate all CSR/WCOJ topological metrics into the standard 20-D float32 row array per edge.

#### **SUBTASK 4.1: VADER Sentiment Scoring**
* Extract text for edge e_i and compute normalized sentiment weight w_s in [-1.0, 1.0].

#### **SUBTASK 4.2: 20-D Vector Layout Assembly**
For each edge e_i in window [T_start, T_end], map structural and temporal variables into an array of shape [20]:
* **Cols 0–4:** One-hot Motif Type (from WCOJ Engine)
* **Cols 5–8:** CSR Centrality Metrics (`tree_depth`, `parent_out_degree`, `csr_node_count`, `root_distance`)
* **Cols 9–12:** Sentiment Features (w_s, `compound`, `pos`, `neg`)
* **Cols 13–15:** Structural Reactivation Flags (I_reactivate, T_idle, N_reactivated_descendants)
* **Col 16:** WCOJ Motif Completion Speed Δt_M
* **Cols 17–18:** Context Scalars (`window_index`, `cumulative_prior_comments`)
* **Col 19:** Inter-Arrival Gap Δt_edge

#### **SUBTASK 4.3: Sequence Matrix Stacking**
* Stack all K edge vectors into a 2D matrix X_seq in R^(K x 20).

---

### TASK 5: High-Throughput Serialization & Schema Validation

**Goal:** Output matrices safely for downstream model ingestion.

#### **SUBTASK 5.1: Vector Serialization**
* Serialize X_seq to a JSON-formatted matrix string or Parquet byte array.

#### **SUBTASK 5.2: Atomic File Output**
* Write record to `pipeline_a_features.csv` / Parquet with header:
  `[post_id, window_index, cumulative_prior_comments, edge_count_K, X_seq_json]`

#### **SUBTASK 5.3: Automated Unit & Performance Tests**
Include PyTest suites verifying:
* Dynamic CSR `indptr` consistency and zero out-of-bounds indexing.
* WCOJ Leapfrog Triejoin correctness on known ground-truth graph cliques/cycles.
* Feature array shape invariant ([K, 20] float32 tensor structure).
