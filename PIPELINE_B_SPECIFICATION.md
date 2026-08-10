# PIPELINE B SPECIFICATION OVERVIEW

- **Input:** Raw dataset snapshot containing full comment threads at T = 24 hours (e.g., Barca.json).
- **Output:** Ground-truth label file (`pipeline_b_labels.csv`) mapping each `post_id` to its virality status (`y_viral` in {0.0, 1.0}) and continuous regression metrics.
- **Execution:** Asynchronous offline processing engine.

---

### TASK 1: Dataset Snapshot Ingestion & Tree Flattening

#### **SUBTASK 1.1: Root Post & Thread Identifier Extraction**
* **Objective:** Extract post-level metadata and construct the immutable, globally shared unique primary key (`post_id`).
* **Procedure:**
  1. Parse root post metadata fields: `post_user` (or `author`), `post_time` (Unix timestamp in seconds), and `subreddit`.
  2. Construct `post_id` using the strict cross-pipeline string contract:
     `post_id = post_user + "_" + str(int(post_time))`
  3. Validate `post_id` string formatting to ensure 100% string-matching compatibility during downstream dataset inner-join.

#### **SUBTASK 1.2: Recursive Comment Tree Traversal**
* **Objective:** Flatten nested, hierarchical comment structures (comments and nested replies) into an unnested linear event list.
* **Procedure:**
  1. Implement a recursive Depth-First Search (DFS) or Breadth-First Search (BFS) traverser starting at the root post node.
  2. For every comment/reply node encountered, extract: `comment_id`, `parent_id`, `author_id`, `comment_time` (Unix timestamp), and `body_text`.
  3. Append extracted nodes into a flat interaction array E = {e_1, e_2, ..., e_M}.

#### **SUBTASK 1.3: 24-Hour Cutoff & Leakage Filter**
* **Objective:** Enforce a strict 24-hour temporal boundary (86,400 seconds) to eliminate target leakage from comments created after the target evaluation horizon.
* **Procedure:**
  1. Calculate the relative arrival time for each comment e_i: Δt_i = t_comment - t_post.
  2. Retain only comments satisfying 0 <= Δt_i <= 86,400 seconds.
  3. Discard comments where Δt_i < 0 (clock-skew anomalies) or Δt_i > 86,400 (late arrivals).
  4. Yield the clean, bounded 24-hour comment set E_24h.

---

### TASK 2: 24-Hour Cascade Metrics Computation

#### **SUBTASK 2.1: Total Volume Aggregation (C_24h)**
* **Objective:** Compute the absolute count of valid comments received strictly within 24 hours.
* **Mathematical Definition:**
  `C_24h = sum_{i=1..M} I(0 <= (t_i - t_post) <= 86400)`
* **Procedure:**
  1. Filter out deleted, removed, or automoderator comments (e.g., authors matching "[deleted]", "[removed]", or "AutoModerator").
  2. Deduplicate records by `comment_id`.
  3. Count total valid entries remaining in E_24h to establish C_24h.

#### **SUBTASK 2.2: Unique Participant Expansion (U_24h)**
* **Objective:** Quantify distinct human participants and evaluate organic vs. artificially padded discussion.
* **Mathematical Definition:**
  `U_24h = | Unique(A \ {Special_Accounts}) |`
  where A is the multiset of author IDs in E_24h.
* **Procedure:**
  1. Extract all author IDs from E_24h, excluding system bots and deleted placeholders.
  2. Compute set cardinality to get U_24h.
  3. Calculate the Organic Participant Ratio:
     `R_organic = U_24h / C_24h`
  4. Calculate the Max User Concentration:
     `M_user_max = max_{u in A} Count(u) / C_24h`

#### **SUBTASK 2.3: Peak Arrival Velocity & Acceleration (Δt_peak)**
* **Objective:** Measure maximum throughput and peak velocity timing across 1-hour temporal bins.
* **Mathematical Definition:**
  Divide the 24-hour horizon into 24 non-overlapping 1-hour bins B_1, ..., B_24. Let V(k) be the comment count in bin B_k.
  `V_max = max_{k in {1..24}} V(k)`
  `Δt_peak = (argmax_{k in {1..24}} V(k)) * 3600  (in seconds)`
* **Procedure:**
  1. Assign each comment e_i in E_24h to bin k = floor(Δt_i / 3600) + 1.
  2. Compute maximum hourly volume V_max.
  3. Extract time-to-peak Δt_peak as the elapsed time to the highest-volume hour.
  4. Calculate initial acceleration: `A_initial = V(2) - V(1)`.

---

### TASK 3: Context-Aware Subreddit Percentile Engine

#### **SUBTASK 3.1: Subreddit Grouping & Distribution Analysis**
* **Objective:** Group posts by subreddit to account for baseline engagement disparities between large and small communities.
* **Procedure:**
  1. Partition the post population P into sub-populations P_s based on subreddit.
  2. Collect all 24-hour comment volumes C_24h within each subreddit group s.
  3. Verify sample size N_s = |P_s| per subreddit to ensure statistical validity.

#### **SUBTASK 3.2: Dynamic Cutoff Calculation (P_percentile)**
* **Objective:** Calculate the 95th percentile (P95) volume threshold per subreddit.
* **Mathematical Definition:**
  `Cutoff_s = Percentile({C_24h | post in P_s}, 95)`
* **Procedure:**
  1. Sort C_24h values within subreddit s in ascending order.
  2. Compute rank R = ceil(0.95 * N_s).
  3. Interpolate value at rank R to set Cutoff_s. Posts reaching or exceeding this cutoff are classified as viral relative to their specific community norm.

#### **SUBTASK 3.3: Small-Community Fallback Guard**
* **Objective:** Prevent low-volume noise in small subreddits from being incorrectly flagged as viral.
* **Procedure:**
  1. Check if Cutoff_s < Cutoff_global_floor (e.g., minimum 15 comments).
  2. If true or if N_s < 30 posts, override Cutoff_s with the global floor:
     `Cutoff_s_final = max(Cutoff_s, 15.0)`

---

### TASK 4: Binary & Continuous Label Assignment

#### **SUBTASK 4.1: Binary Virality Labeling (y_viral)**
* **Objective:** Assign the final classification target per thread.
* **Mathematical Definition:**
  `y_viral = 1.0 if C_24h >= Cutoff_s_final else 0.0`
* **Procedure:**
  1. Compare thread C_24h against Cutoff_s_final.
  2. Output float representation 1.0 (viral) or 0.0 (non-viral).

#### **SUBTASK 4.2: Continuous Target Transformation (y_reg)**
* **Objective:** Generate a normalized, log-scaled regression target for auxiliary loss functions.
* **Mathematical Definition:**
  `y_reg = log10(C_24h + 1)`
* **Procedure:**
  1. Apply log10(x + 1) shift-transformation to handle zero-comment threads without undefined math errors.
  2. Store transformed scalar as continuous ground truth.

#### **SUBTASK 4.3: Class Imbalance Ratio Validation**
* **Objective:** Monitor overall positive class distribution across the dataset.
* **Procedure:**
  1. Calculate global positive ratio: `Ratio_pos = sum(y_viral) / N_total`.
  2. Assert that Ratio_pos falls within expected domain limits (5% to 10%).

---

### TASK 5: CSV Storage Engine & Contract Validation

#### **SUBTASK 5.1: Schema Formatting**
* **Objective:** Assemble all computed features into the standardized target schema.
* **Required Output Fields:**
  - `post_id` (String - Shared join key)
  - `subreddit` (String)
  - `total_comments_24h` (Integer - C_24h)
  - `unique_users_24h` (Integer - U_24h)
  - `peak_hourly_velocity` (Integer - V_max)
  - `time_to_peak_seconds` (Integer - Δt_peak)
  - `p95_cutoff` (Float - Cutoff_s_final)
  - `y_viral` (Float - 1.0 or 0.0)
  - `y_reg` (Float - Log-transformed target)

#### **SUBTASK 5.2: Atomic File Serialization**
* **Objective:** Export clean, unindexed CSV dataset (`pipeline_b_labels.csv`).
* **Procedure:**
  1. Sort records alphabetically by `post_id` to guarantee deterministic output order.
  2. Write records to disk with UTF-8 encoding.

#### **SUBTASK 5.3: Automated Validation Checks**
* **Objective:** Run automated sanity tests on the generated output file.
* **Procedure:**
  1. **Leakage Test:** Verify zero comments dated > 86,400 seconds were included in C_24h.
  2. **Key Uniqueness Test:** Verify `post_id` contains zero duplicate rows.
  3. **Null Check:** Verify 0% presence of missing, NaN, or null values across all columns.
