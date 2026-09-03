# High-Performance Graph Analytics & Feature Extraction Pipeline

This repository contains a high-throughput Python data pipeline designed to ingest nested JSON comment threads and transform them into structured 20-D feature matrices for downstream machine learning tasks.

## Task Overview

The pipeline implements the `PIPELINE_A_SPECIFICATION`, which requires processing raw streams/micro-batches of comments into a sequence of structural and temporal features. The core engine relies on a dynamic Compressed Sparse Row (CSR) graph representation and Worst-Case Optimal Join (WCOJ) motif detection.

The execution is divided into five primary tasks:

*   **Task 1: Ingestion & Dynamic CSR:** Parses top-level post metadata and flattens nested comments into chronologically sorted temporal edges. It filters these edges into specific time windows (e.g., 15-minute intervals) and builds an in-memory dynamic CSR adjacency structure for fast traversal. Parent/ancestor tracking is **cycle-safe**: a node is only assigned a parent once, self-referencing edges (a user replying to their own earlier comment) are not treated as tree edges, and the ancestor walk used for descendant counting is bounded by a visited-set guard so it cannot loop indefinitely even on malformed or reciprocal reply patterns.
*   **Task 2: WCOJ Motif Engine:** Computes higher-order graph motifs (Single Direct Reply, Sibling Star Fork, Deep Directed Chain, Ping Pong Dyad, Reciprocal Cycle) using Leapfrog Triejoin approximations over the CSR indices, mapping the result to a 5-dimensional one-hot vector.
*   **Task 3: Temporal Dynamics:** Leverages the CSR arrays to calculate temporal velocities, such as the inter-arrival gap between comments, the motif completion speed, and a binary reactivation flag for dormant subgraphs.
*   **Task 4: Content Analysis & Assembly:** Utilizes VADER sentiment analysis to extract normalized sentiment weights. It then aggregates the structural, temporal, and sentiment metrics into a standardized 20-D float32 array per edge.
*   **Task 5: Serialization & Output:** Serializes the resulting 2D matrix (K x 20) into a JSON-formatted string and writes the output atomically to a CSV file.

## Setup & Installation

**1. Prerequisites**
Ensure you have Python 3.7 or higher installed on your system.

**2. Install Dependencies**
The pipeline requires `numpy` for efficient array manipulation and `vaderSentiment` for text analysis. Install the required packages using `pip`:

```bash
pip install numpy vaderSentiment
```

## Usage

Process a single thread file into a feature CSV:

```python
from Task1 import execute_pipeline

execute_pipeline("ActivationSound.json", "ActivationSound_features.csv")
```

Recursively process an entire directory of `.json` thread files, preserving folder structure in the output:

```python
from Task1 import process_directory

process_directory(data_dir="data", output_dir="output_features1")
```

Or simply run the script directly, which processes everything under `data/` and writes results to `output_features1/`:

```bash
python Task1.py
```

## Output Format

Each output CSV contains one row per (post, time-window) batch, with the following columns:

| Column | Description |
|---|---|
| `post_id` | Identifier of the source post/thread |
| `window_index` | Index of the fixed-size time window (default 900s) within the thread |
| `cumulative_prior_comments` | Running count of comments seen before this window |
| `edge_count_K` | Number of edges (comments) in this window's batch |
| `X_seq_json` | JSON-encoded `K x 20` float matrix, one 20-D feature row per edge |

### 20-D Feature Vector Layout

| Index | Feature |
|---|---|
| 0–4 | One-hot motif vector: Single Reply, Sibling Star, Deep Chain, Ping-Pong Dyad, Reciprocal Cycle |
| 5 | Tree depth of the destination node |
| 6 | Out-degree of the parent node |
| 7 | Total node count in the CSR graph so far |
| 8 | Temporal distance from thread root |
| 9–12 | Sentiment features (weighted compound, compound, positive, negative) |
| 13–15 | Reactivation flag, idle time, ancestor descendant count |
| 16 | Motif completion speed (`dt_m`) |
| 17–18 | Window index, cumulative prior comment count |
| 19 | Inter-arrival gap (`dt_edge`) |

## Known Issues & Fixes

### Fixed: infinite loop on cyclic parent relationships

**Symptom:** the pipeline would hang indefinitely partway through processing (often on the very first thread in a dataset), producing a truncated or near-empty output CSV with far fewer rows than the number of input posts.

**Cause:** `CSRGraph` tracked parent/ancestor relationships keyed by *user*, not by individual comment. Because the same user can legitimately appear at multiple positions in a reply tree — replying to their own earlier comment, or exchanging replies with another user across separate branches — the parent pointers could form a cycle (including a trivial self-loop). The ancestor-walk used to update `descendant_counts` had no cycle protection, so it looped forever once such a cycle existed.

**Fix:** `CSRGraph.add_edge()` now:
- Skips self-loop edges (`u == v`) when assigning a parent.
- Assigns a parent to a node only the first time it's encountered (no re-parenting).
- Guards the ancestor-count walk with a `visited` set so it always terminates, even if an unforeseen cyclic pattern occurs.

This restores correct processing of the full input dataset with no change to the motif logic, sentiment features, or CSV output schema.