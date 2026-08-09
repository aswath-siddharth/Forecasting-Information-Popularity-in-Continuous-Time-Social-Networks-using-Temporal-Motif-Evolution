# High-Performance Graph Analytics & Feature Extraction Pipeline

This repository contains a high-throughput Python data pipeline designed to ingest nested JSON comment threads and transform them into structured 20-D feature matrices for downstream machine learning tasks. 

## Task Overview

The pipeline implements the `PIPELINE_A_SPECIFICATION`, which requires processing raw streams/micro-batches of comments into a sequence of structural and temporal features. The core engine relies on a dynamic Compressed Sparse Row (CSR) graph representation and Worst-Case Optimal Join (WCOJ) motif detection.

The execution is divided into five primary tasks:

*   **Task 1: Ingestion & Dynamic CSR:** Parses top-level post metadata and flattens nested comments into chronologically sorted temporal edges. It filters these edges into specific time windows (e.g., 15-minute intervals) and builds an in-memory dynamic CSR adjacency structure for fast traversal.
*   **Task 2: WCOJ Motif Engine:** Computes higher-order graph motifs (Single Direct Reply, Sibling Star Fork, Deep Directed Chain, Ping Pong Dyad, Reciprocal Cycle) using Leapfrog Triejoin approximations over the CSR indices, mapping the result to a 5-dimensional one-hot vector.
*   **Task 3: Temporal Dynamics:** Leverages the CSR arrays to calculate temporal velocities, such as the inter-arrival gap between comments, the motif completion speed, and a binary reactivation flag for dormant subgraphs.
*   **Task 4: Content Analysis & Assembly:** Utilizes VADER sentiment analysis to extract normalized sentiment weights[cite: 2]. It then aggregates the structural, temporal, and sentiment metrics into a standardized 20-D float32 array per edge[cite: 2].
*   **Task 5: Serialization & Output:** Serializes the resulting 2D matrix (K x 20) into a JSON-formatted string and writes the output atomically to a CSV file[cite: 2].

## Setup & Installation

**1. Prerequisites**
Ensure you have Python 3.7 or higher installed on your system. 

**2. Install Dependencies**
The pipeline requires `numpy` for efficient array manipulation and `vaderSentiment` for text analysis. Install the required packages using `pip`:

```bash
pip install numpy vaderSentiment