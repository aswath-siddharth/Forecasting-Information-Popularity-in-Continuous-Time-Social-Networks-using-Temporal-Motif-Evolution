import os
import json
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =====================================================================
# DATA STRUCTURES
# =====================================================================

@dataclass
class Edge:
    src_user: str
    dst_user: str
    timestamp: float
    text: str
    parent_user: str

@dataclass
class WindowBatch:
    post_id: str
    window_idx: int
    cumulative_prior_comments: int
    edges: List[Edge]

# =====================================================================
# STEP 1: INGESTION, FILTERING & DYNAMIC CSR
# =====================================================================

class ThreadParser:
    def __init__(self, post_data: Dict[str, Any], post_index: int):
        self.post_data = post_data
        self.post_id = post_data.get("post_id", f"post_{post_index}")
        self.post_user = post_data.get("post_user", "root")
        self.post_time = float(post_data.get("post_time", 0.0))

    def flatten_comments(self) -> List[Edge]:
        edges: List[Edge] = []
        for comment in self.post_data.get("comments", []):
            self._parse_node(comment, self.post_user, edges)
        edges.sort(key=lambda x: x.timestamp)
        return edges

    def _parse_node(self, node: Dict[str, Any], parent_user: str, edge_list: List[Edge]) -> None:
        current_user = node.get("user")
        if current_user:
            edge_list.append(Edge(
                src_user=parent_user, dst_user=current_user,
                timestamp=float(node.get("time", 0.0)), text=node.get("body", ""),
                parent_user=parent_user
            ))
            for reply in node.get("replies", []):
                self._parse_node(reply, current_user, edge_list)

class StreamWindowFilter:
    def __init__(self, window_size_seconds: float = 900.0):
        self.window_size_seconds = window_size_seconds

    def process_thread(self, post_id: str, post_time: float, edges: List[Edge]) -> List[WindowBatch]:
        if not edges: return []
        window_groups: Dict[int, List[Edge]] = {}
        for edge in edges:
            w_idx = max(0, int((edge.timestamp - post_time) // self.window_size_seconds))
            window_groups.setdefault(w_idx, []).append(edge)
            
        batches, cumulative = [], 0
        for w_idx in range(max(window_groups.keys()) + 1):
            if w_edges := window_groups.get(w_idx, []):
                batches.append(WindowBatch(post_id, w_idx, cumulative, w_edges))
                cumulative += len(w_edges)
        return batches

class CSRGraph:
    def __init__(self):
        self.node_to_id, self.id_to_node = {}, {}
        self.adj_list: Dict[int, List[Tuple[int, float]]] = {}
        self.indptr = np.array([0], dtype=np.int32)
        self.indices = np.array([], dtype=np.int32)
        self.timestamps = np.array([], dtype=np.float64)
        self.node_history: Dict[int, List[float]] = {}
        self.node_depth: Dict[int, int] = {}
        self.parent_map: Dict[int, int] = {}
        self.descendant_counts: Dict[int, int] = {}

    def add_edge(self, u_str: str, v_str: str, timestamp: float):
        u = self._get_or_create(u_str)
        v = self._get_or_create(v_str)
        self.adj_list[u].append((v, timestamp))
        self.node_history.setdefault(v, []).append(timestamp)

        # Set parent pointer for ancestor traversal.
        # Skip self-loops (a user replying to their own earlier comment, u == v)
        # and skip re-parenting a node that has already been assigned a parent
        # elsewhere in the tree — both of these can otherwise create cycles in
        # parent_map (since nodes are keyed by *user*, and the same user can
        # legitimately appear at multiple positions in a reply tree).
        if u != v and v not in self.parent_map:
            self.parent_map[v] = u

        # Incremental tree depth from root post
        parent_depth = self.node_depth.setdefault(u, 0)
        if v not in self.node_depth:
            self.node_depth[v] = parent_depth + 1

        # Increment descendant counts for all ancestors in tree.
        # Guard with a `visited` set so that any residual cycle (or a very
        # deep/pathological chain) can never spin forever.
        curr = u
        visited = set()
        while curr is not None and curr not in visited:
            visited.add(curr)
            self.descendant_counts[curr] = self.descendant_counts.get(curr, 0) + 1
            curr = self.parent_map.get(curr)

        return u, v

    def _get_or_create(self, node_str: str) -> int:
        if node_str not in self.node_to_id:
            n_id = len(self.node_to_id)
            self.node_to_id[node_str] = n_id
            self.id_to_node[n_id] = node_str
            self.adj_list[n_id] = []
        return self.node_to_id[node_str]

    def build_csr(self):
        indptr, indices, timestamps = [0], [], []
        for n_id in range(len(self.node_to_id)):
            neighbors = sorted(self.adj_list.get(n_id, []), key=lambda x: x[0])
            for v, t in neighbors:
                indices.append(v)
                timestamps.append(t)
            indptr.append(len(indices))
        self.indptr = np.array(indptr, dtype=np.int32)
        self.indices = np.array(indices, dtype=np.int32)
        self.timestamps = np.array(timestamps, dtype=np.float64)

    def get_neighbors(self, u_id: int) -> list[int]:
        """Return dynamic out-neighbors of node u_id in O(1)."""
        return [v for v, _ in self.adj_list.get(u_id, [])]

# =====================================================================
# STEP 2 & 3: WCOJ ENGINE & TEMPORAL DYNAMICS
# =====================================================================

class MotifAndTemporalEngine:
    def __init__(self, csr: CSRGraph, chain_depth_threshold: int = 3):
        self.csr = csr
        self.chain_depth_threshold = chain_depth_threshold

    def analyze_edge(self, u_id: int, v_id: int, t_i: float, t_prev: float, post_time: float = 0.0) -> Tuple[List[float], float, Tuple[float, float, float]]:
        """Causally evaluate topological motif and temporal dynamics *before* inserting current edge."""
        u_neighbors = self.csr.get_neighbors(u_id)
        v_neighbors = self.csr.get_neighbors(v_id)
        
        # 1. Multi-Hot Topological Motif Indicators (Causal)
        # Reciprocal 3-Cycle: u -> v completes cycle if v -> w and w -> u exist in graph
        is_cycle = 1.0 if any(u_id in self.csr.get_neighbors(w) for w in v_neighbors) else 0.0
        # Ping-Pong Dyad: u -> v completes dyad if v -> u already exists
        is_dyad = 1.0 if (u_id in v_neighbors) else 0.0
        # Deep Chain: child will be at depth >= K (i.e. parent_depth + 1 >= K)
        parent_depth = self.csr.node_depth.get(u_id, 0)
        is_chain = 1.0 if (parent_depth + 1 >= self.chain_depth_threshold) else 0.0
        # Sibling Star Fork: parent u already has >= 1 child, so adding this edge makes >= 2 children
        is_star = 1.0 if (len(u_neighbors) >= 1) else 0.0
        # Single Direct Reply: active only if NO higher-order motif is triggered
        is_single = 1.0 if (is_cycle == 0.0 and is_dyad == 0.0 and is_chain == 0.0 and is_star == 0.0) else 0.0

        motif_vector = [is_single, is_star, is_chain, is_dyad, is_cycle]

        # 2. Temporal Dynamics
        dt_edge = t_i - t_prev if t_prev > 0 else 0.0
        
        # Motif Completion Speed (Earliest edge trigger on parent u)
        history_u = self.csr.node_history.get(u_id, [t_i])
        t_trigger = min(history_u)
        dt_m = t_i - t_trigger

        # 3. Reactivation & Dormancy Engine (T_threshold = 300s)
        history_v_prev = [t for t in self.csr.node_history.get(v_id, []) if t < t_i]
        if history_v_prev:
            # Returning user: idle time since their last comment
            t_idle = t_i - max(history_v_prev)
            i_reactivate = 1.0 if t_idle > 300.0 else 0.0
        else:
            # First-time user joining: elapsed time since post inception
            t_idle = max(0.0, t_i - post_time)
            i_reactivate = 0.0

        # Subtree / descendant count of parent u
        descendant_count = float(self.csr.descendant_counts.get(u_id, 0))
        reactivation_flags = (i_reactivate, t_idle, descendant_count)

        return motif_vector, dt_m, reactivation_flags

# =====================================================================
# STEP 4: CONTENT ANALYSIS & 20-D MATRIX ASSEMBLY
# =====================================================================

class FeatureAssembler:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def build_20d_vector(self, edge: Edge, motif_one_hot: List[float], csr_metrics: List[float], 
                         reactivation: Tuple[float, float, float], dt_m: float, 
                         w_idx: int, cumulative: int, dt_edge: float) -> np.ndarray:
        
        # Sentiment
        vs = self.analyzer.polarity_scores(edge.text)
        w_s = vs['compound'] # Normalized weight
        sentiment_features = [w_s, vs['compound'], vs['pos'], vs['neg']]

        # Vector Assembly
        vec = []
        vec.extend(motif_one_hot)                    # 0-4
        vec.extend(csr_metrics)                      # 5-8 (depth, out_deg, count, t_from_root)
        vec.extend(sentiment_features)               # 9-12
        vec.extend(list(reactivation))               # 13-15 (I_react, T_idle, desc)
        vec.append(dt_m)                             # 16
        vec.extend([float(w_idx), float(cumulative)])# 17-18
        vec.append(dt_edge)                          # 19
        
        return np.array(vec, dtype=np.float32)

# =====================================================================
# STEP 5: SERIALIZATION & ATOMIC OUTPUT
# =====================================================================

def execute_pipeline(input_json: str, output_csv: str):
    """Processes a single JSON file and writes features to a CSV."""
    with open(input_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    window_filter = StreamWindowFilter()
    feature_assembler = FeatureAssembler()
    
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["post_id", "window_index", "cumulative_prior_comments", "edge_count_K", "X_seq_json"])

        for post_idx, post in enumerate(dataset):
            parser = ThreadParser(post, post_idx)
            edges = parser.flatten_comments()
            batches = window_filter.process_thread(parser.post_id, parser.post_time, edges)
            
            csr = CSRGraph()
            engine = MotifAndTemporalEngine(csr)
            t_prev = 0.0
            
            for batch in batches:
                X_seq = []

                for e in batch.edges:
                    # 1. Register node IDs
                    u_id = csr._get_or_create(e.src_user)
                    v_id = csr._get_or_create(e.dst_user)
                    
                    # 2. Causally evaluate Motif & Time features (BEFORE inserting current edge into graph)
                    motif, dt_m, react = engine.analyze_edge(u_id, v_id, e.timestamp, t_prev, post_time=parser.post_time)
                    
                    # 3. Add edge to graph (updates depth, parent map, ancestor descendant counts, adj_list)
                    csr.add_edge(e.src_user, e.dst_user, e.timestamp)

                    # 4. Centrality Metrics (Real computed tree depth, out-degree, node count, temporal root distance)
                    v_depth = float(csr.node_depth.get(v_id, 0))
                    t_from_root = max(0.0, float(e.timestamp - parser.post_time))
                    csr_metrics = [
                        v_depth,                                # Col 5: tree_depth
                        float(len(csr.get_neighbors(u_id))),    # Col 6: parent_out_degree
                        float(len(csr.node_to_id)),             # Col 7: csr_node_count
                        t_from_root                             # Col 8: temporal_distance_from_root (replaces duplicate depth)
                    ]

                    # 5. 20-D Matrix Stacking
                    vec = feature_assembler.build_20d_vector(
                        edge=e, motif_one_hot=motif, csr_metrics=csr_metrics, 
                        reactivation=react, dt_m=dt_m, w_idx=batch.window_idx, 
                        cumulative=batch.cumulative_prior_comments, dt_edge=(e.timestamp - t_prev if t_prev else 0.0)
                    )
                    X_seq.append(vec.tolist())
                    t_prev = e.timestamp

                # Build contiguous CSR array buffers for current batch
                csr.build_csr()

                # Serialization to CSV
                X_seq_json = json.dumps(X_seq)
                writer.writerow([
                    batch.post_id, 
                    batch.window_idx, 
                    batch.cumulative_prior_comments, 
                    len(batch.edges), 
                    X_seq_json
                ])

# =====================================================================
# BATCH DIRECTORY PROCESSING
# =====================================================================

def process_directory(data_dir: str, output_dir: str):
    """
    Recursively finds all .json files in data_dir, processes them, 
    and writes the output to output_dir while maintaining folder structure.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    
    if not data_path.exists():
        print(f"Error: The directory '{data_dir}' does not exist.")
        return

    # Find all .json files recursively
    json_files = list(data_path.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files to process.\n" + "="*50)
    
    for json_file in json_files:
        # Get the relative path to recreate the folder structure in the output directory
        rel_path = json_file.relative_to(data_path)
        output_file_dir = out_path / rel_path.parent
        
        # Ensure the target output directory exists
        output_file_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the output CSV name
        output_csv = output_file_dir / f"{json_file.stem}_features.csv"
        
        print(f"Processing: {json_file.name} ...")
        try:
            execute_pipeline(str(json_file), str(output_csv))
            print(f"  -> Saved to {output_csv}")
        except Exception as e:
            print(f"  -> [FAILED] Error processing {json_file.name}: {e}")

if __name__ == "__main__":
    # Define root input folder and where the CSVs should be generated
    INPUT_DATA_DIR = "data"
    OUTPUT_DATA_DIR = "output_features1"
    
    process_directory(INPUT_DATA_DIR, OUTPUT_DATA_DIR)
    print("\nAll processing complete.")