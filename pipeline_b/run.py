import os
import json
import csv
import math
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any

# =====================================================================
# DATA CLASSES
# =====================================================================

class CommentNode:
    def __init__(self, author: str, time: float, body: str, parent_id: str, comment_id: str):
        self.author = author
        self.time = time
        self.body = body
        self.parent_id = parent_id
        self.comment_id = comment_id

class PostData:
    def __init__(self, post_index: int, post_user: str, post_time: float, subreddit: str, raw_comments: List[Dict[str, Any]], post_id_mode: str):
        self.post_index = post_index
        self.post_user = post_user
        self.post_time = post_time
        self.subreddit = subreddit
        
        # Determine post_id
        if post_id_mode == "index":
            self.post_id = f"post_{post_index}"
        else:
            self.post_id = f"{post_user}_{int(post_time)}"
            
        self.raw_comments = raw_comments
        self.flat_comments: List[CommentNode] = []
        self.filtered_comments: List[CommentNode] = []
        
        # Computed features
        self.c_24h = 0
        self.u_24h = 0
        self.v_max = 0
        self.dt_peak = 0
        self.a_initial = 0
        
        # Output label targets
        self.cutoff_s_final = 0.0
        self.y_viral = 0.0
        self.y_reg = 0.0

# =====================================================================
# INGESTION & TREE FLATTENING
# =====================================================================

def flatten_comments(comments_list: List[Dict[str, Any]], post_id: str, post_user: str) -> List[CommentNode]:
    """Flattens nested JSON comments tree using Depth-First Search (DFS)."""
    flat = []
    comment_counter = 0

    def traverse(node: Dict[str, Any], parent_id: str):
        nonlocal comment_counter
        author = node.get("user")
        if not author:
            return  # Skip invalid nodes lacking author info
        
        time_val = float(node.get("time", 0.0))
        body = node.get("body", "")
        comment_id = f"{post_id}_c{comment_counter}"
        comment_counter += 1
        
        flat.append(CommentNode(
            author=author,
            time=time_val,
            body=body,
            parent_id=parent_id,
            comment_id=comment_id
        ))
        
        for reply in node.get("replies", []):
            traverse(reply, comment_id)

    for comment in comments_list:
        traverse(comment, post_user)
        
    return flat

# =====================================================================
# CASCADE METRICS COMPUTATION
# =====================================================================

def compute_cascade_metrics(post: PostData):
    """Processes 24h filter and calculates cascade statistics."""
    # Subtask 1.2 & 1.3: Recursive traversal & 24-Hour filter
    post.flat_comments = flatten_comments(post.raw_comments, post.post_id, post.post_user)
    
    # Keep only comments created within the [0, 86400] seconds window relative to post time
    post.filtered_comments = []
    for c in post.flat_comments:
        dt = c.time - post.post_time
        if 0.0 <= dt <= 86400.0:
            post.filtered_comments.append(c)

    # Subtask 2.1: Volume Aggregation (C_24h)
    # Filter out system accounts, deleted, removed, and AutoModerator
    valid_comments = []
    for c in post.filtered_comments:
        author_lower = c.author.lower()
        if author_lower in ["[deleted]", "[removed]", "automoderator"]:
            continue
        valid_comments.append(c)
        
    # Deduplicate by comment_id
    seen_ids = set()
    deduped_comments = []
    for c in valid_comments:
        if c.comment_id not in seen_ids:
            seen_ids.add(c.comment_id)
            deduped_comments.append(c)
            
    post.c_24h = len(deduped_comments)
    
    # Subtask 2.2: Unique Participant Expansion (U_24h)
    unique_authors = set(c.author for c in deduped_comments)
    post.u_24h = len(unique_authors)
    
    # Subtask 2.3: Binned arrival velocity & acceleration
    bins = [0] * 24
    for c in deduped_comments:
        dt = c.time - post.post_time
        # Map [0, 86400] to bins 0..23 (handling boundary case 86400 cleanly)
        bin_idx = min(23, int(dt // 3600))
        bins[bin_idx] += 1
        
    post.v_max = max(bins) if bins else 0
    
    # Time-to-peak (1-indexed rank of highest-volume bin in seconds)
    if post.v_max > 0:
        peak_bin_1indexed = bins.index(post.v_max) + 1
        post.dt_peak = peak_bin_1indexed * 3600
    else:
        post.dt_peak = 0
        
    # Initial acceleration: V(2) - V(1)
    post.a_initial = bins[1] - bins[0] if len(bins) > 1 else 0

# =====================================================================
# SUBREDDIT PERCENTILE & LABEL ASSIGNMENT
# =====================================================================

def calculate_percentile_cutoffs(posts: List[PostData], cutoff_floor: float = 15.0):
    """Computes subreddit-specific 95th percentile volume cutoffs with small-community fallback."""
    # Group posts by subreddit
    subreddits_posts: Dict[str, List[PostData]] = {}
    for post in posts:
        subreddits_posts.setdefault(post.subreddit, []).append(post)
        
    # Process each subreddit group
    for subreddit, group_posts in subreddits_posts.items():
        n_s = len(group_posts)
        volumes = sorted(p.c_24h for p in group_posts)
        
        # Calculate rank R = ceil(0.95 * N_s)
        rank_idx = int(math.ceil(0.95 * n_s)) - 1
        # Access sorted volume at rank (0-indexed)
        cutoff_s = float(volumes[rank_idx]) if rank_idx < len(volumes) else 0.0
        
        # Apply small-community guard
        if cutoff_s < cutoff_floor or n_s < 30:
            cutoff_s_final = max(cutoff_s, cutoff_floor)
        else:
            cutoff_s_final = cutoff_s
            
        # Assign cutoff and compute labels for all posts in the group
        for post in group_posts:
            post.cutoff_s_final = cutoff_s_final
            
            # Subtask 4.1: Binary Virality (y_viral in {0.0, 1.0})
            post.y_viral = 1.0 if post.c_24h >= cutoff_s_final else 0.0
            
            # Subtask 4.2: Continuous Target Transformation (y_reg)
            post.y_reg = float(math.log10(post.c_24h + 1))

# =====================================================================
# AUTOMATED VALIDATION CHECKS
# =====================================================================

def validate_pipeline_output(posts: List[PostData]):
    """Executes automated sanity checks as defined in Subtasks 4.3 & 5.3."""
    print("Running automated validation checks...")
    
    # 1. Leakage Test: Verify zero comments > 86400 seconds are included in calculation
    for post in posts:
        for c in post.filtered_comments:
            dt = c.time - post.post_time
            assert 0.0 <= dt <= 86400.0, f"Leakage detected: Comment arrival delta {dt}s is outside [0, 86400] boundary!"
            
    # 2. Key Uniqueness Test: Verify post_id contains zero duplicate rows
    post_ids = [p.post_id for p in posts]
    assert len(post_ids) == len(set(post_ids)), f"Duplicate post_ids found in generated results! Unique IDs: {len(set(post_ids))} vs Total: {len(post_ids)}"
    
    # 3. Null Check: Verify 0% presence of missing/null values across all properties
    for post in posts:
        assert post.post_id is not None and post.post_id != "", "Missing post_id value"
        assert post.subreddit is not None and post.subreddit != "", f"Missing subreddit for post {post.post_id}"
        assert post.c_24h is not None, f"Missing total_comments_24h for post {post.post_id}"
        assert post.u_24h is not None, f"Missing unique_users_24h for post {post.post_id}"
        assert post.v_max is not None, f"Missing peak_hourly_velocity for post {post.post_id}"
        assert post.dt_peak is not None, f"Missing time_to_peak_seconds for post {post.post_id}"
        assert post.cutoff_s_final is not None, f"Missing p95_cutoff for post {post.post_id}"
        assert post.y_viral is not None, f"Missing y_viral for post {post.post_id}"
        assert post.y_reg is not None, f"Missing y_reg for post {post.post_id}"
        
    # 4. Class Imbalance Ratio Validation
    ratio_pos = sum(p.y_viral for p in posts) / len(posts)
    print(f"  - Positive Virality Ratio: {ratio_pos:.4%}")
    # Run imbalance assert only if sample is large enough to be statistically robust
    if len(posts) >= 30:
        assert 0.05 <= ratio_pos <= 0.10, f"Class imbalance ratio {ratio_pos:.4%} is outside expected bounds [5%, 10%]"
    else:
        print("  - [Warning] Skipping imbalance ratio assertion due to small sample size (<30 posts)")

    print("All validation checks passed successfully!")

# =====================================================================
# ENGINE EXECUTION PIPELINE
# =====================================================================

def process_file(json_path: Path, post_id_mode: str, global_post_counter: int) -> Tuple[List[PostData], int]:
    """Ingests a single JSON thread file and yields posts with initial cascade metrics."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # If the JSON file contains a single post object instead of a list, wrap it
    if isinstance(data, dict):
        data = [data]
        
    posts = []
    subreddit_name = json_path.stem  # Default subreddit name from filename stem
    
    for post_idx, post_dict in enumerate(data):
        # Determine the subreddit
        subreddit = post_dict.get("subreddit", subreddit_name)
        
        post_user = post_dict.get("post_user", post_dict.get("author", "unknown"))
        post_time = float(post_dict.get("post_time", 0.0))
        comments = post_dict.get("comments", [])
        
        # Create PostData object
        post_obj = PostData(
            post_index=global_post_counter,
            post_user=post_user,
            post_time=post_time,
            subreddit=subreddit,
            raw_comments=comments,
            post_id_mode=post_id_mode
        )
        global_post_counter += 1
        
        # Calculate intermediate cascade features
        compute_cascade_metrics(post_obj)
        posts.append(post_obj)
        
    return posts, global_post_counter

def main():
    parser = argparse.ArgumentParser(description="Pipeline B: Target Label Generation Engine.")
    parser.add_argument("--input", required=True, help="Path to input JSON file or directory containing JSON files.")
    parser.add_argument("--output", default="pipeline_b_labels.csv", help="Target output file path for y labels CSV.")
    parser.add_argument("--post-id-mode", choices=["contract", "index"], default="contract",
                        help="Format of post_id: 'contract' (user_time) or 'index' (post_index to match Pipeline A).")
    parser.add_argument("--cutoff-floor", type=float, default=15.0, help="Minimum volume threshold fallback for virality.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
        
    # Find all JSON files
    if input_path.is_dir():
        json_files = sorted(list(input_path.rglob("*.json")))
    else:
        json_files = [input_path]
        
    print(f"Ingesting {len(json_files)} JSON file(s)...")
    
    all_posts: List[PostData] = []
    global_counter = 0
    
    for j_file in json_files:
        posts, global_counter = process_file(j_file, args.post_id_mode, global_counter)
        all_posts.extend(posts)
        
    print(f"Total posts ingested: {len(all_posts)}")
    
    # Calculate subreddit percentiles and final targets
    calculate_percentile_cutoffs(all_posts, args.cutoff_floor)
    
    # Validate targets and metrics
    validate_pipeline_output(all_posts)
    
    # Sort post alphabetically by post_id
    all_posts.sort(key=lambda x: x.post_id)
    
    # Subtask 5.2: Atomic file serialization
    # Write to a temporary file in the output directory
    temp_output_path = output_path.with_suffix(".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(temp_output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Write headers
        writer.writerow([
            "post_id",
            "subreddit",
            "total_comments_24h",
            "unique_users_24h",
            "peak_hourly_velocity",
            "time_to_peak_seconds",
            "p95_cutoff",
            "y_viral",
            "y_reg"
        ])
        
        for post in all_posts:
            writer.writerow([
                post.post_id,
                post.subreddit,
                post.c_24h,
                post.u_24h,
                post.v_max,
                post.dt_peak,
                post.cutoff_s_final,
                post.y_viral,
                post.y_reg
            ])
            
    # Perform atomic replace
    if output_path.exists():
        output_path.unlink()
    temp_output_path.rename(output_path)
    
    print(f"Saved {len(all_posts)} post labels successfully to {output_path}!")

if __name__ == "__main__":
    main()
