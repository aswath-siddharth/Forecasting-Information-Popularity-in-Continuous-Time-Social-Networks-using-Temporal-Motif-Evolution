from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .data import (
    CascadeDataset,
    StreamingCascadeDataset,
    discover_file_pairs,
    load_labels_map,
    load_single_file_data,
    pad_collate_fn,
)
from .model import CascadeAttentionGRU, PoolingBaseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Pipeline C cascade virality models.")
    parser.add_argument(
        "--features",
        default="pipeline_a_features.csv",
        help="Path to feature CSV file (e.g. pipeline_a_features.csv) or directory containing feature CSVs (e.g. output_features).",
    )
    parser.add_argument(
        "--labels",
        default="pipeline_b_labels.csv",
        help="Path to target labels CSV file (e.g. pipeline_b_labels.csv or pipeline_b_labels_full.csv).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to raw JSON dataset directory (used for ID contract resolution when loading directory features).",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Optional category filter (e.g. 'Sports', 'Technology', 'Animals and Pets', 'Gaming', etc.).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on number of subreddit feature files to process.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Optional subsample fraction (0.0 to 1.0) of records to train on.",
    )
    parser.add_argument("--output-dir", default="pipeline_c/outputs")
    parser.add_argument("--sequence-column", default="X_seq_json")
    parser.add_argument("--target-column", default="y_viral")
    parser.add_argument("--feature-dim", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def make_splits(df: pd.DataFrame, target_column: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    has_multiple_classes = df[target_column].nunique() > 1
    stratify_col = df[target_column] if has_multiple_classes else None

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=stratify_col,
    )

    temp_stratify = temp_df[target_column] if (has_multiple_classes and temp_df[target_column].nunique() > 1) else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_stratify,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_single_file_loader(
    df: pd.DataFrame,
    args: argparse.Namespace,
    shuffle: bool,
    is_cuda: bool,
) -> DataLoader:
    dataset = CascadeDataset(
        df,
        sequence_column=args.sequence_column,
        target_column=args.target_column,
        feature_dim=args.feature_dim,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=pad_collate_fn,
        pin_memory=is_cuda,
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        if batch is None:
            continue
        x, padding_mask, labels = batch
        if x.nelement() == 0 or labels.nelement() == 0:
            continue

        x = x.to(device)
        padding_mask = padding_mask.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(x, padding_mask)
            loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    probabilities: list[float] = []
    labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            x, padding_mask, y = batch
            if x.nelement() == 0:
                continue
            logits = model(x.to(device), padding_mask.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().reshape(-1).tolist())
            labels.extend(y.numpy().reshape(-1).astype(int).tolist())

    y_true = np.asarray(labels)
    y_prob = np.asarray(probabilities)
    y_pred = (y_prob >= 0.5).astype(int)

    has_both_classes = len(np.unique(y_true)) == 2 if len(y_true) > 0 else False
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if has_both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if has_both_classes else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else float("nan"),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) > 0 else float("nan"),
        "total_test_samples": int(len(y_true)),
    }


def build_model(name: str, args: argparse.Namespace) -> torch.nn.Module:
    if name == "attention_gru":
        return CascadeAttentionGRU(
            input_dim=args.feature_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            attention_dim=args.attention_dim,
            dropout=args.dropout,
        )
    if name == "mean_pool":
        return PoolingBaseline("mean", input_dim=args.feature_dim, dropout=args.dropout)
    if name == "max_pool":
        return PoolingBaseline("max", input_dim=args.feature_dim, dropout=args.dropout)
    raise ValueError(f"Unknown model name: {name}")


def train_one_model(
    name: str,
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    pos_weight: torch.Tensor,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    model = build_model(name, args).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, float | int | str]] = []
    checkpoint_path = output_dir / f"best_{name}_model.pt"

    print(f"\n--- Training {name} ---")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        history.append({"model": name, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"[{name}] epoch {epoch:03d}/{args.epochs:03d} -> train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"[{name}] early stopping triggered after {epoch} epochs")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    metrics = evaluate_model(model, test_loader, device)
    metrics["best_val_loss"] = float(best_val_loss)

    if name == "attention_gru":
        torch.save(model.state_dict(), output_dir / "best_cascade_model.pt")

    return metrics, history


def save_loss_curves(history_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(9, 5))
    for model_name, model_history in history_df.groupby("model"):
        plt.plot(model_history["epoch"], model_history["train_loss"], label=f"{model_name} train")
        plt.plot(model_history["epoch"], model_history["val_loss"], linestyle="--", label=f"{model_name} val")
    plt.xlabel("Epoch")
    plt.ylabel("BCEWithLogits Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features_path = Path(args.features)
    labels_path = Path(args.labels)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    is_cuda = (device.type == "cuda")

    if features_path.is_dir():
        print(f"Discovered features directory: {features_path}")
        train_pairs, val_pairs, test_pairs = discover_file_pairs(
            features_dir=features_path,
            data_dir=args.data_dir,
            category=args.category,
            max_files=args.max_files,
            seed=args.seed,
        )
        total_files = len(train_pairs) + len(val_pairs) + len(test_pairs)
        print(f"Total subreddit files: {total_files} (Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)})")

        labels_map = load_labels_map(labels_path, target_column=args.target_column)
        positives = sum(labels_map.values())
        negatives = len(labels_map) - positives
        pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)

        train_dataset = StreamingCascadeDataset(
            train_pairs,
            labels_map,
            sequence_column=args.sequence_column,
            target_column=args.target_column,
            feature_dim=args.feature_dim,
            shuffle_files=True,
            sample_ratio=args.sample_ratio,
        )
        val_dataset = StreamingCascadeDataset(
            val_pairs,
            labels_map,
            sequence_column=args.sequence_column,
            target_column=args.target_column,
            feature_dim=args.feature_dim,
            shuffle_files=False,
            sample_ratio=args.sample_ratio,
        )
        test_dataset = StreamingCascadeDataset(
            test_pairs,
            labels_map,
            sequence_column=args.sequence_column,
            target_column=args.target_column,
            feature_dim=args.feature_dim,
            shuffle_files=False,
            sample_ratio=args.sample_ratio,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=pad_collate_fn,
            pin_memory=is_cuda,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=pad_collate_fn,
            pin_memory=is_cuda,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=pad_collate_fn,
            pin_memory=is_cuda,
        )

        dataset_meta = {
            "mode": "streaming_directory",
            "total_files": total_files,
            "train_files": len(train_pairs),
            "val_files": len(val_pairs),
            "test_files": len(test_pairs),
            "labels_count": len(labels_map),
        }
    else:
        # Single-file mode
        print(f"Loading single features file: {features_path}")
        merged_df, join_stats = load_single_file_data(
            features_path,
            labels_path,
            sequence_column=args.sequence_column,
            target_column=args.target_column,
            sample_ratio=args.sample_ratio,
        )
        train_df, val_df, test_df = make_splits(merged_df, args.target_column, args.seed)
        train_loader = make_single_file_loader(train_df, args, shuffle=True, is_cuda=is_cuda)
        val_loader = make_single_file_loader(val_df, args, shuffle=False, is_cuda=is_cuda)
        test_loader = make_single_file_loader(test_df, args, shuffle=False, is_cuda=is_cuda)

        positives = float(train_df[args.target_column].sum())
        negatives = float(len(train_df) - positives)
        pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)

        dataset_meta = {
            "mode": "single_file",
            "feature_rows": join_stats.feature_rows,
            "joined_rows": join_stats.joined_rows,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        }

    print(f"device={device} | pos_weight={pos_weight.item():.4f} | streaming_workers={args.num_workers}")
    print(f"dataset info: {dataset_meta}")

    all_metrics: dict[str, dict[str, float]] = {}
    all_history: list[dict[str, float | int | str]] = []
    for model_name in ("attention_gru", "mean_pool", "max_pool"):
        metrics, history = train_one_model(
            model_name,
            args,
            train_loader,
            val_loader,
            test_loader,
            pos_weight,
            device,
            output_dir,
        )
        all_metrics[model_name] = metrics
        all_history.extend(history)

    history_df = pd.DataFrame(all_history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    save_loss_curves(history_df, output_dir)

    report = {
        "dataset_meta": dataset_meta,
        "metrics": all_metrics,
    }
    metadata = {
        "input_feature_dim": args.feature_dim,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "attention_dim": args.attention_dim,
        "dropout": args.dropout,
        "threshold": 0.5,
        "pos_weight": pos_weight.item(),
        "best_model": "attention_gru",
        "checkpoint": "best_cascade_model.pt",
        "metrics": all_metrics["attention_gru"],
    }

    (output_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n--- Final Evaluation Report ---")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
