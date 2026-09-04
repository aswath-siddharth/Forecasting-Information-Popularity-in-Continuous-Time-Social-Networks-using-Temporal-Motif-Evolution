from __future__ import annotations

import argparse
import json
import random
import math
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
    LengthGroupedBatchSampler,
    SequenceScaler,
    StreamingCascadeDataset,
    discover_file_pairs,
    load_labels_map,
    load_single_file_data,
    pad_collate_fn,
    parse_sequence,
)
from .model import BinaryFocalLossWithLogits, CascadeAttentionGRU, PoolingBaseline


class PlateauCosineLR:
    """Learning rate scheduler that monitors a metric (e.g. val_loss) and applies smooth
    Cosine Annealing transitions when a plateau is detected.

    When a metric stops improving after `patience` epochs, instead of an instantaneous step drop,
    it smoothly decays LR from start_lr to target_lr = max(start_lr * factor, min_lr) over
    `cosine_epochs` following a half-cosine curve:
        lr(t) = target_lr + 0.5 * (start_lr - target_lr) * (1 + cos(pi * t / N))
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = "min",
        patience: int = 3,
        cosine_epochs: int = 5,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        threshold: float = 1e-4,
    ) -> None:
        if factor >= 1.0 or factor <= 0.0:
            raise ValueError("factor must be strictly between 0 and 1.")
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'.")

        self.optimizer = optimizer
        self.mode = mode
        self.patience = patience
        self.cosine_epochs = max(1, cosine_epochs)
        self.factor = factor
        self.min_lr = min_lr
        self.threshold = threshold

        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.bad_epochs = 0
        self.is_decaying = False
        self.decay_step = 0
        self.start_lrs: list[float] = []
        self.target_lrs: list[float] = []

    def _is_improvement(self, metric: float) -> bool:
        if self.mode == "min":
            return metric < (self.best_metric - self.threshold)
        return metric > (self.best_metric + self.threshold)

    def step(self, metric: float) -> None:
        if self.is_decaying:
            self.decay_step += 1
            alpha = min(1.0, self.decay_step / self.cosine_epochs)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * alpha))

            for param_group, start_lr, target_lr in zip(self.optimizer.param_groups, self.start_lrs, self.target_lrs):
                new_lr = target_lr + (start_lr - target_lr) * cosine_factor
                param_group["lr"] = new_lr

            if self.decay_step >= self.cosine_epochs:
                self.is_decaying = False
                self.bad_epochs = 0
                if self._is_improvement(metric):
                    self.best_metric = metric
        else:
            if self._is_improvement(metric):
                self.best_metric = metric
                self.bad_epochs = 0
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= self.patience:
                    self.is_decaying = True
                    self.decay_step = 0
                    self.start_lrs = [group["lr"] for group in self.optimizer.param_groups]
                    self.target_lrs = [max(lr * self.factor, self.min_lr) for lr in self.start_lrs]


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
    parser.add_argument(
        "--loss-type",
        choices=["focal", "bce_damped", "bce_raw"],
        default="focal",
        help="Loss function for imbalanced virality classification: 'focal' (recommended), 'bce_damped' (sqrt pos_weight), or 'bce_raw'.",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focusing parameter gamma for Focal Loss.")
    parser.add_argument("--focal-alpha", type=float, default=0.75, help="Alpha class balance weight for Focal Loss.")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3, help="Plateau and early stopping patience (epochs).")
    parser.add_argument("--scheduler-type", choices=["plateau_cosine", "plateau"], default="plateau_cosine")
    parser.add_argument("--cosine-epochs", type=int, default=3, help="Duration of cosine decay transitions.")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--use-bucket-sampler", action="store_true", default=True, help="Use LengthGroupedBatchSampler to minimize padding waste and inter-batch variance.")
    parser.add_argument("--no-bucket-sampler", action="store_false", dest="use_bucket_sampler", help="Disable LengthGroupedBatchSampler.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate (default: 0.0003).")
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
    scaler: SequenceScaler | None = None,
) -> DataLoader:
    dataset = CascadeDataset(
        df,
        sequence_column=args.sequence_column,
        target_column=args.target_column,
        feature_dim=args.feature_dim,
        scaler=scaler,
    )
    if getattr(args, "use_bucket_sampler", True):
        lengths = [
            len(parse_sequence(row, args.feature_dim))
            for row in df[args.sequence_column]
        ]
        batch_sampler = LengthGroupedBatchSampler(
            lengths=lengths,
            batch_size=args.batch_size,
            shuffle=shuffle,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            collate_fn=pad_collate_fn,
            pin_memory=is_cuda,
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
    prior_prob: float,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    model = build_model(name, args).to(device)
    if hasattr(model, "init_prior_bias"):
        model.init_prior_bias(prior_prob)

    if args.loss_type == "focal":
        criterion = BinaryFocalLossWithLogits(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler_type == "plateau_cosine":
        scheduler = PlateauCosineLR(
            optimizer,
            mode="min",
            patience=args.patience,
            cosine_epochs=args.cosine_epochs,
            factor=args.lr_factor,
            min_lr=1e-6,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=args.patience, factor=args.lr_factor
        )

    best_val_loss = float("inf")
    stale_epochs = 0
    history: list[dict[str, float | int | str]] = []
    checkpoint_path = output_dir / f"best_{name}_model.pt"

    print(f"\n--- Training {name} (Scheduler: {args.scheduler_type}) ---")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "model": name,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
        })

        print(
            f"[{name}] epoch {epoch:03d}/{args.epochs:03d} -> "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={current_lr:.6f}"
        )

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
        positives = float(sum(labels_map.values()))
        negatives = float(len(labels_map) - positives)
        prior_prob = positives / max(positives + negatives, 1.0)
        raw_pos_ratio = negatives / max(positives, 1.0)
        damped_pos_ratio = math.sqrt(raw_pos_ratio)

        if args.loss_type == "bce_damped":
            pos_weight = torch.tensor([damped_pos_ratio], dtype=torch.float32)
        else:
            pos_weight = torch.tensor([raw_pos_ratio], dtype=torch.float32)

        # Fit SequenceScaler on sample of training sequences to prevent data leakage
        def _get_train_seqs():
            count = 0
            for feat_file, _ in train_pairs[:20]:
                try:
                    with open(feat_file, "r", encoding="utf-8") as ff:
                        reader = csv.DictReader(ff)
                        for row in reader:
                            seq_str = row.get(args.sequence_column)
                            if seq_str:
                                yield parse_sequence(seq_str, args.feature_dim)
                                count += 1
                                if count >= 2000:
                                    return
                except Exception:
                    continue

        scaler = SequenceScaler().fit(_get_train_seqs())
        print(f"Fitted SequenceScaler on training split (mean={scaler.mean}, scale={scaler.scale})")

        train_dataset = StreamingCascadeDataset(
            train_pairs,
            labels_map,
            sequence_column=args.sequence_column,
            target_column=args.target_column,
            feature_dim=args.feature_dim,
            shuffle_files=True,
            sample_ratio=args.sample_ratio,
            scaler=scaler,
        )
        val_dataset = StreamingCascadeDataset(
            val_pairs,
            labels_map,
            sequence_column=args.sequence_column,
            target_column=args.target_column,
            feature_dim=args.feature_dim,
            shuffle_files=False,
            sample_ratio=args.sample_ratio,
            scaler=scaler,
        )
        test_dataset = StreamingCascadeDataset(
            test_pairs,
            labels_map,
            sequence_column=args.sequence_column,
            target_column=args.target_column,
            feature_dim=args.feature_dim,
            shuffle_files=False,
            sample_ratio=args.sample_ratio,
            scaler=scaler,
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

        # Fit SequenceScaler on training split only to prevent data leakage
        train_sequences = [
            parse_sequence(row, args.feature_dim)
            for row in train_df[args.sequence_column]
        ]
        scaler = SequenceScaler().fit(train_sequences)
        print(f"Fitted SequenceScaler on training split (mean={scaler.mean}, scale={scaler.scale})")

        train_loader = make_single_file_loader(train_df, args, shuffle=True, is_cuda=is_cuda, scaler=scaler)
        val_loader = make_single_file_loader(val_df, args, shuffle=False, is_cuda=is_cuda, scaler=scaler)
        test_loader = make_single_file_loader(test_df, args, shuffle=False, is_cuda=is_cuda, scaler=scaler)

        positives = float(train_df[args.target_column].sum())
        negatives = float(len(train_df) - positives)
        prior_prob = positives / max(positives + negatives, 1.0)
        raw_pos_ratio = negatives / max(positives, 1.0)
        damped_pos_ratio = math.sqrt(raw_pos_ratio)

        if args.loss_type == "bce_damped":
            pos_weight = torch.tensor([damped_pos_ratio], dtype=torch.float32)
        else:
            pos_weight = torch.tensor([raw_pos_ratio], dtype=torch.float32)

        dataset_meta = {
            "mode": "single_file",
            "feature_rows": join_stats.feature_rows,
            "joined_rows": join_stats.joined_rows,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        }

    print(
        f"device={device} | loss_type={args.loss_type} | prior_prob={prior_prob:.4f} | "
        f"pos_weight={pos_weight.item():.4f} (raw={raw_pos_ratio:.4f}, damped={damped_pos_ratio:.4f}) | "
        f"streaming_workers={args.num_workers}"
    )
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
            prior_prob,
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
