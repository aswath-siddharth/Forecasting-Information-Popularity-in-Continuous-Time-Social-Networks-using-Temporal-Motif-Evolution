"""Run complete DeepCas benchmark on Twitter dataset for 1-day and 2-day observation windows."""

import argparse
import json
from pathlib import Path
from deepcas.train import train_deepcas


def main():
    parser = argparse.ArgumentParser(description="Run full DeepCas benchmark on Twitter dataset.")
    parser.add_argument("--window", choices=["1d", "2d", "both"], default="both", help="Which window to run")
    parser.add_argument("--epochs", type=int, default=60, help="Max training epochs (default: 60)")
    parser.add_argument("--early-stop", type=int, default=8, help="Early stopping patience (default: 8)")
    parser.add_argument("--patience", type=int, default=3, help="LR plateau decay patience (default: 3)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--seed", type=int, default=42, help="Seed (default: 42)")
    args = parser.parse_args()

    results = {}

    if args.window in ["1d", "both"]:
        print("\n" + "=" * 75)
        print(">>> STARTING EXPERIMENT 1: 1-DAY OBSERVATION WINDOW (T_obs = 86,400s) <<<")
        print("=" * 75)
        res_1d = train_deepcas(
            data_dir="data_twitter/processed_1d",
            output_dir="outputs_deepcas/outputs_1d",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lr_patience=args.patience,
            early_stop_patience=args.early_stop,
            seed=args.seed,
        )
        results["1d"] = res_1d

    if args.window in ["2d", "both"]:
        print("\n" + "=" * 75)
        print(">>> STARTING EXPERIMENT 2: 2-DAY OBSERVATION WINDOW (T_obs = 172,800s) <<<")
        print("=" * 75)
        res_2d = train_deepcas(
            data_dir="data_twitter/processed_2d",
            output_dir="outputs_deepcas/outputs_2d",
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lr_patience=args.patience,
            early_stop_patience=args.early_stop,
            seed=args.seed,
        )
        results["2d"] = res_2d

    # Save consolidated summary
    summary_path = Path("outputs_deepcas/benchmark_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print(" ALL EXPERIMENTS COMPLETE! SUMMARY SAVED TO " + str(summary_path))
    print("=" * 75)


if __name__ == "__main__":
    main()
