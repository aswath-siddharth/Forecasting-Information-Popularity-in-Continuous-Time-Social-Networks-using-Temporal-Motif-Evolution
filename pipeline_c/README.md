# Pipeline C: Cascade Virality Model

This folder implements the Pipeline C specification:

- Ingests single-file (`pipeline_a_features.csv`) or directory (`output_features/`) feature inputs.
- Automatically resolves contract post IDs against raw dataset JSONs and `pipeline_b_labels_full.csv`.
- Parses JSON sequence matrices into variable-length PyTorch tensors.
- Trains a Bidirectional GRU with learned attention pooling (`CascadeAttentionGRU`).
- Trains mean-pooling and max-pooling ablation baselines (`PoolingBaseline`).
- Writes checkpoints, loss curves, metadata, and evaluation reports.

## Setup

Install the Python dependencies:

```powershell
python -m pip install -r pipeline_c/requirements.txt
```

## Running Pipeline C

### 1. Sample Run (Single Subreddit)
```powershell
python -m pipeline_c.train --features pipeline_a_features.csv --labels pipeline_b_labels.csv --output-dir pipeline_c/outputs --epochs 50 --batch-size 64 --device cuda
```

### 2. Category-Specific Run (e.g. Sports, Technology, Gaming)
```powershell
python -m pipeline_c.train --features output_features --labels pipeline_b_labels_full.csv --category Sports --output-dir pipeline_c/outputs_sports --epochs 30 --batch-size 128 --device cuda
```

### 3. Full Dataset Run (All 4,050 Preprocessed Subreddits)
```powershell
python -m pipeline_c.train --features output_features --labels pipeline_b_labels_full.csv --output-dir pipeline_c/outputs_full --epochs 20 --batch-size 128 --device cuda
```

### CLI Options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--features` | `pipeline_a_features.csv` | Feature file path or directory (e.g. `output_features`). |
| `--labels` | `pipeline_b_labels.csv` | Labels file path (e.g. `pipeline_b_labels.csv` or `pipeline_b_labels_full.csv`). |
| `--data-dir` | `data` | Path to raw JSON files used for ID contract resolution. |
| `--category` | `None` | Optional category filter (e.g. `Sports`, `Technology`, `Gaming`, `Animals and Pets`). |
| `--max-files` | `None` | Limit number of subreddit CSV files to ingest. |
| `--sample-ratio` | `1.0` | Subsample fraction (0.0 to 1.0) of records. |
| `--epochs` | `50` | Number of training epochs. |
| `--batch-size` | `64` | Training batch size. |
| `--lr` | `1e-3` | Learning rate for AdamW optimizer. |
| `--device` | `None` (auto: CUDA/CPU) | Compute device. |

## Outputs

By default, outputs are written to the target `--output-dir`:

- `best_cascade_model.pt` (Best Attention-GRU checkpoint)
- `best_mean_pool_model.pt`
- `best_max_pool_model.pt`
- `training_history.csv`
- `loss_curves.png`
- `evaluation_report.json`
- `metadata.json`
