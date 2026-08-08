# Pipeline C: Cascade Virality Model

This folder implements the Pipeline C specification:

- joins `pipeline_a_features.csv` with `pipeline_b_labels.csv` on `post_id`
- parses JSON sequence matrices into variable-length tensors
- trains a bidirectional GRU with learned attention pooling
- trains mean-pooling and max-pooling ablation baselines
- writes checkpoints, loss curves, metadata, and evaluation reports

## Setup

Install the Python dependencies:

```powershell
python -m pip install -r pipeline_c/requirements.txt
```

## Expected Inputs

From the repository root:

- `pipeline_a_features.csv`
- `pipeline_b_labels.csv`

`pipeline_a_features.csv` must contain `post_id` and `X_seq_json`.
`pipeline_b_labels.csv` must contain `post_id` and `y_viral`.

## Run

```powershell
python -m pipeline_c.train --features pipeline_a_features.csv --labels pipeline_b_labels.csv --output-dir pipeline_c/outputs
```

Useful options:

```powershell
python -m pipeline_c.train --epochs 50 --batch-size 32 --hidden-size 64 --device cuda
```

If `--device` is omitted, the script uses CUDA when available and otherwise CPU.

## Outputs

By default, outputs are written to `pipeline_c/outputs/`:

- `best_cascade_model.pt`
- `best_mean_pool_model.pt`
- `best_max_pool_model.pt`
- `training_history.csv`
- `loss_curves.png`
- `evaluation_report.json`
- `metadata.json`

