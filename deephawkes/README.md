# DeepHawkes Controlled Baseline Comparison (`deephawkes/`)

Controlled reimplementation of **DeepHawkes** (Cao et al., *ACM CIKM 2017*) in PyTorch, evaluated head-to-head against **CasMotif** (`pipeline_w/`) on the exact same Sina Weibo splits, targets, loss, and training harness — without modifying `pipeline_w/`.

## Controlled Protocol (Held Identical Across Both Models)

| Category | Setting |
| :--- | :--- |
| **Splits** | Strictly chronological **70% / 15% / 15%** aligned 1-to-1 to `weibo_{train,val,test}.npz` (`29,530 / 6,328 / 6,328` at 1h; `16,509 / 3,538 / 3,538` at 15m) |
| **Windows & Horizon** | $T_\text{obs} \in \{3600\text{s (1h)},\; 900\text{s (15m)}\} \to T_\text{pred} = 86,400\text{s (24h)}$ |
| **Target** | Future increment only: $\Delta P = R_{24\text{h}} - R_{T_\text{obs}}$, transformed as $\log_2(\Delta P + 1)$ |
| **Loss & Primary Metric** | $\text{MSLE}_{\log_2}$ (MSE in $\log_2$ space) |
| **Batch Size** | `64` |
| **Optimizer** | `AdamW` (`lr = 1e-3`, `weight_decay = 1e-4`; DeepHawkes sparse embedding table uses `0.1×` LR + active-batch L2 `0.01`, matching `CaoQi92/DeepHawkes`) |
| **LR Schedule** | `ReduceLROnPlateau(factor=0.5, patience=3)` |
| **Gradient Clipping** | `max_norm = 1.0` |
| **Early Stopping** | `patience = 8`, `max_epochs = 60`, restoring best validation MSLE checkpoint |
| **Seeds & Zero Leakage** | Seeds `42, 43, 44` (mean ± std + paired $t$-test / Wilcoxon / 95% bootstrap CI); `SequenceScaler` (CasMotif) and user vocabulary (DeepHawkes) fitted on the training split only |

## Files

- `data.py` — Extracts diffusion paths aligned to existing `weibo_{train,val,test}.npz` cascade IDs and builds a train-only user vocabulary.
- `model.py` — PyTorch port of `CaoQi92/DeepHawkes/deep_learning/model_sparse.py` (`Uniform(0, 1)` 50-D user embeddings + 0.2 dropout, 64-unit GRU path encoder, 6 equal non-parametric time-decay intervals, additive sum pooling, `64 -> 32 -> 16 -> 1` ReLU MLP).
- `train.py` — Training & evaluation harness under the controlled protocol.
- `compare.py` — Asserts exact cascade-ID & target alignment across all seeds of both models and computes summary tables + paired statistical tests.
- `run_benchmark.py` — Single-command end-to-end runner for 1h and 15m across seeds `42, 43, 44`.

## Running

```powershell
# Full benchmark (1h + 15m, seeds 42, 43, 44 for DeepHawkes + CasMotif + paired tests)
python -m deephawkes.run_benchmark

# Or 1h only first:
python -m deephawkes.run_benchmark --windows 3600
```
