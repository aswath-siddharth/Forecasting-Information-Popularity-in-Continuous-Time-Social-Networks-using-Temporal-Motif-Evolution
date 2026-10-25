# Pipeline W — Published-Benchmark Validation (DeepHawkes Sina Weibo)

Runs the project's temporal-motif + BiGRU + attention architecture on the dataset
released with a published paper, so the model can be reported against numbers that
already exist in the literature.

**Paper:** Cao, Shen, Cen, Ouyang, Cheng. *DeepHawkes: Bridging the Gap between
Prediction and Understanding of Information Cascades.* CIKM 2017.
DOI [10.1145/3132847.3132973](https://doi.org/10.1145/3132847.3132973)

**Dataset:** `dataset_weibo.txt` — 119,313 Sina Weibo messages from 1 June 2016,
with timestamped retweet propagation paths and 24-hour retweet counts.

---

## Getting the data

The authors' own link is a **Baidu Pan** share
(`https://pan.baidu.com/s/1c2rnvJq`, password `ijp6`), which in practice needs a
Chinese phone number to download from.

The same `dataset_weibo.txt` is redistributed in the dataset bundle for **CCGL**
(Xu et al., TKDE 2022, <https://github.com/Xovee/ccgl>), which mirrors to Dropbox,
Google Drive and OneDrive. The Dropbox mirror needs no account:

```bash
curl -L -o weibo_data/datasets.zip \
  "https://www.dropbox.com/s/0kadkjyuwffcuw2/datasets.zip?dl=1"     # ~333 MB
unzip -j weibo_data/datasets.zip '*weibo*' -d weibo_data/
```

Cite the DeepHawkes paper for the dataset regardless of which mirror you pull it
from — the authors ask for that, and it is the paper we are benchmarking against.

## The honest framing (read before writing any slide)

DeepHawkes and CasCN publish results at **1-hour, 2-hour and 3-hour** observation
windows. We run at **15 minutes**. That is a strictly harder early-prediction
setting with far less observed information, so:

> ✅ "We use the publicly released Sina Weibo dataset from the published DeepHawkes
> paper and introduce a 15-minute early-observation setting to test whether our
> temporal-motif architecture can forecast 24-hour cascade popularity using
> substantially less initial information."

> ❌ "DeepHawkes used a 15-minute observation window." — they did not.

> ❌ Putting our 15-min MSLE next to their 1-hour MSLE as if the two are equivalent.

To produce a genuinely head-to-head row, re-run the parser with
`--observation-time 3600` and compare that against their published 1-hour number.
Both rows in one table is the strongest, most defensible presentation.

## Protocol fidelity

These filters are copied from `gen_sequence/gen_shortestpath.py` in the authors'
released code, so our cascade population matches theirs:

| Step | DeepHawkes | Ours |
| --- | --- | --- |
| Publish-hour filter | drop `hour <= 7 or hour >= 19` (Beijing) | same |
| Malformed nodes | drop paths containing node `-1` | same |
| Cascade size | keep `10 <= observed <= 1000` | same |
| Split | chronological by publish time, 14/20 · 3/20 · 3/20 | same |
| Training target | `log2(increment + 1)`, increment = 24h total − observed | same |
| Loss | MSE in log2 space (= MSLE) | same |
| **Observation window** | **3600 s** | **900 s** ← the only deliberate change |

Because the split is chronological and one cascade is one sequence, a cascade can
never appear in two splits — this avoids the record-level leakage that affected the
earlier Reddit run.

## Features — 15 dimensions

The Reddit pipeline uses 20 columns, four of which are VADER sentiment. The released
Weibo data carries **no message text**, so those four are dropped rather than
zero-filled, and the constant `window_index` column is dropped too (we only ever use
window 0). That leaves 15 live columns:

| Col | Feature | Col | Feature |
| --- | --- | --- | --- |
| 0–4 | motif one-hot: single, star, chain, dyad, cycle | 10 | `t_idle` |
| 5 | `tree_depth` | 11 | `descendant_count` (of parent) |
| 6 | `parent_out_degree` | 12 | `dt_motif` (motif completion speed) |
| 7 | `node_count` | 13 | `cumulative_prior` |
| 8 | `t_from_root` | 14 | `dt_edge` (inter-arrival gap) |
| 9 | `i_reactivate` | | |

Motifs are detected **causally** — evaluated against the graph state *before* the
current edge is inserted, so no feature can see its own effect.

## Running it

```bash
# 1. Build features from the raw dataset (15-minute window)
python -m pipeline_w.weibo_parser \
    --input weibo_data/dataset_weibo.txt \
    --output-dir weibo_data/processed_900 \
    --observation-time 900

# 2. Train + evaluate (Apple MPS is picked up automatically)
python -m pipeline_w.train_weibo \
    --data-dir weibo_data/processed_900 \
    --output-dir pipeline_w/outputs_900

# 3. For the directly-comparable 1-hour row
python -m pipeline_w.weibo_parser --input weibo_data/dataset_weibo.txt \
    --output-dir weibo_data/processed_3600 --observation-time 3600
python -m pipeline_w.train_weibo --data-dir weibo_data/processed_3600 \
    --output-dir pipeline_w/outputs_3600
```

## Metrics

Primary (regression, matches the published task):

- **MSLE (log2)** — the DeepHawkes objective verbatim; this is the comparable number.
- **MSLE (ln)** — same quantity in natural-log units, for papers that use ln.
- **MAE / RMSE** — on raw retweet counts.
- **R² / PCC** — in log2 space.

`mean_baseline` (always predict the training mean) is reported alongside. A model
that does not beat it has learned nothing, so it is the sanity floor for every run.

Secondary (classification — **our adaptation, not the DeepHawkes task**): the top-k%
cutoff is taken from the *training* split, and the decision threshold is tuned on the
*validation* split, never on test. Accuracy/Precision/Recall/F1 depend on that
threshold; ROC-AUC and PR-AUC do not.

## Files

- `weibo_parser.py` — raw `.txt` → motif/temporal features → `weibo_{train,val,test}.npz`
- `data.py` — dataset, train-split-only scaler, padding collate
- `model.py` — regression head on the same encoder as `pipeline_c`, plus pooling ablations
- `train_weibo.py` — training loop, metrics, report writer
