# Controlled Reddit Benchmark: CasMotif vs DeepHawkes (`reddit_benchmark/`)

Controlled multi-seed evaluation of **CasMotif** (15-D causal temporal motifs + attentive
Bi-GRU) against **DeepHawkes** (Cao et al., CIKM 2017) on the novel multi-subreddit Reddit
comment-cascade dataset (`data/`, 4,050 subreddits across 13 categories).

All existing folders (`pipeline_a/`, `pipeline_b/`, `pipeline_c/`, `pipeline_w/`, `deephawkes/`)
are left untouched.

## Controlled Protocol (Identical to the Sina Weibo Benchmark)

- **Splits:** Strictly chronological `70% / 15% / 15%` by `post_time` ($t_0$), aligned 1-to-1 between CasMotif and DeepHawkes.
- **Filter:** `10 <= observed_size <= 1000` (root post + comments with $0 \le \Delta t < T_\text{obs}$; deleted/bot accounts excluded while preserving child subtrees).
- **Windows & Horizon:** $T_\text{obs} \in \{3600\text{s (1 h)},\; 900\text{s (15 m)}\} \to T_\text{pred} = 86,400\text{s (24 h)}$.
- **Target & Loss:** Future increment $\Delta P = R_{24\text{h}} - R_{T_\text{obs}}$, $\log_2(\Delta P + 1)$, trained & evaluated with $\text{MSLE}_{\log_2}$.
- **Optimization Harness:** Batch size `64`, `AdamW(lr=1e-3, weight_decay=1e-4)`, `ReduceLROnPlateau(factor=0.5, patience=3)`, grad clip `1.0`, early stop `patience=8` / max `60` epochs, best validation checkpoint restored.
- **Seeds & Zero Leakage:** Seeds `42, 43, 44` + paired $t$-test / Wilcoxon / 10,000-sample bootstrap 95% CI; `SequenceScaler` (CasMotif) and user vocabulary (DeepHawkes) fitted strictly on the training split only.

## Running

```bash
# Full end-to-end run (parses data/ in parallel, runs 1h + 15m across seeds 42, 43, 44, compares, plots chart)
python -m reddit_benchmark.run_benchmark
```
