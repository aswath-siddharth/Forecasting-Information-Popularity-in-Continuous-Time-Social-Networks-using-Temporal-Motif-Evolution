# Controlled Multi-Seed Comparison: CasMotif vs DeepHawkes (Novel Reddit Dataset)

Head-to-head controlled evaluation of **CasMotif** (15-D causal temporal motifs + attentive
Bi-GRU, `49,824` params) against **DeepHawkes** (Cao et al., ACM CIKM 2017) on our novel
multi-subreddit Reddit comment-cascade dataset (`data/`, **4,050 subreddits across 13
categories**, **1,200,698 posts scanned**).

---

## Dataset Characteristics & Controlled Protocol

| Dimension | 1-Hour Window (`3600s`) | 15-Minute Window (`900s`) |
| :--- | :---: | :---: |
| **Total Cascades (`10 ≤ N_obs ≤ 1000`)** | **246,232** | **33,551** |
| **Chronological Split (`70 / 15 / 15` by $t_0$)** | `172,362 / 36,934 / 36,936` | `23,485 / 5,032 / 5,034` |
| **Target & Loss** | $\Delta P = R_{24\text{h}} - R_{T_\text{obs}}$, $\text{MSLE}_{\log_2}$ | Same |
| **Optimization Harness** | Batch `64`, `AdamW(lr=1e-3, wd=1e-4)`, `ReduceLROnPlateau(0.5, patience=3)`, grad clip `1.0`, early stop `8` / max `60` epochs, best-val ckpt | Same |
| **Seeds** | `42, 43, 44` (+ 3-model ensemble) | `42, 43, 44` (+ 3-model ensemble) |
| **Train-Only DeepHawkes User Vocab** | `885,488` users (`44.30M` total params) | `140,978` users (`7.07M` total params) |
| **Known User Token Rate (Train / Val / Test)** | `94.3% / 37.6% / 26.9%` | `89.8% / 45.7% / 25.6%` |
| **Train Motif Rates** (`star / chain / dyad / cycle / single`) | `72.4% / 19.5% / 15.1% / 6.1% / 14.0%` | `77.8% / 12.3% / 8.1% / 3.6% / 13.3%` |

> **Dataset Note (200-comment scraping cap):** Raw threads in `data/` are capped at ~200 scraped comments (`max ΔP = 191`), truncating the upper popularity tail relative to Sina Weibo (`max ΔP = 36,876`). Consequently, baseline variance (`MSLE = 0.834` at 1 h, `0.635` at 15 m) and absolute $R^2$ ceilings are smaller than on Weibo; both models face the exact same targets and cap.

---

## Results — 1-Hour Observation Window (`3600s`, 36,936 identical test cascades)

| Model | Params (Total / Non-Emb) | MSLE ($\log_2$) $\downarrow$ | MSLE ($\ln$) $\downarrow$ | $R^2$ ($\log_2$) $\uparrow$ | PCC $\uparrow$ | MAPE ($\log_2$) $\downarrow$ | MAE $\downarrow$ | MedAE $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Mean Baseline | 0 | 0.8342 | 0.4008 | -0.0005 | — | 0.1251 | 46.64 | 47.96 |
| DeepHawkes, seed 42 | 44,299,303 / 24,903 | 0.7102 | 0.3412 | 0.1483 | 0.4181 | 0.1080 | 38.22 | 34.06 |
| DeepHawkes, seed 43 | 44,299,303 / 24,903 | 0.7191 | 0.3455 | 0.1376 | 0.4116 | 0.1099 | 39.20 | 36.89 |
| DeepHawkes, seed 44 | 44,299,303 / 24,903 | 0.7000 | 0.3363 | 0.1605 | 0.4328 | 0.1069 | 37.74 | 33.54 |
| **DeepHawkes (3-seed mean)** | **44.30M / 24.9k** | **0.7098 ± 0.0096** | **0.3410 ± 0.0046** | **0.1488 ± 0.0115** | **0.4209** | **0.1083** | **38.39 ± 0.75** | **34.83** |
| CasMotif, seed 42 | 49,824 | 0.6310 | 0.3032 | 0.2433 | 0.5034 | 0.1032 | 37.07 | 33.59 |
| CasMotif, seed 43 | 49,824 | 0.6165 | 0.2962 | 0.2607 | 0.5134 | 0.1009 | 36.53 | 31.79 |
| CasMotif, seed 44 | 49,824 | 0.6196 | 0.2977 | 0.2569 | 0.5122 | 0.1016 | 36.70 | 32.78 |
| **CasMotif (3-seed mean)** | **49,824** | **0.6224 ± 0.0076** | **0.2990 ± 0.0037** | **0.2536 ± 0.0091** | **0.5096** | **0.1019** | **36.77 ± 0.28** | **32.72** |
| DeepHawkes (3-model ensemble) | 132.9M / 74.7k | 0.7015 | 0.3370 | 0.1587 | 0.4314 | 0.1076 | 38.15 | 34.67 |
| **CasMotif (3-model ensemble)** | **149,472** | **0.6173** | **0.2966** | **0.2597** | **0.5170** | **0.1016** | **36.67** | **32.70** |

### Paired Significance Tests (1 h, $n = 36,936$, 10,000-sample bootstrap)
| Comparison | $\Delta\text{MSLE}$ (Ours − DH) | Relative Reduction | 95% Bootstrap CI | Paired $t$-test $p$ | Wilcoxon $p$ | CasMotif Better On |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **3-seed mean per-cascade error** | **−0.0874** | **12.31%** | **[−0.0937, −0.0812]** | **$1.04 \times 10^{-163}$** | **$1.62 \times 10^{-71}$** | **53.8%** |
| **3-model ensemble vs ensemble** | **−0.0842** | **12.01%** | **[−0.0905, −0.0780]** | **$3.83 \times 10^{-153}$** | **$9.49 \times 10^{-42}$** | **51.7%** |
| Seed 42 single run | −0.0792 | 11.16% | [−0.0858, −0.0727] | $4.54 \times 10^{-125}$ | $1.74 \times 10^{-46}$ | 50.9% |

---

## Results — 15-Minute Ultra-Early Window (`900s`, 5,034 identical test cascades)

| Model | Params (Total / Non-Emb) | MSLE ($\log_2$) $\downarrow$ | MSLE ($\ln$) $\downarrow$ | $R^2$ ($\log_2$) $\uparrow$ | PCC $\uparrow$ | MAPE ($\log_2$) $\downarrow$ | MAE $\downarrow$ | MedAE $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Mean Baseline | 0 | 0.6350 | 0.3051 | -0.0006 | — | 0.0990 | 46.12 | 48.53 |
| DeepHawkes, seed 42 | 7,073,803 / 24,903 | 0.5844 | 0.2808 | 0.0792 | 0.3080 | 0.0895 | 40.14 | 39.65 |
| DeepHawkes, seed 43 | 7,073,803 / 24,903 | 0.5855 | 0.2813 | 0.0774 | 0.2964 | 0.0905 | 41.00 | 41.48 |
| DeepHawkes, seed 44 | 7,073,803 / 24,903 | 0.5849 | 0.2810 | 0.0783 | 0.2937 | 0.0913 | 41.66 | 42.68 |
| **DeepHawkes (3-seed mean)** | **7.07M / 24.9k** | **0.5849 ± 0.0006** | **0.2810 ± 0.0003** | **0.0783 ± 0.0009** | **0.2994** | **0.0904** | **40.93 ± 0.76** | **41.27** |
| CasMotif, seed 42 | 49,824 | 0.5734 | 0.2755 | 0.0965 | 0.3186 | 0.0871 | 38.93 | 37.08 |
| CasMotif, seed 43 | 49,824 | 0.5754 | 0.2764 | 0.0934 | 0.3161 | 0.0880 | 39.08 | 36.09 |
| CasMotif, seed 44 | 49,824 | 0.5688 | 0.2733 | 0.1037 | 0.3277 | 0.0872 | 39.06 | 36.94 |
| **CasMotif (3-seed mean)** | **49,824** | **0.5725 ± 0.0034** | **0.2751 ± 0.0016** | **0.0979 ± 0.0053** | **0.3208** | **0.0875** | **39.02 ± 0.08** | **36.70** |
| DeepHawkes (3-model ensemble) | 21.2M / 74.7k | 0.5843 | 0.2807 | 0.0793 | 0.3005 | 0.0904 | 40.90 | 41.34 |
| **CasMotif (3-model ensemble)** | **149,472** | **0.5694** | **0.2736** | **0.1027** | **0.3275** | **0.0871** | **38.86** | **36.84** |

### Paired Significance Tests (15 m, $n = 5,034$, 10,000-sample bootstrap)
| Comparison | $\Delta\text{MSLE}$ (Ours − DH) | Relative Reduction | 95% Bootstrap CI | Paired $t$-test $p$ | Wilcoxon $p$ | CasMotif Better On |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **3-seed mean per-cascade error** | **−0.0124** | **2.12%** | **[−0.0222, −0.0027]** | **$0.0125$** | **$6.75 \times 10^{-11}$** | **56.8%** |
| **3-model ensemble vs ensemble** | **−0.0149** | **2.54%** | **[−0.0246, −0.0052]** | **$0.00276$** | **$7.44 \times 10^{-14}$** | **57.4%** |
| Seed 42 single run | −0.0110 | 1.88% | [−0.0223, +0.0005] | 0.0550 | $4.74 \times 10^{-3}$ | 53.5% |

---

## Cross-Platform Synthesis (Sina Weibo + Novel Reddit)

| Platform & Window | Test Cascades | DeepHawkes MSLE ($R^2$) | **CasMotif MSLE ($R^2$)** | MSLE Reduction | Paired $t$-test $p$ | Parameter Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sina Weibo — 1 h** | 6,328 | `2.6167` (`0.505`) | **`1.9446` (`0.632`)** | **−25.68%** | $2.7 \times 10^{-29}$ | **300× smaller** |
| **Sina Weibo — 15 m** | 3,538 | `4.2737` (`0.257`) | **`2.4494` (`0.574`)** | **−42.69%** | $1.4 \times 10^{-27}$ | **111× smaller** |
| **Reddit (4,050 subs) — 1 h** | 36,936 | `0.7098` (`0.149`) | **`0.6224` (`0.254`)** | **−12.31%** | $1.0 \times 10^{-163}$ | **889× smaller** |
| **Reddit (4,050 subs) — 15 m** | 5,034 | `0.5849` (`0.078`) | **`0.5725` (`0.098`)** | **−2.12%** | $1.25 \times 10^{-2}$ | **142× smaller** |

CasMotif generalizes across **both** microblogging retweet cascades (Sina Weibo) and multi-community discussion trees (Reddit), outperforming DeepHawkes on **every metric across all 4 controlled benchmarks** while remaining a fixed **49,824-parameter (`0.20 MB`) inductive model** even as the user base scales to `1.56M` unique users.

