# PIPELINE C SPECIFICATION OVERVIEW

- **Inputs:** `pipeline_a_features.csv` (contains `post_id` and K x 20 sequence matrices) and `pipeline_b_labels.csv` (contains `post_id` and `y_viral` ground-truth targets).
- **Outputs:** Trained PyTorch model checkpoint (`best_cascade_model.pt`), loss curves, and evaluation report (ROC-AUC, F1-Score, Precision-Recall).
- **Core Architecture:** Sequence Input [B, K, 20] -> Bidirectional GRU -> Learned Attention Pooling -> Bottleneck MLP -> Binary Logit

---

### TASK 1: Dataset Merging, Parsing & PyTorch Ingestion

#### **SUBTASK 1.1: CSV Inner-Join Engine**
* **Objective:** Merge feature matrices and ground-truth targets into a single aligned dataset while dropping unmatched records.
* **Procedure:**
  1. Load `pipeline_a_features.csv` and deserialize the JSON-encoded X_seq strings back into K x 20 floating-point lists/arrays.
  2. Load `pipeline_b_labels.csv`.
  3. Perform an Inner Join on `post_id` (`pd.merge(df_a, df_b, on='post_id', how='inner')`).
  4. Log join statistics (e.g., number of successfully paired cascades vs. dropped orphans).

#### **SUBTASK 1.2: Custom CascadeDataset Implementation**
* **Objective:** Wrap joined data in a native `torch.utils.data.Dataset` class to handle tensor conversion on demand.
* **Procedure:**
  1. Define `__len__` returning total merged cascade count N.
  2. Define `__getitem__(idx)` returning tuple (X_i, y_i):
     - Convert X_i into `torch.FloatTensor` of shape [K_i, 20] (where K_i is variable comment sequence length).
     - Convert y_i into `torch.FloatTensor` of shape [1].

#### **SUBTASK 1.3: Dynamic Sequence Padding & Collate Function (`pad_collate_fn`)**
* **Objective:** Batch variable-length K_i x 20 matrices into uniform 3D tensors and generate corresponding boolean padding masks.
* **Procedure:**
  1. Inspect batch to find `K_max = max({K_1, K_2, ..., K_B})`.
  2. Pad sequence matrices with zero vectors (0 in R^20) up to K_max using `torch.nn.utils.rnn.pad_sequence`.
  3. Construct a Boolean Padding Mask M in B^(B x K_max):
     - M[b, j] = False if position j contains a valid comment
     - M[b, j] = True if position j is padded zero-fill
  4. Return batched tuple: `padded_X` [B, K_max, 20], `padding_mask` [B, K_max], and `labels` [B, 1].

---

### TASK 2: Hybrid GRU + Learned Attention Architecture Construction

#### **SUBTASK 2.1: Temporal Sequence Encoding Layer (GRU)**
* **Objective:** Capture sequential, chronological narrative dynamics across the comment stream using a fast Gated Recurrent Unit.
* **Mathematical Definition:**
  `H = GRU(X_batch)` where `H in R^(B x K_max x 2*D_hidden)`
* **Procedure:**
  1. Pass padded batch [B, K_max, 20] through a 1- or 2-layer Bidirectional GRU (`batch_first=True`).
  2. Apply `torch.nn.utils.rnn.pack_padded_sequence` before GRU and `pad_packed_sequence` after GRU to optimize GPU computation over padded regions.
  3. Output sequence hidden states H of shape [B, K_max, H_dim] (where `H_dim = 2 * hidden_size`).

#### **SUBTASK 2.2: Learnable Attentive Pooling Module**
* **Objective:** Compress variable-length hidden sequences into a rich, fixed-size summary vector c in R^H_dim by scoring the importance of every edge without O(K^2) matrix multiplications.
* **Mathematical Definition:**
  Given sequence hidden states h_j in R^H_dim for comments j in {1..K_max}:
  `s_j = w_2^T * tanh(W_1 * h_j + b_1) + b_2`
  `alpha_j = exp(s_j) / sum_{k=1..K} exp(s_k)` (with mask M[b, j] set to -1e9 prior to softmax)
  `c_b = sum_{j=1..K_max} (alpha_j * h_j)` in R^H_dim
* **Procedure:**
  1. Pass H through linear projection layers to compute scalar score logits s_j.
  2. Mask padded positions by setting `s[b, j] = -1e9` where `M[b, j] = True`.
  3. Compute normalized attention weights alpha_j via `torch.softmax`.
  4. Perform weighted batch matrix multiplication to yield fixed summary vector c in R^(B x H_dim).

#### **SUBTASK 2.3: Bottleneck MLP Classification Head**
* **Objective:** Map non-linear feature interactions from context vector c to a final scalar virality score.
* **Architecture Layout:**
  `c in R^H_dim -> Dense(H_dim -> 64) + ReLU -> z_1 -> Dropout(0.3) + Dense(64 -> 32) + ReLU -> z_2 -> Dense(32 -> 1) -> y_hat_logit`
* **Procedure:**
  1. Add Dropout (p = 0.3) between dense layers for regularization.
  2. Output raw unnormalized logit tensor `y_hat_logit` in R^(B x 1).

---

### TASK 3: Training Loop & Optimization Setup

#### **SUBTASK 3.1: Loss Function & Imbalance Handling**
* **Objective:** Set up numerically stable binary classification loss with class-imbalance weighting.
* **Procedure:**
  1. Calculate positive class weighting factor `pos_weight` from Pipeline B distribution:
     `pos_weight = N_negative / N_positive`
  2. Instantiate `torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)` to prevent gradient dominance by the non-viral majority class.

#### **SUBTASK 3.2: Optimizer & Learning Rate Schedule**
* **Objective:** Enable stable convergence across GRU, Attention, and MLP layers.
* **Procedure:**
  1. Instantiate `AdamW` optimizer with learning rate eta = 1e-3 and weight decay lambda = 1e-4.
  2. Implement learning rate scheduler `CosineAnnealingLR` or `ReduceLROnPlateau` monitoring validation loss.

#### **SUBTASK 3.3: Stratified Splitting & Validation Pipeline**
* **Objective:** Train and evaluate without data contamination.
* **Procedure:**
  1. Perform an 80 / 10 / 10 Stratified Train / Validation / Test split based on `y_viral`.
  2. Implement training loop executing forward pass, loss calculation, backpropagation (`loss.backward()`), gradient clipping (`torch.nn.utils.clip_grad_norm_(max_norm=1.0)`), and optimizer step.
  3. Save model weights whenever validation loss reaches a new minimum (Early Stopping with patience = 10 epochs).

---

### TASK 4: Downstream Evaluation & Metrics Logging

#### **SUBTASK 4.1: Discrimination Performance Metrics**
* **Objective:** Evaluate classifier performance on held-out test dataset.
* **Procedure:**
  1. Pass test batch through model and apply sigmoid: `p_hat = sigmoid(y_hat_logit)`.
  2. Compute ROC-AUC (Area Under ROC Curve).
  3. Compute PR-AUC (Precision-Recall Area Under Curve - essential for imbalanced cascade data).
  4. At threshold tau = 0.5, compute F1-Score, Precision, and Recall.

#### **SUBTASK 4.2: Ablation Study Execution**
* **Objective:** Validate that GRU + Attentive Pooling outperforms simple baselines.
* **Procedure:**
  1. Train Baseline 1: Mean Pooling + MLP.
  2. Train Baseline 2: Max Pooling + MLP.
  3. Compare test set ROC-AUC across models to quantify performance gains achieved by learned attention.

#### **SUBTASK 4.3: Model Checkpoint Export**
* **Objective:** Export model weights and hyperparameter config for production inference.
* **Procedure:**
  1. Save final state dictionary to `best_cascade_model.pt`.
  2. Save metadata JSON containing input feature dimensions, layer sizes, and threshold metrics.
