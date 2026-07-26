"""
SEPAGen - Phase 4: Transformer Model
======================================
GPT-style autoregressive Transformer for SEPA payment
sequence modelling and APP fraud anomaly detection.

Architecture:
  TransactionEmbedding → PositionalEncoding →
  N × CausalTransformerBlock → OutputProjection → AnomalyScorer

Run this in Google Colab with GPU runtime (T4 or better).

Usage:
  python phase4_transformer.py

Requirements:
  pip install torch numpy pandas scikit-learn matplotlib
"""

import json
import math
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

# Model hyperparameters
D_MODEL    = 64     # embedding dimension
N_HEADS    = 4      # attention heads (D_MODEL must be divisible by N_HEADS)
N_LAYERS   = 2      # number of transformer blocks
D_FF       = 256    # feed forward hidden dimension
SEQ_LEN    = 20     # transactions of context
DROPOUT    = 0.1    # dropout rate

# Training hyperparameters
BATCH_SIZE    = 256
LEARNING_RATE = 3e-4
N_EPOCHS      = 10
WARMUP_STEPS  = 1000

# Tokeniser settings (must match Phase 3)
VOCAB_SIZE    = 66    # 65 + PAD token
PAD_TOKEN     = 65
TOKENS_PER_TXN = 8

# Files
TOKENISER_FILE = "tokeniser.json"
TRAIN_FILE     = "sequences_train.csv"
VAL_FILE       = "sequences_val.csv"
TEST_FILE      = "sequences_test.csv"
MODEL_SAVE     = "sepagen_model.pt"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
print(f"Model config: d_model={D_MODEL}, n_heads={N_HEADS}, "
      f"n_layers={N_LAYERS}, d_ff={D_FF}")


# ─────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────

class SEPADataset(Dataset):
    """
    PyTorch Dataset for SEPA transaction sequences.

    Loads sequences from CSV file generated in Phase 3.
    Each sample:
      input_tokens:  (SEQ_LEN, TOKENS_PER_TXN) — context window
      target_tokens: (TOKENS_PER_TXN,)          — next transaction
      is_fraud:      int                         — label (eval only)
    """

    def __init__(self, csv_file, max_samples=None):
        """
        Args:
            csv_file:    path to sequences_*.csv from Phase 3
            max_samples: optionally limit dataset size (for quick testing)
        """
        print(f"Loading {csv_file}...")
        self.inputs   = []
        self.targets  = []
        self.is_fraud = []
        self.metadata = []

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_samples and i >= max_samples:
                    break
                self.inputs.append(json.loads(row["input_tokens"]))
                self.targets.append(json.loads(row["target_tokens"]))
                self.is_fraud.append(int(row["is_fraud"]))
                self.metadata.append({
                    "account_id": row["account_id"],
                    "timestamp":  row["timestamp"],
                    "fraud_type": row["fraud_type"],
                    "amount":     row["amount"],
                })

        self.inputs   = torch.tensor(self.inputs,   dtype=torch.long)
        self.targets  = torch.tensor(self.targets,  dtype=torch.long)
        self.is_fraud = torch.tensor(self.is_fraud, dtype=torch.long)

        fraud_count = self.is_fraud.sum().item()
        print(f"  Loaded {len(self):,} sequences  "
              f"(fraud={fraud_count:,}  "
              f"normal={len(self)-fraud_count:,})")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input":    self.inputs[idx],     # (SEQ_LEN, 8)
            "target":   self.targets[idx],    # (8,)
            "is_fraud": self.is_fraud[idx],   # scalar
        }


# ─────────────────────────────────────────
# MODEL COMPONENTS
# ─────────────────────────────────────────

class TransactionEmbedding(nn.Module):
    """
    Converts 8 tokens per transaction into a single embedding vector.

    Each of the 8 fields (amount, country, category, etc.) has its
    own embedding lookup. We sum all 8 embeddings to get one vector
    of size d_model representing the full transaction.

    Why sum (not concatenate)?
      - Concatenating 8 × d_model/8 vectors gives same dimension
        but summing is simpler and works well in practice
      - Each field embedding learns to capture its own signal
      - Summation lets the model combine signals flexibly

    Shape:
      input:  (batch, seq_len, 8)   — 8 token IDs per transaction
      output: (batch, seq_len, d_model) — one vector per transaction
    """

    def __init__(self, vocab_size, d_model, pad_token):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = d_model,
            padding_idx    = pad_token,   # PAD tokens get zero embedding
        )
        self.d_model   = d_model
        self.pad_token = pad_token

    def forward(self, x):
        # x: (batch, seq_len, 8)
        embedded = self.embedding(x)          # (batch, seq_len, 8, d_model)
        summed   = embedded.sum(dim=-2)       # (batch, seq_len, d_model)
        # Scale by sqrt(d_model) — standard practice from "Attention is All You Need"
        return summed * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Adds a unique position fingerprint to each transaction in the sequence.

    Without this, the Transformer has no sense of order — transaction 1
    and transaction 20 look identical to the attention mechanism.

    We use sinusoidal encoding from "Attention is All You Need":
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Each position gets a unique vector of sine/cosine values.
    Nearby positions have similar encodings; distant positions differ.

    Shape:
      input:  (batch, seq_len, d_model)
      output: (batch, seq_len, d_model)  — same, with position added
    """

    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build positional encoding matrix — shape (max_len, d_model)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()   # (max_len, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)   # even dimensions
        pe[:, 1::2] = torch.cos(pos * div)   # odd dimensions

        # Register as buffer (not a parameter — not updated by optimiser)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal masking.

    Causal = each position can only attend to itself and previous positions.
    This ensures the model can't "cheat" by looking at future transactions.

    The attention formula:
      Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) × V

    With causal mask applied before softmax:
      future positions → -infinity → softmax → 0 attention weight

    Shape:
      input:  (batch, seq_len, d_model)
      output: (batch, seq_len, d_model)
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads   # dimension per head

        # Q, K, V projections (combined into one matrix for efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape   # batch, seq_len, d_model

        # Compute Q, K, V for all heads in one matrix multiply
        qkv = self.qkv_proj(x)                       # (B, T, 3*d_model)
        q, k, v = qkv.split(self.d_model, dim=-1)    # each: (B, T, d_model)

        # Reshape for multi-head: (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        scale  = math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, n_heads, T, T)

        # Causal mask — upper triangle → -infinity
        # This prevents attending to future transactions
        mask   = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
        ).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        # Softmax → attention weights
        attn_weights = F.softmax(scores, dim=-1)     # (B, n_heads, T, T)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)           # (B, n_heads, T, d_head)

        # Merge heads back
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, d_model)
        return self.out_proj(out), attn_weights


class FeedForward(nn.Module):
    """
    Position-wise feed forward network.
    Applied independently to each position (transaction) after attention.

    Structure: Linear → GELU → Dropout → Linear
    Expands to d_ff then projects back to d_model.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CausalTransformerBlock(nn.Module):
    """
    One Transformer block = Attention + FeedForward + LayerNorm.

    Structure (Pre-LN variant — more stable training):
      x → LayerNorm → Attention → residual add
      x → LayerNorm → FeedForward → residual add

    Residual connections (x + sublayer(x)) prevent vanishing gradients
    and allow the model to learn incrementally on top of the identity.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, dropout)
        self.ff    = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, return_attn=False):
        # Attention with residual
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out

        # Feed forward with residual
        x = x + self.ff(self.norm2(x))

        if return_attn:
            return x, attn_weights
        return x, None


class SEPAGenModel(nn.Module):
    """
    Full SEPAGen Transformer model.

    Architecture:
      1. TransactionEmbedding  — 8 tokens → d_model vector per transaction
      2. PositionalEncoding    — add position fingerprint
      3. N × TransformerBlock  — learn temporal patterns via attention
      4. LayerNorm             — final normalisation
      5. OutputProjection      — d_model → vocab_size logits per token

    Training objective:
      For each position in the sequence, predict the next transaction's tokens.
      Loss = average cross-entropy across all 8 token predictions.

    Inference:
      Feed a sequence of N transactions → get log-likelihood of transaction N+1.
      Low log-likelihood = anomalous = potential fraud.
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers,
                 d_ff, seq_len, pad_token, dropout=0.1):
        super().__init__()

        self.embedding = TransactionEmbedding(vocab_size, d_model, pad_token)
        self.pos_enc   = PositionalEncoding(d_model, max_len=seq_len+1,
                                            dropout=dropout)
        self.blocks    = nn.ModuleList([
            CausalTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm      = nn.LayerNorm(d_model)

        # Output projection: d_model → vocab_size logits
        # Predicts probability of each possible token value
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights: output projection shares weights with embedding
        # (common practice in language models — saves parameters)
        self.output_proj.weight = self.embedding.embedding.weight

        # Initialise weights
        self.apply(self._init_weights)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"SEPAGenModel: {total_params:,} parameters")

    def _init_weights(self, module):
        """Standard GPT weight initialisation."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, return_attn=False):
        """
        Forward pass.

        Args:
            x: (batch, seq_len, 8) — input token sequences
            return_attn: if True, return attention weights for explainability

        Returns:
            logits: (batch, seq_len, vocab_size) — token probabilities
            attn_weights: list of attention weight tensors (if return_attn)
        """
        # Embed all 8 tokens per transaction → one vector
        x = self.embedding(x)     # (batch, seq_len, d_model)

        # Add positional encoding
        x = self.pos_enc(x)       # (batch, seq_len, d_model)

        # Pass through transformer blocks
        all_attn = []
        for block in self.blocks:
            x, attn = block(x, return_attn=return_attn)
            if return_attn and attn is not None:
                all_attn.append(attn)

        # Final normalisation
        x = self.norm(x)          # (batch, seq_len, d_model)

        # Project to vocabulary
        logits = self.output_proj(x)  # (batch, seq_len, vocab_size)

        return logits, all_attn


# ─────────────────────────────────────────
# ANOMALY SCORER
# ─────────────────────────────────────────

class AnomalyScorer:
    """
    Computes anomaly scores for transactions using the trained model.

    Score = mean log-likelihood of the transaction's 8 tokens
            given the preceding sequence of normal transactions.

    Low score → transaction is unlikely given the model's learned
                distribution of normal behaviour → potential fraud.

    Higher score = more normal
    Lower score  = more anomalous
    """

    def __init__(self, model, device):
        self.model  = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def score_batch(self, inputs, targets):
        """
        Compute anomaly scores for a batch of sequences.

        Args:
            inputs:  (batch, seq_len, 8) — context transactions
            targets: (batch, 8)          — transaction to score

        Returns:
            scores: (batch,) — log-likelihood per sequence
                    Higher = more normal, Lower = more anomalous
        """
        inputs  = inputs.to(self.device)
        targets = targets.to(self.device)

        # Get logits from model
        # We use the LAST position's output to score the target
        # (the model's prediction of what comes after the full context)
        logits, _ = self.model(inputs)          # (batch, seq_len, vocab_size)
        last_logits = logits[:, -1, :]           # (batch, vocab_size)

        # Compute log-probability for each of the 8 target tokens
        log_probs = F.log_softmax(last_logits, dim=-1)  # (batch, vocab_size)

        # For each token field in the target, get its log-probability
        # targets shape: (batch, 8) — 8 token IDs
        # We average log-prob across all 8 tokens
        token_log_probs = []
        for field_idx in range(targets.shape[1]):
            field_tgt = targets[:, field_idx]              # (batch,)
            field_lp  = log_probs.gather(
                1, field_tgt.unsqueeze(1)
            ).squeeze(1)                                   # (batch,)
            token_log_probs.append(field_lp)

        # Stack and average: mean log-likelihood across all 8 fields
        stacked = torch.stack(token_log_probs, dim=1)     # (batch, 8)
        scores  = stacked.mean(dim=1)                     # (batch,)

        return scores.cpu().numpy()


# ─────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────

def compute_loss(model, batch, device):
    """
    Compute cross-entropy loss for a batch.

    For each position in the input sequence, the model predicts
    the tokens at the NEXT position. This is autoregressive training.

    input:   sequence positions [0, 1, ..., T-1]
    predict: sequence positions [1, 2, ..., T]

    We shift by 1: input[:-1] → predict input[1:]
    """
    inputs  = batch["input"].to(device)    # (B, seq_len, 8)
    targets = batch["input"].to(device)    # same — we predict next token

    # Autoregressive: use positions 0..T-2 to predict positions 1..T-1
    inp = inputs[:, :-1, :]    # (B, seq_len-1, 8) — context
    tgt = targets[:, 1:, :]    # (B, seq_len-1, 8) — next transactions

    logits, _ = model(inp)     # (B, seq_len-1, vocab_size)

    B, T, V = logits.shape

    # Compute loss for each of the 8 token fields
    total_loss = 0.0
    for field_idx in range(TOKENS_PER_TXN):
        field_logits  = logits                        # (B, T, V)
        field_targets = tgt[:, :, field_idx]          # (B, T)

        # Cross-entropy loss
        loss = F.cross_entropy(
            field_logits.view(B * T, V),
            field_targets.reshape(B * T),
            ignore_index=PAD_TOKEN,     # ignore PAD tokens in loss
        )
        total_loss += loss

    return total_loss / TOKENS_PER_TXN   # average across fields


def train_epoch(model, loader, optimiser, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches  = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        optimiser.zero_grad()

        loss = compute_loss(model, batch, device)
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimiser.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        n_batches  += 1

        # Progress every 200 batches
        if (batch_idx + 1) % 200 == 0:
            avg_loss   = total_loss / n_batches
            elapsed    = time.time() - start_time
            lr         = optimiser.param_groups[0]["lr"]
            print(f"  Epoch {epoch} | batch {batch_idx+1:,}/{len(loader):,} | "
                  f"loss={avg_loss:.4f} | lr={lr:.2e} | {elapsed:.0f}s")

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, device):
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        loss       = compute_loss(model, batch, device)
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches


# ─────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────

@torch.no_grad()
def evaluate_fraud_detection(model, test_loader, device):
    """
    Evaluate fraud detection performance on test set.

    Computes anomaly scores for all test sequences,
    then evaluates using AUROC, PR-AUC, and F1 at 1% FPR.
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        roc_curve, precision_recall_curve
    )

    scorer    = AnomalyScorer(model, device)
    all_scores = []
    all_labels = []

    print("\nScoring test sequences...")
    for batch in test_loader:
        scores = scorer.score_batch(batch["input"], batch["target"])
        # Negate: lower log-likelihood = higher fraud score
        all_scores.extend(-scores)
        all_labels.extend(batch["is_fraud"].numpy())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # Metrics
    auroc = roc_auc_score(all_labels, all_scores)
    auprc = average_precision_score(all_labels, all_scores)

    # Threshold at 1% FPR
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    idx_1pct  = np.argmin(np.abs(fpr - 0.01))
    threshold = thresholds[idx_1pct]
    preds_1pct = (all_scores >= threshold).astype(int)

    tp = ((preds_1pct == 1) & (all_labels == 1)).sum()
    fp = ((preds_1pct == 1) & (all_labels == 0)).sum()
    fn = ((preds_1pct == 0) & (all_labels == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    print("\n" + "="*55)
    print("SEPAGEN FRAUD DETECTION RESULTS")
    print("="*55)
    print(f"AUROC:               {auroc:.4f}")
    print(f"PR-AUC:              {auprc:.4f}")
    print(f"Threshold @ 1% FPR:  {threshold:.4f}")
    print(f"Precision @ 1% FPR:  {precision:.4f}")
    print(f"Recall @ 1% FPR:     {recall:.4f}")
    print(f"F1 @ 1% FPR:         {f1:.4f}")
    print(f"\nBaseline (Isolation Forest):")
    print(f"  AUROC:  0.9514")
    print(f"  PR-AUC: 0.1368")
    print(f"\nSEPAGen improvement over baseline:")
    print(f"  AUROC:  {auroc - 0.9514:+.4f}")
    print(f"  PR-AUC: {auprc - 0.1368:+.4f}")
    print("="*55)

    return {
        "auroc": auroc, "auprc": auprc,
        "f1": f1, "precision": precision, "recall": recall,
        "threshold": threshold,
        "scores": all_scores, "labels": all_labels,
    }


# ─────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────

def main():

    print("="*55)
    print("SEPAGen — Phase 4: Transformer Training")
    print("="*55)
    print(f"Device: {DEVICE}\n")

    # ── Load datasets ──
    # For quick testing use max_samples — remove for full training
    train_dataset = SEPADataset(TRAIN_FILE)   # 1.17M sequences
    val_dataset   = SEPADataset(VAL_FILE)
    test_dataset  = SEPADataset(TEST_FILE)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2
    )

    # ── Build model ──
    model = SEPAGenModel(
        vocab_size = VOCAB_SIZE,
        d_model    = D_MODEL,
        n_heads    = N_HEADS,
        n_layers   = N_LAYERS,
        d_ff       = D_FF,
        seq_len    = SEQ_LEN,
        pad_token  = PAD_TOKEN,
        dropout    = DROPOUT,
    ).to(DEVICE)

    # ── Optimiser ──
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr           = LEARNING_RATE,
        weight_decay = 0.01,
        betas        = (0.9, 0.95),
    )

    # Linear warmup then cosine decay scheduler
    total_steps  = len(train_loader) * N_EPOCHS
    warmup_steps = WARMUP_STEPS

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    # ── Training loop ──
    print(f"\nStarting training for {N_EPOCHS} epochs...")
    print(f"Train batches per epoch: {len(train_loader):,}\n")

    best_val_loss  = float("inf")
    train_losses   = []
    val_losses     = []

    for epoch in range(1, N_EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimiser, scheduler, DEVICE, epoch
        )
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, DEVICE)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch}/{N_EPOCHS} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"time={epoch_time:.0f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_loss":   val_loss,
                "config": {
                    "vocab_size": VOCAB_SIZE,
                    "d_model":    D_MODEL,
                    "n_heads":    N_HEADS,
                    "n_layers":   N_LAYERS,
                    "d_ff":       D_FF,
                    "seq_len":    SEQ_LEN,
                    "pad_token":  PAD_TOKEN,
                    "dropout":    DROPOUT,
                }
            }, MODEL_SAVE)
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")

    # ── Evaluate on test set ──
    print("\nLoading best model for evaluation...")
    checkpoint = torch.load(MODEL_SAVE, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    results = evaluate_fraud_detection(model, test_loader, DEVICE)

    # ── Save training curves ──
    with open("training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for ep, (tl, vl) in enumerate(zip(train_losses, val_losses), 1):
            writer.writerow([ep, tl, vl])
    print("\n✅ Training log saved to training_log.csv")

    return model, results


# ─────────────────────────────────────────
# QUICK TEST (no training — just verify model works)
# ─────────────────────────────────────────

def quick_test():
    """
    Verify model architecture works correctly
    without loading data or training.
    Run this first to check everything is set up.
    """
    print("Running quick architecture test...")

    model = SEPAGenModel(
        vocab_size = VOCAB_SIZE,
        d_model    = D_MODEL,
        n_heads    = N_HEADS,
        n_layers   = N_LAYERS,
        d_ff       = D_FF,
        seq_len    = SEQ_LEN,
        pad_token  = PAD_TOKEN,
        dropout    = DROPOUT,
    ).to(DEVICE)

    # Dummy batch
    B   = 4      # batch size
    T   = SEQ_LEN
    inp = torch.randint(0, VOCAB_SIZE, (B, T, TOKENS_PER_TXN)).to(DEVICE)
    tgt = torch.randint(0, VOCAB_SIZE, (B, TOKENS_PER_TXN)).to(DEVICE)

    # Forward pass
    logits, attn_weights = model(inp, return_attn=True)

    print(f"\n✅ Forward pass successful")
    print(f"   Input shape:        {tuple(inp.shape)}")
    print(f"   Logits shape:       {tuple(logits.shape)}")
    print(f"   Attention layers:   {len(attn_weights)}")
    print(f"   Attention shape:    {tuple(attn_weights[0].shape)}")

    # Test anomaly scorer
    scorer = AnomalyScorer(model, DEVICE)
    scores = scorer.score_batch(inp, tgt)
    print(f"   Anomaly scores:     {scores.round(3)}")
    print(f"\n✅ Model architecture verified — ready for training")


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        # Quick architecture test — no data needed
        quick_test()
    else:
        # Full training
        model, results = main()
