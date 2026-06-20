"""
hybrid_model.py — XGBoost-transformer hybrid model.

Pipeline:
  1. Transformer encoder ingests a (lookback, n_features) window and
     produces a fixed-size embedding via mean-pooling.
  2. A linear head pretrains the transformer end-to-end on the 3-class
     {SELL, HOLD, BUY} target so the encoder learns useful representations.
  3. XGBoost is then trained on [last-bar features ⊕ transformer embedding]
     to make the final classification.

The transformer captures multi-day temporal patterns the tree model cannot
see from a single bar; XGBoost extracts non-linear combinations on top.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TransformerCfg:
    lookback: int = 20
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 128
    dropout: float = 0.2
    embed_dim: int = 64
    num_classes: int = 3


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEncoder(nn.Module):
    """Sequence encoder producing both an embedding and class logits."""

    def __init__(self, n_features: int, cfg: TransformerCfg):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(n_features, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model, max_len=cfg.lookback + 4)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        # Embedding projection: combine mean + last token
        self.embed_proj = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.embed_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.classifier = nn.Linear(cfg.embed_dim, cfg.num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled embedding (no classifier head)."""
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        h = self.norm(h)
        # Pool: mean over the sequence + last token
        pooled = torch.cat([h.mean(dim=1), h[:, -1, :]], dim=-1)
        return self.embed_proj(pooled)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.encode(x)
        logits = self.classifier(embed)
        return logits, embed


def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int,
    symbol_starts: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a flat (N, F) feature matrix into (M, lookback, F) sequences.

    Each sequence ending at index i predicts y[i]. Sequences that would
    cross a symbol boundary (per symbol_starts) are skipped.

    Returns (sequences, last_bar_features, labels). last_bar_features is
    X[i] — i.e. the same features the existing XGBoost sees — kept aligned
    with the sequence labels so they can be concatenated with embeddings.
    """
    n, n_feat = X.shape
    if symbol_starts is None:
        symbol_starts = [0]
    # Each symbol owns [start, next_start)
    starts = list(symbol_starts) + [n]

    seqs: list[np.ndarray] = []
    lasts: list[np.ndarray] = []
    labels: list[int] = []

    for s_idx in range(len(starts) - 1):
        s_lo, s_hi = starts[s_idx], starts[s_idx + 1]
        # First valid end index is s_lo + lookback - 1
        for i in range(s_lo + lookback - 1, s_hi):
            seqs.append(X[i - lookback + 1 : i + 1])
            lasts.append(X[i])
            labels.append(y[i])

    return (
        np.stack(seqs).astype(np.float32),
        np.stack(lasts).astype(np.float32),
        np.array(labels, dtype=np.int64),
    )


def train_transformer(
    model: TransformerEncoder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    patience: int = 12,
    verbose: bool = True,
) -> dict:
    """Train the transformer with early stopping on val loss."""
    model = model.to(device)

    # Optional class re-weighting (sqrt-inverse keeps natural prior signal but
    # still gives minority classes some leverage). Set use_class_weights=False
    # in train_transformer's caller for fully natural priors.
    class_counts = np.bincount(y_train, minlength=model.cfg.num_classes)
    total = len(y_train)
    # Mild reweighting via sqrt; pure inverse weighting destroys the prior
    weights = np.sqrt(total / (model.cfg.num_classes * np.clip(class_counts, 1, None)))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stale = 0
    history: list[dict] = []

    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(xb)
            tr_correct += (logits.argmax(-1) == yb).sum().item()
            tr_total += len(xb)
        sched.step()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device)
                yb = yb.to(device)
                logits, _ = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * len(xb)
                val_correct += (logits.argmax(-1) == yb).sum().item()
                val_total += len(xb)

        tr_loss /= max(tr_total, 1)
        val_loss /= max(val_total, 1)
        tr_acc = tr_correct / max(tr_total, 1)
        val_acc = val_correct / max(val_total, 1)
        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": val_loss,
                        "train_acc": tr_acc, "val_acc": val_acc})

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  ep {ep:3d}  train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
                  f"train_acc={tr_acc:.3f} val_acc={val_acc:.3f}")

        if stale >= patience:
            if verbose:
                print(f"  early stop @ ep {ep} (best val_loss={best_val_loss:.4f})")
            break

    model.load_state_dict(best_state)
    return {"best_val_loss": best_val_loss, "best_val_acc": best_val_acc, "history": history}


@torch.no_grad()
def extract_embeddings(
    model: TransformerEncoder, X: np.ndarray, batch_size: int = 256, device: str = "cpu"
) -> np.ndarray:
    model.eval()
    model = model.to(device)
    out: list[np.ndarray] = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(device)
        emb = model.encode(xb).cpu().numpy()
        out.append(emb)
    return np.vstack(out)
