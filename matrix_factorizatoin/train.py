# matrix_factorization_router.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Data
# -----------------------------
class PromptModelOutcomeDataset(Dataset):
    """
    Supervised data: (prompt_embedding, model_id) -> y in {0,1}
    Use model_id=0 for single-model training.
    """
    def __init__(
        self,
        q_emb: np.ndarray,          # shape [N, dq]
        y: np.ndarray,              # shape [N] values {0,1}
        model_id: Optional[np.ndarray] = None,  # shape [N] ints in [0, num_models)
        dtype: np.dtype = np.float32,
    ):
        assert q_emb.ndim == 2, "q_emb must be [N, dq]"
        assert y.ndim == 1 and y.shape[0] == q_emb.shape[0]
        if model_id is None:
            model_id = np.zeros((q_emb.shape[0],), dtype=np.int64)
        assert model_id.ndim == 1 and model_id.shape[0] == q_emb.shape[0]

        self.q_emb = q_emb.astype(dtype, copy=False)
        self.y = y.astype(dtype, copy=False)
        self.model_id = model_id.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return self.q_emb.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.q_emb[idx]),              # [dq]
            torch.tensor(self.model_id[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class PairwisePreferenceDataset(Dataset):
    """
    Pairwise preference data: for each prompt q,
    we have (winner_model_id, loser_model_id) and label=1 (winner beats loser).
    You can also support ties by sampling both directions as 0.5 labels, etc.
    """
    def __init__(
        self,
        q_emb: np.ndarray,          # [N, dq]
        winner_id: np.ndarray,      # [N]
        loser_id: np.ndarray,       # [N]
        y: Optional[np.ndarray] = None,  # [N] in {0,1} ; default 1s
        dtype: np.dtype = np.float32,
    ):
        assert q_emb.ndim == 2
        n = q_emb.shape[0]
        assert winner_id.shape == (n,)
        assert loser_id.shape == (n,)
        if y is None:
            y = np.ones((n,), dtype=dtype)

        self.q_emb = q_emb.astype(dtype, copy=False)
        self.winner_id = winner_id.astype(np.int64, copy=False)
        self.loser_id = loser_id.astype(np.int64, copy=False)
        self.y = y.astype(dtype, copy=False)

    def __len__(self) -> int:
        return self.q_emb.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.q_emb[idx]),              # [dq]
            torch.tensor(self.winner_id[idx], dtype=torch.long),
            torch.tensor(self.loser_id[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


# -----------------------------
# Model: Matrix-factorization scoring function δ(M,q)
# RouteLLM-style: δ(M,q) = w2^T ( vm ⊙ (W1^T vq + b) )
# -----------------------------
class MatrixFactorizationScorer(nn.Module):
    def __init__(self, num_models: int, dq: int, dm: int):
        super().__init__()
        self.model_emb = nn.Embedding(num_models, dm)   # vm
        self.proj = nn.Linear(dq, dm, bias=True)        # W1^T vq + b
        self.w2 = nn.Linear(dm, 1, bias=False)          # w2^T (...)

        # mild init helps
        nn.init.normal_(self.model_emb.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.w2.weight)

    def delta(self, q: torch.Tensor, model_id: torch.Tensor) -> torch.Tensor:
        """
        q: [B, dq]
        model_id: [B]
        returns δ: [B]
        """
        vm = self.model_emb(model_id)        # [B, dm]
        pq = self.proj(q)                    # [B, dm]
        h = vm * pq                          # Hadamard product
        s = self.w2(h).squeeze(-1)           # [B]
        return s

    @torch.no_grad()
    def predict_success_proba(self, q: torch.Tensor, model_id: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.delta(q, model_id))

    @torch.no_grad()
    def predict_win_proba(self, q: torch.Tensor, a_id: torch.Tensor, b_id: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.delta(q, a_id) - self.delta(q, b_id))


# -----------------------------
# Training
# -----------------------------
@dataclass
class TrainConfig:
    dm: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 10
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    seed: int = 42


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_supervised_success(
    q_emb: np.ndarray,
    y: np.ndarray,
    num_models: int,
    model_id: Optional[np.ndarray] = None,
    cfg: TrainConfig = TrainConfig(),
) -> MatrixFactorizationScorer:
    """
    Train p(y=1|q,M)=sigmoid(δ(M,q)) using BCE.
    """
    set_seed(cfg.seed)

    ds = PromptModelOutcomeDataset(q_emb=q_emb, y=y, model_id=model_id)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=(cfg.device.startswith("cuda")))

    dq = q_emb.shape[1]
    model = MatrixFactorizationScorer(num_models=num_models, dq=dq, dm=cfg.dm).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.train()
    for epoch in range(cfg.epochs):
        total_loss = 0.0
        n = 0
        for q, mid, yb in dl:
            q = q.to(cfg.device, non_blocking=True)
            mid = mid.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)

            logits = model.delta(q, mid)  # δ(M,q)
            loss = F.binary_cross_entropy_with_logits(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            bs = q.size(0)
            total_loss += float(loss.item()) * bs
            n += bs

        print(f"epoch {epoch+1:02d}/{cfg.epochs} | loss={total_loss/max(n,1):.4f}")

    return model


def train_pairwise_preferences(
    q_emb: np.ndarray,
    winner_id: np.ndarray,
    loser_id: np.ndarray,
    num_models: int,
    y: Optional[np.ndarray] = None,
    cfg: TrainConfig = TrainConfig(),
) -> MatrixFactorizationScorer:
    """
    Train p(winner beats loser | q)=sigmoid(δ(w,q)-δ(l,q)) using BCE.
    This matches RouteLLM Eq. 11 with δ from Eq. 12. :contentReference[oaicite:1]{index=1}
    """
    set_seed(cfg.seed)

    ds = PairwisePreferenceDataset(q_emb=q_emb, winner_id=winner_id, loser_id=loser_id, y=y)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                    num_workers=cfg.num_workers, pin_memory=(cfg.device.startswith("cuda")))

    dq = q_emb.shape[1]
    model = MatrixFactorizationScorer(num_models=num_models, dq=dq, dm=cfg.dm).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.train()
    for epoch in range(cfg.epochs):
        total_loss = 0.0
        n = 0
        for q, win_id, lose_id, yb in dl:
            q = q.to(cfg.device, non_blocking=True)
            win_id = win_id.to(cfg.device, non_blocking=True)
            lose_id = lose_id.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)

            logits = model.delta(q, win_id) - model.delta(q, lose_id)
            loss = F.binary_cross_entropy_with_logits(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            bs = q.size(0)
            total_loss += float(loss.item()) * bs
            n += bs

        print(f"epoch {epoch+1:02d}/{cfg.epochs} | loss={total_loss/max(n,1):.4f}")

    return model


# -----------------------------
# Scoring function wrapper you can save/use
# -----------------------------
@torch.no_grad()
def score_prompts(
    model: MatrixFactorizationScorer,
    q_emb: np.ndarray,
    model_id: Optional[np.ndarray] = None,
    device: Optional[str] = None,
    batch_size: int = 512,
) -> np.ndarray:
    """
    Returns probabilities p(success=1 | prompt, model_id).
    """
    if device is None:
        device = next(model.parameters()).device.type

    if model_id is None:
        model_id = np.zeros((q_emb.shape[0],), dtype=np.int64)

    model.eval()
    probs = []
    for i in range(0, q_emb.shape[0], batch_size):
        q = torch.from_numpy(q_emb[i:i+batch_size]).to(device)
        mid = torch.from_numpy(model_id[i:i+batch_size]).long().to(device)
        p = model.predict_success_proba(q, mid).cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


def save_checkpoint(model: MatrixFactorizationScorer, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
    ckpt = {"state_dict": model.state_dict()}
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path: str, num_models: int, dq: int, dm: int, device: str = "cpu") -> MatrixFactorizationScorer:
    model = MatrixFactorizationScorer(num_models=num_models, dq=dq, dm=dm).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
