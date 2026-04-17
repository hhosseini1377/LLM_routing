"""
Train a routing model using sentence-transformers embeddings + matrix factorization.

Loads a dataset with prompts and binary (or binarizable) labels, embeds prompts,
and trains MatrixFactorizationScorer for single-model success prediction.

Supported via --dataset (see routing_dataset.dataset_paths):
  - mmlu_original_pro_auxiliary_gsm8k
  - lmsys_chat1m
  - hotpotqa (and others with prompts + correct_labels)

Usage:
    python -m matrix_factorizatoin.train_hotpotqa_embed --dataset mmlu_original_pro_auxiliary_gsm8k
    python -m matrix_factorizatoin.train_hotpotqa_embed --dataset lmsys_chat1m
    python -m matrix_factorizatoin.train_hotpotqa_embed  # legacy: HotpotQA paths

Requires:
    pip install sentence-transformers
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from routing_dataset.dataset_paths import get_dataset_files

# Default embedding model (sentence-transformers)
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"  # 384 dims, fast


def load_prompt_label_data(
    file_path: Path,
    *,
    labels_from_scores: bool = False,
    score_threshold: float = 7.0,
) -> tuple[list[str], np.ndarray]:
    """
    Load routing-style dataset from pickle (DataFrame or dict).

    Prompts: 'prompts' or 'prompt'.
    Labels: 'correct_labels', 'success_labels', 'em', or 'labels'; or if
    ``labels_from_scores`` and 'scores' is present (e.g. LMSYS judge 0–10), use
    (scores >= score_threshold) as binary labels.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    if hasattr(data, "columns"):
        df = data
        if "prompts" in df.columns:
            prompts = df["prompts"].tolist()
        elif "prompt" in df.columns:
            prompts = df["prompt"].tolist()
        else:
            raise KeyError(f"No prompt column. Available: {list(df.columns)}")

        if "correct_labels" in df.columns:
            labels = np.array(df["correct_labels"], dtype=np.float32)
        elif "success_labels" in df.columns:
            labels = np.array(df["success_labels"], dtype=np.float32)
        elif "em" in df.columns:
            labels = np.array(df["em"], dtype=np.float32)
        elif "labels" in df.columns:
            labels = np.array(df["labels"], dtype=np.float32)
        elif labels_from_scores and "scores" in df.columns:
            labels = (np.array(df["scores"], dtype=np.float32) >= score_threshold).astype(
                np.float32
            )
        else:
            raise KeyError(
                f"No label column (correct_labels/success_labels/em/labels/scores). "
                f"Available: {list(df.columns)}"
            )
    else:
        prompts = data.get("prompts") or data.get("prompt")
        if prompts is None:
            raise KeyError(f"Missing prompts. Keys: {list(data.keys())}")
        prompts = list(prompts)

        if data.get("correct_labels") is not None:
            labels = np.array(data["correct_labels"], dtype=np.float32)
        elif data.get("success_labels") is not None:
            labels = np.array(data["success_labels"], dtype=np.float32)
        elif data.get("em") is not None:
            labels = np.array(data["em"], dtype=np.float32)
        elif data.get("labels") is not None:
            labels = np.array(data["labels"], dtype=np.float32)
        elif labels_from_scores and data.get("scores") is not None:
            labels = (
                np.array(data["scores"], dtype=np.float32) >= score_threshold
            ).astype(np.float32)
        else:
            raise KeyError(
                f"Missing labels. Keys: {list(data.keys())}"
            )

    assert len(prompts) == len(labels), "prompts and labels length mismatch"
    labels = np.asarray(labels, dtype=np.float32).squeeze()
    if labels.ndim > 1:
        labels = labels.ravel()
    return prompts, labels


def load_hotpotqa_data(file_path: Path) -> tuple[list[str], np.ndarray]:
    """Backward-compatible alias for :func:`load_prompt_label_data`."""
    return load_prompt_label_data(file_path)


def embed_prompts(prompts: list[str], model_name: str = DEFAULT_EMBED_MODEL) -> np.ndarray:
    """
    Embed prompts using sentence-transformers (free, runs locally).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers")

    print(f"  Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        prompts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    )
    return embeddings.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Train matrix-factorization router with sentence-transformer prompt embeddings"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Dataset name for routing_dataset.dataset_paths.get_dataset_files, e.g. "
            "mmlu_original_pro_auxiliary_gsm8k, lmsys_chat1m, hotpotqa. "
            "If set, train/val paths are resolved automatically (overrides default HotpotQA paths)."
        ),
    )
    parser.add_argument(
        "--dataset_model",
        type=str,
        default="qwen8b",
        help="Model tag for get_dataset_files (default: qwen8b)",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help="Train pkl path (relative to project root). Used when --dataset is not set.",
    )
    parser.add_argument(
        "--val_path",
        type=Path,
        default=None,
        help="Validation pkl path (relative to project root). Used when --dataset is not set.",
    )
    parser.add_argument(
        "--labels_from_scores",
        action="store_true",
        help="If set, derive binary labels from 'scores' column (score >= --score_threshold).",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=7.0,
        help="When --labels_from_scores: label=1 iff scores >= this value (default: 7).",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("./matrix_factorizatoin/checkpoints"))
    parser.add_argument(
        "--embed_model",
        type=str,
        default=DEFAULT_EMBED_MODEL,
        help="sentence-transformers model (e.g. all-MiniLM-L6-v2, all-mpnet-base-v2)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dm", type=int, default=256, help="Model embedding dimension")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent

    if args.dataset is not None:
        train_rel, val_rel, _ = get_dataset_files(args.dataset, args.dataset_model)
        data_path = project_root / train_rel
        val_path = project_root / val_rel
        run_tag = args.dataset
    elif args.data_path is not None:
        data_path = args.data_path if args.data_path.is_absolute() else project_root / args.data_path
        if args.val_path is None:
            raise ValueError("When using --data_path, also pass --val_path (or use --dataset).")
        val_path = args.val_path if args.val_path.is_absolute() else project_root / args.val_path
        run_tag = data_path.stem
    else:
        data_path = project_root / "routing_dataset/datasets/hotpotqa/hotpotqa_qwen8b_train.pkl"
        val_path = project_root / "routing_dataset/datasets/hotpotqa/hotpotqa_qwen8b_val.pkl"
        run_tag = "hotpotqa"

    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print(f"Loading dataset: {data_path}")
    prompts, labels = load_prompt_label_data(
        data_path,
        labels_from_scores=args.labels_from_scores,
        score_threshold=args.score_threshold,
    )
    print(f"  Loaded {len(prompts)} prompts, labels shape {labels.shape}")

    print("Embedding prompts with sentence-transformers...")
    q_emb = embed_prompts(prompts, model_name=args.embed_model)
    embed_dim = q_emb.shape[1]
    print(f"  Embeddings shape: {q_emb.shape}")

    # Load validation set for per-epoch evaluation
    val_emb = None
    val_labels = None
    if val_path.exists():
        val_prompts, val_labels = load_prompt_label_data(
            val_path,
            labels_from_scores=args.labels_from_scores,
            score_threshold=args.score_threshold,
        )
        print(f"Loading validation embeddings ({len(val_prompts)} prompts)...")
        val_emb = embed_prompts(val_prompts, model_name=args.embed_model)
    else:
        print(f"Validation set not found (skipping per-epoch eval): {val_path}")

    # Train with validation after each epoch
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score, average_precision_score

    from matrix_factorizatoin.train import (
        TrainConfig,
        PromptModelOutcomeDataset,
        MatrixFactorizationScorer,
        score_prompts,
        save_checkpoint,
        set_seed,
    )

    cfg = TrainConfig(
        dm=args.dm,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )
    set_seed(cfg.seed)

    ds = PromptModelOutcomeDataset(q_emb=q_emb, y=labels, model_id=None)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
    )

    model = MatrixFactorizationScorer(num_models=1, dq=embed_dim, dm=cfg.dm).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    print(f"\nTraining MatrixFactorizationScorer (num_models=1, dm={cfg.dm}, epochs={cfg.epochs})...")
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for q, mid, yb in dl:
            q = q.to(cfg.device, non_blocking=True)
            mid = mid.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)

            logits = model.delta(q, mid)
            loss = F.binary_cross_entropy_with_logits(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            bs = q.size(0)
            total_loss += float(loss.item()) * bs
            n += bs

        avg_loss = total_loss / max(n, 1)
        if val_emb is not None and val_labels is not None:
            val_probs = score_prompts(model, val_emb, model_id=None, device=cfg.device)
            roc_auc = roc_auc_score(val_labels, val_probs)
            pr_auc_pos = average_precision_score(val_labels, val_probs, pos_label=1)  # class 1 (success)
            pr_auc_neg = average_precision_score(1 - val_labels, 1 - val_probs)       # class 0 (fail)
            print(f"epoch {epoch+1:02d}/{cfg.epochs} | loss={avg_loss:.4f} | val_roc_auc={roc_auc:.4f} | val_pr_auc_pos={pr_auc_pos:.4f} | val_pr_auc_neg={pr_auc_neg:.4f}")
        else:
            print(f"epoch {epoch+1:02d}/{cfg.epochs} | loss={avg_loss:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = run_tag.replace("/", "_")
    ckpt_path = output_dir / f"{safe_tag}_embed_routing.pt"
    save_checkpoint(
        model,
        str(ckpt_path),
        extra={
            "embed_dim": embed_dim,
            "embed_model": args.embed_model,
            "num_models": 1,
            "dm": args.dm,
            "dataset": run_tag,
            "dataset_model": args.dataset_model,
        },
    )
    print(f"\nSaved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
