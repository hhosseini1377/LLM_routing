"""
Train a routing model on HotpotQA using text-embedding-3-small embeddings.

Loads hotpotqa_qwen8b_train.pkl (prompts + success_labels/correct_labels),
embeds each prompt with OpenAI text-embedding-3-small, and trains
MatrixFactorizationScorer for single-model success prediction.

Usage:
    python -m matrix_factorizatoin.train_hotpotqa_embed

Requires:
    - OPENAI_API_KEY environment variable
    - pip install openai tenacity
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from routing_dataset.dataset_paths import FINAL_HOTPOTQA_QWEN8B_TRAIN_FILE

# Embedding model
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536  # default for text-embedding-3-small
BATCH_SIZE_EMBED = 100  # OpenAI recommends batches of up to 2048 inputs


def load_hotpotqa_data(file_path: Path) -> tuple[list[str], np.ndarray]:
    """
    Load HotpotQA dataset from pickle.
    Supports 'prompts'/'prompt' and 'correct_labels'/'success_labels'.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    if hasattr(data, "columns"):
        # DataFrame
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
        else:
            raise KeyError(f"No label column. Available: {list(df.columns)}")
    else:
        # Dict
        prompts = data.get("prompts") or data.get("prompt")
        labels = np.array(
            data.get("correct_labels") or data.get("success_labels") or data.get("em"),
            dtype=np.float32,
        )
        if prompts is None or labels is None:
            raise KeyError(f"Missing prompts or labels. Keys: {list(data.keys())}")
        prompts = list(prompts)

    assert len(prompts) == len(labels), "prompts and labels length mismatch"
    return prompts, labels


def embed_prompts(prompts: list[str], model: str = EMBED_MODEL) -> np.ndarray:
    """
    Embed prompts using OpenAI text-embedding-3-small.
    Batches requests to respect rate limits.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai: pip install openai")

    try:
        from tenacity import retry, stop_after_attempt, wait_random_exponential
        _retry = retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    except ImportError:
        def _retry(f):
            return f

    client = OpenAI()
    embeddings = []

    @_retry
    def _embed_batch(texts: list[str]) -> list[list[float]]:
        r = client.embeddings.create(input=texts, model=model)
        return [d.embedding for d in r.data]

    for i in range(0, len(prompts), BATCH_SIZE_EMBED):
        batch = prompts[i : i + BATCH_SIZE_EMBED]
        batch_emb = _embed_batch(batch)
        embeddings.extend(batch_emb)
        if (i + BATCH_SIZE_EMBED) % 500 == 0 or i + BATCH_SIZE_EMBED >= len(prompts):
            print(f"  Embedded {min(i + BATCH_SIZE_EMBED, len(prompts))}/{len(prompts)} prompts")

    return np.array(embeddings, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train routing model on HotpotQA with OpenAI embeddings")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("routing_dataset/datasets/hotpotqa/hotpotqa_qwen8b_train.pkl"),
        help="Path to hotpotqa train pkl (relative to project root)",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("./matrix_factorizatoin/checkpoints"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dm", type=int, default=256, help="Model embedding dimension")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    data_path = args.data_path if args.data_path.is_absolute() else project_root / args.data_path
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print(f"Loading dataset: {data_path}")
    prompts, labels = load_hotpotqa_data(data_path)
    print(f"  Loaded {len(prompts)} prompts, labels shape {labels.shape}")

    print("Embedding prompts with text-embedding-3-small...")
    q_emb = embed_prompts(prompts)
    print(f"  Embeddings shape: {q_emb.shape} (expected [N, {EMBED_DIM}])")

    # Train: single model (num_models=1), model_id=None (all zeros)
    from matrix_factorizatoin.train import (
        TrainConfig,
        train_supervised_success,
        save_checkpoint,
    )

    cfg = TrainConfig(
        dm=args.dm,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(f"\nTraining MatrixFactorizationScorer (num_models=1, dm={cfg.dm}, epochs={cfg.epochs})...")
    model = train_supervised_success(
        q_emb=q_emb,
        y=labels,
        num_models=1,
        model_id=None,
        cfg=cfg,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "hotpotqa_embed_routing.pt"
    save_checkpoint(
        model,
        str(ckpt_path),
        extra={"embed_dim": EMBED_DIM, "num_models": 1, "dm": args.dm},
    )
    print(f"\nSaved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
