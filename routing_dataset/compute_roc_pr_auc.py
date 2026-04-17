"""
Compute ROC-AUC and PR-AUC (positive and negative) for binary classification.

Useful for evaluating success prediction outputs (e.g. self_evaluation_score vs correct_label).

Usage:
    python -m routing_dataset.compute_roc_pr_auc --input_file path/to/success_pred.pkl
    python -m routing_dataset.compute_roc_pr_auc --y_true 1,0,1,0 --y_score 0.9,0.2,0.8,0.3

Can also be imported:
    from routing_dataset.compute_roc_pr_auc import compute_roc_pr_auc
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


def compute_roc_pr_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pos_label: int = 1,
) -> Tuple[float, float, float]:
    """
    Compute ROC-AUC, positive PR-AUC, and negative PR-AUC.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_score: Predicted scores or probabilities (higher = more likely positive).
        pos_label: Label of the positive class (default 1).

    Returns:
        Tuple of (roc_auc, pr_auc_positive, pr_auc_negative).

    - ROC-AUC: Area under ROC curve (threshold-independent).
    - PR-AUC positive: Precision-Recall AUC when class 1 is "positive".
    - PR-AUC negative: Precision-Recall AUC when class 0 is "positive".
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    if len(y_true) != len(y_score):
        raise ValueError(
            f"y_true and y_score length mismatch: {len(y_true)} vs {len(y_score)}"
        )

    n_pos = np.sum(y_true == pos_label)
    n_neg = np.sum(y_true != pos_label)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Need both classes present. Found pos={n_pos}, neg={n_neg}"
        )

    # ROC-AUC (sklearn assumes positive class has larger label)
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = float("nan")

    # PR-AUC for positive class (class 1)
    try:
        pr_auc_pos = average_precision_score(
            y_true, y_score, pos_label=pos_label
        )
    except ValueError:
        pr_auc_pos = float("nan")

    # PR-AUC for negative class (class 0): flip labels and scores
    try:
        y_true_neg = 1 - y_true if pos_label == 1 else (y_true == 0).astype(int)
        y_score_neg = 1 - y_score
        pr_auc_neg = average_precision_score(
            y_true_neg, y_score_neg, pos_label=1
        )
    except ValueError:
        pr_auc_neg = float("nan")

    return roc_auc, pr_auc_pos, pr_auc_neg


def load_from_pkl(
    file_path: Path,
    label_col: str = "correct_label",
    score_col: str = "self_evaluation_score",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load y_true and y_score from a pickle file (DataFrame or dict).

    Args:
        file_path: Path to .pkl file.
        label_col: Column name for ground truth. Tries 'correct_label', 'correct_labels', 'labels'.
        score_col: Column name for predicted scores. Tries 'self_evaluation_score', 'probabilities', 'scores'.

    Returns:
        Tuple of (y_true, y_score).
    """
    data = pd.read_pickle(file_path)

    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)

    # Resolve label column
    for c in (label_col, "correct_label", "correct_labels", "labels"):
        if c in df.columns:
            y_true = np.asarray(df[c]).ravel()
            if y_true.dtype == bool:
                y_true = y_true.astype(int)
            break
    else:
        raise KeyError(
            f"No label column found. Tried: {label_col}, correct_label, correct_labels, labels. "
            f"Available: {list(df.columns)}"
        )

    # Resolve score column (can be binary 0/1 or probabilities)
    for c in (score_col, "self_evaluation_score", "probabilities", "scores"):
        if c in df.columns:
            y_score = np.asarray(df[c], dtype=float).ravel()
            break
    else:
        raise KeyError(
            f"No score column found. Tried: {score_col}, self_evaluation_score, probabilities, scores. "
            f"Available: {list(df.columns)}"
        )

    return y_true, y_score


def main():
    parser = argparse.ArgumentParser(
        description="Compute ROC-AUC and PR-AUC (positive/negative) for binary classification"
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        default=None,
        help="Path to pkl file with correct_label and self_evaluation_score (or score_col)",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="correct_label",
        help="Column name for ground truth labels",
    )
    parser.add_argument(
        "--score_col",
        type=str,
        default="self_evaluation_score",
        help="Column name for predicted scores/probabilities",
    )
    parser.add_argument(
        "--y_true",
        type=str,
        default=None,
        help="Comma-separated ground truth labels (alternative to input_file)",
    )
    parser.add_argument(
        "--y_score",
        type=str,
        default=None,
        help="Comma-separated predicted scores (alternative to input_file)",
    )
    args = parser.parse_args()

    if args.input_file is not None:
        path = (
            args.input_file
            if args.input_file.is_absolute()
            else Path(__file__).resolve().parent.parent / args.input_file
        )
        if not path.exists():
            print(f"Error: File not found: {path}")
            sys.exit(1)
        y_true, y_score = load_from_pkl(
            path, label_col=args.label_col, score_col=args.score_col
        )
    elif args.y_true is not None and args.y_score is not None:
        y_true = np.array([int(x.strip()) for x in args.y_true.split(",")])
        y_score = np.array([float(x.strip()) for x in args.y_score.split(",")])
    else:
        parser.error("Provide --input_file OR both --y_true and --y_score")

    roc_auc, pr_auc_pos, pr_auc_neg = compute_roc_pr_auc(y_true, y_score)

    print("ROC-AUC:           {:.4f}".format(roc_auc))
    print("PR-AUC (positive): {:.4f}".format(pr_auc_pos))
    print("PR-AUC (negative): {:.4f}".format(pr_auc_neg))


if __name__ == "__main__":
    main()
