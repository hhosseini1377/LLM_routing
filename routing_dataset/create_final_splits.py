"""
Combine MMLU, MMLU-Pro, and GSM8K datasets and split into train/validation/test.

This script:
1. Loads the three combined dataset files (MMLU, MMLU-Pro, GSM8K)
2. Combines them into a single dataset
3. Splits into train/val/test with 80/10/10 ratio
4. Saves the splits
"""

import pickle
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from routing_dataset.dataset_paths import *


def load_dataset(file_path: Path) -> Dict:
    """Load a dataset from pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def combine_datasets(datasets: list, dataset_names: list) -> Dict:
    """
    Combine multiple datasets into one.
    
    Args:
        datasets: List of dataset dictionaries
        dataset_names: List of dataset names for tracking
    
    Returns:
        Combined dataset dictionary with 'prompts', 'ground_truths', 'correct_labels', and 'dataset_source'
    """
    combined = {
        'prompts': [],
        'ground_truths': [],
        'correct_labels': [],
        'dataset_source': []  # Track which dataset each sample came from
    }
    
    for dataset, name in zip(datasets, dataset_names):
        n_samples = len(dataset['prompts'])
        combined['prompts'].extend(dataset['prompts'])
        combined['ground_truths'].extend(dataset['ground_truths'])
        combined['correct_labels'].extend(dataset['correct_labels'])
        combined['dataset_source'].extend([name] * n_samples)
    
    return combined


def split_dataset(
    dataset: Dict,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify_by: str = 'correct_labels'
) -> Tuple[Dict, Dict, Dict]:
    """
    Split a dataset into train/val/test sets using stratified sampling.
    
    Stratified splitting ensures that the distribution of classes (correct_labels)
    is maintained across train/val/test splits.
    
    Args:
        dataset: Dataset dictionary
        train_ratio: Ratio for training set (default: 0.8)
        val_ratio: Ratio for validation set (default: 0.1)
        test_ratio: Ratio for test set (default: 0.1)
        seed: Random seed for reproducibility
        stratify_by: Key to use for stratification (default: 'correct_labels')
    
    Returns:
        Tuple of (train, val, test) datasets
    """
    # Validate ratios sum to 1.0
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    np.random.seed(seed)
    n_samples = len(dataset['prompts'])
    
    # Get stratification labels
    if stratify_by not in dataset:
        raise ValueError(f"Stratification key '{stratify_by}' not found in dataset")
    
    stratify_labels = np.array(dataset[stratify_by])
    
    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(stratify_labels, return_counts=True)
    print(f"  Stratifying by '{stratify_by}':")
    for cls, count in zip(unique_classes, class_counts):
        print(f"    Class {cls}: {count:,} samples ({count/n_samples*100:.2f}%)")
    
    # Split indices by class
    train_indices = []
    val_indices = []
    test_indices = []
    
    for cls in unique_classes:
        # Get all indices for this class
        class_indices = np.where(stratify_labels == cls)[0]
        n_class = len(class_indices)
        
        # Shuffle class indices
        np.random.shuffle(class_indices)
        
        # Calculate split sizes for this class
        n_train_class = int(n_class * train_ratio)
        n_val_class = int(n_class * val_ratio)
        n_test_class = n_class - n_train_class - n_val_class
        
        # Split indices for this class
        train_indices.extend(class_indices[:n_train_class])
        val_indices.extend(class_indices[n_train_class:n_train_class + n_val_class])
        test_indices.extend(class_indices[n_train_class + n_val_class:])
    
    # Shuffle final indices to mix classes
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    def create_split(indices):
        return {
            'prompts': [dataset['prompts'][i] for i in indices],
            'ground_truths': [dataset['ground_truths'][i] for i in indices],
            'correct_labels': [dataset['correct_labels'][i] for i in indices],
            'dataset_source': [dataset['dataset_source'][i] for i in indices] if 'dataset_source' in dataset else None
        }
    
    return create_split(train_indices), create_split(val_indices), create_split(test_indices)


def create_final_splits(
    output_dir: Path = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Combine MMLU, MMLU-Pro, and GSM8K datasets and split into train/val/test.
    
    Args:
        output_dir: Directory to save the split files. If None, uses DATA_DIR / "final_splits"
        train_ratio: Ratio for training set (default: 0.8)
        val_ratio: Ratio for validation set (default: 0.1)
        test_ratio: Ratio for test set (default: 0.1)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing dataset dictionaries
    """
    if output_dir is None:
        output_dir = DATA_DIR / "final_splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Creating Final Train/Val/Test Splits")
    print("=" * 60)
    print(f"Split ratio: {train_ratio*100:.0f}% train / {val_ratio*100:.0f}% val / {test_ratio*100:.0f}% test")
    print(f"Random seed: {seed}")
    
    # Load combined datasets
    print("\nLoading combined datasets...")
    mmlu_combined = load_dataset(MMLU_COMBINED_FILE)
    mmlu_pro_combined = load_dataset(MMLU_PRO_COMBINED_FILE)
    gsm8k_combined = load_dataset(GSM8K_COMBINED_FILE)
    
    print(f"  MMLU: {len(mmlu_combined['prompts']):,} samples")
    print(f"  MMLU-Pro: {len(mmlu_pro_combined['prompts']):,} samples")
    print(f"  GSM8K: {len(gsm8k_combined['prompts']):,} samples")
    
    # Combine all three datasets
    print("\nCombining all datasets...")
    all_combined = combine_datasets(
        [mmlu_combined, mmlu_pro_combined, gsm8k_combined],
        ["MMLU", "MMLU-Pro", "GSM8K"]
    )
    
    total_samples = len(all_combined['prompts'])
    print(f"  Total combined: {total_samples:,} samples")
    
    # Count by dataset source
    from collections import Counter
    source_counts = Counter(all_combined['dataset_source'])
    print(f"  By dataset:")
    for source, count in sorted(source_counts.items()):
        print(f"    {source}: {count:,}")
    
    # Split into train/val/test (stratified by correct_labels)
    print(f"\nSplitting into train/val/test ({train_ratio*100:.0f}/{val_ratio*100:.0f}/{test_ratio*100:.0f})...")
    print("Using stratified splitting based on correct_labels...")
    train_split, val_split, test_split = split_dataset(
        all_combined,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratify_by='correct_labels'
    )
    
    # Print statistics for each split
    def print_stats(name, dataset):
        total = len(dataset['prompts'])
        correct = sum(dataset['correct_labels'])
        accuracy = correct / total if total > 0 else 0
        
        # Count by dataset source
        if dataset['dataset_source']:
            source_counts = Counter(dataset['dataset_source'])
        
        print(f"\n{name}:")
        print(f"  Total samples: {total:,}")
        print(f"  Correct: {correct:,}")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        if dataset['dataset_source']:
            print(f"  By dataset:")
            for source, count in sorted(source_counts.items()):
                print(f"    {source}: {count:,}")
    
    print_stats("TRAINING", train_split)
    print_stats("VALIDATION", val_split)
    print_stats("TEST", test_split)
    
    # Save splits
    print(f"\nSaving splits to {output_dir}...")
    
    train_file = output_dir / "train.pkl"
    val_file = output_dir / "val.pkl"
    test_file = output_dir / "test.pkl"
    
    with open(train_file, 'wb') as f:
        pickle.dump(train_split, f)
    print(f"  Saved: {train_file}")
    
    with open(val_file, 'wb') as f:
        pickle.dump(val_split, f)
    print(f"  Saved: {val_file}")
    
    with open(test_file, 'wb') as f:
        pickle.dump(test_split, f)
    print(f"  Saved: {test_file}")
    
    return {
        'train': train_split,
        'val': val_split,
        'test': test_split
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine datasets and create train/val/test splits")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio (default: 0.1)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: DATA_DIR/final_splits)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    splits = create_final_splits(
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    print("\n✅ Done! Final splits created successfully.")

