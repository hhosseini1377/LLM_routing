"""
Create combined train/validation/test splits from MMLU, MMLU-Pro, and GSM8K datasets.

This script:
1. Omits auxiliary/train splits (MMLU Auxiliary, GSM8K Train)
2. Combines only test/validation/dev splits from each dataset
3. Splits the combined dataset into train/val/test (70/15/15)
4. Maintains balanced representation from all three dataset types
5. Preserves difficulty distribution for CPX routing learning

Datasets used:
- MMLU: Test, Validation, Dev (omits Auxiliary)
- MMLU-Pro: Test, Validation
- GSM8K: Test (omits Train)
"""

import pickle
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from routing_dataset.dataset_paths import *


def load_dataset(file_path: Path) -> Dict:
    """Load a dataset from pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def combine_datasets(datasets: List[Dict], dataset_names: List[str]) -> Dict:
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


def split_dataset(dataset: Dict, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42) -> Tuple[Dict, Dict, Dict]:
    """
    Split a dataset into train/val/test sets.
    
    Args:
        dataset: Dataset dictionary
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train, val, test) datasets
    """
    np.random.seed(seed)
    n_samples = len(dataset['prompts'])
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    n_val = int(n_samples * val_ratio)
    n_test = int(n_samples * test_ratio)
    n_train = n_samples - n_val - n_test
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    def create_split(indices):
        return {
            'prompts': [dataset['prompts'][i] for i in indices],
            'ground_truths': [dataset['ground_truths'][i] for i in indices],
            'correct_labels': [dataset['correct_labels'][i] for i in indices],
            'dataset_source': [dataset['dataset_source'][i] for i in indices] if 'dataset_source' in dataset else None
        }
    
    return create_split(train_indices), create_split(val_indices), create_split(test_indices)


def create_combined_splits(
    output_dir: Path = None,
    strategy: str = "preserve_test",
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Create combined train/val/test splits from all datasets.
    
    Args:
        output_dir: Directory to save the split files. If None, uses DATA_DIR.
        strategy: Split strategy - "preserve_test" (recommended) or "complete_resplit"
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing dataset dictionaries
    """
    if output_dir is None:
        output_dir = DATA_DIR / "combined"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading datasets...")
    print("Note: Omitting auxiliary/train splits - only using test/validation/dev splits")
    
    # Load only test/validation/dev splits (omit auxiliary/train)
    mmlu_test = load_dataset(MMLU_TEST_QWEN8B_CORRECT_RESULTS_FILE)
    mmlu_val = load_dataset(MMLU_VALIDATION_QWEN8B_CORRECT_RESULTS_FILE)
    mmlu_dev = load_dataset(MMLU_DEV_QWEN8B_CORRECT_RESULTS_FILE)
    mmlu_pro_test = load_dataset(MMLU_PRO_TEST_QWEN8B_CORRECT_RESULTS_FILE)
    mmlu_pro_val = load_dataset(MMLU_PRO_VALIDATION_QWEN8B_CORRECT_RESULTS_FILE)
    gsm8k_test = load_dataset(GSM8K_TEST_QWEN8B_CORRECT_RESULTS_FILE)
    
    if strategy == "preserve_test":
        print("\nUsing Strategy 1: Preserve Original Test Sets")
        print("=" * 60)
        print("Combining test/validation/dev splits, then splitting into train/val/test")
        
        # Combine all test/validation/dev splits
        print("\nCombining all available splits...")
        all_datasets = [
            (mmlu_test, "MMLU_Test"),
            (mmlu_val, "MMLU_Validation"),
            (mmlu_dev, "MMLU_Dev"),
            (mmlu_pro_test, "MMLU-Pro_Test"),
            (mmlu_pro_val, "MMLU-Pro_Validation"),
            (gsm8k_test, "GSM8K_Test")
        ]
        all_combined = combine_datasets(
            [d[0] for d in all_datasets],
            [d[1] for d in all_datasets]
        )
        
        # Split into train/val/test (70/15/15)
        print("Splitting combined dataset into train/val/test (70/15/15)...")
        train_combined, val_combined, test_combined = split_dataset(
            all_combined, val_ratio=0.15, test_ratio=0.15, seed=seed
        )
        
    elif strategy == "complete_resplit":
        print("\nUsing Strategy 2: Complete Re-split")
        print("=" * 60)
        print("Combining test/validation/dev splits, then splitting into train/val/test")
        
        # Combine all test/validation/dev splits (same as preserve_test now)
        print("\nCombining all available splits...")
        all_datasets = [
            (mmlu_test, "MMLU_Test"),
            (mmlu_val, "MMLU_Validation"),
            (mmlu_dev, "MMLU_Dev"),
            (mmlu_pro_test, "MMLU-Pro_Test"),
            (mmlu_pro_val, "MMLU-Pro_Validation"),
            (gsm8k_test, "GSM8K_Test")
        ]
        all_combined = combine_datasets(
            [d[0] for d in all_datasets],
            [d[1] for d in all_datasets]
        )
        
        # Shuffle and split
        print("Splitting into train/val/test (70/15/15)...")
        train_combined, val_combined, test_combined = split_dataset(
            all_combined, val_ratio=0.15, test_ratio=0.15, seed=seed
        )
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'preserve_test' or 'complete_resplit'")
    
    # Print statistics
    def print_stats(name, dataset):
        total = len(dataset['prompts'])
        correct = sum(dataset['correct_labels'])
        accuracy = correct / total if total > 0 else 0
        
        # Count by dataset source
        if 'dataset_source' in dataset and dataset['dataset_source']:
            source_counts = {}
            for source in dataset['dataset_source']:
                source_counts[source] = source_counts.get(source, 0) + 1
        
        print(f"\n{name}:")
        print(f"  Total samples: {total:,}")
        print(f"  Correct: {correct:,}")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        if 'dataset_source' in dataset and dataset['dataset_source']:
            print(f"  By dataset:")
            for source, count in sorted(source_counts.items()):
                print(f"    {source}: {count:,}")
    
    print_stats("TRAINING", train_combined)
    print_stats("VALIDATION", val_combined)
    print_stats("TEST", test_combined)
    
    # Save splits
    print(f"\nSaving splits to {output_dir}...")
    
    train_file = output_dir / "combined_train.pkl"
    val_file = output_dir / "combined_val.pkl"
    test_file = output_dir / "combined_test.pkl"
    
    with open(train_file, 'wb') as f:
        pickle.dump(train_combined, f)
    print(f"  Saved: {train_file}")
    
    with open(val_file, 'wb') as f:
        pickle.dump(val_combined, f)
    print(f"  Saved: {val_file}")
    
    with open(test_file, 'wb') as f:
        pickle.dump(test_combined, f)
    print(f"  Saved: {test_file}")
    
    return {
        'train': train_combined,
        'val': val_combined,
        'test': test_combined
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create combined train/val/test splits")
    parser.add_argument(
        "--strategy",
        type=str,
        default="preserve_test",
        choices=["preserve_test", "complete_resplit"],
        help="Split strategy: 'preserve_test' (recommended) or 'complete_resplit'"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: DATA_DIR/combined)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    splits = create_combined_splits(
        output_dir=output_dir,
        strategy=args.strategy,
        seed=args.seed
    )
    
    print("\n✅ Done! Combined splits created successfully.")

