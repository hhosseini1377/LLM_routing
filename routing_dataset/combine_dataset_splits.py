"""
Combine dataset splits into single files for each dataset type.

This script combines:
1. MMLU: Test + Validation + Dev (omits Auxiliary) → mmlu_combined.pkl
2. MMLU-Pro: Test + Validation → mmlu_pro_combined.pkl
3. GSM8K: Test → gsm8k_combined.pkl
"""

import pickle
import sys
from pathlib import Path
from typing import Dict, List

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
        Combined dataset dictionary with 'prompts', 'ground_truths', 'correct_labels', and 'split_source'
    """
    combined = {
        'prompts': [],
        'ground_truths': [],
        'correct_labels': [],
        'split_source': []  # Track which split each sample came from
    }
    
    for dataset, name in zip(datasets, dataset_names):
        n_samples = len(dataset['prompts'])
        combined['prompts'].extend(dataset['prompts'])
        combined['ground_truths'].extend(dataset['ground_truths'])
        combined['correct_labels'].extend(dataset['correct_labels'])
        combined['split_source'].extend([name] * n_samples)
    
    return combined


def combine_mmlu_splits(output_file: Path = None) -> Dict:
    """
    Combine MMLU Test, Validation, and Dev splits (omits Auxiliary).
    
    Args:
        output_file: Path to save the combined file. If None, uses MMLU_DATA_DIR / "mmlu_combined.pkl"
    
    Returns:
        Combined dataset dictionary
    """
    if output_file is None:
        output_file = MMLU_DATA_DIR / "mmlu_combined.pkl"
    
    print("Loading MMLU splits...")
    mmlu_test = load_dataset(MMLU_TEST_QWEN8B_CORRECT_RESULTS_FILE)
    mmlu_val = load_dataset(MMLU_VALIDATION_QWEN8B_CORRECT_RESULTS_FILE)
    mmlu_dev = load_dataset(MMLU_DEV_QWEN8B_CORRECT_RESULTS_FILE)
    
    print("Combining MMLU splits...")
    combined = combine_datasets(
        [mmlu_test, mmlu_val, mmlu_dev],
        ["test", "validation", "dev"]
    )
    
    # Print statistics
    total = len(combined['prompts'])
    correct = sum(combined['correct_labels'])
    accuracy = correct / total if total > 0 else 0
    
    # Count by split
    split_counts = {}
    for split in combined['split_source']:
        split_counts[split] = split_counts.get(split, 0) + 1
    
    print(f"\nMMLU Combined:")
    print(f"  Total samples: {total:,}")
    print(f"  Correct: {correct:,}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  By split:")
    for split, count in sorted(split_counts.items()):
        print(f"    {split}: {count:,}")
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(combined, f)
    print(f"\n  Saved to: {output_file}")
    
    return combined


def combine_mmlu_pro_splits(output_file: Path = None) -> Dict:
    """
    Combine MMLU-Pro Test and Validation splits.
    
    Args:
        output_file: Path to save the combined file. If None, uses MMLU_DATA_DIR / "mmlu_pro_combined.pkl"
    
    Returns:
        Combined dataset dictionary
    """
    if output_file is None:
        output_file = MMLU_DATA_DIR / "mmlu_pro_combined.pkl"
    
    print("\nLoading MMLU-Pro splits...")
    mmlu_pro_test = load_dataset(MMLU_PRO_TEST_QWEN8B_CORRECT_RESULTS_FILE)
    mmlu_pro_val = load_dataset(MMLU_PRO_VALIDATION_QWEN8B_CORRECT_RESULTS_FILE)
    
    print("Combining MMLU-Pro splits...")
    combined = combine_datasets(
        [mmlu_pro_test, mmlu_pro_val],
        ["test", "validation"]
    )
    
    # Print statistics
    total = len(combined['prompts'])
    correct = sum(combined['correct_labels'])
    accuracy = correct / total if total > 0 else 0
    
    # Count by split
    split_counts = {}
    for split in combined['split_source']:
        split_counts[split] = split_counts.get(split, 0) + 1
    
    print(f"\nMMLU-Pro Combined:")
    print(f"  Total samples: {total:,}")
    print(f"  Correct: {correct:,}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  By split:")
    for split, count in sorted(split_counts.items()):
        print(f"    {split}: {count:,}")
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(combined, f)
    print(f"\n  Saved to: {output_file}")
    
    return combined


def combine_gsm8k_splits(output_file: Path = None) -> Dict:
    """
    Combine GSM8K Test split (omits Train).
    
    Args:
        output_file: Path to save the combined file. If None, uses GSM8K_DATA_DIR / "gsm8k_combined.pkl"
    
    Returns:
        Combined dataset dictionary
    """
    if output_file is None:
        output_file = GSM8K_DATA_DIR / "gsm8k_combined.pkl"
    
    print("\nLoading GSM8K split...")
    gsm8k_test = load_dataset(GSM8K_TEST_QWEN8B_CORRECT_RESULTS_FILE)
    gsm8k_train = load_dataset(GSM8K_TRAIN_QWEN8B_CORRECT_RESULTS_FILE)
    print("Combining GSM8K split...")
    # For consistency, wrap in list even though it's just one split
    combined = combine_datasets(
        [gsm8k_test, gsm8k_train],
        ["test", "train"]
    )
    
    # Print statistics
    total = len(combined['prompts'])
    correct = sum(combined['correct_labels'])
    accuracy = correct / total if total > 0 else 0
    
    print(f"\nGSM8K Combined:")
    print(f"  Total samples: {total:,}")
    print(f"  Correct: {correct:,}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(combined, f)
    print(f"\n  Saved to: {output_file}")
    
    return combined


def main():
    """Combine all dataset splits."""
    print("=" * 60)
    print("Combining Dataset Splits")
    print("=" * 60)
    
    # Combine MMLU splits
    mmlu_combined = combine_mmlu_splits()
    
    # Combine MMLU-Pro splits
    mmlu_pro_combined = combine_mmlu_pro_splits()
    
    # Combine GSM8K split
    gsm8k_combined = combine_gsm8k_splits()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"MMLU Combined:     {len(mmlu_combined['prompts']):,} samples")
    print(f"MMLU-Pro Combined: {len(mmlu_pro_combined['prompts']):,} samples")
    print(f"GSM8K Combined:    {len(gsm8k_combined['prompts']):,} samples")
    print(f"\nTotal: {len(mmlu_combined['prompts']) + len(mmlu_pro_combined['prompts']) + len(gsm8k_combined['prompts']):,} samples")
    print("\n✅ Done! All datasets combined successfully.")


if __name__ == "__main__":
    main()

