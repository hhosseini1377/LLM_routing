"""
Distributed Weighted Random Sampler for PyTorch Distributed Training.

This sampler combines WeightedRandomSampler with DistributedSampler to enable
weighted sampling in distributed training scenarios.
"""

import torch
from torch.utils.data import Sampler
from typing import List, Optional, Dict
import torch.distributed as dist


class DistributedWeightedSampler(Sampler):
    """
    A distributed sampler that applies weights to samples while ensuring
    each process gets a non-overlapping subset of the data.
    
    This is the recommended way to use weighted sampling in distributed training.
    
    Important: Generates MORE samples than dataset size (1.5x by default) to ensure:
    - Minority class is oversampled (goal of weighted sampling)
    - Majority class samples still appear (avoid dropping them completely)
    - Better coverage of all samples in the dataset
    
    Args:
        dataset: Dataset to sample from
        weights: List of weights for each sample in the dataset (length must match dataset size)
        num_replicas: Number of processes participating in distributed training
        rank: Rank of the current process
        replacement: Whether to sample with replacement (default: True)
        seed: Random seed for reproducibility (default: 0)
        drop_last: Whether to drop the last incomplete batch (default: False)
        oversample_factor: Factor to multiply dataset size when generating samples (default: 1.5)
                          Higher values ensure better coverage of majority class samples
    """
    
    def __init__(
        self,
        dataset,
        weights: List[float],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        replacement: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        oversample_factor: float = 1.5,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed
        self.drop_last = drop_last
        self.oversample_factor = oversample_factor
        
        # Validate weights
        if len(weights) != len(dataset):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match dataset size ({len(dataset)})"
            )
        
        # Convert weights to tensor
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        
        # Calculate samples per replica
        num_samples = len(self.dataset)
        
        # Generate MORE samples than dataset size to ensure:
        # 1. Minority class is oversampled (goal of weighted sampling)
        # 2. Majority class samples still appear (avoid dropping them completely)
        # With replacement=True, we can sample same index multiple times
        # Generating more samples ensures better coverage of all samples
        total_samples_to_generate = int(num_samples * oversample_factor)
        
        # Ensure total_size is divisible by num_replicas
        self.total_size = (total_samples_to_generate // self.num_replicas) * self.num_replicas
        self.num_samples = self.total_size // self.num_replicas
        
        # Calculate indices for this rank
        # First, generate weighted samples for the entire dataset
        # Then, partition them across replicas
        self._generate_indices()
    
    def _generate_indices(self):
        """Generate weighted sample indices and partition them across replicas."""
        # Set seed for reproducibility
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        
        # Generate weighted samples for entire dataset
        # This ensures consistent sampling across epochs when seed is fixed
        all_indices = torch.multinomial(
            self.weights,
            num_samples=self.total_size,
            replacement=self.replacement,
            generator=generator
        ).tolist()
        
        # Partition indices for this rank
        # Each rank gets a contiguous chunk
        start_idx = self.rank * self.num_samples
        end_idx = start_idx + self.num_samples
        self.indices = all_indices[start_idx:end_idx]
    
    def __iter__(self):
        """Return an iterator over the indices for this rank."""
        return iter(self.indices)
    
    def __len__(self):
        """Return the number of samples for this rank."""
        return self.num_samples
    
    def set_epoch(self, epoch: int):
        """
        Set the epoch for this sampler.
        
        When called, this will regenerate the indices with a new seed
        based on the epoch number. This ensures different shuffling each epoch.
        
        Args:
            epoch: Current epoch number
        """
        # Update seed based on epoch to get different samples each epoch
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        
        # Regenerate weighted samples
        all_indices = torch.multinomial(
            self.weights,
            num_samples=self.total_size,
            replacement=self.replacement,
            generator=generator
        ).tolist()
        
        # Partition for this rank
        start_idx = self.rank * self.num_samples
        end_idx = start_idx + self.num_samples
        self.indices = all_indices[start_idx:end_idx]


def compute_class_weights(labels: List[int], power: float = 1.0) -> List[float]:
    """
    Compute class weights for weighted sampling.
    
    Args:
        labels: List of class labels (0 or 1 for binary classification)
        power: Power to apply to weights (1.0=standard, 0.5=sqrt=gentle, 1.5=more aggressive)
    
    Returns:
        List of weights for each sample
    """
    import numpy as np
    from collections import Counter
    
    # Count classes
    class_counts = Counter(labels)
    n_samples = len(labels)
    n_classes = len(class_counts)
    
    # Compute inverse frequency weights
    class_weights = {}
    for cls in class_counts:
        # Standard inverse frequency: n_samples / (n_classes * count)
        weight = n_samples / (n_classes * class_counts[cls])
        # Apply power transformation
        weight = weight ** power
        class_weights[cls] = weight
    
    # Assign weight to each sample based on its class
    weights = [class_weights[label] for label in labels]
    
    return weights


def compute_sample_weights_from_labels(
    labels: List[int],
    class_weight_power: float = 1.0
) -> List[float]:
    """
    Compute sample weights from labels for use with DistributedWeightedSampler.
    
    This function computes weights based on CLASS frequencies (0 vs 1).
    Samples with the same class label get the same weight.
    
    How it works:
    1. Counts how many samples belong to each class
    2. Computes inverse frequency: weight = n_samples / (n_classes * class_count)
    3. Applies power transformation
    4. Assigns weight to each sample based on its class
    
    Args:
        labels: List of class labels (e.g., [0, 1, 0, 1, ...])
        class_weight_power: Power to apply to weights (default: 1.0)
                           - 1.0: Standard inverse frequency
                           - 0.5: Square root (gentler weighting)
                           - 1.5: More aggressive weighting
    
    Returns:
        List of weights, one per sample
    
    Example:
        >>> labels = [0, 0, 0, 1, 1]  # 3 class 0, 2 class 1
        >>> weights = compute_sample_weights_from_labels(labels)
        >>> # Class 0 weight: 5/(2*3) = 0.833
        >>> # Class 1 weight: 5/(2*2) = 1.25
        >>> # Returns: [0.833, 0.833, 0.833, 1.25, 1.25]
    """
    return compute_class_weights(labels, power=class_weight_power)


def compute_sample_weights_from_dataset_source(
    dataset_sources: List[str],
    dataset_weight_power: float = 1.0,
    custom_weights: Optional[Dict[str, float]] = None
) -> List[float]:
    """
    Compute sample weights based on dataset source (MMLU, MMLU-Pro, GSM8K).
    
    This function weights samples based on which dataset they came from,
    allowing you to balance representation across different datasets.
    
    How it works:
    1. Counts how many samples from each dataset
    2. Computes inverse frequency weights OR uses custom weights
    3. Applies power transformation
    4. Assigns weight to each sample based on its dataset source
    
    Args:
        dataset_sources: List of dataset names (e.g., ["MMLU", "MMLU-Pro", "GSM8K", ...])
        dataset_weight_power: Power to apply to weights (default: 1.0)
                            - 1.0: Standard inverse frequency
                            - 0.5: Square root (gentler weighting)
                            - 1.5: More aggressive weighting
        custom_weights: Optional dict to manually set weights for each dataset
                       (e.g., {"MMLU": 1.0, "MMLU-Pro": 2.0, "GSM8K": 0.5})
                       If provided, overrides inverse frequency calculation
    
    Returns:
        List of weights, one per sample
    
    Example:
        >>> sources = ["MMLU", "MMLU", "MMLU-Pro", "GSM8K", "GSM8K"]
        >>> # MMLU: 2 samples, MMLU-Pro: 1, GSM8K: 2
        >>> weights = compute_sample_weights_from_dataset_source(sources)
        >>> # MMLU weight: 5/(3*2) = 0.833
        >>> # MMLU-Pro weight: 5/(3*1) = 1.667
        >>> # GSM8K weight: 5/(3*2) = 0.833
        >>> # Returns: [0.833, 0.833, 1.667, 0.833, 0.833]
        
        >>> # Or use custom weights:
        >>> custom = {"MMLU": 1.0, "MMLU-Pro": 2.0, "GSM8K": 0.5}
        >>> weights = compute_sample_weights_from_dataset_source(sources, custom_weights=custom)
        >>> # Returns: [1.0, 1.0, 2.0, 0.5, 0.5]
    """
    from collections import Counter
    
    # Count samples per dataset
    dataset_counts = Counter(dataset_sources)
    n_samples = len(dataset_sources)
    n_datasets = len(dataset_counts)
    
    if custom_weights is not None:
        # Use custom weights if provided
        dataset_weights = {}
        for dataset in dataset_counts:
            if dataset in custom_weights:
                weight = custom_weights[dataset]
                # Apply power transformation
                weight = weight ** dataset_weight_power
                dataset_weights[dataset] = weight
            else:
                # If dataset not in custom_weights, use inverse frequency
                weight = n_samples / (n_datasets * dataset_counts[dataset])
                weight = weight ** dataset_weight_power
                dataset_weights[dataset] = weight
    else:
        # Compute inverse frequency weights
        dataset_weights = {}
        for dataset in dataset_counts:
            # Standard inverse frequency: n_samples / (n_datasets * count)
            weight = n_samples / (n_datasets * dataset_counts[dataset])
            # Apply power transformation
            weight = weight ** dataset_weight_power
            dataset_weights[dataset] = weight
    
    # Assign weight to each sample based on its dataset source
    weights = [dataset_weights[source] for source in dataset_sources]
    
    return weights


def compute_sample_weights_from_combination(
    dataset_sources: List[str],
    labels: List[int],
    combination_weight_power: float = 1.0
) -> List[float]:
    """
    Compute sample weights based on the combination of dataset source AND label.
    
    This function weights samples based on (dataset_source, label) pairs,
    allowing you to balance representation across both datasets AND success/failure cases.
    
    Goal: Balance batches so each dataset has roughly equal representation of:
    - Failures (label=0)
    - Successes (label=1)
    
    How it works:
    1. Creates (dataset_source, label) pairs for each sample
    2. Counts how many samples belong to each combination
    3. Computes inverse frequency weights for each combination
    4. Applies power transformation
    5. Assigns weight to each sample based on its combination
    
    Args:
        dataset_sources: List of dataset names (e.g., ["MMLU", "MMLU-Pro", "GSM8K", ...])
        labels: List of class labels (0 or 1 for binary classification)
        combination_weight_power: Power to apply to weights (default: 1.0)
                                - 1.0: Standard inverse frequency
                                - 0.5: Square root (gentler weighting)
                                - 1.5: More aggressive weighting
    
    Returns:
        List of weights, one per sample
    
    Example:
        >>> sources = ["GSM8K", "GSM8K", "MMLU", "MMLU", "MMLU-Pro"]
        >>> labels = [0, 1, 0, 1, 0]
        >>> # (GSM8K, 0): 1 sample
        >>> # (GSM8K, 1): 1 sample
        >>> # (MMLU, 0): 1 sample
        >>> # (MMLU, 1): 1 sample
        >>> # (MMLU-Pro, 0): 1 sample
        >>> weights = compute_sample_weights_from_combination(sources, labels)
        >>> # All combinations have equal count, so all weights are equal
    """
    from collections import Counter
    
    if len(dataset_sources) != len(labels):
        raise ValueError(
            f"Length mismatch: dataset_sources ({len(dataset_sources)}) != labels ({len(labels)})"
        )
    
    # Convert labels to ints (handle float labels like 0.0, 1.0)
    labels_int = [int(label) for label in labels]
    
    # Create (dataset_source, label) pairs
    combinations = [(source, label) for source, label in zip(dataset_sources, labels_int)]
    
    # Count samples per combination
    combination_counts = Counter(combinations)
    n_samples = len(combinations)
    n_combinations = len(combination_counts)
    
    # Compute inverse frequency weights for each combination
    combination_weights = {}
    for combination in combination_counts:
        # Standard inverse frequency: n_samples / (n_combinations * count)
        weight = n_samples / (n_combinations * combination_counts[combination])
        # Apply power transformation
        weight = weight ** combination_weight_power
        combination_weights[combination] = weight
    
    # Assign weight to each sample based on its combination
    weights = [combination_weights[(source, label)] for source, label in zip(dataset_sources, labels_int)]
    
    return weights

