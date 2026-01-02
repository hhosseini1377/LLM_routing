"""
Dataset loader functions for CPX and BERT routing model training.

This module provides a unified interface for loading different dataset types,
normalizing their return values to a consistent 6-tuple structure:
(train_texts, train_labels, train_dataset_sources, validation_texts, validation_labels, validation_dataset_sources)

Simple datasets return None for dataset_sources.

When cpx_tokens is False, loads datasets without CPX tokens (for BERT routing).
When cpx_tokens is a list or None, loads datasets with CPX tokens (for CPX routing).
"""
from cpx_model.cpx_causal_utils import (
    load_mmlu_data_with_cpx,
    load_gsm8k_data_with_cpx,
    load_mix_data_with_cpx,
    load_imdb_data_with_cpx,
    load_combined_data_with_cpx,
    load_gsm8k_data,
    load_mmlu_data,
    load_mix_data,
    load_imdb_data,
    load_combined_data
)
from routing_dataset.dataset_paths import get_dataset_files, FINAL_TRAIN_FILE, FINAL_VAL_FILE


def _load_gsm8k(cpx_tokens, dataset_name=None, dataset_model_name=None):
    """Load GSM8K dataset with or without CPX tokens."""
    if cpx_tokens is False:
        # Load without CPX tokens (for BERT routing)
        train_texts, train_labels, validation_texts, validation_labels = load_gsm8k_data()
    else:
        # Load with CPX tokens (for CPX routing)
        train_texts, train_labels, validation_texts, validation_labels = load_gsm8k_data_with_cpx(cpx_tokens)
    return train_texts, train_labels, None, validation_texts, validation_labels, None


def _load_mix(cpx_tokens, dataset_name=None, dataset_model_name=None):
    """Load MIX dataset with or without CPX tokens."""
    if cpx_tokens is False:
        # Load without CPX tokens (for BERT routing)
        train_texts, train_labels, validation_texts, validation_labels = load_mix_data()
    else:
        # Load with CPX tokens (for CPX routing)
        train_texts, train_labels, validation_texts, validation_labels = load_mix_data_with_cpx(cpx_tokens)
    return train_texts, train_labels, None, validation_texts, validation_labels, None


def _load_imdb(cpx_tokens, dataset_name=None, dataset_model_name=None):
    """Load IMDb dataset with or without CPX tokens."""
    if cpx_tokens is False:
        # Load without CPX tokens (for BERT routing)
        train_texts, train_labels, validation_texts, validation_labels = load_imdb_data()
    else:
        # Load with CPX tokens (for CPX routing)
        train_texts, train_labels, validation_texts, validation_labels = load_imdb_data_with_cpx(cpx_tokens)
    return train_texts, train_labels, None, validation_texts, validation_labels, None


def _load_single_dataset(cpx_tokens, dataset_name=None, dataset_model_name=None):
    """Load MMLU dataset with or without CPX tokens."""
    if cpx_tokens is False:
        # Load without CPX tokens (for BERT routing)
        train_texts, train_labels, validation_texts, validation_labels = load_mmlu_data(
            dataset_name=dataset_name, dataset_model_name=dataset_model_name
        )
    else:
        # Load with CPX tokens (for CPX routing)
        train_texts, train_labels, validation_texts, validation_labels = load_mmlu_data_with_cpx(
            cpx_tokens, dataset_name=dataset_name, dataset_model_name=dataset_model_name
        )
        print(f"Loaded MMLU dataset: dataset_name='{dataset_name}', dataset_model_name='{dataset_model_name}'")
    return train_texts, train_labels, None, validation_texts, validation_labels, None


def _load_combined(cpx_tokens, dataset_name=None, dataset_model_name=None):
    """Load combined dataset (MMLU + MMLU-Pro + GSM8K) with or without CPX tokens."""
    if cpx_tokens is False:
        train_path, validation_path, _ = get_dataset_files(dataset_name, dataset_model_name)
        # Load without CPX tokens (for BERT routing)
        # Use FINAL_TRAIN_FILE and FINAL_VAL_FILE for backward compatibility
        train_texts, train_labels, train_dataset_sources, validation_texts, validation_labels, validation_dataset_sources = \
            load_combined_data(str(train_path), str(validation_path))
    else:
        # Load with CPX tokens (for CPX routing)
        train_path, validation_path, _ = get_dataset_files(dataset_name, dataset_model_name)
        train_texts, train_labels, train_dataset_sources, validation_texts, validation_labels, validation_dataset_sources = \
            load_combined_data_with_cpx(str(train_path), str(validation_path), cpx_tokens)
        print(f"Loaded combined dataset: dataset_name='{dataset_name}', dataset_model_name='{dataset_model_name}'")
    return train_texts, train_labels, train_dataset_sources, validation_texts, validation_labels, validation_dataset_sources


# Dictionary mapping dataset_type to loader functions
DATASET_LOADERS = {
    'gsm8k': _load_gsm8k,
    'mix': _load_mix,
    'imdb': _load_imdb,
    'mmlu': _load_single_dataset,
    'combined': _load_combined,
    'mmlu_original_auxiliary': _load_single_dataset,
    'mmlu_original_pro_auxiliary': _load_single_dataset,
    'mmlu_auxiliary': _load_single_dataset,
    'mmlu_original_pro_auxiliary_gsm8k': _load_combined,
    'hotpotqa': _load_single_dataset,
    'mmlu_original_pro_auxiliary_gsm8k_hotpotqa': _load_single_dataset,
    'apps': _load_single_dataset,
    'lmsys_chat1m': _load_single_dataset,
}


def get_dataset_loader(dataset_type):
    """
    Get the loader function for a given dataset type.
    
    Args:
        dataset_type: Type of dataset ('gsm8k', 'mix', 'imdb', 'mmlu', 'combined')
    
    Returns:
        Loader function that takes (cpx_tokens, dataset_name, dataset_model_name) and returns
        (train_texts, train_labels, train_dataset_sources, validation_texts, validation_labels, validation_dataset_sources)
        
        Note: Pass cpx_tokens=False to load without CPX tokens (for BERT routing).
              Pass cpx_tokens as a list or None to load with CPX tokens (for CPX routing).
    
    Raises:
        ValueError: If dataset_type is not supported
    """
    if dataset_type not in DATASET_LOADERS:
        raise ValueError(f"Invalid dataset type: {dataset_type}. Supported types: {list(DATASET_LOADERS.keys())}")
    return DATASET_LOADERS[dataset_type]

