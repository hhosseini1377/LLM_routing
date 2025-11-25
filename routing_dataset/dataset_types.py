"""
Type definitions and constants for dataset splits.

This module contains type aliases and mappings for different dataset splits,
making it easy to extend support for new datasets.
"""
from typing import Literal, Dict
from pathlib import Path
from routing_dataset.dataset_paths import (
    MMLU_AUXILIARY_PROMPTS_FILE,
    MMLU_TEST_PROMPTS_FILE,
    MMLU_VALIDATION_PROMPTS_FILE,
    MMLU_DEV_PROMPTS_FILE,
    MMLU_PRO_TEST_PROMPTS_FILE,
    MMLU_PRO_VALIDATION_PROMPTS_FILE,
    GSM8K_TRAIN_PROMPTS_FILE,
    GSM8K_TEST_PROMPTS_FILE,
)

# Type aliases for valid split names
MMLUSplit = Literal["auxiliary", "test", "validation", "dev"]
MMLUProSplit = Literal["test", "validation"]
GSMSplit = Literal["train", "test"]

# Mapping from MMLU-Pro split names to output file paths
MMLU_PRO_SPLIT_TO_OUTPUT_FILE: Dict[MMLUProSplit, Path] = {
    "test": MMLU_PRO_TEST_PROMPTS_FILE,
    "validation": MMLU_PRO_VALIDATION_PROMPTS_FILE,
}
# Mapping from split names to output file paths
MMLU_SPLIT_TO_OUTPUT_FILE: Dict[MMLUSplit, Path] = {
    "auxiliary": MMLU_AUXILIARY_PROMPTS_FILE,
    "test": MMLU_TEST_PROMPTS_FILE,
    "validation": MMLU_VALIDATION_PROMPTS_FILE,
    "dev": MMLU_DEV_PROMPTS_FILE,
}

# Mapping from GSM split names to output file paths
GSM_SPLIT_TO_OUTPUT_FILE: Dict[GSMSplit, Path] = {
    "train": GSM8K_TRAIN_PROMPTS_FILE,
    "test": GSM8K_TEST_PROMPTS_FILE,
}


