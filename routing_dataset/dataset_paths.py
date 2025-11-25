"""
Path constants for dataset files.

This module contains Path objects pointing to dataset files and directories.
All paths are relative to the project root.
"""
from pathlib import Path

DATA_DIR = Path("./routing_dataset/datasets")

MMLU_DATA_DIR = DATA_DIR / "mmlu"
GSM8K_DATA_DIR = DATA_DIR / "gsm8k"

# MMLU prompts 
MMLU_AUXILIARY_PROMPTS_FILE = MMLU_DATA_DIR / "mmlu_auxiliary_prompts.pkl"
MMLU_TEST_PROMPTS_FILE = MMLU_DATA_DIR / "mmlu_test_prompts.pkl"
MMLU_VALIDATION_PROMPTS_FILE = MMLU_DATA_DIR / "mmlu_validation_prompts.pkl"
MMLU_DEV_PROMPTS_FILE = MMLU_DATA_DIR / "mmlu_dev_prompts.pkl"

# MMLU results on Qwen8B
MMLU_AUXILIARY_QWEN8B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_auxiliary_qwen8b_results.pkl"
MMLU_TEST_QWEN8B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_test_qwen8b_results.pkl"
MMLU_VALIDATION_QWEN8B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_validation_qwen8b_results.pkl"
MMLU_DEV_QWEN8B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_dev_qwen8b_results.pkl"

# MMLU with correct labels on Qwen8B
MMLU_AUXILIARY_QWEN8B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_auxiliary_qwen8b_correct_results.pkl"
MMLU_TEST_QWEN8B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_test_qwen8b_correct_results.pkl"
MMLU_VALIDATION_QWEN8B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_validation_qwen8b_correct_results.pkl"
MMLU_DEV_QWEN8B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_dev_qwen8b_correct_results.pkl"

# MMLU-Pro prompts
MMLU_PRO_TEST_PROMPTS_FILE = MMLU_DATA_DIR / "mmlu_pro_test_prompts.pkl"
MMLU_PRO_VALIDATION_PROMPTS_FILE = MMLU_DATA_DIR / "mmlu_pro_validation_prompts.pkl"

# MMLU-Pro results on Qwen8B
MMLU_PRO_TEST_QWEN8B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_pro_test_qwen8b_results.pkl"
MMLU_PRO_VALIDATION_QWEN8B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_pro_validation_qwen8b_results.pkl"

# MMLU-Pro with correct labels on Qwen8B
MMLU_PRO_TEST_QWEN8B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_pro_test_qwen8b_correct_results.pkl"
MMLU_PRO_VALIDATION_QWEN8B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_pro_validation_qwen8b_correct_results.pkl"

# GSM8K prompts
# Note: GSM8K dataset uses "main" config with "train" and "test" splits
GSM8K_TRAIN_PROMPTS_FILE = GSM8K_DATA_DIR / "gsm8k_train_prompts.pkl"
GSM8K_TEST_PROMPTS_FILE = GSM8K_DATA_DIR / "gsm8k_test_prompts.pkl"

# GSM8K results on Qwen8B
GSM8K_TRAIN_QWEN8B_RESULTS_FILE = GSM8K_DATA_DIR / "gsm8k_train_qwen8b_results.pkl"
GSM8K_TEST_QWEN8B_RESULTS_FILE = GSM8K_DATA_DIR / "gsm8k_test_qwen8b_results.pkl"

# GSM8K with correct labels on Qwen8B
GSM8K_TRAIN_QWEN8B_CORRECT_RESULTS_FILE = GSM8K_DATA_DIR / "gsm8k_train_qwen8b_correct_results.pkl"
GSM8K_TEST_QWEN8B_CORRECT_RESULTS_FILE = GSM8K_DATA_DIR / "gsm8k_test_qwen8b_correct_results.pkl"

# Combined dataset files (combining multiple splits)
MMLU_COMBINED_FILE = MMLU_DATA_DIR / "mmlu_combined.pkl"
MMLU_PRO_COMBINED_FILE = MMLU_DATA_DIR / "mmlu_pro_combined.pkl"
GSM8K_COMBINED_FILE = GSM8K_DATA_DIR / "gsm8k_combined.pkl"

# Final splits files (train/val/test)
FINAL_SPLITS_DIR = DATA_DIR / "final_splits"
FINAL_TRAIN_FILE = FINAL_SPLITS_DIR / "train.pkl"
FINAL_VAL_FILE = FINAL_SPLITS_DIR / "val.pkl"
FINAL_TEST_FILE = FINAL_SPLITS_DIR / "test.pkl"