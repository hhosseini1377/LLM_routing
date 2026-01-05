"""
Path constants for dataset files.

This module contains Path objects pointing to dataset files and directories.
All paths are relative to the project root.
"""
from pathlib import Path

DATA_DIR = Path("./routing_dataset/datasets")

MMLU_DATA_DIR = DATA_DIR / "mmlu"
GSM8K_DATA_DIR = DATA_DIR / "gsm8k"
HOTPOTQA_DATA_DIR = DATA_DIR / "hotpotqa"
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

# MMLU results on Qwen3-4B
MMLU_AUXILIARY_QWEN34B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_auxiliary_qwen34b_results.pkl"
MMLU_TEST_QWEN34B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_test_qwen34b_results.pkl"
MMLU_VALIDATION_QWEN34B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_validation_qwen34b_results.pkl"
MMLU_DEV_QWEN34B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_dev_qwen34b_results.pkl"

# MMLU results on Qwen3-1.7B
MMLU_AUXILIARY_QWEN17B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_auxiliary_qwen17b_results.pkl"
MMLU_TEST_QWEN17B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_test_qwen17b_results.pkl"
MMLU_VALIDATION_QWEN17B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_validation_qwen17b_results.pkl"
MMLU_DEV_QWEN17B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_dev_qwen17b_results.pkl"

# MMLU results on Qwen3-1.7B
MMLU_AUXILIARY_QWEN17B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_auxiliary_qwen17b_results.pkl"
MMLU_TEST_QWEN17B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_test_qwen17b_results.pkl"
MMLU_VALIDATION_QWEN17B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_validation_qwen17b_results.pkl"
MMLU_DEV_QWEN17B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_dev_qwen17b_results.pkl"

# MMLU with correct labels on Qwen8B
MMLU_AUXILIARY_QWEN8B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_auxiliary_qwen8b_correct_results.pkl"
MMLU_TEST_QWEN8B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_test_qwen8b_correct_results.pkl"
MMLU_VALIDATION_QWEN8B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_validation_qwen8b_correct_results.pkl"
MMLU_DEV_QWEN8B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_dev_qwen8b_correct_results.pkl"

# MMLU with correct labels on Qwen3-4B
MMLU_AUXILIARY_QWEN34B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_auxiliary_qwen34b_correct_results.pkl"
MMLU_TEST_QWEN34B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_test_qwen34b_correct_results.pkl"
MMLU_VALIDATION_QWEN34B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_validation_qwen34b_correct_results.pkl"
MMLU_DEV_QWEN34B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_dev_qwen34b_correct_results.pkl"

# MMLU with correct labels on Qwen3-1.7B
MMLU_AUXILIARY_QWEN17B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_auxiliary_qwen17b_correct_results.pkl"
MMLU_TEST_QWEN17B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_test_qwen17b_correct_results.pkl"
MMLU_VALIDATION_QWEN17B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_validation_qwen17b_correct_results.pkl"
MMLU_DEV_QWEN17B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_dev_qwen17b_correct_results.pkl"

# MMLU-Pro prompts
MMLU_PRO_TEST_PROMPTS_FILE = MMLU_DATA_DIR / "mmlu_pro_test_prompts.pkl"
MMLU_PRO_VALIDATION_PROMPTS_FILE = MMLU_DATA_DIR / "mmlu_pro_validation_prompts.pkl"

# MMLU-Pro results on Qwen8B
MMLU_PRO_TEST_QWEN8B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_pro_test_qwen8b_results.pkl"
MMLU_PRO_VALIDATION_QWEN8B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_pro_validation_qwen8b_results.pkl"

# MMLU-Pro with correct labels on Qwen8B
MMLU_PRO_TEST_QWEN8B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_pro_test_qwen8b_correct_results.pkl"
MMLU_PRO_VALIDATION_QWEN8B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_pro_validation_qwen8b_correct_results.pkl"

# MMLU-Pro results on Qwen3-1.7B
MMLU_PRO_TEST_QWEN17B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_pro_test_qwen17b_results.pkl"
MMLU_PRO_VALIDATION_QWEN17B_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_pro_validation_qwen17b_results.pkl"

# MMLU-Pro with correct labels on Qwen3-1.7B
MMLU_PRO_TEST_QWEN17B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_pro_test_qwen17b_correct_results.pkl"
MMLU_PRO_VALIDATION_QWEN17B_CORRECT_RESULTS_FILE = MMLU_DATA_DIR / "mmlu_pro_validation_qwen17b_correct_results.pkl"

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

# Final auxiliary splits files (train/val/test)
FINAL_AUXILIARY_TRAIN_FILE = FINAL_SPLITS_DIR / "auxiliary_train.pkl"
FINAL_AUXILIARY_VAL_FILE = FINAL_SPLITS_DIR / "auxiliary_val.pkl"
FINAL_AUXILIARY_TEST_FILE = FINAL_SPLITS_DIR / "auxiliary_test.pkl"

# Final axuliary splits files (train/val/test) on Qwen3-1.7B
FINAL_AUXILIARY_QWEN17B_TRAIN_FILE = FINAL_SPLITS_DIR / "auxiliary_qwen17b_train.pkl"
FINAL_AUXILIARY_QWEN17B_VAL_FILE = FINAL_SPLITS_DIR / "auxiliary_qwen17b_val.pkl"
FINAL_AUXILIARY_QWEN17B_TEST_FILE = FINAL_SPLITS_DIR / "auxiliary_qwen17b_test.pkl"

# Final axuliary splits files (train/val/test) on Qwen3-4B
FINAL_AUXILIARY_QWEN34B_TRAIN_FILE = FINAL_SPLITS_DIR / "auxiliary_qwen34b_train.pkl"
FINAL_AUXILIARY_QWEN34B_VAL_FILE = FINAL_SPLITS_DIR / "auxiliary_qwen34b_val.pkl"
FINAL_AUXILIARY_QWEN34B_TEST_FILE = FINAL_SPLITS_DIR / "auxiliary_qwen34b_test.pkl"

FINAL_MMLU_ALL_QWEN8B_TRAIN_FILE = FINAL_SPLITS_DIR / "mmlu_all_qwen8b_train.pkl"
FINAL_MMLU_ALL_QWEN8B_VAL_FILE = FINAL_SPLITS_DIR / "mmlu_all_qwen8b_val.pkl"
FINAL_MMLU_ALL_QWEN8B_TEST_FILE = FINAL_SPLITS_DIR / "mmlu_all_qwen8b_test.pkl"

# MMLU + MMLU-PRO + MMLU AUXILIARY dataset on Qwen8B
FINAL_MMLU_ALL_PRO_QWEN8B_TRAIN_FILE = FINAL_SPLITS_DIR / "mmlu_all_pro_qwen8b_train.pkl"
FINAL_MMLU_ALL_PRO_QWEN8B_VAL_FILE = FINAL_SPLITS_DIR / "mmlu_all_pro_qwen8b_val.pkl"
FINAL_MMLU_ALL_PRO_QWEN8B_TEST_FILE = FINAL_SPLITS_DIR / "mmlu_all_pro_qwen8b_test.pkl"
FINAL_MMLU_ALL_PRO_QWEN8B_TEST_FILE_BALANCED = FINAL_SPLITS_DIR / "mmlu_all_pro_qwen8b_test_balanced.pkl"
FINAL_MMLU_ALL_PRO_QWEN8B_TEST_FILE_512_LONG = FINAL_SPLITS_DIR / "mmlu_all_pro_qwen8b_test_balanced_512_long.pkl"

# MMLU + MMLU-PRO + MMLU AUXILIARY dataset on Qwen3-1.7B
FINAL_MMLU_ALL_PRO_QWEN17B_TRAIN_FILE = FINAL_SPLITS_DIR / "mmlu_all_pro_qwen17b_train.pkl"
FINAL_MMLU_ALL_PRO_QWEN17B_VAL_FILE = FINAL_SPLITS_DIR / "mmlu_all_pro_qwen17b_val.pkl"
FINAL_MMLU_ALL_PRO_QWEN17B_TEST_FILE = FINAL_SPLITS_DIR / "mmlu_all_pro_qwen17b_test.pkl"
FINAL_MMLU_ALL_PRO_QWEN17B_TEST_FILE_BALANCED = FINAL_SPLITS_DIR / "mmlu_all_pro_qwen17b_test_balanced.pkl"

# MMLU + MMLU-PRO + MMLU AUXILIARY + GSM8K dataset on Qwen8B
FINAL_MMLU_ALL_PRO_GSM8K_QWEN8B_TRAIN_FILE = FINAL_SPLITS_DIR / "mmlu_all_pro_gsm8k_qwen8b_train.pkl"
FINAL_MMLU_ALL_PRO_GSM8K_QWEN8B_VAL_FILE = FINAL_SPLITS_DIR / "mmlu_all_pro_gsm8k_qwen8b_val.pkl"
FINAL_MMLU_ALL_PRO_GSM8K_QWEN8B_TEST_FILE = FINAL_SPLITS_DIR / "mmlu_all_pro_gsm8k_qwen8b_test.pkl"

# HotpotQA with correct labels on Qwen8B (Original dataset contains validation and trian sets)
HOTPOTQA_QWEN8B_VALIDATION_CORRECT_LABELS_FILE = HOTPOTQA_DATA_DIR / "hotpotqa_qwen8b_validation_correct_results.pkl"
HOTPOTQA_QWEN8B_TRAIN_CORRECT_LABELS_FILE = HOTPOTQA_DATA_DIR / "hotpotqa_qwen8b_train_correct_results.pkl"
HOTPOTQA_QWEN8B_CORRECT_LABELS_FILE = HOTPOTQA_DATA_DIR / "hotpotqa_qwen8b_correct_results.pkl"

# HotpotQA dataset on Qwen8B
FINAL_HOTPOTQA_QWEN8B_TRAIN_FILE = HOTPOTQA_DATA_DIR / "hotpotqa_qwen8b_train.pkl"
FINAL_HOTPOTQA_QWEN8B_VAL_FILE = HOTPOTQA_DATA_DIR / "hotpotqa_qwen8b_val.pkl"
FINAL_HOTPOTQA_QWEN8B_TEST_FILE = HOTPOTQA_DATA_DIR / "hotpotqa_qwen8b_test_cleaned.pkl"

# HotpotQA with correct labels on Qwen1.7B (Original dataset contains validation and trian sets)
HOTPOTQA_QWEN34B_VALIDATION_CORRECT_LABELS_FILE = HOTPOTQA_DATA_DIR / "hotpotqa_qwen34b_validation_correct_results.pkl"
HOTPOTQA_QWEN34B_TRAIN_CORRECT_LABELS_FILE = HOTPOTQA_DATA_DIR / "hotpotqa_qwen34b_train_correct_results.pkl"
HOTPOTQA_QWEN34B_CORRECT_LABELS_FILE = HOTPOTQA_DATA_DIR / "hotpotqa_qwen34b_correct_results.pkl"

# HotpotQA with correct labels on Qwen1.7B (Original dataset contains validation and trian sets)
HOTPOTQA_QWEN17B_VALIDATION_CORRECT_LABELS_FILE = HOTPOTQA_DATA_DIR / "hotpotqa_qwen17b_validation_correct_results.pkl"
HOTPOTQA_QWEN17B_TRAIN_CORRECT_LABELS_FILE = HOTPOTQA_DATA_DIR / "hotpotqa_qwen17b_train_correct_results.pkl"
HOTPOTQA_QWEN17B_CORRECT_LABELS_FILE = HOTPOTQA_DATA_DIR / "hotpotqa_qwen17b_correct_results.pkl"

# HotpotQA dataset on Qwen3-1.7B
FINAL_HOTPOTQA_QWEN17B_TRAIN_FILE = HOTPOTQA_DATA_DIR / "hotpotqa_qwen17b_train.pkl"
FINAL_HOTPOTQA_QWEN17B_VAL_FILE = HOTPOTQA_DATA_DIR / "hotpotqa_qwen17b_val.pkl"
FINAL_HOTPOTQA_QWEN17B_TEST_FILE = HOTPOTQA_DATA_DIR / "hotpotqa_qwen17b_test_cleaned.pkl"

# MMLU + MMLU-PRO + MMLU AUXILIARY + GSM8K + HotpotQA dataset on Qwen8B
FINAL_MMLU_ALL_PRO_GSM8K_QWEN8B_HOTPOTQA_QWEN8B_TRAIN_FILE = FINAL_SPLITS_DIR / "mmlu_all_pro_gsm8k_qwen8b_hotpotqa_qwen8b_train.pkl"
FINAL_MMLU_ALL_PRO_GSM8K_QWEN8B_HOTPOTQA_QWEN8B_VAL_FILE = FINAL_SPLITS_DIR / "mmlu_all_pro_gsm8k_qwen8b_hotpotqa_qwen8b_val.pkl"
FINAL_MMLU_ALL_PRO_GSM8K_QWEN8B_HOTPOTQA_QWEN8B_TEST_FILE = FINAL_SPLITS_DIR / "mmlu_all_pro_gsm8k_qwen8b_hotpotqa_qwen8b_test.pkl"

# Apps dataset
APPS_DATA_DIR = DATA_DIR / "apps"

# Apps dataset on Qwen8B
APPS_PROMPTS_FILE = APPS_DATA_DIR / "apps_prompts.pkl"
APPS_PROMPTS_WITH_ANSWERS_QWEN8B_FILE = APPS_DATA_DIR / "apps_prompts_with_answers_qwen8b.pkl"
APPS_PROMPTS_WITH_CORRECT_LABELS_QWEN8B_FILE = APPS_DATA_DIR / "apps_prompts_with_correct_labels_qwen8b.pkl"
APPS_QWEN8B_TRAIN_FILE = APPS_DATA_DIR / "apps_qwen8b_train.pkl"
APPS_QWEN8B_VAL_FILE = APPS_DATA_DIR / "apps_qwen8b_val.pkl"
APPS_QWEN8B_TEST_FILE = APPS_DATA_DIR / "apps_qwen8b_test.pkl"

# CNN Daily Mail dataset
CNN_DAILY_MAIL_DATA_DIR = DATA_DIR / "cnn_dailymail"
CNN_DAILY_MAIL_PROMPTS_FILE = CNN_DAILY_MAIL_DATA_DIR / "cnn_dailymail_prompts.pkl"
CNN_DAILY_MAIL_PROMPTS_30000_FILE = CNN_DAILY_MAIL_DATA_DIR / "cnn_dailymail_prompts_30000.pkl"
CNN_DAILY_MAIL_PROMPTS_WITH_ANSWERS_QWEN8B_FILE = CNN_DAILY_MAIL_DATA_DIR / "cnn_dailymail_prompts_with_answers_qwen8b.pkl"
CNN_DAILY_MAIL_PROMPTS_WITH_CORRECT_LABELS_QWEN8B_FILE = CNN_DAILY_MAIL_DATA_DIR / "cnn_dailymail_prompts_with_correct_labels_qwen8b.pkl"
CNN_DAILY_MAIL_QWEN8B_TRAIN_FILE = CNN_DAILY_MAIL_DATA_DIR / "cnn_dailymail_qwen8b_train.pkl"
CNN_DAILY_MAIL_QWEN8B_VAL_FILE = CNN_DAILY_MAIL_DATA_DIR / "cnn_dailymail_qwen8b_val.pkl"
CNN_DAILY_MAIL_QWEN8B_TEST_FILE = CNN_DAILY_MAIL_DATA_DIR / "cnn_dailymail_qwen8b_test.pkl"

# LMSYS-Chat-1M dataset
LMSYS_CHAT1M_DATA_DIR = DATA_DIR / "lmsys_chat1m"
LMSYS_CHAT1M_PROMPTS_FILE = LMSYS_CHAT1M_DATA_DIR / "lmsys_chat1m_prompts.pkl"
LMSYS_CHAT1M_PROMPTS_100K_FILE = LMSYS_CHAT1M_DATA_DIR / "lmsys_chat1m_prompts_100k.pkl"
LMSYS_CHAT1M_PROMPTS_100K_FILE_CLEANED = LMSYS_CHAT1M_DATA_DIR / "lmsys_chat1m_prompts_100k_cleaned.pkl"
LMSYS_CHAT1M_PROMPTS_100K_WITH_ANSWERS_QWEN8B_FILE = LMSYS_CHAT1M_DATA_DIR / "lmsys_chat1m_prompts_100k_with_answers_qwen8b.pkl"
LMSYS_CHAT1M_PROMPTS_100K_WITH_SCORES_QWEN8B_FILE = LMSYS_CHAT1M_DATA_DIR / "lmsys_chat1m_prompts_100k_with_scores_qwen8b.pkl"
LMSYS_CHAT1M_PROMPTS_FILE_CLEANED = LMSYS_CHAT1M_DATA_DIR / "lmsys_chat1m_prompts_cleaned.pkl"
LMSYS_CHAT1M_PROMPTS_WITH_ANSWERS_QWEN8B_FILE = LMSYS_CHAT1M_DATA_DIR / "lmsys_chat1m_prompts_with_answers_qwen8b.pkl"
LMSYS_CHAT1M_PROMPTS_WITH_SCORES_QWEN8B_FILE = LMSYS_CHAT1M_DATA_DIR / "lmsys_chat1m_prompts_with_scores_qwen8b.pkl"
# LMSYS-Chat-1M train, test, validation files
LMSYS_CHAT1M_TRAIN_FILE = LMSYS_CHAT1M_DATA_DIR / "lmsys_chat1m_train.pkl"
LMSYS_CHAT1M_TEST_FILE = LMSYS_CHAT1M_DATA_DIR / "lmsys_chat1m_test.pkl"
LMSYS_CHAT1M_VALIDATION_FILE = LMSYS_CHAT1M_DATA_DIR / "lmsys_chat1m_validation.pkl"

# Dataset file mapping: (dataset_name, dataset_model_name) -> (train_file, val_file, test_file)
# dataset_name: e.g., 'auxiliary', 'test', 'validation'
# dataset_model_name: e.g., None (default), 'qwen17b', 'qwen34b', 'qwen4' (alias for qwen34b), 'qwen8b'
DATASET_FILE_MAP = {
    # Auxiliary dataset - no model specified (default)
    #TODO: add qwen8b to the auxiliary dataset
    ('mmlu_auxiliary', 'qwen8b'): (
        FINAL_AUXILIARY_TRAIN_FILE,
        FINAL_AUXILIARY_VAL_FILE,
        FINAL_AUXILIARY_TEST_FILE
    ),
    # Auxiliary dataset - Qwen3-1.7B
    ('mmlu_auxiliary', 'qwen17b'): (
        FINAL_AUXILIARY_QWEN17B_TRAIN_FILE,
        FINAL_AUXILIARY_QWEN17B_VAL_FILE,
        FINAL_AUXILIARY_QWEN17B_TEST_FILE
    ),
    # Auxiliary dataset - Qwen3-4B
    ('mmlu_auxiliary', 'qwen34b'): (
        FINAL_AUXILIARY_QWEN34B_TRAIN_FILE,
        FINAL_AUXILIARY_QWEN34B_VAL_FILE,
        FINAL_AUXILIARY_QWEN34B_TEST_FILE
    ),
    # Auxiliary dataset - Qwen4 (alias for qwen34b)
    ('mmlu_auxiliary', 'qwen4'): (
        FINAL_AUXILIARY_QWEN34B_TRAIN_FILE,
        FINAL_AUXILIARY_QWEN34B_VAL_FILE,
        FINAL_AUXILIARY_QWEN34B_TEST_FILE
    ),

    # MMLU all dataset - Qwen8B
    ('mmlu_original_auxiliary', 'qwen8b'): (
        FINAL_MMLU_ALL_QWEN8B_TRAIN_FILE,
        FINAL_MMLU_ALL_QWEN8B_VAL_FILE,
        FINAL_MMLU_ALL_QWEN8B_TEST_FILE
    ),
    
    # Combined dataset - no model specified (default)
    ('combined', 'qwen8b'): (
        FINAL_TRAIN_FILE,
        FINAL_VAL_FILE,
        FINAL_TEST_FILE
    ),
    # MMLU + MMLU-PRO + MMLU AUXILIARY dataset - Qwen8B
    ('mmlu_original_pro_auxiliary', 'qwen8b'): (
        FINAL_MMLU_ALL_PRO_QWEN8B_TRAIN_FILE,
        FINAL_MMLU_ALL_PRO_QWEN8B_VAL_FILE,
        FINAL_MMLU_ALL_PRO_QWEN8B_TEST_FILE
    ),
    # MMLU + MMLU-PRO + MMLU AUXILIARY dataset - Qwen3-1.7B
    ('mmlu_original_pro_auxiliary', 'qwen17b'): (
        FINAL_MMLU_ALL_PRO_QWEN17B_TRAIN_FILE,
        FINAL_MMLU_ALL_PRO_QWEN17B_VAL_FILE,
        FINAL_MMLU_ALL_PRO_QWEN17B_TEST_FILE
    ),

    # MMLU + MMLU-PRO + MMLU AUXILIARY + GSM8K dataset - Qwen8B
    ('mmlu_original_pro_auxiliary_gsm8k', 'qwen8b'): (
        FINAL_MMLU_ALL_PRO_GSM8K_QWEN8B_TRAIN_FILE,
        FINAL_MMLU_ALL_PRO_GSM8K_QWEN8B_VAL_FILE,
        FINAL_MMLU_ALL_PRO_GSM8K_QWEN8B_TEST_FILE
    ),

    # HotpotQA dataset - Qwen8B
    ('hotpotqa', 'qwen8b'): (
        FINAL_HOTPOTQA_QWEN8B_TRAIN_FILE,
        FINAL_HOTPOTQA_QWEN8B_VAL_FILE,
        FINAL_HOTPOTQA_QWEN8B_TEST_FILE
    ),

    # HotpotQA dataset - Qwen3-1.7B
    ('hotpotqa', 'qwen17b'): (
        FINAL_HOTPOTQA_QWEN17B_TRAIN_FILE,
        FINAL_HOTPOTQA_QWEN17B_VAL_FILE,
        FINAL_HOTPOTQA_QWEN17B_TEST_FILE
    ),

    # MMLU + MMLU-PRO + MMLU AUXILIARY + GSM8K + HotpotQA dataset - Qwen8B
    ('mmlu_original_pro_auxiliary_gsm8k_hotpotqa', 'qwen8b'): (
        FINAL_MMLU_ALL_PRO_GSM8K_QWEN8B_HOTPOTQA_QWEN8B_TRAIN_FILE,
        FINAL_MMLU_ALL_PRO_GSM8K_QWEN8B_HOTPOTQA_QWEN8B_VAL_FILE,
        FINAL_MMLU_ALL_PRO_GSM8K_QWEN8B_HOTPOTQA_QWEN8B_TEST_FILE
    ),

    # Apps dataset - Qwen8B
    ('apps', 'qwen8b'): (
        APPS_QWEN8B_TRAIN_FILE,
        APPS_QWEN8B_VAL_FILE,
        APPS_QWEN8B_TEST_FILE
    ),

    # LMSYS-Chat-1M dataset - Qwen8B
    ('lmsys_chat1m', 'qwen8b'): (
        LMSYS_CHAT1M_TRAIN_FILE,
        LMSYS_CHAT1M_VALIDATION_FILE,
        LMSYS_CHAT1M_TEST_FILE
    ),

    ('imdb', 'qwen8b'): (
        None
    ),

    # MNLI dataset - Qwen8B
    ('mnli', 'qwen8b'): (
        None
    ),

    # ANLI dataset - Qwen8B
    ('anli', 'qwen8b'): (
        None
    ),
}

AVAILABLE_DATASET_NAMES = list(DATASET_FILE_MAP.keys())

def get_dataset_files(dataset_name: str, dataset_model_name: str = None):
    """
    Get dataset file paths based on dataset name and model name.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'auxiliary', 'combined')
        dataset_model_name: Name of the model (e.g., 'qwen4', 'qwen17b', 'qwen34b', None)
    
    Returns:
        tuple: (train_file_path, val_file_path, test_file_path)
    
    Raises:
        ValueError: If the combination of dataset_name and dataset_model_name is not found
    """
    key = (dataset_name, dataset_model_name)
    
    if key not in DATASET_FILE_MAP:
        # Try with None as model_name for backward compatibility
        fallback_key = (dataset_name, None)
        if fallback_key in DATASET_FILE_MAP:
            return DATASET_FILE_MAP[fallback_key]
        else:
            available = list(DATASET_FILE_MAP.keys())
            raise ValueError(
                f"Dataset combination not found: dataset_name='{dataset_name}', "
                f"dataset_model_name='{dataset_model_name}'. "
                f"Available combinations: {available}"
            )
    
    return DATASET_FILE_MAP[key]