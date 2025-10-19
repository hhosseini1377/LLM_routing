from dataclasses import dataclass


@dataclass
class CPXDatasetConfig:
    """Configuration for CPX datasets (model-agnostic)"""
    DATA_DIR = "./generate_dataset/datasets"
    TRAIN_FILE = "train_routerbench_0shot_512_left_truncated_cleaned.pkl"
    TEST_FILE = "test_routerbench_0shot_512_left_truncated_cleaned.pkl"
    MMLU_DATA_DIR = "./generate_dataset/datasets/MMLU"
    MMLU_TRAIN_FILE = "mmlu_auxiliary_train_n5_t0.8.pkl"
    MMLU_TEST_FILE = "mmlu_auxiliary_test_n5_t0.8.pkl"
    MMLU_VALIDATION_FILE = "mmlu_auxiliary_validation_n5_t0.8.pkl"


@dataclass
class CPXTrainingConfig:
    """General training configuration for CPX wrapper (works with any Causal LM)"""
    # CPX-specific settings
    cpx_token = '[CPX]'
    is_cpx_token_trainable = True
    
    # Training settings
    METRIC = "f1"
    LOSS = "bce"
    dropout_rate = 0.1
    classifier_dropout = True
    learning_rate = 1e-5
    weight_decay = 0.01
    num_labels = 1  # For binary classification
    
    # Evaluation settings
    evaluation_batch_size = 8
    evaluation_size = 1000
    
    # Model freezing settings
    layers_to_freeze = 2
    freeze_layers = True
    
    # Paths
    LOG_DIR = "./cpx_model/results_logs"
    MODEL_DIR = "./cpx_model/finetuned_models"
    
    # Training hyperparameters
    data_size = "None"
    strategy = "cls"
    context_window = 8192
    num_epochs = 200
    scheduler = "linear"
    warmup_steps = 0.1
    gradient_checkpointing = True  # Enable gradient checkpointing to reduce memory usage


# Backward compatibility aliases
CPXMistralDatasetConfig = CPXDatasetConfig
MistralTrainingConfig = CPXTrainingConfig