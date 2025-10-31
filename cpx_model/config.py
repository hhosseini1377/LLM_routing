from dataclasses import dataclass, field

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
    GSM8K_DATA_DIR = "./generate_dataset/datasets/GSM8K"
    GSM8K_TRAIN_FILE = "gsm8k_generated_data_train.pkl"
    GSM8K_TEST_FILE = "gsm8k_generated_data_test.pkl"
    MIX_DATA_DIR = "./generate_dataset/datasets/mix"
    MIX_TRAIN_FILE = "mmlu_and_gsm8k_with_correct_train.pkl"
    MIX_TEST_FILE = "mmlu_and_gsm8k_with_correct_test.pkl"
    MIX_VALIDATION_FILE = "mmlu_and_gsm8k_with_correct_val.pkl"
@dataclass
class CPXTrainingConfig:
    """General training configuration for CPX wrapper (works with any Causal LM)"""
    dataset: str = "gsm8k"
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    
    # CPX-specific settings
    cpx_token: str = '[CPX]'
    is_cpx_token_trainable: bool = True
    cpx_token_id: int = None
    
    # Training settings
    METRIC: str = "f1"
    LOSS: str = "bce"
    dropout_rate: float = 0.1
    classifier_dropout: bool = True
    use_lora: bool = True
    mask_lora_for_non_cpx: bool = False
    
    # Component-specific learning rates (optimized for CPX + LoRA)
    classifier_lr: float = 5e-4        # New classifier head
    embedding_lr: float = 1e-4         # CPX token embedding
    lora_lr: float = 2e-4              # LoRA adapter layers

    
    # Weight decay
    weight_decay: float = 0.01
    embedding_weight_decay: float = 0.0  # No regularization for single embedding
    
    num_labels: int = 1  # For binary classification
    
    # Evaluation settings
    evaluation_batch_size: int = 32
    evaluation_size: int = 1000
    
    # Paths
    LOG_DIR: str = "./cpx_model/results_logs"
    MODEL_DIR: str = "./cpx_model/finetuned_models"
    
    # Training hyperparameters
    context_window: int = 1024  # Reduced from 8192 to fit in memory (4x less memory usage)
    scheduler: str = "linear"  # Options: "linear", "cosine", "ReduceLROnPlateau"
    warmup_steps: float = 0.1
    gradient_checkpointing: bool = True  # Enable gradient checkpointing to reduce memory usage
    max_grad_norm: float = 1.0  # Gradient clipping threshold
    patience: int = 3

    # LoRA settings
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ['q_proj', 'o_proj'])
    lora_bias: str = "none"
    lora_task_type: str = "CAUSAL_LM"
    freeze_LoRA_layers: bool = False
    freeze_LoRA_start_layer_idx: int = 0