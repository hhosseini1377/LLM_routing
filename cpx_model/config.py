from dataclasses import dataclass, field
from typing import Union, List
@dataclass
class CPXDatasetConfig:
    """Configuration for CPX datasets (model-agnostic)"""
    DATA_DIR = "./generate_dataset/datasets"
    TRAIN_FILE = "train_routerbench_0shot_512_left_truncated_cleaned.pkl"
    TEST_FILE = "test_routerbench_0shot_512_left_truncated_cleaned.pkl"
    MMLU_DATA_DIR = "./generate_dataset/datasets/MMLU"
    MMLU_TRAIN_FILE = "mmlu_auxiliary_and_all_with_correct_counts_n5_train.pkl"
    MMLU_TEST_FILE = "mmlu_auxiliary_and_all_with_correct_counts_n5_val.pkl"
    MMLU_VALIDATION_FILE = "mmlu_auxiliary_and_all_with_correct_counts_n5_val.pkl"
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
    cpx_tokens: List[str] = field(default_factory=lambda: ['[CPX1]', '[CPX2]'])  # Single token string or list of tokens: ['[CPX]'] or ['[CPX1]', '[CPX2]', '[CPX3]']
    is_cpx_token_trainable: bool = True
    cpx_token_ids: List[int] = None  # list[int] for multiple tokens
    cpx_aggregation: str = 'mean'  # Options: 'mean', 'max', 'sum', 'attention', 'first'
    
    # Training settings
    METRIC: str = "f1"
    LOSS: str = "bce"
    dropout_rate: float = 0.1
    classifier_dropout: bool = True
    use_lora: bool = True
    mask_lora_for_non_cpx: bool = True
    use_class_weights: bool = False  # Enable class weighting for imbalanced datasets
    class_weight_power: float = 0.5  # Power to apply to class weights (1.0=standard, 0.5=sqrt=gentle, 1.5=more aggressive)
    
    # Component-specific learning rates (optimized for CPX + LoRA)
    # Reduced slightly to prevent overfitting based on your results
    classifier_lr: float = 3e-4        # Reduced from 5e-4 to prevent overfitting
    embedding_lr: float = 1e-4         # CPX token embedding (unchanged)
    lora_lr: float = 1.5e-4              # Reduced from 2e-4 to prevent overfitting

    
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
    warmup_steps: float = 0.05
    gradient_checkpointing: bool = True  # Enable gradient checkpointing to reduce memory usage
    max_grad_norm: float = 1.0  # Gradient clipping threshold
    patience: int = 3  # More aggressive early stopping to prevent overfitting
    amsgrad: bool = True  # Enable AMSGrad for better convergence stability
    label_smoothing: float = 0.1  # Add label smoothing to improve calibration and reduce overfitting

    # LoRA settings
    # Increased rank for better capacity with masked LoRA (only CPX token benefits)
    lora_r: int = 16  # Increased from 8 for better expressivity
    lora_alpha: int = 32  # Keep alpha = 2*r ratio for stable scaling
    lora_dropout: float = 0.15  # Slightly increased to reduce overfitting
    # Optimized for CPX token classification with masked LoRA:
    # - q_proj: CRITICAL - Controls what the CPX token attends to in context (main info extraction)
    # - o_proj: CRITICAL - Final attention output that flows to hidden states
    # - gate_proj: Important for Mistral's SwiGLU gating mechanism
    # - up_proj, down_proj: MLP transformations after attention
    # Note: v_proj only affects V[cpx], not V[context], so limited benefit for context extraction
    # Note: k_proj only affects how CPX presents itself as a key, less important for classification
    lora_target_modules: list[str] = field(default_factory=lambda: ['q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
)
    lora_bias: str = "none"
    lora_task_type: str = "CAUSAL_LM"
    freeze_LoRA_layers: bool = False
    freeze_LoRA_start_layer_idx: int = 0