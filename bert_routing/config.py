from dataclasses import dataclass

generator_config_with_sampling = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "repetition_penalty": 1.2,
    "do_sample": True,
}

generator_config_without_sampling = {
    "max_new_tokens": 1024,
    "temperature": 0.0,
    "top_p": 1.0,
    "do_sample": False,
}

@dataclass
class TrainingConfig:
    model_name: str = "deberta"
    data_size: str = "None"
    evaluation_size: str = "None"
    dataset_name: str = "mmlu"
    METRIC: str = "f1"
    LOSS: str = "bce"
    dropout_rate: float = 0.1
    classifier_dropout: bool = True
    weight_decay: float = 0.01
    evaluation_batch_size: int = 128
    layers_to_freeze: int = 4
    freeze_layers: bool = False
    LOG_DIR: str = "./bert_routing/results_logs"
    MODEL_DIR: str = "./bert_routing/finetuned_models"
    strategy: str = "cls"
    context_window: int = 512
    num_epochs: int = 10
    scheduler: str = "cosine"
    warmup_steps: float = 0.1
    classifier_type: str = "linear"  # Options: "linear" or "mlp"
    mlp_hidden_size: int = 512  # Hidden layer size for MLP classifier
    
    # Optimizer improvements
    max_grad_norm: float = 1.0  # Gradient clipping threshold for stability
    betas: tuple[float, float] = (0.9, 0.999)  # Adam/AdamW beta parameters
    eps: float = 1e-8  # Adam epsilon for numerical stability
    amsgrad: bool = False  # AMSGrad variant for Adam

    # Learning rates
    embedding_lr: float = 1e-5
    classifier_lr: float = 1e-4
    model_lr: float = 2e-5

    # Freeze the embedding layer
    freeze_embedding: bool = False
    
    # Weighted sampling options
    use_weighted_sampling: bool = False  # Enable weighted random sampling
    dataset_weight_power: float = 1.0  # Power to apply to dataset source weights
    sampling_weight_power: float = None  # Power to apply to weights for weighted sampling. If None, uses class_weight_power for backward compatibility.
    loss_weight_power: float = None  # Power to apply to class weights in loss function. If None, uses class_weight_power for backward compatibility.
    class_weight_power: float = 1.0  # DEPRECATED: Use sampling_weight_power and loss_weight_power instead. Kept for backward compatibility.
    use_class_weights: bool = False  # Enable class weighting in loss function (BCEWithLogitsLoss pos_weight)
    
class DatasetConfig:
    DATA_DIR = "./generate_dataset/datasets"
    TRAIN_FILE = "train_routerbench_0shot_512_left_truncated_cleaned.pkl"
    TEST_FILE = "test_routerbench_0shot_512_left_truncated_cleaned.pkl"
    MMLU_DATA_DIR = "./generate_dataset/datasets/MMLU"
    MIX_DATA_DIR = "./generate_dataset/datasets/mix"
    MMLU_TRAIN_FILE = "mmlu_auxiliary_train_n5_t0.8.pkl"
    MMLU_TEST_FILE = "mmlu_auxiliary_test_n5_t0.8.pkl"
    MMLU_VALIDATION_FILE = "mmlu_auxiliary_validation_n5_t0.8.pkl"
    MIX_TRAIN_FILE = "mmlu_and_gsm8k_with_correct_train.pkl"
    MIX_VALIDATION_FILE = "mmlu_and_gsm8k_with_correct_val.pkl"

MODEL_REGISTRY = {
    "llama3_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "deepseek_7b": "deepseek-ai/deepseek-llm-7b-chat",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "openchat_3.5": "openchat/openchat-3.5-1210",
    "phi_2": "microsoft/phi-2",
    "ultralm_13b": "openbmb/UltraLM-13B",
    'qwen-3b': "Qwen/Qwen2.5-VL-3B-Instruct"
}