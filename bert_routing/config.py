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

#TODO: Specifiy the attributes as class variables
@dataclass
class TrainingConfig:
    model_name: str = "distilbert"
    data_size: str = "None"
    dataset: str = "gsm8k"
    METRIC = "f1"
    LOSS = "bce"
    dropout_rate = 0.1
    classifier_dropout = True
    weight_decay = 0.01
    evaluation_batch_size = 128
    layers_to_freeze = 4
    freeze_layers = False
    LOG_DIR = "./bert_routing/results_logs"
    MODEL_DIR = "./bert_routing/finetuned_models"
    data_size = "None"
    strategy = "cls"
    context_window = 512
    num_epochs = 200
    scheduler = "cosine"
    warmup_steps = 0.1
    classifier_type = "linear"  # Options: "linear" or "mlp"
    mlp_hidden_size = 512  # Hidden layer size for MLP classifier
    
    # Optimizer improvements
    max_grad_norm = 1.0  # Gradient clipping threshold for stability
    betas = (0.9, 0.999)  # Adam/AdamW beta parameters
    eps = 1e-8  # Adam epsilon for numerical stability
    amsgrad = False  # AMSGrad variant for Adam

    # Learning rates
    embedding_lr = 1e-5
    classifier_lr = 1e-4
    model_lr = 2e-5
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