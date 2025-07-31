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

class TrainingConfig:
    METRIC = "f1"
    LOSS = "bce"
    dropout_rate = 0.1
    classifier_dropout = True
    learning_rate = 3e-5
    weight_decay = 0.01
    evaluation_batch_size = 128
    layers_to_freeze = 2
    freeze_layers = True


class DatasetConfig:
    DATA_DIR = "./datasets"
    TRAIN_FILE = "train_routerbench_combined.pkl"
    TEST_FILE = "test_routerbench_combined.pkl"

MODEL_REGISTRY = {
    "llama3_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "deepseek_7b": "deepseek-ai/deepseek-llm-7b-chat",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "openchat_3.5": "openchat/openchat-3.5-1210",
    "phi_2": "microsoft/phi-2",
    "ultralm_13b": "openbmb/UltraLM-13B",
    'qwen-3b': "Qwen/Qwen2.5-VL-3B-Instruct"
}