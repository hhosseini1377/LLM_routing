from dataclasses import dataclass

from datasets.utils.version import dataclasses

@dataclass
class CPXMistralDatasetConfig:
    DATA_DIR = "./generate_dataset/datasets"
    TRAIN_FILE = "train_routerbench_0shot_512_left_truncated_cleaned.pkl"
    TEST_FILE = "test_routerbench_0shot_512_left_truncated_cleaned.pkl"
    MMLU_DATA_DIR = "./generate_dataset/datasets/MMLU"
    MMLU_TRAIN_FILE = "mmlu_auxiliary_train.pkl"
    MMLU_TEST_FILE = "mmlu_auxiliary_test.pkl"
    MMLU_VALIDATION_FILE = "mmlu_auxiliary_validation.pkl"

@dataclass
class MistralTrainingConfig:
    cpx_token = '[CPX]'
    METRIC = "f1"
    LOSS = "bce"
    dropout_rate = 0.1
    classifier_dropout = True
    learning_rate = 1e-5  # Reduced learning rate for stability
    weight_decay = 0.01
    evaluation_batch_size = 128
    layers_to_freeze = 2
    freeze_layers = True
    LOG_DIR = "./cpx_model/cpxmistral/results_logs"
    MODEL_DIR = "./cpx_model/cpxmistral/finetuned_models"
    model_name = "distilbert"
    data_size = "None"
    strategy = "cls"
    context_window = 8192
    num_epochs = 200
    scheduler = "linear"
    warmup_steps = 0.1