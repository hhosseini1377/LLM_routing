from routing_dataset.judge_apps_vllm import judge_apps_with_vllm
from routing_dataset.dataset_paths import (
    APPS_PROMPTS_WITH_ANSWERS_QWEN8B_FILE,
    APPS_PROMPTS_WITH_CORRECT_LABELS_QWEN8B_FILE,
)

results = judge_apps_with_vllm(
    input_file=APPS_PROMPTS_WITH_ANSWERS_QWEN8B_FILE,
    output_file=APPS_PROMPTS_WITH_CORRECT_LABELS_QWEN8B_FILE,
    judge_model="Qwen/Qwen2.5-32B-Instruct",
    tensor_parallel_size=4,
    temperature=0.0,
    max_tokens=512,
)