from routing_dataset.run_prompts import run_prompts_with_vllm, add_correct_labels
from routing_dataset.configs import SamplerConfig
from dataclasses import asdict
from routing_dataset.dataset_paths import *
from routing_dataset.dataset_types import MMLUSplit
from datasets import load_from_disk, Dataset
from routing_dataset.load_dataset import load_mmlu_pro_dataset, load_mmlu_split, load_gsm8k_split
from routing_dataset.run_prompts import run_prompts_with_vllm
import pickle

def main():

    # load the normal mmlu pro test dataset
    dataset = load_mmlu_pro_dataset("test", output_file=MMLU_PRO_TEST_PROMPTS_FILE, model_type="qwen")
    results = run_prompts_with_vllm(dataset, model_name="Qwen/Qwen3-8B", temperature=0, top_p=0.9, max_tokens=512, gpu_memory_utilization=0.8, max_num_seqs=2048, tensor_parallel_size=2)
    with open(MMLU_PRO_TEST_QWEN8B_RESULTS_FILE, 'wb') as f:
        pickle.dump(results, f)
    add_correct_labels(MMLU_PRO_TEST_QWEN8B_RESULTS_FILE, MMLU_PRO_TEST_QWEN8B_CORRECT_RESULTS_FILE)

    # load the normal mmlu pro validation dataset
    dataset = load_mmlu_pro_dataset("validation", output_file=MMLU_PRO_VALIDATION_PROMPTS_FILE, model_type="qwen")
    results = run_prompts_with_vllm(dataset, model_name="Qwen/Qwen3-8B", temperature=0, top_p=0.9, max_tokens=512, gpu_memory_utilization=0.8, max_num_seqs=2048, tensor_parallel_size=2)
    with open(MMLU_PRO_VALIDATION_QWEN8B_RESULTS_FILE, 'wb') as f:
        pickle.dump(results, f)
    add_correct_labels(MMLU_PRO_VALIDATION_QWEN8B_RESULTS_FILE, MMLU_PRO_VALIDATION_QWEN8B_CORRECT_RESULTS_FILE)

if __name__ == "__main__":
    main()