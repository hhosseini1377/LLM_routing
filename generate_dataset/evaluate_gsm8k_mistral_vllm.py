"""
Evaluate Mistral-7B-Instruct on the GSM8K benchmark using vLLM.
Saves results in both JSON and Pickle (.pkl) formats.

Requirements:
    pip install vllm datasets tqdm
"""

import re
import pickle
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm
from generate_dataset.generation_config import GSM8KGenerationConfig
# -----------------------------------------------------------
# 1. Load GSM8K dataset
# -----------------------------------------------------------
def load_gsm8k():
    dataset = load_dataset("gsm8k", "main")
    return dataset["test"]  # 1,319 samples


# -----------------------------------------------------------
# 2. Prompt formatting
# -----------------------------------------------------------
def format_prompt(question: str) -> str:
    return (
        "You are a helpful reasoning assistant.\n"
        f"Question: {question}\n"
        "Please show your reasoning step by step and end with 'Answer: <number>'."
    )

# -----------------------------------------------------------
# 3. Regex-based answer extraction
# -----------------------------------------------------------
def extract_final_answer(text: str):
    """Extract numeric answer after 'Answer:'."""
    match = re.search(r"Answer:\s*([0-9,.\-]+)", text)
    return match.group(1) if match else None


def get_ground_truth(answer_text: str):
    """Extract ground-truth number from the GSM8K field."""
    match = re.search(r"The answer is ([0-9,.\-]+)", answer_text)
    return match.group(1) if match else None


# -----------------------------------------------------------
# 4. Evaluation with vLLM
# -----------------------------------------------------------
def evaluate_vllm(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    save_prefix="mistral_gsm8k_results_vllm"
):
    print(f"Loading vLLM model: {model_name}")

    # Load model
    sampling_params = SamplingParams(temperature=GSM8KGenerationConfig.temperature, top_p=GSM8KGenerationConfig.top_p, max_tokens=GSM8KGenerationConfig.max_new_tokens)
    llm = LLM(model=GSM8KGenerationConfig.model_name, dtype=GSM8KGenerationConfig.dtype, max_num_seqs=GSM8KGenerationConfig.max_num_sequences, gpu_memory_utilization=GSM8KGenerationConfig.gpu_memory_utilization, tokenizer_mode="mistral")


    # Load dataset
    test_data = load_gsm8k()
    prompts = [format_prompt(x["question"]) for x in test_data]

    # Run batched inference
    print(f"Running inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    # Collect predictions
    results, correct = [], 0
    for sample, output in tqdm(zip(test_data, outputs), total=len(outputs), desc="Evaluating"):
        pred_text = output.outputs[0].text.strip()
        pred = extract_final_answer(pred_text)
        gt = get_ground_truth(sample["answer"])

        is_correct = (pred == gt)
        correct += int(is_correct)

        results.append({
            "question": sample["question"],
            "ground_truth": gt,
            "predicted": pred,
            "correct": is_correct,
            "model_output": pred_text
        })

    # Compute accuracy
    acc = correct / len(results)
    print(f"\n✅ Accuracy: {acc:.2%} ({correct}/{len(results)})")

    # Save to Pickle (.pkl)
    pkl_path = f"{save_prefix}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {pkl_path}")

# -----------------------------------------------------------
# 5. Entry point
# -----------------------------------------------------------
if __name__ == "__main__":
    evaluate_vllm(model_name=GSM8KGenerationConfig.model_name, 
    save_prefix=f'{GSM8KGenerationConfig.dataset_folder}/gsm8k_generated_data')
