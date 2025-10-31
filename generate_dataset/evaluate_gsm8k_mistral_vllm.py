"""
Evaluate Mistral-7B-Instruct on the GSM8K benchmark using vLLM.
Saves results in both JSON and Pickle (.pkl) formats.

Requirements:
    pip install vllm datasets tqdm
"""

import argparse

import re
import pickle
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm
from generate_dataset.generation_config import GSM8KGenerationConfig
from typing import Optional
# -----------------------------------------------------------
# 1. Load GSM8K dataset
# -----------------------------------------------------------
def load_gsm8k(split: str = "test"):
    if split == 'auxiliary':
        dataset = load_dataset("gsm8k", "auxiliary")
        return dataset
    else:
        dataset = load_dataset("gsm8k", "main")
        return dataset[split]  

# -----------------------------------------------------------
# 2. Prompt formatting
# -----------------------------------------------------------
def format_prompt(question: str, few_shot_examples: Optional[list[str]] = None) -> str:
    """
    Format a GSM8K-style math question for Mistral-7B or similar models.

    Args:
        question (str): The math word problem (GSM8K question text).
        few_shot_examples (list[str], optional): List of example QA strings in the same format
            (each containing a reasoning and final answer like '#### 5'). Defaults to None.

    Returns:
        str: A fully formatted prompt ready for model inference.
    """
    header = (
        "You are a helpful and careful reasoning assistant.\n"
        "Solve the following math word problem step by step.\n"
        "After finishing, write the final numeric answer on a new line starting with '####'.\n"
        "Do NOT include words or explanations after '####'.\n"
    )

    # Optional few-shot section
    examples_text = ""
    if few_shot_examples:
        examples_text = "\n".join(
            f"Example {i+1}:\n{ex.strip()}\n" for i, ex in enumerate(few_shot_examples)
        )

    # Combine everything into the final prompt
    prompt = f"{header}\n{examples_text}\nQuestion:\n{question.strip()}\n\nLet's reason step by step."
    return prompt


# -----------------------------------------------------------
# 3. Regex-based answer extraction
# -----------------------------------------------------------
def extract_final_answer(text: str):
    """Extract numeric answer after 'Answer:'."""
    match = re.search(r"####\s*([0-9,.\-]+)", text)
    return match.group(1) if match else None


def get_ground_truth(answer_text: str):
    """Extract ground-truth number from the GSM8K field."""
    match = re.search(r"####\s*([0-9,.\-]+)", answer_text)
    return match.group(1) if match else None


# -----------------------------------------------------------
# 4. Evaluation with vLLM
# -----------------------------------------------------------
def evaluate_vllm(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    save_prefix="mistral_gsm8k_results_vllm",
    split: str = "test"
):
    print(f"Loading vLLM model: {model_name}")

    # Load model
    sampling_params = SamplingParams(temperature=GSM8KGenerationConfig.temperature, top_p=GSM8KGenerationConfig.top_p, max_tokens=GSM8KGenerationConfig.max_new_tokens)
    llm = LLM(
        model=GSM8KGenerationConfig.model_name, 
        dtype=GSM8KGenerationConfig.dtype, 
        max_num_seqs=GSM8KGenerationConfig.max_num_sequences, 
        gpu_memory_utilization=GSM8KGenerationConfig.gpu_memory_utilization, 
        tokenizer_mode="mistral",
    )

    # Load dataset
    test_data = load_gsm8k(split)
    prompts = [format_prompt(x["question"]) for x in test_data]

    # Run batched inference
    print(f"Running inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    # Collect predictions
    results, correct = [], 0
    for sample, output in tqdm(zip(test_data, outputs), total=len(outputs), desc="Evaluating"):
        pred_text = output.outputs[0].text.strip()
        pred = extract_final_answer(pred_text)
        if pred is None:
            continue
        gt = get_ground_truth(sample["answer"])

        is_correct = (pred == gt)
        correct += int(is_correct)

        results.append({
            "question": sample["question"],
            "answer": sample["answer"],
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["test", "train", "validation", "auxiliary"])
    args = parser.parse_args()      
    evaluate_vllm(
        model_name=GSM8KGenerationConfig.model_name, 
        save_prefix=f'{GSM8KGenerationConfig.dataset_folder}/gsm8k_generated_data_{args.split}',
        split=args.split
    )
