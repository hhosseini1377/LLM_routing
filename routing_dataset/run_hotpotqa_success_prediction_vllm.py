"""
Success prediction pipeline for HotpotQA using vLLM.

This script:
1. Loads prompts from hotpotqa_qwen8b_val.pkl
2. Runs prompts on Qwen3-8B to generate answers
3. Self-evaluates: gives prompt + answer to Qwen3-8B and asks if correct (output "1" or "0")
4. Saves results with prompt, output, and self_evaluation_score

The correct_label from the dataset is preserved in the output for later evaluation
of the self-prediction quality, but is NOT used during the pipeline.

Usage:
    python -m routing_dataset.run_hotpotqa_success_prediction_vllm
    python -m routing_dataset.run_hotpotqa_success_prediction_vllm --input_file path/to/input.pkl --output_file path/to/output.pkl
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from vllm import LLM, SamplingParams

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from routing_dataset.dataset_paths import FINAL_HOTPOTQA_QWEN8B_VAL_FILE


def format_self_eval_prompt_qwen(
    original_prompt: str,
    model_answer: str,
    use_no_think: bool = True,
) -> str:
    """
    Format a self-evaluation prompt for Qwen: given prompt + answer, ask if correct.

    Args:
        original_prompt: The full prompt (context + question) used for generation
        model_answer: The model's generated answer
        use_no_think: Whether to use /no_think for Qwen3 models

    Returns:
        Formatted prompt string in ChatML format
    """
    if use_no_think:
        system_instruction = (
            "/no_think You are an expert evaluator. Given a question with context and a model's answer, "
            "determine if the answer is correct based on the provided context. "
            "Consider the answer correct if it accurately answers the question using information from the context. "
            "Respond with ONLY a single digit: 1 if correct, 0 if incorrect. No explanation."
        )
    else:
        system_instruction = (
            "You are an expert evaluator. Given a question with context and a model's answer, "
            "determine if the answer is correct based on the provided context. "
            "Consider the answer correct if it accurately answers the question using information from the context. "
            "Respond with ONLY a single digit: 1 if correct, 0 if incorrect. No explanation."
        )

    # Extract the user content from the original prompt (between user tags) for clarity
    # The original prompt has: system, user (context+question), assistant
    # We include the full original prompt so the model has all context
    user_content = (
        f"Original question and context:\n{original_prompt}\n\n"
        f"Model's answer: {model_answer}\n\n"
        "Is this answer correct based on the context? Respond with 1 (correct) or 0 (incorrect) only."
    )

    prompt = (
        "<|im_start|>system\n"
        f"{system_instruction}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_content}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt


def extract_self_eval_score(text: str) -> Tuple[int, str]:
    """
    Extract 1 or 0 from model output for self-evaluation.

    Args:
        text: Raw model output

    Returns:
        Tuple of (score: 0 or 1, raw_text for debugging)
    """
    text = text.strip()
    # Look for first occurrence of 1 or 0
    if re.search(r"\b1\b", text):
        # Check if 0 appears before 1 (e.g., "0" or "incorrect: 0")
        zero_match = re.search(r"\b0\b", text)
        one_match = re.search(r"\b1\b", text)
        if zero_match and one_match:
            if zero_match.start() < one_match.start():
                return 0, text
        return 1, text
    if re.search(r"\b0\b", text):
        return 0, text
    # Fallback: check for "correct" / "incorrect" keywords
    lower = text.lower()
    if "correct" in lower and "incorrect" not in lower[: lower.find("correct")]:
        return 1, text
    if "incorrect" in lower:
        return 0, text
    # Default to 0 if unparseable
    return 0, text


def run_success_prediction_pipeline(
    input_file: Path = FINAL_HOTPOTQA_QWEN8B_VAL_FILE,
    output_file: Optional[Path] = None,
    model_name: str = "Qwen/Qwen3-8B",
    temperature: float = 0.0,
    max_tokens_answer: int = 256,
    max_tokens_eval: int = 16,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.8,
    max_num_seqs: int = 256,
    tensor_parallel_size: int = 1,
    batch_size: Optional[int] = None,
    use_no_think: bool = True,
    progress_interval: int = 500,
) -> pd.DataFrame:
    """
    Run the full success prediction pipeline: generate answers, then self-evaluate.

    Args:
        input_file: Path to hotpotqa pkl (DataFrame with 'prompts' column)
        output_file: Path to save results. If None, saves to input_file dir with _success_pred suffix
        model_name: Model for generation and self-evaluation
        temperature: Sampling temperature (0 for deterministic)
        max_tokens_answer: Max tokens for answer generation
        max_tokens_eval: Max tokens for self-evaluation (1 or 0)
        dtype: Model dtype
        gpu_memory_utilization: vLLM GPU memory fraction
        max_num_seqs: vLLM max sequences per batch
        tensor_parallel_size: Number of GPUs for tensor parallelism
        batch_size: Optional batch size for processing (None = vLLM internal batching)
        use_no_think: Use /no_think for Qwen3
        progress_interval: Print progress every N examples

    Returns:
        DataFrame with prompt, output, self_evaluation_score, and original columns (e.g. correct_label)
    """
    if output_file is None:
        output_file = input_file.parent / (
            input_file.stem + "_success_pred.pkl"
        )

    # Load dataset
    print(f"Loading dataset from {input_file}...")
    df = pd.read_pickle(input_file)

    # Support both 'prompts' and 'prompt' column names
    if "prompts" in df.columns:
        prompts = df["prompts"].tolist()
    elif "prompt" in df.columns:
        prompts = df["prompt"].tolist()
    else:
        raise KeyError(
            f"Dataset must have 'prompts' or 'prompt' column. Found: {list(df.columns)}"
        )

    n = len(prompts)
    print(f"Loaded {n} prompts")

    # Initialize vLLM
    print(f"\nLoading vLLM model: {model_name}")
    if tensor_parallel_size > 1:
        print(f"Using {tensor_parallel_size} GPUs for tensor parallelism")

    llm_kwargs = {
        "model": model_name,
        "dtype": dtype,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
    }
    if tensor_parallel_size > 1:
        llm_kwargs["tensor_parallel_size"] = tensor_parallel_size

    llm = LLM(**llm_kwargs)

    # Step 1: Generate answers
    sampling_answer = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens_answer,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    print(f"\nStep 1: Generating answers (max_tokens={max_tokens_answer})...")
    if batch_size:
        outputs = []
        for i in range(0, n, batch_size):
            batch = prompts[i : i + batch_size]
            print(
                f"  Batch {i // batch_size + 1}/{(n + batch_size - 1) // batch_size}"
            )
            out = llm.generate(batch, sampling_answer)
            outputs.extend(out)
    else:
        outputs = llm.generate(prompts, sampling_answer)

    responses = [out.outputs[0].text.strip() for out in outputs]
    print(f"Generated {len(responses)} responses")

    # Step 2: Build self-evaluation prompts
    print("\nStep 2: Building self-evaluation prompts...")
    eval_prompts = [
        format_self_eval_prompt_qwen(p, r, use_no_think=use_no_think)
        for p, r in zip(prompts, responses)
    ]

    # Step 3: Run self-evaluation
    sampling_eval = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens_eval,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    print(f"Running self-evaluation (max_tokens={max_tokens_eval})...")
    if batch_size:
        eval_outputs = []
        for i in range(0, n, batch_size):
            batch = eval_prompts[i : i + batch_size]
            print(
                f"  Batch {i // batch_size + 1}/{(n + batch_size - 1) // batch_size}"
            )
            out = llm.generate(batch, sampling_eval)
            eval_outputs.extend(out)
    else:
        eval_outputs = llm.generate(eval_prompts, sampling_eval)

    # Extract self-evaluation scores
    self_eval_scores = []
    for i, out in enumerate(eval_outputs):
        text = out.outputs[0].text.strip()
        score, _ = extract_self_eval_score(text)
        self_eval_scores.append(score)
        if (i + 1) % progress_interval == 0:
            acc = sum(self_eval_scores) / len(self_eval_scores)
            print(f"  Processed {i + 1}/{n} | self-eval positive rate: {acc:.4f}")

    # Build result DataFrame
    result = pd.DataFrame(
        {
            "prompt": prompts,
            "output": responses,
            "self_evaluation_score": self_eval_scores,
        }
    )

    # Preserve original columns (e.g. correct_label, id, question, context, answer)
    # for later evaluation - but don't use them in the pipeline
    preserve_cols = [
        c
        for c in df.columns
        if c not in ("prompt", "prompts", "output", "self_evaluation_score")
    ]
    for col in preserve_cols:
        result[col] = df[col].values

    # Preserve correct_labels as correct_label (int) for later evaluation
    if "correct_labels" in df.columns:
        result["correct_label"] = df["correct_labels"].astype(int).values

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_pickle(output_file)
    print(f"\nSaved results to {output_file}")
    print(f"Columns: {list(result.columns)}")
    print(
        f"Self-evaluation positive rate: {sum(self_eval_scores) / n:.4f} ({sum(self_eval_scores)}/{n})"
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="HotpotQA success prediction pipeline: generate + self-evaluate with vLLM"
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        default=FINAL_HOTPOTQA_QWEN8B_VAL_FILE,
        help="Path to hotpotqa pkl (e.g. hotpotqa_qwen8b_val.pkl)",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=None,
        help="Output path. Default: input stem + _success_pred.pkl",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model for generation and self-evaluation",
    )
    parser.add_argument(
        "--max_tokens_answer",
        type=int,
        default=256,
        help="Max tokens for answer generation",
    )
    parser.add_argument(
        "--max_tokens_eval",
        type=int,
        default=16,
        help="Max tokens for self-evaluation output",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (None = vLLM internal batching)",
    )
    parser.add_argument(
        "--no_no_think",
        action="store_true",
        help="Disable /no_think for Qwen3 models",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    input_file = (
        args.input_file
        if args.input_file.is_absolute()
        else project_root / args.input_file
    )
    if args.output_file is None:
        output_file = None
    elif args.output_file.is_absolute():
        output_file = args.output_file
    else:
        output_file = project_root / args.output_file

    run_success_prediction_pipeline(
        input_file=input_file,
        output_file=output_file,
        model_name=args.model,
        max_tokens_answer=args.max_tokens_answer,
        max_tokens_eval=args.max_tokens_eval,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        use_no_think=not args.no_no_think,
    )


if __name__ == "__main__":
    main()
