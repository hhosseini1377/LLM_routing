"""
Script to filter LMSYS-Chat-1M prompts using an LLM judge.

This script:
1. Loads prompts from LMSYS_CHAT1M_PROMPTS_100K_FILE
2. Uses an LLM judge to evaluate each prompt as VALID or INVALID
3. Filters to keep only VALID prompts
4. Saves filtered prompts to LMSYS_CHAT1M_PROMPTS_100K_FILE_CLEANED
"""

import pickle
import sys
from pathlib import Path
from typing import List, Tuple
from vllm import LLM, SamplingParams

from routing_dataset.dataset_paths import (
    LMSYS_CHAT1M_PROMPTS_FILE,
    LMSYS_CHAT1M_PROMPTS_FILE_CLEANED,
)


def format_prompt_filter_prompt_qwen(
    prompt: str,
    use_no_think: bool = True
) -> str:
    """
    Format a prompt filtering prompt for Qwen judge model.
    
    Args:
        prompt: The user prompt to evaluate
        use_no_think: Whether to use /no_think command for Qwen3 models
    
    Returns:
        Formatted prompt string in ChatML format
    """
    # System instruction
    if use_no_think:
        system_instruction = (
            "/no_think You are filtering user prompts for a prompt-only evaluation setting. "
            "A prompt is INVALID if it: "
            "(1) depends on prior conversation context (e.g., refers to \"your previous answer\", \"above\", \"revise your response\"), or "
            "(2) is nonsensical / spam / too incomplete to answer as a standalone request. "
            "Otherwise it is VALID. "
            "Return only one token: VALID or INVALID."
        )
    else:
        system_instruction = (
            "You are filtering user prompts for a prompt-only evaluation setting. "
            "A prompt is INVALID if it: "
            "(1) depends on prior conversation context (e.g., refers to \"your previous answer\", \"above\", \"revise your response\"), or "
            "(2) is nonsensical / spam / too incomplete to answer as a standalone request. "
            "Otherwise it is VALID. "
            "Return only one token: VALID or INVALID."
        )
    
    # User content
    user_content = f"PROMPT:\n{prompt}"
    
    # Format using ChatML (Qwen format)
    formatted_prompt = (
        "<|im_start|>system\n"
        f"{system_instruction}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_content}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    return formatted_prompt


def extract_judgment(text: str) -> Tuple[str, str]:
    """
    Extract VALID/INVALID judgment from model output.
    
    Args:
        text: Raw model output text
    
    Returns:
        Tuple of (judgment, raw_text) where judgment is "VALID" or "INVALID"
    """
    text = text.strip().upper()
    
    # Look for VALID or INVALID in the response
    if "VALID" in text:
        # Check if it's INVALID (which contains VALID as substring)
        if text.startswith("INVALID") or "INVALID" in text:
            # If both appear, check which comes first
            valid_pos = text.find("VALID")
            invalid_pos = text.find("INVALID")
            if invalid_pos != -1 and (valid_pos == -1 or invalid_pos < valid_pos):
                return "INVALID", text
        return "VALID", text
    elif "INVALID" in text:
        return "INVALID", text
    
    # Fallback: check first word or common patterns
    first_word = text.split()[0] if text.split() else ""
    if first_word == "VALID":
        return "VALID", text
    elif first_word == "INVALID":
        return "INVALID", text
    
    # Default to INVALID if unclear
    return "INVALID", text


def filter_prompts_with_vllm(
    input_file: Path = LMSYS_CHAT1M_PROMPTS_FILE,
    output_file: Path = LMSYS_CHAT1M_PROMPTS_FILE_CLEANED,
    judge_model: str = "Qwen/Qwen3-8B",
    temperature: float = 0.0,
    max_tokens: int = 10,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.8,
    max_num_seqs: int = 256,
    tensor_parallel_size: int = 1,
    use_no_think: bool = True,
    batch_size: int = None,
) -> List[str]:
    """
    Filter LMSYS prompts using an LLM judge.
    
    Args:
        input_file: Path to pickled list of prompts
        output_file: Path to save filtered prompts
        judge_model: Model identifier for the judge (default: "Qwen/Qwen3-8B")
        temperature: Sampling temperature (default: 0.0 for deterministic)
        max_tokens: Maximum tokens to generate (default: 10, just need VALID/INVALID)
        dtype: Model dtype (default: "bfloat16")
        gpu_memory_utilization: GPU memory utilization (default: 0.8)
        max_num_seqs: Maximum sequences in parallel (default: 256)
        tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
        use_no_think: Whether to use /no_think command (default: True)
        batch_size: Optional batch size for processing (None = let vLLM handle it)
    
    Returns:
        List of valid prompts
    """
    print(f"Loading prompts from: {input_file}")
    
    # Load prompts
    with open(input_file, 'rb') as f:
        prompts = pickle.load(f)
    
    if not isinstance(prompts, list):
        raise ValueError(f"Expected list of prompts, got {type(prompts)}")
    
    num_prompts = len(prompts)
    print(f"Loaded {num_prompts} prompts")
    
    # Initialize judge model
    print(f"\nLoading judge model: {judge_model}")
    if tensor_parallel_size > 1:
        print(f"Using {tensor_parallel_size} GPUs for tensor parallelism")
    
    llm_kwargs = {
        'model': judge_model,
        'dtype': dtype,
        'max_num_seqs': max_num_seqs,
        'gpu_memory_utilization': gpu_memory_utilization,
        'trust_remote_code': True,
    }
    
    if tensor_parallel_size > 1:
        llm_kwargs['tensor_parallel_size'] = tensor_parallel_size
    
    llm = LLM(**llm_kwargs)
    
    # Build prompts
    print("\nBuilding judge prompts...")
    formatted_prompts = []
    for prompt in prompts:
        formatted_prompt = format_prompt_filter_prompt_qwen(
            prompt=prompt,
            use_no_think=use_no_think
        )
        formatted_prompts.append(formatted_prompt)
    
    print(f"Formatted {len(formatted_prompts)} prompts")
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],  # Stop at end tokens
    )
    
    # Run inference
    print(f"\nRunning inference on {len(formatted_prompts)} prompts...")
    print(f"Generation parameters: temp={temperature}, max_tokens={max_tokens}")
    
    if batch_size:
        # Process in batches if specified
        all_outputs = []
        for i in range(0, len(formatted_prompts), batch_size):
            batch = formatted_prompts[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{(len(formatted_prompts) + batch_size - 1) // batch_size}")
            outputs = llm.generate(batch, sampling_params)
            all_outputs.extend(outputs)
        outputs = all_outputs
    else:
        # Let vLLM handle batching internally
        outputs = llm.generate(formatted_prompts, sampling_params)
    
    # Extract judgments
    print("\nExtracting judgments...")
    judgments = []
    valid_count = 0
    invalid_count = 0
    
    for i, output in enumerate(outputs):
        text = output.outputs[0].text.strip()
        judgment, _ = extract_judgment(text)
        judgments.append(judgment)
        
        if judgment == "VALID":
            valid_count += 1
        else:
            invalid_count += 1
        
        # Print sample judgments
        if i < 5:
            print(f"  Prompt {i}: {judgment} (response: '{text[:50]}')")
    
    print(f"\nJudgment summary:")
    print(f"  VALID: {valid_count} ({100*valid_count/num_prompts:.1f}%)")
    print(f"  INVALID: {invalid_count} ({100*invalid_count/num_prompts:.1f}%)")
    
    # Filter to keep only VALID prompts
    valid_prompts = [
        prompt for prompt, judgment in zip(prompts, judgments)
        if judgment == "VALID"
    ]
    
    print(f"\nFiltered to {len(valid_prompts)} valid prompts")
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving filtered prompts to: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(valid_prompts, f)
    
    print("Done!")
    return valid_prompts


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Filter LMSYS prompts using an LLM judge"
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        default=LMSYS_CHAT1M_PROMPTS_FILE,
        help="Path to input prompts file (pickle)"
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=LMSYS_CHAT1M_PROMPTS_FILE_CLEANED,
        help="Path to output filtered prompts file (pickle)"
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model identifier for the judge"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10,
        help="Maximum tokens to generate (default: 10)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Model dtype (default: bfloat16)"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization (default: 0.8)"
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=256,
        help="Maximum sequences in parallel (default: 256)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--disable_no_think",
        action="store_true",
        help="Disable /no_think mode"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for processing (None = let vLLM handle it)"
    )
    
    args = parser.parse_args()
    
    filter_prompts_with_vllm(
        input_file=args.input_file,
        output_file=args.output_file,
        judge_model=args.judge_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        use_no_think=not args.disable_no_think,
        batch_size=args.batch_size,
    )

