"""
Script to run LMSYS-Chat-1M cleaned prompts using vLLM on Qwen3-8B.

This script:
1. Loads cleaned prompts from LMSYS_CHAT1M_PROMPTS_100K_FILE_CLEANED
2. Formats them using a simple chat prompt template
3. Runs inference using vLLM with Qwen3-8B
4. Saves results to a pickle file
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional
from vllm import LLM, SamplingParams
from routing_dataset.dataset_paths import (
    LMSYS_CHAT1M_PROMPTS_FILE_CLEANED,
    LMSYS_CHAT1M_PROMPTS_WITH_ANSWERS_QWEN8B_FILE,
)


def format_lmsys_prompt_qwen(prompt: str, use_thinking: bool = False) -> str:
    """
    Format an LMSYS prompt into a Qwen-style chat prompt.
    
    Args:
        prompt: The user prompt to answer
        use_thinking: If True, enables thinking mode (/think) for Qwen3 models.
                     If False, disables thinking mode (/no_think) for direct answers.
                     Default: False (direct answers)
    
    Returns:
        Formatted prompt string in ChatML format
    """
    # System instruction
    # For Qwen3 models:
    # - /think: Enables thinking/reasoning mode (shows reasoning process)
    # - /no_think: Disables thinking mode (direct answers, more concise)
    if use_thinking:
        system_instruction = (
            "/think You are a helpful assistant. Answer the user's request clearly and completely. "
            "Think through the problem step by step before providing your answer."
        )
    else:
        system_instruction = (
            "/no_think You are a helpful assistant. Answer the user's request clearly and completely."
        )
    
    # Format using ChatML (Qwen format)
    formatted_prompt = (
        "<|im_start|>system\n"
        f"{system_instruction}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    return formatted_prompt


def run_lmsys_prompts_with_vllm(
    prompts_file: Path = LMSYS_CHAT1M_PROMPTS_FILE_CLEANED,
    output_file: Path = LMSYS_CHAT1M_PROMPTS_WITH_ANSWERS_QWEN8B_FILE,
    model_name: str = "Qwen/Qwen3-8B",
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_tokens: int = 2048,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.8,
    max_num_seqs: int = 256,
    tensor_parallel_size: int = 1,
    batch_size: Optional[int] = None,
    use_thinking: bool = False,
) -> Dict[str, List]:
    """
    Run LMSYS cleaned prompts through vLLM with Qwen3-8B.
    
    Args:
        prompts_file: Path to cleaned prompts pickle file (list of strings)
        output_file: Path to save results pickle file
        model_name: Model name for vLLM (default: "Qwen/Qwen3-8B")
        temperature: Sampling temperature (default: 0.0)
        top_p: Top-p sampling parameter (default: 0.95)
        max_tokens: Maximum number of tokens to generate (default: 2048)
        dtype: Data type for the model (default: "bfloat16")
        gpu_memory_utilization: GPU memory utilization (default: 0.8)
        max_num_seqs: Maximum number of sequences to process in parallel (default: 256)
        tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
        batch_size: Optional batch size for processing (None = let vLLM handle it)
        use_thinking: If True, enables thinking mode (/think) for Qwen3 models.
                     If False, uses /no_think for direct answers (default: False)
    
    Returns:
        Dictionary with 'prompts', 'formatted_prompts', and 'responses'
    """
    print(f"Loading cleaned prompts from: {prompts_file}")
    
    # Load prompts
    with open(prompts_file, 'rb') as f:
        prompts = pickle.load(f)
    
    if not isinstance(prompts, list):
        raise ValueError(f"Expected list of prompts, got {type(prompts)}")
    
    num_prompts = len(prompts)
    print(f"Loaded {num_prompts} prompts")
    
    # Format prompts
    thinking_mode_str = "thinking mode (/think)" if use_thinking else "non-thinking mode (/no_think)"
    print(f"Formatting prompts in {thinking_mode_str}...")
    formatted_prompts = []
    for prompt in prompts:
        formatted_prompt = format_lmsys_prompt_qwen(prompt, use_thinking=use_thinking)
        formatted_prompts.append(formatted_prompt)
    
    print(f"Formatted {len(formatted_prompts)} prompts")
    print(f"Sample prompt length: {len(formatted_prompts[0])} characters")
    print(f"\nSample formatted prompt (first 500 chars):\n{formatted_prompts[0][:500]}...")
    
    # Initialize vLLM
    print(f"\nLoading vLLM model: {model_name}")
    if tensor_parallel_size > 1:
        print(f"Using {tensor_parallel_size} GPUs for tensor parallelism")
    
    llm_kwargs = {
        'model': model_name,
        'dtype': dtype,
        'max_num_seqs': max_num_seqs,
        'gpu_memory_utilization': gpu_memory_utilization,
        'trust_remote_code': True,
    }
    
    if tensor_parallel_size > 1:
        llm_kwargs['tensor_parallel_size'] = tensor_parallel_size
    
    llm = LLM(**llm_kwargs)
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],  # Stop at end tokens
    )
    
    # Run inference
    print(f"\nRunning inference on {len(formatted_prompts)} prompts...")
    print(f"Generation parameters: temp={temperature}, top_p={top_p}, max_tokens={max_tokens}")
    
    if batch_size:
        # Process in batches if specified
        responses = []
        for i in range(0, len(formatted_prompts), batch_size):
            batch = formatted_prompts[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{(len(formatted_prompts) + batch_size - 1) // batch_size}")
            outputs = llm.generate(batch, sampling_params)
            responses.extend([out.outputs[0].text.strip() for out in outputs])
    else:
        # Let vLLM handle batching internally
        outputs = llm.generate(formatted_prompts, sampling_params)
        responses = [out.outputs[0].text.strip() for out in outputs]
    
    print(f"Generated {len(responses)} responses")
    print(f"Average response length: {sum(len(r) for r in responses) / len(responses):.1f} characters")
    print(f"\nSample response (first 300 chars):\n{responses[0][:300]}...")
    
    # Prepare output dictionary
    result_dict = {
        'prompts': prompts,
        'formatted_prompts': formatted_prompts,
        'responses': responses,
    }
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(result_dict, f)
    
    print("Done!")
    return result_dict


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LMSYS cleaned prompts with vLLM on Qwen3-8B")
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=str(LMSYS_CHAT1M_PROMPTS_FILE_CLEANED),
        help="Path to cleaned prompts pickle file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(LMSYS_CHAT1M_PROMPTS_WITH_ANSWERS_QWEN8B_FILE),
        help="Path to save results"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name for vLLM"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter (default: 0.95)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)"
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
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for processing (None = let vLLM handle it)"
    )
    parser.add_argument(
        "--use_thinking",
        action="store_true",
        help="Enable thinking mode (/think) for Qwen3 models. "
             "Default: False (uses /no_think for direct answers)"
    )
    
    args = parser.parse_args()
    
    run_lmsys_prompts_with_vllm(
        prompts_file=Path(args.prompts_file),
        output_file=Path(args.output_file),
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        use_thinking=args.use_thinking,
    )

