"""
Script to run APPS dataset using vLLM on Qwen3-8B.

This script:
1. Loads prompts from APPS_PROMPTS_FILE
2. Formats them using a code generation prompt template
3. Runs inference using vLLM with Qwen3-8B
4. Saves results to a pickle file
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional
from vllm import LLM, SamplingParams
from routing_dataset.dataset_paths import (
    APPS_PROMPTS_FILE,
    APPS_PROMPTS_WITH_ANSWERS_QWEN8B_FILE,
    APPS_DATA_DIR,
)
import pandas as pd


def format_apps_prompt_qwen(
    problem_description: str,
    starter_code: Optional[str] = None,
    use_no_think: bool = True
) -> str:
    """
    Format an APPS problem into a Qwen-style chat prompt for code generation.
    
    Args:
        problem_description: The full problem description from APPS
        starter_code: Optional starter code provided in the problem
        use_no_think: Whether to use /no_think command for Qwen3 models
    
    Returns:
        Formatted prompt string in ChatML format
    """
    # System instruction for code generation
    # The /no_think command tells Qwen3 models to skip thinking/reasoning mode
    # and output code directly, which is important for code generation tasks
    if use_no_think:
        system_instruction = (
            "/no_think You are an expert Python programmer specializing in competitive programming "
            "and algorithmic problem solving. Your task is to write clean, efficient, and correct "
            "Python code that solves the given problem. Read the problem statement carefully, "
            "understand the input/output format, and implement a solution that passes all test cases. "
            "Output only the Python code without any explanation, comments, or markdown formatting."
        )
    else:
        system_instruction = (
            "You are an expert Python programmer specializing in competitive programming "
            "and algorithmic problem solving. Your task is to write clean, efficient, and correct "
            "Python code that solves the given problem. Read the problem statement carefully, "
            "understand the input/output format, and implement a solution that passes all test cases. "
            "Output only the Python code without any explanation, comments, or markdown formatting."
        )
    
    # Build user content
    user_content = f"Problem Description:\n{problem_description}\n"
    
    if starter_code and starter_code.strip():
        user_content += f"\nStarter Code:\n```python\n{starter_code}\n```\n"
        user_content += "\nComplete the solution by implementing the missing parts. "
        user_content += "Your code should read from standard input and write to standard output."
    else:
        user_content += "\nWrite a complete Python solution for this problem. "
        user_content += "Your code should read from standard input and write to standard output."
    
    # Format using ChatML (Qwen format)
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


def run_apps_with_vllm(
    prompts_file: Path = APPS_PROMPTS_FILE,
    output_file: Path = APPS_PROMPTS_WITH_ANSWERS_QWEN8B_FILE,
    model_name: str = "Qwen/Qwen3-8B",
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_tokens: int = 2048,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.8,
    max_num_seqs: int = 256,
    tensor_parallel_size: int = 1,
    use_starter_code: bool = True,
    batch_size: Optional[int] = None,
) -> Dict[str, List]:
    """
    Run APPS dataset prompts through vLLM with Qwen3-8B.
    
    Args:
        prompts_file: Path to the pickled APPS prompts DataFrame
        output_file: Path to save the results
        model_name: Model identifier (default: "Qwen/Qwen3-8B")
        temperature: Sampling temperature (default: 0.0 for deterministic)
        top_p: Top-p sampling parameter (default: 0.95)
        max_tokens: Maximum tokens to generate (default: 2048)
        dtype: Model dtype (default: "bfloat16")
        gpu_memory_utilization: GPU memory utilization (default: 0.8)
        max_num_seqs: Maximum sequences in parallel (default: 256)
        tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
        use_starter_code: Whether to include starter code in prompts (default: True)
        batch_size: Optional batch size for processing (None = let vLLM handle it)
    
    Returns:
        Dictionary with 'prompts', 'formatted_prompts', 'responses', and metadata
    """
    print(f"Loading APPS prompts from: {prompts_file}")
    
    # Load prompts DataFrame
    with open(prompts_file, 'rb') as f:
        apps_df = pickle.load(f)
    
    if not isinstance(apps_df, pd.DataFrame):
        raise ValueError(f"Expected DataFrame, got {type(apps_df)}")
    
    print(f"Loaded {len(apps_df)} APPS problems")
    print(f"Columns: {list(apps_df.columns)}")
    
    # Extract prompts and metadata
    problem_descriptions = apps_df['prompts'].tolist()
    starter_codes = apps_df['starter_code'].tolist() if 'starter_code' in apps_df.columns else [None] * len(apps_df)
    problem_ids = apps_df['problem_id'].tolist() if 'problem_id' in apps_df.columns else list(range(len(apps_df)))
    
    # Format prompts
    print("Formatting prompts...")
    formatted_prompts = []
    for desc, starter in zip(problem_descriptions, starter_codes):
        starter_code = starter if use_starter_code and starter and str(starter).strip() else None
        formatted_prompt = format_apps_prompt_qwen(
            problem_description=desc,
            starter_code=starter_code,
            use_no_think=True
        )
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
        'problem_ids': problem_ids,
        'prompts': problem_descriptions,
        'formatted_prompts': formatted_prompts,
        'responses': responses,
        'starter_codes': starter_codes,
    }
    
    # Add other metadata if available
    if 'solutions' in apps_df.columns:
        result_dict['solutions'] = apps_df['solutions'].tolist()
    if 'difficulty' in apps_df.columns:
        result_dict['difficulty'] = apps_df['difficulty'].tolist()
    if 'url' in apps_df.columns:
        result_dict['url'] = apps_df['url'].tolist()
    if 'input_output' in apps_df.columns:
        result_dict['input_output'] = apps_df['input_output'].tolist()
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(result_dict, f)
    
    print("Done!")
    return result_dict


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run APPS dataset with vLLM on Qwen3-8B")
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=str(APPS_PROMPTS_FILE),
        help="Path to APPS prompts pickle file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(APPS_PROMPTS_WITH_ANSWERS_QWEN8B_FILE),
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
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization"
    )
    parser.add_argument(
        "--no_starter_code",
        action="store_true",
        help="Don't include starter code in prompts"
    )
    
    args = parser.parse_args()
    
    run_apps_with_vllm(
        prompts_file=Path(args.prompts_file),
        output_file=Path(args.output_file),
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        use_starter_code=not args.no_starter_code,
    )

