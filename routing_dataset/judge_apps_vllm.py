"""
Script to judge APPS code generation results using an LLM judge.

This script:
1. Loads APPS results with generated code
2. Uses an LLM judge to evaluate code correctness based on test cases
3. Adds correct_labels to the dataset
"""

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Increase limit for integer string conversion to handle large test case values
# APPS dataset can have very large integers in test cases
sys.set_int_max_str_digits(100000)  # Allow up to 100k digit integers
from routing_dataset.dataset_paths import (
    APPS_PROMPTS_WITH_ANSWERS_QWEN8B_FILE,
    APPS_PROMPTS_WITH_CORRECT_LABELS_QWEN8B_FILE,
    APPS_DATA_DIR,
)


def format_apps_judge_prompt_qwen(
    problem_description: str,
    generated_code: str,
    test_inputs: List[str],
    test_outputs: List[str],
    use_no_think: bool = True
) -> str:
    """
    Format an APPS code evaluation prompt for Qwen judge model.
    
    Args:
        problem_description: The problem statement
        generated_code: The generated Python code to evaluate
        test_inputs: List of test case inputs
        test_outputs: List of expected test case outputs
        use_no_think: Whether to use /no_think command for Qwen3 models
    
    Returns:
        Formatted prompt string in ChatML format
    """
    # System instruction for code evaluation
    if use_no_think:
        system_instruction = (
            "/no_think You are an expert code evaluator specializing in competitive programming. "
            "Your task is to determine if a Python solution correctly solves the given problem "
            "by checking if it produces the expected outputs for the provided test cases. "
            "Evaluate the code's correctness, not its style or efficiency. "
            "Respond only in JSON format with 'is_correct' (boolean) and 'reason' (brief explanation)."
        )
    else:
        system_instruction = (
            "You are an expert code evaluator specializing in competitive programming. "
            "Your task is to determine if a Python solution correctly solves the given problem "
            "by checking if it produces the expected outputs for the provided test cases. "
            "Evaluate the code's correctness, not its style or efficiency. "
            "Respond only in JSON format with 'is_correct' (boolean) and 'reason' (brief explanation)."
        )
    
    # Build test cases section
    # Truncate very long test cases to avoid prompt length and integer conversion issues
    test_cases_text = ""
    max_case_length = 2000  # Maximum characters per test case to include
    
    for i, (test_in, test_out) in enumerate(zip(test_inputs[:5], test_outputs[:5]), 1):  # Limit to first 5 for prompt length
        # Convert to string and truncate if too long
        test_in_str = str(test_in)[:max_case_length]
        test_out_str = str(test_out)[:max_case_length]
        if len(str(test_in)) > max_case_length:
            test_in_str += f"\n... (truncated, original length: {len(str(test_in))} chars)"
        if len(str(test_out)) > max_case_length:
            test_out_str += f"\n... (truncated, original length: {len(str(test_out))} chars)"
        
        test_cases_text += f"\nTest Case {i}:\n"
        test_cases_text += f"Input:\n{test_in_str}\n"
        test_cases_text += f"Expected Output:\n{test_out_str}\n"
    
    if len(test_inputs) > 5:
        test_cases_text += f"\n... and {len(test_inputs) - 5} more test cases.\n"
    
    # Build user content
    user_content = (
        f"Problem Description:\n{problem_description}\n\n"
        f"Generated Code:\n```python\n{generated_code}\n```\n\n"
        f"Test Cases:{test_cases_text}\n\n"
        "Evaluate whether the generated code produces the correct outputs for all test cases. "
        "Consider that:\n"
        "- The code should read from standard input (using input())\n"
        "- The code should write to standard output (using print())\n"
        "- Output formatting must match exactly (whitespace, newlines, etc.)\n"
        "- The code should handle all edge cases\n\n"
        "Respond in JSON format:\n"
        "{\n"
        '  "is_correct": true or false,\n'
        '  "reason": "brief explanation of why the code is correct or incorrect"\n'
        "}"
    )
    
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


def format_apps_judge_prompt_llama(
    problem_description: str,
    generated_code: str,
    test_inputs: List[str],
    test_outputs: List[str],
) -> Tuple[str, str]:
    """
    Format an APPS code evaluation prompt for Llama judge model.
    
    Args:
        problem_description: The problem statement
        generated_code: The generated Python code to evaluate
        test_inputs: List of test case inputs
        test_outputs: List of expected test case outputs
    
    Returns:
        Tuple of (system_message, user_message) for chat template
    """
    # Build test cases section
    # Truncate very long test cases to avoid prompt length and integer conversion issues
    test_cases_text = ""
    max_case_length = 2000  # Maximum characters per test case to include
    
    for i, (test_in, test_out) in enumerate(zip(test_inputs[:5], test_outputs[:5]), 1):
        # Convert to string and truncate if too long
        test_in_str = str(test_in)[:max_case_length]
        test_out_str = str(test_out)[:max_case_length]
        if len(str(test_in)) > max_case_length:
            test_in_str += f"\n... (truncated, original length: {len(str(test_in))} chars)"
        if len(str(test_out)) > max_case_length:
            test_out_str += f"\n... (truncated, original length: {len(str(test_out))} chars)"
        
        test_cases_text += f"\nTest Case {i}:\n"
        test_cases_text += f"Input:\n{test_in_str}\n"
        test_cases_text += f"Expected Output:\n{test_out_str}\n"
    
    if len(test_inputs) > 5:
        test_cases_text += f"\n... and {len(test_inputs) - 5} more test cases.\n"
    
    system_message = (
        "You are an expert code evaluator specializing in competitive programming. "
        "Your task is to determine if a Python solution correctly solves the given problem "
        "by checking if it produces the expected outputs for the provided test cases. "
        "Evaluate the code's correctness, not its style or efficiency. "
        "Respond only in JSON format with 'is_correct' (boolean) and 'reason' (brief explanation)."
    )
    
    user_message = (
        f"Problem Description:\n{problem_description}\n\n"
        f"Generated Code:\n```python\n{generated_code}\n```\n\n"
        f"Test Cases:{test_cases_text}\n\n"
        "Evaluate whether the generated code produces the correct outputs for all test cases. "
        "Consider that:\n"
        "- The code should read from standard input (using input())\n"
        "- The code should write to standard output (using print())\n"
        "- Output formatting must match exactly (whitespace, newlines, etc.)\n"
        "- The code should handle all edge cases\n\n"
        "Respond in JSON format:\n"
        "{\n"
        '  "is_correct": true or false,\n'
        '  "reason": "brief explanation of why the code is correct or incorrect"\n'
        "}"
    )
    
    return system_message, user_message


def extract_judgment(generated_text: str) -> Tuple[bool, str]:
    """
    Extract JSON judgment from generated text.
    
    Args:
        generated_text: Raw text from the judge model
    
    Returns:
        Tuple of (is_correct: bool, reason: str)
    """
    text = generated_text.strip()
    
    # Remove markdown code blocks if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    
    try:
        # Find JSON object
        start_idx = text.index("{")
        end_idx = text.rindex("}") + 1
        json_str = text[start_idx:end_idx]
        result = json.loads(json_str)
        
        is_correct = result.get("is_correct", False)
        # Handle string "true"/"false" as well
        if isinstance(is_correct, str):
            is_correct = is_correct.lower() in ("true", "1", "yes", "correct")
        
        reason = result.get("reason", "")
        return bool(is_correct), reason
    except (ValueError, json.JSONDecodeError, KeyError) as e:
        # Fallback: try to extract boolean from text
        text_lower = text.lower()
        if "is_correct" in text_lower or '"correct": true' in text_lower or "'correct': true" in text_lower:
            return True, f"Extracted from text (JSON parse failed: {str(e)[:50]})"
        elif '"correct": false' in text_lower or "'correct': false" in text_lower:
            return False, f"Extracted from text (JSON parse failed: {str(e)[:50]})"
        else:
            return False, f"JSON parsing failed: {str(e)[:50]}"


def judge_apps_with_vllm(
    input_file: Path = APPS_PROMPTS_WITH_ANSWERS_QWEN8B_FILE,
    output_file: Path = APPS_PROMPTS_WITH_CORRECT_LABELS_QWEN8B_FILE,
    judge_model: str = "Qwen/Qwen2.5-32B-Instruct",
    temperature: float = 0.0,
    max_tokens: int = 512,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.8,
    max_num_seqs: int = 256,
    tensor_parallel_size: int = 4,
    max_test_cases: int = 5,
    use_tokenizer_template: bool = False,
) -> Dict:
    """
    Judge APPS code generation results using an LLM judge.
    
    Args:
        input_file: Path to pickled results with 'responses' and other fields
        output_file: Path to save results with correct_labels
        judge_model: Model identifier for the judge (default: "Qwen/Qwen2.5-32B-Instruct")
        temperature: Sampling temperature (default: 0.0 for deterministic)
        max_tokens: Maximum tokens to generate (default: 512)
        dtype: Model dtype (default: "bfloat16")
        gpu_memory_utilization: GPU memory utilization (default: 0.8)
        max_num_seqs: Maximum sequences in parallel (default: 256)
        tensor_parallel_size: Number of GPUs for tensor parallelism (default: 4)
        max_test_cases: Maximum number of test cases to include in prompt (default: 5)
        use_tokenizer_template: If True, use tokenizer's chat template (for Llama models)
    
    Returns:
        Dictionary with added 'correct_labels' and 'judge_reasons'
    """
    print(f"Loading APPS results from: {input_file}")
    
    # Load results
    with open(input_file, 'rb') as f:
        results = pickle.load(f)
    
    if not isinstance(results, dict):
        raise ValueError(f"Expected dictionary, got {type(results)}")
    
    required_keys = ['prompts', 'responses']
    missing_keys = [k for k in required_keys if k not in results]
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")
    
    num_samples = len(results['prompts'])
    print(f"Loaded {num_samples} samples")
    
    # Parse input_output if available
    test_inputs_list = []
    test_outputs_list = []
    
    if 'input_output' in results:
        for idx, io_str in enumerate(results['input_output']):
            try:
                # Skip None or empty values
                if io_str is None or (isinstance(io_str, str) and not io_str.strip()):
                    test_inputs_list.append([])
                    test_outputs_list.append([])
                    continue
                
                if isinstance(io_str, str):
                    # Skip empty strings
                    if not io_str.strip():
                        test_inputs_list.append([])
                        test_outputs_list.append([])
                        continue
                    # Use parse_constant to handle very large integers as strings
                    io_dict = json.loads(io_str, parse_constant=lambda x: str(x) if isinstance(x, (int, float)) else x)
                elif isinstance(io_str, dict):
                    io_dict = io_str
                else:
                    # Unknown type, skip
                    test_inputs_list.append([])
                    test_outputs_list.append([])
                    continue
                
                # Convert large integers to strings to avoid conversion issues
                inputs = io_dict.get('inputs', [])
                outputs = io_dict.get('outputs', [])
                
                # Ensure inputs and outputs are lists
                if not isinstance(inputs, list):
                    inputs = [inputs] if inputs else []
                if not isinstance(outputs, list):
                    outputs = [outputs] if outputs else []
                
                # Truncate very long test cases to avoid prompt length issues
                test_inputs_list.append([str(inp)[:10000] if len(str(inp)) > 10000 else str(inp) for inp in inputs])
                test_outputs_list.append([str(out)[:10000] if len(str(out)) > 10000 else str(out) for out in outputs])
            except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as e:
                # Only print warning for non-empty values to reduce noise
                if io_str and (not isinstance(io_str, str) or io_str.strip()):
                    if idx < 10:  # Only show first 10 warnings
                        print(f"Warning: Failed to parse input_output at index {idx}: {str(e)[:100]}")
                test_inputs_list.append([])
                test_outputs_list.append([])
    else:
        test_inputs_list = [[]] * num_samples
        test_outputs_list = [[]] * num_samples
    
    print(f"Parsed test cases: {sum(1 for inp in test_inputs_list if inp)}/{len(test_inputs_list)} samples have test cases")
    
    # Initialize judge model
    print(f"\nLoading judge model: {judge_model}")
    if tensor_parallel_size > 1:
        print(f"Using {tensor_parallel_size} GPUs for tensor parallelism")
    
    # Load tokenizer if using template
    tokenizer = None
    if use_tokenizer_template or "llama" in judge_model.lower():
        print("Loading tokenizer for chat template formatting...")
        tokenizer = AutoTokenizer.from_pretrained(judge_model, trust_remote_code=True)
        use_tokenizer_template = True
    
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
    prompts = []
    metadata = []
    
    for i in range(num_samples):
        problem_desc = results['prompts'][i]
        generated_code = results['responses'][i]
        test_inputs = test_inputs_list[i][:max_test_cases] if test_inputs_list[i] else []
        test_outputs = test_outputs_list[i][:max_test_cases] if test_outputs_list[i] else []
        
        if use_tokenizer_template and tokenizer:
            system_msg, user_msg = format_apps_judge_prompt_llama(
                problem_desc, generated_code, test_inputs, test_outputs
            )
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = format_apps_judge_prompt_qwen(
                problem_desc, generated_code, test_inputs, test_outputs,
                use_no_think=True
            )
        
        prompts.append(prompt)
        metadata.append((i, problem_desc[:100], generated_code[:100]))
    
    print(f"Built {len(prompts)} judge prompts")
    
    # Run inference
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    
    print(f"\nRunning judge inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract judgments
    print("Extracting judgments...")
    correct_labels = []
    judge_reasons = []
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        is_correct, reason = extract_judgment(generated_text)
        correct_labels.append(1 if is_correct else 0)
        judge_reasons.append(reason)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(outputs)} samples")
    
    # Add to results
    results['correct_labels'] = correct_labels
    results['judge_reasons'] = judge_reasons
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Print statistics
    num_correct = sum(correct_labels)
    accuracy = num_correct / len(correct_labels) if correct_labels else 0.0
    print(f"\nJudging complete!")
    print(f"Correct: {num_correct}/{len(correct_labels)} ({accuracy:.2%})")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Judge APPS code generation results")
    parser.add_argument(
        "--input_file",
        type=str,
        default=str(APPS_PROMPTS_WITH_ANSWERS_QWEN8B_FILE),
        help="Path to input results file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(APPS_PROMPTS_WITH_CORRECT_LABELS_QWEN8B_FILE),
        help="Path to save results with labels"
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="Qwen/Qwen2.5-32B-Instruct",
        help="Judge model name"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--max_test_cases",
        type=int,
        default=5,
        help="Maximum test cases to include in prompt"
    )
    
    args = parser.parse_args()
    
    judge_apps_with_vllm(
        input_file=Path(args.input_file),
        output_file=Path(args.output_file),
        judge_model=args.judge_model,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_test_cases=args.max_test_cases,
        use_tokenizer_template="llama" in args.judge_model.lower(),
    )

