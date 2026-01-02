"""
Script to judge LMSYS-Chat-1M prompt responses using an LLM judge.

This script:
1. Loads LMSYS results with generated responses
2. Uses an LLM judge to evaluate response quality
3. Adds scores (0-10) and judge_reasons to the dataset
"""

import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd

from routing_dataset.dataset_paths import (
    LMSYS_CHAT1M_PROMPTS_100K_WITH_ANSWERS_QWEN8B_FILE,
    LMSYS_CHAT1M_PROMPTS_100K_WITH_SCORES_QWEN8B_FILE,
)


def format_lmsys_judge_prompt_qwen(
    user_prompt: str,
    generated_response: str,
    use_no_think: bool = True
) -> str:
    """
    Format an LMSYS response evaluation prompt for Qwen judge model.
    
    Args:
        user_prompt: The original user prompt/question
        generated_response: The generated response to evaluate
        use_no_think: Whether to use /no_think command for Qwen3 models
    
    Returns:
        Formatted prompt string in ChatML format
    """
    # System instruction for response evaluation
    if use_no_think:
        system_instruction = (
            "/no_think You are an expert evaluator specializing in conversational AI responses. "
            "Your task is to evaluate the quality of a generated response to a user's prompt. "
            "Consider relevance, accuracy, completeness, clarity, and helpfulness. "
            "Respond only in JSON format with 'score' (integer from 0 to 10) indicating the response quality, "
            "where 0 is very poor and 10 is excellent, and 'reason' (brief explanation)."
        )
    else:
        system_instruction = (
            "You are an expert evaluator specializing in conversational AI responses. "
            "Your task is to evaluate the quality of a generated response to a user's prompt. "
            "Consider relevance, accuracy, completeness, clarity, and helpfulness. "
            "Respond only in JSON format with 'score' (integer from 0 to 10) indicating the response quality, "
            "where 0 is very poor and 10 is excellent, and 'reason' (brief explanation)."
        )
    
    # Build user content
    user_content = (
        f"User Prompt:\n{user_prompt}\n\n"
        f"Generated Response:\n{generated_response}\n\n"
        "Evaluate the quality of the generated response. Consider:\n"
        "- Relevance: Does the response directly address the user's prompt/question?\n"
        "- Accuracy: Is the information provided factually correct and reliable?\n"
        "- Completeness: Does it fully answer the question or address all aspects of the request?\n"
        "- Clarity: Is it well-written, easy to understand, and well-structured?\n"
        "- Helpfulness: Is the response useful and actionable for the user?\n\n"
        "Score the response from 0 to 10:\n"
        "- 0-3: Very poor quality (irrelevant, incorrect, incomplete, or unhelpful)\n"
        "- 4-6: Moderate quality (partially relevant, some accuracy issues, incomplete, or unclear)\n"
        "- 7-8: Good quality (mostly relevant and accurate, minor gaps or clarity issues)\n"
        "- 9-10: Excellent quality (highly relevant, accurate, complete, clear, and helpful)\n\n"
        "Respond in JSON format:\n"
        "{\n"
        '  "score": <integer from 0 to 10>,\n'
        '  "reason": "brief explanation of the score"\n'
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


def format_lmsys_judge_prompt_llama(
    user_prompt: str,
    generated_response: str,
) -> Tuple[str, str]:
    """
    Format an LMSYS response evaluation prompt for Llama judge model.
    
    Args:
        user_prompt: The original user prompt/question
        generated_response: The generated response to evaluate
    
    Returns:
        Tuple of (system_message, user_message) for chat template
    """
    system_message = (
        "You are an expert evaluator specializing in conversational AI responses. "
        "Your task is to evaluate the quality of a generated response to a user's prompt. "
        "Consider relevance, accuracy, completeness, clarity, and helpfulness. "
        "Respond only in JSON format with 'score' (integer from 0 to 10) indicating the response quality, "
        "where 0 is very poor and 10 is excellent, and 'reason' (brief explanation)."
    )
    
    user_message = (
        f"User Prompt:\n{user_prompt}\n\n"
        f"Generated Response:\n{generated_response}\n\n"
        "Evaluate the quality of the generated response. Consider:\n"
        "- Relevance: Does the response directly address the user's prompt/question?\n"
        "- Accuracy: Is the information provided factually correct and reliable?\n"
        "- Completeness: Does it fully answer the question or address all aspects of the request?\n"
        "- Clarity: Is it well-written, easy to understand, and well-structured?\n"
        "- Helpfulness: Is the response useful and actionable for the user?\n\n"
        "Score the response from 0 to 10:\n"
        "- 0-3: Very poor quality (irrelevant, incorrect, incomplete, or unhelpful)\n"
        "- 4-6: Moderate quality (partially relevant, some accuracy issues, incomplete, or unclear)\n"
        "- 7-8: Good quality (mostly relevant and accurate, minor gaps or clarity issues)\n"
        "- 9-10: Excellent quality (highly relevant, accurate, complete, clear, and helpful)\n\n"
        "Respond in JSON format:\n"
        "{\n"
        '  "score": <integer from 0 to 10>,\n'
        '  "reason": "brief explanation of the score"\n'
        "}"
    )
    
    return system_message, user_message


def extract_judgment(text: str) -> Tuple[int, str]:
    """
    Extract score and reason from judge model output.
    
    Args:
        text: Raw text output from judge model
    
    Returns:
        Tuple of (score: int 0-10, reason: str)
    """
    text = text.strip()
    
    # Try to parse as JSON first
    try:
        # Try to find JSON object in the text
        json_match = re.search(r'\{[^}]*"score"[^}]*\}', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
        else:
            result = json.loads(text)
        
        score = int(result.get("score", 0))
        # Clamp score to 0-10 range
        score = max(0, min(10, score))
        
        reason = result.get("reason", "")
        return score, reason
    except (ValueError, json.JSONDecodeError, KeyError) as e:
        # Fallback: try to extract score from text
        # Look for "score": X or score: X patterns
        score_match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if score_match:
            try:
                score = int(float(score_match.group(1)))
                score = max(0, min(10, score))
                return score, f"Extracted from text (JSON parse failed: {str(e)[:50]})"
            except (ValueError, TypeError):
                pass
        
        # If no score found, return 0
        return 0, f"JSON parsing failed: {str(e)[:50]}"


def judge_lmsys_prompts_with_vllm(
    input_file: Path = LMSYS_CHAT1M_PROMPTS_100K_WITH_ANSWERS_QWEN8B_FILE,
    output_file: Path = LMSYS_CHAT1M_PROMPTS_100K_WITH_SCORES_QWEN8B_FILE,
    judge_model: str = "Qwen/Qwen2.5-32B-Instruct",
    temperature: float = 0.0,
    max_tokens: int = 512,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.8,
    max_num_seqs: int = 256,
    tensor_parallel_size: int = 4,
    use_tokenizer_template: bool = False,
    disable_no_think: bool = False,
) -> Dict:
    """
    Judge LMSYS prompt responses using an LLM judge.
    
    Args:
        input_file: Path to pickled results with 'responses' and 'prompts'
        output_file: Path to save results with scores
        judge_model: Model identifier for the judge (default: "Qwen/Qwen2.5-32B-Instruct")
        temperature: Sampling temperature (default: 0.0 for deterministic)
        max_tokens: Maximum tokens to generate (default: 512)
        dtype: Model dtype (default: "bfloat16")
        gpu_memory_utilization: GPU memory utilization (default: 0.8)
        max_num_seqs: Maximum sequences in parallel (default: 256)
        tensor_parallel_size: Number of GPUs for tensor parallelism (default: 4)
        use_tokenizer_template: If True, use tokenizer's chat template (for Llama models)
        disable_no_think: If True, disable /no_think command (default: False)
    
    Returns:
        Dictionary with added 'scores' (list of integers 0-10) and 'judge_reasons'
    """
    print(f"Loading LMSYS results from: {input_file}")
    
    # Load results
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    # Handle both DataFrame and dictionary inputs
    if isinstance(data, pd.DataFrame):
        print("Input is a DataFrame, converting to dictionary format...")
        results = {}
        
        # Extract prompts
        prompt_column = None
        for col in ['prompts', 'prompt', 'user_prompt', 'user_prompts']:
            if col in data.columns:
                prompt_column = col
                break
        
        if prompt_column is None:
            raise ValueError(f"Could not find prompt column. Available columns: {list(data.columns)}")
        
        results['prompts'] = data[prompt_column].tolist()
        
        # Extract responses
        response_column = None
        for col in ['responses', 'response', 'generated_response', 'answers']:
            if col in data.columns:
                response_column = col
                break
        
        if response_column is None:
            raise ValueError(f"Could not find response column. Available columns: {list(data.columns)}")
        
        results['responses'] = data[response_column].tolist()
        
        # Extract other columns if available
        for col in ['formatted_prompts']:
            if col in data.columns:
                results[col] = data[col].tolist()
        
        print(f"Converted DataFrame with columns: {list(data.columns)}")
    elif isinstance(data, dict):
        results = data
    else:
        raise ValueError(f"Expected dictionary or DataFrame, got {type(data)}")
    
    # Validate required fields
    if 'prompts' not in results:
        raise ValueError("Missing 'prompts' field in input data")
    if 'responses' not in results:
        raise ValueError("Missing 'responses' field in input data")
    
    prompts = results['prompts']
    responses = results['responses']
    
    if len(prompts) != len(responses):
        raise ValueError(f"Mismatch: {len(prompts)} prompts but {len(responses)} responses")
    
    num_examples = len(prompts)
    print(f"Loaded {num_examples} prompt-response pairs")
    
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
    
    # Determine if we should use tokenizer template (for Llama models)
    is_llama_model = 'llama' in judge_model.lower() or 'mistral' in judge_model.lower()
    if use_tokenizer_template or is_llama_model:
        print("Using tokenizer chat template for Llama/Mistral models")
        tokenizer = AutoTokenizer.from_pretrained(judge_model, trust_remote_code=True)
        if tokenizer.chat_template is None:
            print("Warning: No chat template found, falling back to Qwen format")
            use_tokenizer_template = False
    
    # Build judge prompts
    print("\nBuilding judge prompts...")
    judge_prompts = []
    
    use_no_think = not disable_no_think
    
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        if use_tokenizer_template:
            system_msg, user_msg = format_lmsys_judge_prompt_llama(prompt, response)
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = format_lmsys_judge_prompt_qwen(prompt, response, use_no_think=use_no_think)
        
        judge_prompts.append(formatted_prompt)
        
        if i < 2:
            print(f"\nSample judge prompt {i+1} (first 500 chars):\n{formatted_prompt[:500]}...")
    
    # Run inference
    print(f"\nRunning judge inference on {len(judge_prompts)} examples...")
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    
    outputs = llm.generate(judge_prompts, sampling_params)
    judge_outputs = [out.outputs[0].text.strip() for out in outputs]
    
    print(f"Extracted {len(judge_outputs)} judge outputs")
    
    # Extract scores and reasons
    print("\nExtracting scores and reasons...")
    scores = []
    judge_reasons = []
    
    for i, output_text in enumerate(judge_outputs):
        score, reason = extract_judgment(output_text)
        scores.append(score)
        judge_reasons.append(reason)
        
        if i < 5:
            print(f"Example {i+1}: score={score}, reason={reason[:100]}...")
    
    # Add scores and reasons to results
    results['scores'] = scores
    results['judge_reasons'] = judge_reasons
    
    # Print statistics
    print(f"\n=== Judging Statistics ===")
    print(f"Total examples: {len(scores)}")
    print(f"Mean score: {sum(scores) / len(scores):.2f}")
    print(f"Std dev: {(sum((s - sum(scores) / len(scores))**2 for s in scores) / len(scores))**0.5:.2f}")
    print(f"Min score: {min(scores)}")
    print(f"Max score: {max(scores)}")
    
    # Score distribution
    score_dist = {}
    for score in scores:
        score_dist[score] = score_dist.get(score, 0) + 1
    print(f"\nScore distribution:")
    for score in sorted(score_dist.keys()):
        count = score_dist[score]
        percentage = 100 * count / len(scores)
        print(f"  {score}: {count} ({percentage:.1f}%)")
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print("Done!")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Judge LMSYS prompt responses using an LLM judge")
    parser.add_argument(
        "--input_file",
        type=str,
        default=str(LMSYS_CHAT1M_PROMPTS_100K_WITH_ANSWERS_QWEN8B_FILE),
        help="Path to input file with prompts and responses"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(LMSYS_CHAT1M_PROMPTS_100K_WITH_SCORES_QWEN8B_FILE),
        help="Path to save results with scores"
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="Qwen/Qwen2.5-32B-Instruct",
        help="Judge model identifier"
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
        default=512,
        help="Maximum tokens to generate (default: 512)"
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
        default=4,
        help="Number of GPUs for tensor parallelism (default: 4)"
    )
    parser.add_argument(
        "--use_tokenizer_template",
        action="store_true",
        help="Use tokenizer's chat template (for Llama models)"
    )
    parser.add_argument(
        "--disable_no_think",
        action="store_true",
        help="Disable /no_think command (for non-Qwen models)"
    )
    
    args = parser.parse_args()
    
    judge_lmsys_prompts_with_vllm(
        input_file=Path(args.input_file),
        output_file=Path(args.output_file),
        judge_model=args.judge_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        use_tokenizer_template=args.use_tokenizer_template,
        disable_no_think=args.disable_no_think,
    )

