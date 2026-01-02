"""
Script to judge CNN/DailyMail summarization results using an LLM judge.

This script:
1. Loads CNN/DailyMail results with generated summaries
2. Uses an LLM judge to evaluate summary quality based on reference summaries
3. Adds scores (0-10) and judge_reasons to the dataset
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd

from routing_dataset.dataset_paths import (
    CNN_DAILY_MAIL_PROMPTS_WITH_ANSWERS_QWEN8B_FILE,
    CNN_DAILY_MAIL_PROMPTS_WITH_CORRECT_LABELS_QWEN8B_FILE,
)


def format_cnn_dailymail_judge_prompt_qwen(
    article_text: str,
    generated_summary: str,
    reference_summary: Optional[str] = None,
    use_no_think: bool = True
) -> str:
    """
    Format a CNN/DailyMail summary evaluation prompt for Qwen judge model.
    
    Args:
        article_text: The original article text
        generated_summary: The generated summary to evaluate
        reference_summary: Optional reference summary (highlights) for comparison
        use_no_think: Whether to use /no_think command for Qwen3 models
    
    Returns:
        Formatted prompt string in ChatML format
    """
    # System instruction for summary evaluation
    if use_no_think:
        system_instruction = (
            "/no_think You are an expert evaluator specializing in news article summarization. "
            "Your task is to evaluate the quality of a generated summary based on the original article. "
            "Consider factual accuracy, coverage of main points, coherence, and conciseness. "
            "Respond only in JSON format with 'score' (integer from 0 to 10) indicating the summary quality, "
            "where 0 is very poor and 10 is excellent, and 'reason' (brief explanation)."
        )
    else:
        system_instruction = (
            "You are an expert evaluator specializing in news article summarization. "
            "Your task is to evaluate the quality of a generated summary based on the original article. "
            "Consider factual accuracy, coverage of main points, coherence, and conciseness. "
            "Respond only in JSON format with 'score' (integer from 0 to 10) indicating the summary quality, "
            "where 0 is very poor and 10 is excellent, and 'reason' (brief explanation)."
        )
    
    # Build user content
    user_content = (
        f"Article:\n{article_text}\n\n"
        f"Generated Summary:\n{generated_summary}\n\n"
    )
    
    if reference_summary:
        user_content += (
            f"Reference Summary (for comparison):\n{reference_summary}\n\n"
        )
    
    user_content += (
        "Evaluate the quality of the generated summary. Consider:\n"
        "- Factual accuracy: Does the summary accurately represent the facts in the article?\n"
        "- Coverage: Does it cover the main points and key information?\n"
        "- Coherence: Is it well-written and easy to understand?\n"
        "- Conciseness: Is it appropriately concise while maintaining important information?\n"
        "- Completeness: Does it provide sufficient context without unnecessary details?\n\n"
        "Score the summary from 0 to 10:\n"
        "- 0-3: Very poor quality (factually incorrect, missing key points, incoherent)\n"
        "- 4-6: Moderate quality (some issues with accuracy, coverage, or clarity)\n"
        "- 7-8: Good quality (mostly accurate and complete, minor issues)\n"
        "- 9-10: Excellent quality (highly accurate, comprehensive, well-written)\n\n"
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


def format_cnn_dailymail_judge_prompt_llama(
    article_text: str,
    generated_summary: str,
    reference_summary: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Format a CNN/DailyMail summary evaluation prompt for Llama judge model.
    
    Args:
        article_text: The original article text
        generated_summary: The generated summary to evaluate
        reference_summary: Optional reference summary (highlights) for comparison
    
    Returns:
        Tuple of (system_message, user_message) for chat template
    """
    system_message = (
        "You are an expert evaluator specializing in news article summarization. "
        "Your task is to evaluate the quality of a generated summary based on the original article. "
        "Consider factual accuracy, coverage of main points, coherence, and conciseness. "
        "Respond only in JSON format with 'score' (integer from 0 to 10) indicating the summary quality, "
        "where 0 is very poor and 10 is excellent, and 'reason' (brief explanation)."
    )
    
    user_message = (
        f"Article:\n{article_text}\n\n"
        f"Generated Summary:\n{generated_summary}\n\n"
    )
    
    if reference_summary:
        user_message += (
            f"Reference Summary (for comparison):\n{reference_summary}\n\n"
        )
    
    user_message += (
        "Evaluate the quality of the generated summary. Consider:\n"
        "- Factual accuracy: Does the summary accurately represent the facts in the article?\n"
        "- Coverage: Does it cover the main points and key information?\n"
        "- Coherence: Is it well-written and easy to understand?\n"
        "- Conciseness: Is it appropriately concise while maintaining important information?\n"
        "- Completeness: Does it provide sufficient context without unnecessary details?\n\n"
        "Score the summary from 0 to 10:\n"
        "- 0-3: Very poor quality (factually incorrect, missing key points, incoherent)\n"
        "- 4-6: Moderate quality (some issues with accuracy, coverage, or clarity)\n"
        "- 7-8: Good quality (mostly accurate and complete, minor issues)\n"
        "- 9-10: Excellent quality (highly accurate, comprehensive, well-written)\n\n"
        "Respond in JSON format:\n"
        "{\n"
        '  "score": <integer from 0 to 10>,\n'
        '  "reason": "brief explanation of the score"\n'
        "}"
    )
    
    return system_message, user_message


def extract_judgment(generated_text: str) -> Tuple[int, str]:
    """
    Extract JSON judgment from generated text.
    
    Args:
        generated_text: Raw text from the judge model
    
    Returns:
        Tuple of (score: int from 0 to 10, reason: str)
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
        
        score = result.get("score", 0)
        # Handle string scores and convert to int
        if isinstance(score, str):
            try:
                score = int(float(score))
            except (ValueError, TypeError):
                score = 0
        elif isinstance(score, float):
            score = int(score)
        elif not isinstance(score, int):
            score = 0
        
        # Clamp score to 0-10 range
        score = max(0, min(10, int(score)))
        
        reason = result.get("reason", "")
        return score, reason
    except (ValueError, json.JSONDecodeError, KeyError) as e:
        # Fallback: try to extract score from text
        import re
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


def judge_cnn_dailymail_with_vllm(
    input_file: Path = CNN_DAILY_MAIL_PROMPTS_WITH_ANSWERS_QWEN8B_FILE,
    output_file: Path = CNN_DAILY_MAIL_PROMPTS_WITH_CORRECT_LABELS_QWEN8B_FILE,
    judge_model: str = "Qwen/Qwen2.5-32B-Instruct",
    temperature: float = 0.0,
    max_tokens: int = 512,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.8,
    max_num_seqs: int = 256,
    tensor_parallel_size: int = 4,
    use_reference_summary: bool = True,
    use_tokenizer_template: bool = False,
    max_article_length: int = 5000,
) -> Dict:
    """
    Judge CNN/DailyMail summarization results using an LLM judge.
    
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
        use_reference_summary: Whether to include reference summary in prompt (default: True)
        use_tokenizer_template: If True, use tokenizer's chat template (for Llama models)
        max_article_length: Maximum article length to include (truncate if longer, default: 5000)
    
    Returns:
        Dictionary with added 'scores' (list of integers 0-10) and 'judge_reasons'
    """
    print(f"Loading CNN/DailyMail results from: {input_file}")
    
    # Load results
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    # Handle both DataFrame and dictionary inputs
    if isinstance(data, pd.DataFrame):
        print("Input is a DataFrame, converting to dictionary format...")
        results = {}
        
        # Extract articles - try common column names
        article_column = None
        for col in ['article', 'text', 'article_text', 'articles']:
            if col in data.columns:
                article_column = col
                break
        
        if article_column is None:
            raise ValueError(f"Could not find article column. Available columns: {list(data.columns)}")
        
        results['articles'] = data[article_column].tolist()
        
        # Extract responses - try common column names
        response_column = None
        for col in ['responses', 'response', 'summary', 'generated_summary']:
            if col in data.columns:
                response_column = col
                break
        
        if response_column is None:
            raise ValueError(f"Could not find response column. Available columns: {list(data.columns)}")
        
        results['responses'] = data[response_column].tolist()
        
        # Extract other columns if available
        for col in ['highlights', 'summary', 'reference_summary', 'reference', 'article_ids', 'formatted_prompts', 'url']:
            if col in data.columns:
                results[col] = data[col].tolist()
        
        print(f"Converted DataFrame with columns: {list(data.columns)}")
    elif isinstance(data, dict):
        results = data
    else:
        raise ValueError(f"Expected dictionary or DataFrame, got {type(data)}")
    
    required_keys = ['articles', 'responses']
    missing_keys = [k for k in required_keys if k not in results]
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")
    
    num_samples = len(results['articles'])
    print(f"Loaded {num_samples} samples")
    
    # Get reference summaries if available
    reference_summaries = None
    if use_reference_summary:
        # Try different possible column names for reference summaries
        for col_name in ['highlights', 'summary', 'reference_summary', 'reference']:
            if col_name in results:
                reference_summaries = results[col_name]
                print(f"Found reference summaries in column: {col_name}")
                break
        
        if reference_summaries is None:
            print("Warning: No reference summaries found. Proceeding without them.")
            use_reference_summary = False
    
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
        article_text = str(results['articles'][i])
        generated_summary = str(results['responses'][i])
        
        # Truncate article if too long
        if len(article_text) > max_article_length:
            article_text = article_text[:max_article_length] + "\n... (truncated)"
        
        reference_summary = None
        if use_reference_summary and reference_summaries:
            reference_summary = str(reference_summaries[i]) if i < len(reference_summaries) else None
        
        if use_tokenizer_template and tokenizer:
            system_msg, user_msg = format_cnn_dailymail_judge_prompt_llama(
                article_text, generated_summary, reference_summary
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
            prompt = format_cnn_dailymail_judge_prompt_qwen(
                article_text, generated_summary, reference_summary,
                use_no_think=True
            )
        
        prompts.append(prompt)
        metadata.append((i, article_text[:100], generated_summary[:100]))
    
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
    scores = []
    judge_reasons = []
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        score, reason = extract_judgment(generated_text)
        scores.append(score)
        judge_reasons.append(reason)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(outputs)} samples")
    
    # Add to results
    results['scores'] = scores
    results['judge_reasons'] = judge_reasons
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Print statistics
    import statistics
    mean_score = statistics.mean(scores) if scores else 0.0
    std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 0
    
    # Count by quality tiers
    very_poor = sum(1 for s in scores if s <= 3)
    moderate = sum(1 for s in scores if 4 <= s <= 6)
    good = sum(1 for s in scores if 7 <= s <= 8)
    excellent = sum(1 for s in scores if s >= 9)
    
    print(f"\nJudging complete!")
    print(f"Score statistics:")
    print(f"  Mean: {mean_score:.2f} ± {std_score:.2f}")
    print(f"  Range: {min_score} - {max_score}")
    print(f"  Distribution:")
    print(f"    Very poor (0-3): {very_poor} ({very_poor/len(scores)*100:.1f}%)")
    print(f"    Moderate (4-6): {moderate} ({moderate/len(scores)*100:.1f}%)")
    print(f"    Good (7-8): {good} ({good/len(scores)*100:.1f}%)")
    print(f"    Excellent (9-10): {excellent} ({excellent/len(scores)*100:.1f}%)")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Judge CNN/DailyMail summarization results")
    parser.add_argument(
        "--input_file",
        type=str,
        default=str(CNN_DAILY_MAIL_PROMPTS_WITH_ANSWERS_QWEN8B_FILE),
        help="Path to input results file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(CNN_DAILY_MAIL_PROMPTS_WITH_CORRECT_LABELS_QWEN8B_FILE),
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
        "--use_reference_summary",
        action="store_true",
        default=True,
        help="Include reference summary in prompt (default: True)"
    )
    parser.add_argument(
        "--no_use_reference_summary",
        dest="use_reference_summary",
        action="store_false",
        help="Do not include reference summary in prompt"
    )
    parser.add_argument(
        "--max_article_length",
        type=int,
        default=5000,
        help="Maximum article length to include (truncate if longer)"
    )
    
    args = parser.parse_args()
    
    judge_cnn_dailymail_with_vllm(
        input_file=Path(args.input_file),
        output_file=Path(args.output_file),
        judge_model=args.judge_model,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        use_reference_summary=args.use_reference_summary,
        use_tokenizer_template="llama" in args.judge_model.lower(),
        max_article_length=args.max_article_length,
    )

