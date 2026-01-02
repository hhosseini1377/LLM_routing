"""
Script to run CNN/DailyMail dataset using vLLM on Qwen3-8B.

This script:
1. Loads prompts from CNN_DAILY_MAIL_PROMPTS_FILE
2. Formats them using a summarization prompt template
3. Runs inference using vLLM with Qwen3-8B
4. Saves results to a pickle file
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional
from vllm import LLM, SamplingParams
from routing_dataset.dataset_paths import (
    CNN_DAILY_MAIL_PROMPTS_FILE,
    CNN_DAILY_MAIL_PROMPTS_WITH_ANSWERS_QWEN8B_FILE,
    CNN_DAILY_MAIL_PROMPTS_30000_FILE,
    CNN_DAILY_MAIL_DATA_DIR,
)
import pandas as pd


def format_cnn_dailymail_prompt_qwen(
    article_text: str,
    use_chatml: bool = True,
    use_no_think: bool = True
) -> str:
    """
    Format a CNN/DailyMail article into a Qwen-style prompt for summarization.
    
    Args:
        article_text: The article text to summarize
        use_chatml: Whether to use ChatML format (default: True for Qwen models)
        use_no_think: Whether to use /no_think command to disable thinking mode (default: True)
    
    Returns:
        Formatted prompt string
    """
    # Simple prompt format as specified by user
    user_content = (
        "Summarize the following article in a few sentences, focusing on the main points.\n\n"
        f"Article:\n{article_text}\n\n"
        "Summary:\n"
    )
    
    if use_chatml:
        # Format using ChatML (Qwen format)
        # The /no_think command tells Qwen3 models to skip thinking/reasoning mode
        # and output the summary directly without reasoning text
        if use_no_think:
            system_instruction = (
                "/no_think You are a helpful assistant that provides concise and accurate summaries "
                "of news articles. Output only the summary without any reasoning or explanation."
            )
        else:
            system_instruction = (
                "You are a helpful assistant that provides concise and accurate summaries "
                "of news articles. Output only the summary without any reasoning or explanation."
            )
        
        prompt = (
            "<|im_start|>system\n"
            f"{system_instruction}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_content}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    else:
        # Plain text format
        prompt = user_content
    
    return prompt


def run_cnn_dailymail_with_vllm(
    prompts_file: Path = CNN_DAILY_MAIL_PROMPTS_30000_FILE,
    output_file: Path = CNN_DAILY_MAIL_PROMPTS_WITH_ANSWERS_QWEN8B_FILE,
    model_name: str = "Qwen/Qwen3-8B",
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_tokens: int = 512,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.8,
    max_num_seqs: int = 256,
    tensor_parallel_size: int = 1,
    batch_size: Optional[int] = None,
    use_no_think: bool = True,
) -> Dict[str, List]:
    """
    Run CNN/DailyMail dataset prompts through vLLM with Qwen3-8B.
    
    Args:
        prompts_file: Path to the pickled CNN/DailyMail prompts DataFrame
        output_file: Path to save the results
        model_name: Model identifier (default: "Qwen/Qwen3-8B")
        temperature: Sampling temperature (default: 0.0 for deterministic)
        top_p: Top-p sampling parameter (default: 0.95)
        max_tokens: Maximum tokens to generate (default: 512 for summaries)
        dtype: Model dtype (default: "bfloat16")
        gpu_memory_utilization: GPU memory utilization (default: 0.8)
        max_num_seqs: Maximum sequences in parallel (default: 256)
        tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
        batch_size: Optional batch size for processing (None = let vLLM handle it)
        use_no_think: Whether to use /no_think to disable thinking mode (default: True)
                     Set to False if you want the model to reason through summaries
    
    Returns:
        Dictionary with 'articles', 'formatted_prompts', 'responses', and metadata
    """
    print(f"Loading CNN/DailyMail prompts from: {prompts_file}")
    
    # Load prompts DataFrame
    with open(prompts_file, 'rb') as f:
        cnn_dailymail_df = pickle.load(f)
    
    if not isinstance(cnn_dailymail_df, pd.DataFrame):
        raise ValueError(f"Expected DataFrame, got {type(cnn_dailymail_df)}")
    
    print(f"Loaded {len(cnn_dailymail_df)} CNN/DailyMail articles")
    print(f"Columns: {list(cnn_dailymail_df.columns)}")
    
    # Extract article texts
    # Try common column names for article text
    article_column = None
    for col in ['article', 'text', 'article_text', 'prompts', 'input']:
        if col in cnn_dailymail_df.columns:
            article_column = col
            break
    
    if article_column is None:
        raise ValueError(f"Could not find article column. Available columns: {list(cnn_dailymail_df.columns)}")
    
    article_texts = cnn_dailymail_df[article_column].tolist()
    
    # Extract IDs if available
    id_column = None
    for col in ['id', 'article_id', 'idx', 'index']:
        if col in cnn_dailymail_df.columns:
            id_column = col
            break
    
    if id_column:
        article_ids = cnn_dailymail_df[id_column].tolist()
    else:
        article_ids = list(range(len(cnn_dailymail_df)))
    
    # Format prompts
    print("Formatting prompts...")
    print(f"Using /no_think mode: {use_no_think}")
    formatted_prompts = []
    for article_text in article_texts:
        formatted_prompt = format_cnn_dailymail_prompt_qwen(
            article_text=str(article_text),
            use_chatml=True,
            use_no_think=use_no_think
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
        'article_ids': article_ids,
        'articles': article_texts,
        'formatted_prompts': formatted_prompts,
        'responses': responses,
    }
    
    # Add other metadata if available
    if 'highlights' in cnn_dailymail_df.columns:
        result_dict['highlights'] = cnn_dailymail_df['highlights'].tolist()
    if 'summary' in cnn_dailymail_df.columns:
        result_dict['summary'] = cnn_dailymail_df['summary'].tolist()
    if 'url' in cnn_dailymail_df.columns:
        result_dict['url'] = cnn_dailymail_df['url'].tolist()
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(result_dict, f)
    
    print("Done!")
    return result_dict


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CNN/DailyMail dataset with vLLM on Qwen3-8B")
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=str(CNN_DAILY_MAIL_PROMPTS_30000_FILE),
        help="Path to CNN/DailyMail prompts pickle file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(CNN_DAILY_MAIL_PROMPTS_WITH_ANSWERS_QWEN8B_FILE),
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
        default=512,
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
        "--disable_no_think",
        action="store_false",
        dest="use_no_think",
        help="Disable /no_think mode to allow model reasoning (default: use_no_think=True)"
    )
    parser.set_defaults(use_no_think=True)
    
    args = parser.parse_args()
    
    run_cnn_dailymail_with_vllm(
        prompts_file=Path(args.prompts_file),
        output_file=Path(args.output_file),
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        use_no_think=args.use_no_think,
    )

