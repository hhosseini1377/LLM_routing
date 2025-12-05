from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import List, Dict, Any, Union
import re
from tqdm import tqdm
# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen3-14B"  # Using the 8B dense model
TARGET_GENERATIONS_PER_PROBLEM = 1
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.7  # Controls diversity; 0.7 balances creativity and faithfulness
SEED = 42
from routing_dataset.dataset_paths import FINAL_TRAIN_FILE
import pickle
from pandas import DataFrame as pd
# --- PIPELINE ---

def load_qwen_model_vllm(
    model_id: str,
    dtype: str = "bfloat16",
    max_num_seqs: int = 256,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1
):
    """
    Loads Qwen3-8B using vLLM for fast batch inference.
    
    Args:
        model_id: Model identifier
        dtype: Data type (bfloat16, float16, etc.)
        max_num_seqs: Maximum number of sequences to process in parallel
        gpu_memory_utilization: GPU memory utilization (0.0-1.0)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        
    Returns:
        LLM instance and tokenizer
    """
    print(f"Loading vLLM model: {model_id}")
    if tensor_parallel_size > 1:
        print(f"Using {tensor_parallel_size} GPUs for tensor parallelism")
    
    # Load tokenizer separately for chat template formatting
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Create LLM instance
    llm_kwargs = {
        'model': model_id,
        'dtype': dtype,
        'max_num_seqs': max_num_seqs,
        'gpu_memory_utilization': gpu_memory_utilization,
        'trust_remote_code': True,
    }
    
    if tensor_parallel_size > 1:
        llm_kwargs['tensor_parallel_size'] = tensor_parallel_size
    
    llm = LLM(**llm_kwargs)
    
    print("Model loaded successfully.")
    return llm, tokenizer

def extract_question_robust(prompt_text: str) -> str:
    """
    Extract question from formatted prompt or return as-is if already a question.
    Handles various formats including chat templates.
    """
    # If it's already a plain question, return it
    if not prompt_text.strip().startswith('<|im_start|>'):
        return prompt_text.strip()
    
    # Find user section
    user_pattern = r'<\|im_start\|>user\s+(.*?)<\|im_end\|>'
    user_match = re.search(user_pattern, prompt_text, re.DOTALL)
    
    if not user_match:
        return prompt_text.strip()
    
    user_content = user_match.group(1).strip()
    
    # Try to find "Question:" prefix (case-insensitive)
    question_patterns = [
        r'Question:\s*(.*)',  # "Question: ..."
        r'question:\s*(.*)',  # "question: ..."
        r'Q:\s*(.*)',         # "Q: ..."
    ]
    
    for pattern in question_patterns:
        match = re.search(pattern, user_content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # If no prefix found, return entire user content
    return user_content.strip()

def generate_paraphrase_prompt(problem_text: str) -> List[Dict[str, str]]:
    """Creates a strict prompt to guide Qwen's paraphrasing, preserving the core math."""
    
    # Extract question if prompt is formatted
    question = extract_question_robust(problem_text)
    
    # The system prompt enforces the behavior: change narrative, keep math logic.
    # Added /no_think to prevent reasoning/thinking output
    system_prompt = (
        "You are an expert data augmentor and creative problem writer. Your task is to "
        "paraphrase the following math problem by changing the setting, characters, and "
        "non-essential objects. However, you MUST ensure the core mathematical structure, "
        "the given numerical values, and the final solution logic remain identical. "
        "Generate ONLY the new, rewritten problem text. /no_think"
    )
    
    user_prompt = f"Original Problem: {question}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return messages

def prepare_prompts_for_batch(
    prompts: List[Union[str, Dict[str, Any]]],
    tokenizer,
    num_generations_per_prompt: int = TARGET_GENERATIONS_PER_PROBLEM
) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    Prepare all prompts for batch processing with vLLM.
    
    Args:
        prompts: List of prompts (strings or dicts)
        tokenizer: Tokenizer for chat template formatting
        num_generations_per_prompt: Number of paraphrases per prompt
        
    Returns:
        Tuple of (formatted_prompts_list, metadata_list)
        metadata_list contains dicts with 'problem_id', 'answer', 'generation_idx'
    """
    formatted_prompts = []
    metadata = []
    
    for idx, prompt in enumerate(prompts):
        # Extract question and metadata
        if isinstance(prompt, dict):
            question = prompt.get('question') or prompt.get('prompt', '')
            problem_id = prompt.get('id', idx)
            answer = prompt.get('answer') or prompt.get('ground_truth')
        else:
            question = prompt
            problem_id = idx
            answer = None
        
        # Extract question if it's a formatted prompt
        question = extract_question_robust(question)
        
        # Create messages for chat template
        messages = generate_paraphrase_prompt(question)
        
        # Format using chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Repeat prompt for multiple generations
        for gen_idx in range(num_generations_per_prompt):
            formatted_prompts.append(formatted_prompt)
            metadata.append({
                'problem_id': problem_id,
                'answer': answer,
                'generation_idx': gen_idx,
                'original_question': question
            })
    
    return formatted_prompts, metadata

def run_augmentation(
    llm: LLM,
    tokenizer,
    prompts: List[Union[str, Dict[str, Any]]],
    num_generations_per_prompt: int = TARGET_GENERATIONS_PER_PROBLEM,
    batch_size: int = None
) -> List[Dict[str, Any]]:
    """
    Generate paraphrases for a list of prompts using vLLM batch inference.
    
    Args:
        llm: Loaded vLLM LLM instance
        tokenizer: Loaded tokenizer
        prompts: List of prompts. Each can be:
            - A string (formatted prompt or plain question)
            - A dict with 'question'/'prompt' key (and optionally 'id', 'answer')
        num_generations_per_prompt: Number of paraphrases per prompt
        batch_size: Optional batch size for processing (None = process all at once)
        
    Returns:
        List of all augmented samples
    """
    print(f"\nPreparing {len(prompts)} prompts for batch processing...")
    print(f"Generating {num_generations_per_prompt} paraphrases per prompt")
    print(f"Total generations: {len(prompts) * num_generations_per_prompt}")
    
    # Prepare all prompts for batch processing
    formatted_prompts, metadata = prepare_prompts_for_batch(
        prompts, tokenizer, num_generations_per_prompt
    )
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
        seed=SEED
    )
    
    # Run batch inference
    print(f"\nRunning batch inference on {len(formatted_prompts)} prompts...")
    
    if batch_size is None:
        # Process all at once (fastest if memory allows)
        outputs = llm.generate(formatted_prompts, sampling_params)
    else:
        # Process in batches
        outputs = []
        for i in tqdm(range(0, len(formatted_prompts), batch_size), desc="Processing batches"):
            batch_prompts = formatted_prompts[i:i + batch_size]
            batch_outputs = llm.generate(batch_prompts, sampling_params)
            outputs.extend(batch_outputs)
    
    # Process outputs and create augmented samples
    print("\nProcessing outputs...")
    all_augmented_data = []
    
    for output, meta in tqdm(zip(outputs, metadata), total=len(outputs), desc="Creating samples"):
        # Extract generated text
        new_problem = output.outputs[0].text.strip()
        
        # Create sample
        sample = {
            "question": new_problem,
            "correct_label": 0  # The label remains 'Fail/Complex' (0)
        }
        
        if meta['problem_id'] is not None:
            sample["original_id"] = meta['problem_id']
        if meta['answer'] is not None:
            sample["answer"] = meta['answer']
        
        all_augmented_data.append(sample)
    
    return all_augmented_data

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Paraphrase GSM8K prompts using vLLM")
    parser.add_argument('--model_id', type=str, default=MODEL_ID, help='Model to use')
    parser.add_argument('--num_generations', type=int, default=TARGET_GENERATIONS_PER_PROBLEM,
                       help=f'Number of paraphrases per prompt (default: {TARGET_GENERATIONS_PER_PROBLEM})')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for processing (None = process all at once)')
    parser.add_argument('--max_num_seqs', type=int, default=256,
                       help='Maximum number of sequences for vLLM (default: 256)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                       help='GPU memory utilization (default: 0.9)')
    parser.add_argument('--tensor_parallel_size', type=int, default=2,
                       help='Number of GPUs for tensor parallelism (default: 1)')
    parser.add_argument('--input_file', type=str, default=None,
                       help='Input pickle file with prompts (optional)')
    parser.add_argument('--output_file', type=str, default='./routing_dataset/datasets/gsm8k/synthetic_data2.pkl',
                       help='Output pickle file path')
    parser.add_argument('--max_prompts', type=int, default=None,
                       help='Maximum number of prompts to process (for testing)')
    args = parser.parse_args()
    
    model_id = args.model_id
    
    # Load model with vLLM
    llm, tokenizer = load_qwen_model_vllm(
        model_id,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size
    )
    
    # Load prompts
    if args.input_file:
        with open(args.input_file, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            prompts = data.get('prompts', [])
        else:
            prompts = data
    else:
        # Default: load from FINAL_TRAIN_FILE
        with open(FINAL_TRAIN_FILE, 'rb') as f:
            final_train = pd(pickle.load(f))
        mask = (final_train['dataset_source'] == 'GSM8K') & (final_train['correct_labels'] == 0)
        final_train_gsm8k = final_train[mask]
        prompts = final_train_gsm8k['prompts'].apply(extract_question_robust).tolist()
    
    # Limit prompts if specified
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    
    print(f"\nProcessing {len(prompts)} prompts...")
    
    # Run augmentation
    synthetic_data = run_augmentation(
        llm,
        tokenizer,
        prompts,
        num_generations_per_prompt=args.num_generations,
        batch_size=args.batch_size
    )
    
    # Save the synthetic data
    print(f"\nSaving results to {args.output_file}...")
    with open(args.output_file, 'wb') as f:
        pickle.dump(synthetic_data, f)
    
    print("\n--- AUGMENTATION COMPLETE ---")
    print(f"Total synthetic samples generated: {len(synthetic_data)}")
    print(f"Input prompts: {len(prompts)}")
    print(f"Generations per prompt: {args.num_generations}")
    print(f"Output saved to: {args.output_file}")

    # NOTE: You should now perform quality control (QC) on the 'question' field
    # of the synthetic_data list to ensure the math logic was preserved.