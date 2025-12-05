import re
from vllm import LLM, SamplingParams
from typing import Dict, List, Optional
import pickle
from routing_dataset.load_dataset import load_mmlu_split, load_mmlu_pro_dataset, load_gsm8k_split
from routing_dataset.dataset_paths import *


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """
    Extract numerical answer from GSM8K model output.
    
    GSM8K answers are typically formatted as:
    - "#### <number>" (standard format)
    - "\\boxed{<number>}" (LaTeX boxed format)
    - "The answer is <number>" or "Final Answer: <number>"
    - Just a number at the end
    
    Args:
        text: Model output text
    
    Returns:
        Extracted numerical answer as string (with commas removed), or None if not found
    """
    if not text:
        return None
    
    # Clean up thinking tokens first (both <think> and <think>)
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r'<think>\s*\n?', '', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'<think>\s*\n?', '', cleaned_text, flags=re.IGNORECASE)
    
    # Pattern 1: LaTeX boxed format: "\\boxed{<number>}" or "$\\boxed{<number>}$"
    match = re.search(r'\\boxed\{([0-9,.\-]+)\}', cleaned_text)
    if match:
        return match.group(1).replace(',', '').strip()
    
    # Pattern 2: Standard GSM8K format: "#### <number>"
    match = re.search(r'####\s*([0-9,.\-]+)', cleaned_text)
    if match:
        return match.group(1).replace(',', '').strip()
    
    # Pattern 3: "Final Answer:" section (often appears after reasoning)
    # Look for "Final Answer:" or "Final Answer" followed by a number
    match = re.search(r'Final\s+Answer\s*:?\s*([0-9,.\-]+)', cleaned_text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '').strip()
    
    # Pattern 4: "The answer is <number>" or "Answer: <number>"
    match = re.search(r'(?:The answer is|Answer|answer)\s*:?\s*\$?\s*([0-9,.\-]+)', cleaned_text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '').strip()
    
    # Pattern 5: Number after "Final Answer:" section header (with markdown formatting)
    # This handles cases where there's a "### **Final Answer:**" header followed by reasoning and then a number
    final_answer_section = re.search(r'Final\s+Answer\s*:.*?([0-9,.\-]+)', cleaned_text, re.DOTALL | re.IGNORECASE)
    if final_answer_section:
        # Extract the number from the final answer section
        number_match = re.search(r'([0-9,.\-]+)', final_answer_section.group(0))
        if number_match:
            return number_match.group(1).replace(',', '').strip()
    
    # Pattern 6: Number at the end of the text (last number in the response)
    # Look for numbers that appear after reasoning (usually at the end)
    numbers = re.findall(r'([0-9,.\-]+)', cleaned_text)
    if numbers:
        # Return the last number found (most likely the final answer)
        return numbers[-1].replace(',', '').strip()
    
    return None


def is_gsm8k_dataset(ground_truths: List[str]) -> bool:
    """
    Detect if dataset is GSM8K (numerical answers) vs MMLU (letter answers).
    
    Args:
        ground_truths: List of ground truth labels
    
    Returns:
        True if dataset appears to be GSM8K (contains numbers), False otherwise
    """
    if not ground_truths:
        return False
    
    # Check first few ground truths to determine dataset type
    sample_size = min(10, len(ground_truths))
    numeric_count = 0
    letter_count = 0
    
    for gt in ground_truths[:sample_size]:
        # Check if ground truth is numeric (not a single letter A-J)
        if isinstance(gt, str):
            cleaned = gt.replace(',', '').strip()
            # Check if it's a single letter A-J (MMLU format)
            if re.match(r'^[A-J]$', cleaned, re.IGNORECASE):
                letter_count += 1
            # Check if it's a number
            elif re.match(r'^-?\d+\.?\d*$', cleaned):
                numeric_count += 1
        elif isinstance(gt, (int, float)):
            numeric_count += 1
    
    # If majority are letters (A-J), it's MMLU/MMLU-Pro
    if letter_count > sample_size / 2:
        return False
    # If majority are numeric, it's GSM8K
    if numeric_count > sample_size / 2:
        return True
    # Default: if more numeric than letters, assume GSM8K
    return numeric_count > letter_count

def run_prompts_with_vllm(
    dataset: Dict[str, List],
    model_name: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 128,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.8,
    max_num_seqs: int = 256,
    tokenizer_mode: Optional[str] = None,
    tensor_parallel_size: int = 1,
    extract_answer: bool = False,
) -> Dict[str, List]:
    """
    Runs prompts using vLLM and concatenates the results to the dataset.
    
    Supports multi-GPU inference through tensor parallelism.
    
    Args:
        dataset: Dictionary with 'prompts' and 'ground_truths' keys.
                 Each value should be a list of the same length.
        model_name: Name of the model to use for inference (e.g., "mistralai/Mistral-7B-Instruct-v0.3")
        temperature: Sampling temperature (default: 0.7)
        top_p: Top-p sampling parameter (default: 0.9)
        max_tokens: Maximum number of tokens to generate (default: 128)
        dtype: Data type for the model (default: "bfloat16")
        gpu_memory_utilization: GPU memory utilization (default: 0.8)
        max_num_seqs: Maximum number of sequences to process in parallel (default: 256)
        tokenizer_mode: Tokenizer mode (e.g., "mistral"). If None, uses default.
        tensor_parallel_size: Number of GPUs to use for tensor parallelism (default: 1).
                            Set to number of available GPUs for multi-GPU inference.
                            For example, use 2 for 2 GPUs, 4 for 4 GPUs, etc.
        extract_answer: (Deprecated) Always extracts answers. This parameter is kept for backward compatibility.
    
    Returns:
        Dictionary with original keys plus:
        - 'responses': Raw model outputs (list of strings)
        - 'extracted_answers': Extracted answer labels 
          (for MMLU: list of 'A'/'B'/'C'/'D'/'E'... or None)
          (for GSM8K: list of numerical strings or None)
    """
    # Validate input
    if type(dataset) != dict:
        if 'prompts' not in dataset.features or 'ground_truths' not in dataset.features:
            raise ValueError("Dataset must contain 'prompts' and 'ground_truths' keys")
        dataset = dataset.to_dict()
    
    prompts = dataset['prompts']
    ground_truths = dataset['ground_truths']
    
    if len(prompts) != len(ground_truths):
        raise ValueError(f"Length mismatch: prompts ({len(prompts)}) != ground_truths ({len(ground_truths)})")
    
    if len(prompts) == 0:
        print("Warning: Empty dataset provided")
        dataset['responses'] = []
        dataset['extracted_answers'] = []
        return dataset
    
    # Detect dataset type (GSM8K vs MMLU)
    is_gsm8k = is_gsm8k_dataset(ground_truths)
    dataset_type = "GSM8K" if is_gsm8k else "MMLU"
    print(f"Detected dataset type: {dataset_type}")
    if len(ground_truths) > 0:
        print(f"Sample ground truths: {ground_truths[:5]}")
    
    print(f"Loading vLLM model: {model_name}")
    if tensor_parallel_size > 1:
        print(f"Using {tensor_parallel_size} GPUs for tensor parallelism")
    print(f"Processing {len(prompts)} prompts...")
    
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    
    # Create LLM instance
    llm_kwargs = {
        'model': model_name,
        'dtype': dtype,
        'max_num_seqs': max_num_seqs,
        'gpu_memory_utilization': gpu_memory_utilization,
        'tensor_parallel_size': tensor_parallel_size,
    }
    
    if tokenizer_mode is not None:
        llm_kwargs['tokenizer_mode'] = tokenizer_mode
    
    llm = LLM(**llm_kwargs)
    
    # Run inference
    print("Running inference...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract responses and answer labels
    responses = []
    extracted_answers = []
    
    for output in outputs:
        response_text = output.outputs[0].text.strip()
        
        # Always keep raw output
        responses.append(response_text)
        
        # Extract answer based on dataset type
        extracted_label = None
        
        if is_gsm8k:
            # Extract numerical answer for GSM8K
            extracted_label = extract_gsm8k_answer(response_text)
        else:
            # Extract letter answer for MMLU/MMLU-Pro
            # Clean up thinking tokens for extraction
            cleaned_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE)
            cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove standalone thinking tags
            cleaned_text = re.sub(r'<think>\s*\n?', '', cleaned_text, flags=re.IGNORECASE)
            cleaned_text = re.sub(r'<think>\s*\n?', '', cleaned_text, flags=re.IGNORECASE)
            
            # Remove common thinking prefixes
            cleaned_text = re.sub(r'^(Okay, let|Okay, so|Let me|Let\'s see)[^\n]*\n?', '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
            cleaned_text = cleaned_text.strip()
            
            # Extract answer letter (A, B, C, D, ..., J) - try multiple patterns
            answer_match = None
            
            # Pattern 1: Letter after "Answer:" or similar (highest priority)
            answer_match = re.search(r'(?:Answer|answer|Answer:|The answer is|correct answer is|choice is)\s*:?\s*([A-J])', cleaned_text, re.IGNORECASE)
            
            # Pattern 2: Standalone letter (A-J for MMLU-Pro) - use word boundary or punctuation
            if not answer_match:
                # Match letter that's either at word boundary or followed by punctuation/whitespace/end
                answer_match = re.search(r'(?:^|\s|\(|\)|\.|,|;|:|\?|!)([A-J])(?:\s|$|\)|\.|,|;|:|\?|!)', cleaned_text.upper())
            
            # Pattern 3: Letter at start of cleaned text
            if not answer_match:
                answer_match = re.search(r'^\s*([A-J])(?:\s|$|\)|\.|,|;|:|\?|!)', cleaned_text.upper(), re.MULTILINE)
            
            # Pattern 4: Letter after common phrases
            if not answer_match:
                answer_match = re.search(r'(?:is|should be|would be)\s+([A-J])(?:\s|$|\)|\.|,|;|:|\?|!)', cleaned_text, re.IGNORECASE)
            
            if answer_match:
                extracted_label = answer_match.group(1)
            else:
                # Last resort: try to find ANY letter A-J in the original response (without word boundaries)
                any_letter = re.search(r'([A-J])', response_text.upper())
                if any_letter:
                    extracted_label = any_letter.group(1)
                # If still no match, extracted_label remains None
        
        extracted_answers.append(extracted_label)
    
    print(f"Generated {len(responses)} responses")
    print(f"Extracted {sum(1 for a in extracted_answers if a is not None)} answers out of {len(extracted_answers)} responses")
    
    # Add both raw responses and extracted answers to dataset
    dataset['responses'] = responses
    dataset['extracted_answers'] = extracted_answers
    
    return dataset

def add_correct_labels(input_file: str, output_file: str) -> Dict[str, List]:
    """
    Adds correct labels to the dataset.
    
    Args:
        input_file: Path to the input file.
        output_file: Path to the output file.
    """
    with open(input_file, 'rb') as f:
        dataset = pickle.load(f)
    ground_truths = dataset['ground_truths']
    extracted_answers = dataset['extracted_answers']
    correct_labels = []
    for i in range(len(extracted_answers)):
        if extracted_answers[i] == ground_truths[i]:
            correct_labels.append(1)
        else:
            correct_labels.append(0)
    dataset['correct_labels'] = correct_labels
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)

def load_run_label_dataset():
    model_name = "Qwen/Qwen3-1.7B"
    tensor_parallel_size = 2
    if tensor_parallel_size > 1:
        print(f"Using {tensor_parallel_size} GPUs for tensor parallelism")
        # load the normal mmlu test dataset
    # dataset = load_mmlu_split("test", output_file=MMLU_TEST_PROMPTS_FILE, model_type="qwen")
    # results = run_prompts_with_vllm(dataset, model_name=model_name, temperature=0, top_p=0.9, max_tokens=512, gpu_memory_utilization=0.8, max_num_seqs=256, tensor_parallel_size=tensor_parallel_size)
    # with open(MMLU_TEST_QWEN17B_RESULTS_FILE, 'wb') as f:
    #     pickle.dump(results, f)
    # add_correct_labels(MMLU_TEST_QWEN17B_RESULTS_FILE, MMLU_TEST_QWEN17B_CORRECT_RESULTS_FILE)

    # # load the normal mmlu validation dataset
    # dataset = load_mmlu_split("validation", output_file=MMLU_VALIDATION_PROMPTS_FILE, model_type="qwen")
    # results = run_prompts_with_vllm(dataset, model_name=model_name, temperature=0, top_p=0.9, max_tokens=512, gpu_memory_utilization=0.8, max_num_seqs=256, tensor_parallel_size=tensor_parallel_size)
    # with open(MMLU_VALIDATION_QWEN17B_RESULTS_FILE, 'wb') as f:
    #     pickle.dump(results, f)
    # add_correct_labels(MMLU_VALIDATION_QWEN17B_RESULTS_FILE, MMLU_VALIDATION_QWEN17B_CORRECT_RESULTS_FILE)

    # # load the normal mmlu dev dataset
    # dataset = load_mmlu_split("dev", output_file=MMLU_DEV_PROMPTS_FILE, model_type="qwen")
    # results = run_prompts_with_vllm(dataset, model_name=model_name, temperature=0, top_p=0.9, max_tokens=512, gpu_memory_utilization=0.8, max_num_seqs=256, tensor_parallel_size=tensor_parallel_size)
    # with open(MMLU_DEV_QWEN17B_RESULTS_FILE, 'wb') as f:
    #     pickle.dump(results, f)
    # add_correct_labels(MMLU_DEV_QWEN17B_RESULTS_FILE, MMLU_DEV_QWEN17B_CORRECT_RESULTS_FILE)

    # # load the normal mmlu auxiliary dataset
    # dataset = load_mmlu_split("auxiliary_train", output_file=MMLU_AUXILIARY_PROMPTS_FILE, model_type="qwen")
    # results = run_prompts_with_vllm(dataset, model_name=model_name, temperature=0, top_p=0.9, max_tokens=512, gpu_memory_utilization=0.8, max_num_seqs=256, tensor_parallel_size=tensor_parallel_size)
    # with open(MMLU_AUXILIARY_QWEN17B_RESULTS_FILE, 'wb') as f:
    #     pickle.dump(results, f)
    # add_correct_labels(MMLU_AUXILIARY_QWEN17B_RESULTS_FILE, MMLU_AUXILIARY_QWEN17B_CORRECT_RESULTS_FILE)

    # # load the normal mmlu pro test dataset
    dataset = load_mmlu_pro_dataset("test", output_file=MMLU_PRO_TEST_PROMPTS_FILE, model_type="qwen")
    results = run_prompts_with_vllm(dataset, model_name=model_name, temperature=0, top_p=0.9, max_tokens=512, gpu_memory_utilization=0.8, max_num_seqs=256, tensor_parallel_size=tensor_parallel_size)
    with open(MMLU_PRO_TEST_QWEN17B_RESULTS_FILE, 'wb') as f:
        pickle.dump(results, f)
    add_correct_labels(MMLU_PRO_TEST_QWEN17B_RESULTS_FILE, MMLU_PRO_TEST_QWEN17B_CORRECT_RESULTS_FILE)

    # # load the normal mmlu pro validation dataset
    # dataset = load_mmlu_pro_dataset("validation", output_file=MMLU_PRO_VALIDATION_PROMPTS_FILE, model_type="qwen")
    # results = run_prompts_with_vllm(dataset, model_name=model_name, temperature=0, top_p=0.9, max_tokens=512, gpu_memory_utilization=0.8, max_num_seqs=256, tensor_parallel_size=tensor_parallel_size)
    # with open(MMLU_PRO_VALIDATION_QWEN17B_RESULTS_FILE, 'wb') as f:
    #     pickle.dump(results, f)
    # add_correct_labels(MMLU_PRO_VALIDATION_QWEN17B_RESULTS_FILE, MMLU_PRO_VALIDATION_QWEN17B_CORRECT_RESULTS_FILE)

if __name__ == "__main__":
    load_run_label_dataset()