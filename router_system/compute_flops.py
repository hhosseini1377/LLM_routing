
from transformers import AutoConfig, AutoTokenizer
import torch
import numpy as np
import pickle
import json
import os
from datasets import Dataset as HFDataset
import itertools
from tqdm import tqdm

def analyze_output_token_lengths(output_path: str):
    with open(output_path, 'rb') as f:
        data = pickle.load(f)
    outputs = [item['answers'][0] for item in data]
    # Compute the number of words in the outputs
    word_count = [len(output.split()) for output in outputs]
    return np.mean(word_count)

def analyze_prompts_token_lengths(
    file_path: str,
    model_name: str = None,
    tokenizer=None,
    prompt_key: str = None,
    max_length: int = None,
    pad_token_id: int = None
):
    """
    Read a file, extract prompts, tokenize them, and find the size of non-padding tokens.
    
    Supports multiple file formats:
    - Pickle files (.pkl): Can contain dicts, lists, or HuggingFace Datasets
    - JSON files (.json): Can contain lists of dicts or single dict
    - Text files (.txt): One prompt per line
    
    Args:
        file_path: Path to the file containing prompts
        model_name: HuggingFace model name (used to load tokenizer if tokenizer not provided)
        tokenizer: Pre-loaded tokenizer (if None, will load from model_name)
        prompt_key: Key to extract prompts from dict (e.g., 'prompt', 'text', 'question').
                   If None, will try common keys automatically.
        max_length: Maximum sequence length for tokenization (default: tokenizer's max_length)
        pad_token_id: Padding token ID (auto-detected from tokenizer if None)
    
    Returns:
        dict: Dictionary containing:
            - 'prompts': List of prompt texts
            - 'true_lengths': List of true sequence lengths (non-padding tokens)
            - 'padded_lengths': List of padded sequence lengths (if padding was used)
            - 'statistics': Dictionary with statistics (mean, min, max, etc.)
            - 'tokenizer_info': Information about the tokenizer used
    """
    # Load tokenizer
    if tokenizer is None:
        if model_name is None:
            raise ValueError("Either model_name or tokenizer must be provided")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Get pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
    
    # Set max_length
    if max_length is None:
        max_length = getattr(tokenizer, 'model_max_length', 512)
    
    # Read file based on extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pkl':
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    elif file_ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif file_ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported: .pkl, .json, .txt")
    
    # Extract prompts based on data type
    prompts = []
    
    if isinstance(data, HFDataset):
        # HuggingFace Dataset
        column_names = data.column_names
        if prompt_key:
            if prompt_key not in column_names:
                raise ValueError(f"Key '{prompt_key}' not found. Available columns: {column_names}")
            prompts = data[prompt_key]
        else:
            # Try common keys
            for key in ['prompt', 'text', 'question', 'input', 'context']:
                if key in column_names:
                    prompts = data[key]
                    break
            if not prompts:
                raise ValueError(f"Could not find prompt field. Available columns: {column_names}")
    
    elif isinstance(data, dict):
        # Dictionary
        if prompt_key:
            if prompt_key not in data:
                raise ValueError(f"Key '{prompt_key}' not found. Available keys: {list(data.keys())}")
            prompts = data[prompt_key]
        else:
            # Try common keys
            for key in ['prompt', 'text', 'question', 'input', 'context', 'prompts']:
                if key in data:
                    prompts = data[key]
                    break
            if not prompts:
                raise ValueError(f"Could not find prompt field. Available keys: {list(data.keys())}")
    
    elif isinstance(data, list):
        # List of strings or list of dicts
        if len(data) > 0 and isinstance(data[0], str):
            prompts = data
        elif len(data) > 0 and isinstance(data[0], dict):
            if prompt_key:
                prompts = [item.get(prompt_key, '') for item in data]
            else:
                # Try common keys
                for key in ['prompt', 'text', 'question', 'input', 'context']:
                    if key in data[0]:
                        prompts = [item.get(key, '') for item in data]
                        break
                if not prompts:
                    raise ValueError(f"Could not find prompt field in dict items. Available keys: {list(data[0].keys())}")
        else:
            prompts = data
    
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")
    
    # Ensure prompts is a list
    if not isinstance(prompts, list):
        prompts = [prompts]
    
    # Tokenize all prompts
    print(f"Tokenizing {len(prompts)} prompts...")
    encoded = tokenizer(
        prompts,
        padding='max_length' if max_length else True,
        truncation=True if max_length else False,
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids']
    attention_mask = encoded.get('attention_mask', None)
    
    # Get true sequence lengths
    true_lengths = get_true_sequence_length(
        input_ids.numpy(),
        attention_mask=attention_mask.numpy() if attention_mask is not None else None,
        pad_token_id=pad_token_id
    )
    
    # Convert to list if single value
    if isinstance(true_lengths, (int, np.integer)):
        true_lengths = [int(true_lengths)]
    else:
        true_lengths = [int(tl) for tl in true_lengths]
    
    # Get padded lengths
    padded_lengths = [input_ids.shape[1]] * len(prompts) if len(input_ids.shape) > 1 else [len(input_ids)]
    
    # Calculate statistics
    true_lengths_array = np.array(true_lengths)
    statistics = {
        'total_prompts': len(prompts),
        'mean_length': float(np.mean(true_lengths_array)),
        'median_length': float(np.median(true_lengths_array)),
        'min_length': int(np.min(true_lengths_array)),
        'max_length': int(np.max(true_lengths_array)),
        'std_length': float(np.std(true_lengths_array)),
        'padded_length': padded_lengths[0] if padded_lengths else None,
        'total_tokens': int(np.sum(true_lengths_array)),
        'total_padding_tokens': int(np.sum(padded_lengths) - np.sum(true_lengths_array)) if padded_lengths else 0
    }
    
    return {
        'prompts': prompts,
        'true_lengths': true_lengths,
        'padded_lengths': padded_lengths,
        'statistics': statistics,
        'tokenizer_info': {
            'model_name': model_name or 'custom',
            'pad_token_id': pad_token_id,
            'max_length': max_length,
            'vocab_size': len(tokenizer)
        }
    }


def print_prompt_analysis(results: dict):
    """
    Pretty print prompt analysis results.
    
    Args:
        results: Results dictionary from analyze_prompts_token_lengths()
    """
    stats = results['statistics']
    tokenizer_info = results['tokenizer_info']
    
    print("=" * 80)
    print("PROMPT TOKEN LENGTH ANALYSIS")
    print("=" * 80)
    print(f"\nTokenizer Info:")
    print(f"  Model: {tokenizer_info['model_name']}")
    print(f"  Vocab Size: {tokenizer_info['vocab_size']:,}")
    print(f"  Max Length: {tokenizer_info['max_length']:,}")
    print(f"  Pad Token ID: {tokenizer_info['pad_token_id']}")
    
    print(f"\nStatistics:")
    print(f"  Total Prompts: {stats['total_prompts']:,}")
    print(f"  Mean Length: {stats['mean_length']:.2f} tokens")
    print(f"  Median Length: {stats['median_length']:.2f} tokens")
    print(f"  Min Length: {stats['min_length']:,} tokens")
    print(f"  Max Length: {stats['max_length']:,} tokens")
    print(f"  Std Dev: {stats['std_length']:.2f} tokens")
    
    if stats['padded_length']:
        print(f"\nPadding Info:")
        print(f"  Padded Length: {stats['padded_length']:,} tokens")
        print(f"  Total True Tokens: {stats['total_tokens']:,}")
        print(f"  Total Padding Tokens: {stats['total_padding_tokens']:,}")
        padding_percentage = (stats['total_padding_tokens'] / (stats['total_prompts'] * stats['padded_length'])) * 100
        print(f"  Padding Percentage: {padding_percentage:.2f}%")
    
    print("=" * 80)


def get_true_sequence_length(input_ids, attention_mask=None, pad_token_id=None):
    """
    Get the true sequence length (excluding padding) from tokenized inputs.
    
    This is useful for accurate FLOPs estimation, as padding tokens don't
    contribute to meaningful computation in attention operations.
    
    Args:
        input_ids: Token IDs tensor/array of shape [batch_size, seq_len] or [seq_len]
        attention_mask: Attention mask tensor/array (1 for real tokens, 0 for padding).
                      If None, will infer from pad_token_id.
        pad_token_id: Padding token ID. If None and attention_mask is None,
                      assumes no padding.
    
    Returns:
        int or list: True sequence length(s). If batch_size > 1, returns list.
    """
    input_ids = np.asarray(input_ids)
    
    # Handle single sequence
    if len(input_ids.shape) == 1:
        if attention_mask is not None:
            attention_mask = np.asarray(attention_mask)
            return int(attention_mask.sum())
        elif pad_token_id is not None:
            return int(np.sum(input_ids != pad_token_id))
        else:
            return len(input_ids)
    
    # Handle batch
    batch_size = input_ids.shape[0]
    true_lengths = []
    
    if attention_mask is not None:
        attention_mask = np.asarray(attention_mask)
        for i in range(batch_size):
            true_lengths.append(int(attention_mask[i].sum()))
    elif pad_token_id is not None:
        for i in range(batch_size):
            true_lengths.append(int(np.sum(input_ids[i] != pad_token_id)))
    else:
        for i in range(batch_size):
            true_lengths.append(len(input_ids[i]))
    
    return true_lengths if batch_size > 1 else true_lengths[0]

def estimate_llm_flops(model_size_b: float,
                       input_len: int,
                       output_len: int):
    """
    Estimate FLOPs for prefill and autoregressive decoding.

    Args:
        model_size_b (float): Model size in billions of parameters (e.g. 7 for 7B).
        input_len (int): Average prompt length.
        output_len (int): Average generated output tokens.

    Returns:
        dict: FLOPs for prefill, decoding, and total.
    """

    # Convert billions of parameters → raw param count
    P = model_size_b * 1e9

    L = input_len
    T = output_len

    # Prefill: 2 * P * L
    prefill_flops = 2 * P * L

    # Decode: P * T(T+1)
    decode_flops = P * T * (T + 1)

    total_flops = prefill_flops + decode_flops

    return {
        "prefill_flops": prefill_flops,
        "decode_flops": decode_flops,
        "total_flops": total_flops
    }



def compute_average_flops_hierarchical_routing(
    m1_prefill_flops: float,
    m1_decode_flops: float,
    m2_prefill_flops: float,
    m2_decode_flops: float,
    n1: int,
    n2: int,
    n3: int
):
    """
    Compute average FLOPs per prompt for a routing scenario with two models.
    
    Routing scenarios:
    - n1 prompts: prefilling + decoding on M1
    - n2 prompts: prefilling + decoding on M2
    - n3 prompts: prefilling on M1, then prefilling + decoding on M2
    
    Args:
        m1_prefill_flops: Total prefilling FLOPs for model M1 (per prompt)
        m1_decode_flops: Total decoding FLOPs for model M1 (per prompt)
        m2_prefill_flops: Total prefilling FLOPs for model M2 (per prompt)
        m2_decode_flops: Total decoding FLOPs for model M2 (per prompt)
        n1: Number of prompts that use M1 for both prefilling and decoding
        n2: Number of prompts that use M2 for both prefilling and decoding
        n3: Number of prompts that use M1 for prefilling, then M2 for prefilling + decoding
    
    Returns:
        dict: Dictionary containing:
            - 'total_flops': Total FLOPs across all prompts
            - 'average_flops_per_prompt': Average FLOPs per prompt
            - 'm1_total_flops': Total FLOPs on M1
            - 'm2_total_flops': Total FLOPs on M2
            - 'n1_flops': Total FLOPs for n1 prompts
            - 'n2_flops': Total FLOPs for n2 prompts
            - 'n3_flops': Total FLOPs for n3 prompts
            - 'total_prompts': Total number of prompts (n1 + n2 + n3)
            - 'breakdown': Detailed breakdown by scenario
    """
    # Scenario 1: n1 prompts use M1 for prefilling + decoding
    n1_total_flops = n1 * (m1_prefill_flops + m1_decode_flops)
    
    # Scenario 2: n2 prompts use M2 for prefilling + decoding
    n2_total_flops = n2 * (m2_prefill_flops + m2_decode_flops)
    
    # Scenario 3: n3 prompts use M1 for prefilling, then M2 for prefilling + decoding
    n3_total_flops = n3 * (m1_prefill_flops + m2_prefill_flops + m2_decode_flops)
    
    # Total FLOPs
    total_flops = n1_total_flops + n2_total_flops + n3_total_flops
    
    # Total number of prompts
    total_prompts = n1 + n2 + n3
    
    # Average FLOPs per prompt
    average_flops_per_prompt = total_flops / total_prompts if total_prompts > 0 else 0
    
    # M1 and M2 total FLOPs
    m1_total_flops = n1 * (m1_prefill_flops + m1_decode_flops) + n3 * m1_prefill_flops
    m2_total_flops = n2 * (m2_prefill_flops + m2_decode_flops) + n3 * (m2_prefill_flops + m2_decode_flops)
    
    return {
        'total_flops': total_flops,
        'average_flops_per_prompt': average_flops_per_prompt,
        'm1_total_flops': m1_total_flops,
        'm2_total_flops': m2_total_flops,
        'n1_flops': n1_total_flops,
        'n2_flops': n2_total_flops,
        'n3_flops': n3_total_flops,
        'total_prompts': total_prompts,
        'breakdown': {
            'n1': {
                'count': n1,
                'flops_per_prompt': m1_prefill_flops + m1_decode_flops,
                'total_flops': n1_total_flops,
                'description': 'M1 prefilling + M1 decoding'
            },
            'n2': {
                'count': n2,
                'flops_per_prompt': m2_prefill_flops + m2_decode_flops,
                'total_flops': n2_total_flops,
                'description': 'M2 prefilling + M2 decoding'
            },
            'n3': {
                'count': n3,
                'flops_per_prompt': m1_prefill_flops + m2_prefill_flops + m2_decode_flops,
                'total_flops': n3_total_flops,
                'description': 'M1 prefilling + M2 prefilling + M2 decoding'
            }
        }
    }



def compute_n1_n2_n3_count_hierarchical_routing(bert_probabilities, cpx_probabilities, small_threshold, large_threshold):
    """
    Compute the number of prompts for each routing scenario.
    
    Routing scenarios:
    - n1 prompts: prefilling + decoding on M1
    - n2 prompts: prefilling + decoding on M2
    - n3 prompts: prefilling on M1, then prefilling + decoding on M2
    """
    n1_count = 0
    n2_count = 0
    n3_count = 0
    for i in range(len(bert_probabilities)):
        if bert_probabilities[i] < small_threshold:
            n2_count += 1
        elif bert_probabilities[i] > small_threshold and cpx_probabilities[i] > large_threshold:
            n1_count += 1
        else:
            n3_count += 1
    return n1_count, n2_count, n3_count

def compute_flops_for_different_thresholds_hierarchical_routing(input_path: str, cpx_prob_dir, bert_prob_dir, small_model_size, large_model_size):

    # Define the thresholds for the bert and cpx models
    small_thresholds = list(np.arange(0.01, 0.99, 0.01))
    large_thresholds = list(np.arange(0.01, 0.99, 0.01))

    # Load the probabilities for the bert and cpx models
    with open(bert_prob_dir, 'rb') as f:
        bert_validation_data = pickle.load(f)
    bert_probabilities = bert_validation_data['probabilities']
    labels = bert_validation_data['labels']

    # Load the probabilities for the cpx model
    with open(cpx_prob_dir, 'rb') as f:
        cpx_validation_data = pickle.load(f)

    cpx_probabilities = cpx_validation_data['probabilities']

    # Compute all possible combinations of thresholds
    all_combinations = list(itertools.product(small_thresholds, large_thresholds))

    prompts_results = analyze_prompts_token_lengths(
        file_path=input_path,
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        prompt_key='prompt',  # Optional: specify key if known
        max_length=512  # Optional: set max length
    )
    
    # Compute the average prompts and output lengths
    average_prompts_length = np.mean(prompts_results['true_lengths'])
    average_output_length = analyze_output_token_lengths(input_path) * 1.33

    thresholds_results = {}
    # Compute the FLOPs for each routing scenario
    for combination in tqdm(all_combinations, desc="Computing FLOPs for different thresholds"):
        small_threshold, large_threshold = combination[0], combination[1]
        n1_count, n2_count, n3_count = compute_n1_n2_n3_count_hierarchical_routing(bert_probabilities, cpx_probabilities, small_threshold, large_threshold)
        m1_flops = estimate_llm_flops(small_model_size, average_prompts_length, average_output_length)['total_flops']
        m2_flops = estimate_llm_flops(large_model_size, average_prompts_length, average_output_length)['total_flops']
        n1_flops = n1_count * m1_flops
        n2_flops = n2_count * m2_flops
        n3_flops = n3_count * (m1_flops + m2_flops)
        total_flops = n1_flops + n2_flops + n3_flops
        average_flops_per_prompt = total_flops / (n1_count + n2_count + n3_count)
        thresholds_results[combination] = {
            'n1_count': n1_count,
            'n2_count': n2_count,
            'n3_count': n3_count,
            'm1_flops': m1_flops,
            'm2_flops': m2_flops,
            'n1_flops': n1_flops,
            'n2_flops': n2_flops,
            'n3_flops': n3_flops,
            'total_flops': total_flops,
            'average_flops_per_prompt': average_flops_per_prompt
        }
    return thresholds_results

def compute_reliability_for_threshold_hierarchical_routing(cpx_probabilities, bert_probabilities, labels, small_threshold, large_threshold):
    total_false_positives = 0
    for i in range(len(cpx_probabilities)):
        if bert_probabilities[i] > small_threshold and cpx_probabilities[i] > large_threshold and labels[i] == 0:
            total_false_positives += 1
    return 1 - (total_false_positives / len(cpx_probabilities))

def compute_reliability_for_different_thresholds_hierarchical_routing(cpx_prob_dir, bert_prob_dir, small_model_size, large_model_size):
    # Define the thresholds for the bert and cpx models
    small_thresholds = list(np.arange(0.01, 0.99, 0.01))
    large_thresholds = list(np.arange(0.01, 0.99, 0.01))

    # Load the probabilities for the bert and cpx models
    with open(bert_prob_dir, 'rb') as f:
        bert_validation_data = pickle.load(f)
    bert_probabilities = bert_validation_data['probabilities']
    labels = bert_validation_data['labels']

    # Load the probabilities for the cpx model
    with open(cpx_prob_dir, 'rb') as f:
        cpx_validation_data = pickle.load(f)

    cpx_probabilities = cpx_validation_data['probabilities']

    # Compute all possible combinations of thresholds
    all_combinations = list(itertools.product(small_thresholds, large_thresholds))

    reliability_results = {}
    for combination in tqdm(all_combinations, desc="Computing reliability for different thresholds"):
        small_threshold, large_threshold = combination[0], combination[1]
        reliability = compute_reliability_for_threshold_hierarchical_routing(cpx_probabilities, bert_probabilities, labels, small_threshold, large_threshold)
        reliability_results[combination] = reliability

    return reliability_results

def reliability_cost_tradeoff_for_different_thresholds_hierarchical_routing(input_path: str, cpx_prob_dir, bert_prob_dir, small_model_size, large_model_size):
    reliability_results = compute_reliability_for_different_thresholds_hierarchical_routing(cpx_prob_dir, bert_prob_dir, small_model_size, large_model_size)
    flops_results = compute_flops_for_different_thresholds_hierarchical_routing(input_path, cpx_prob_dir, bert_prob_dir, small_model_size, large_model_size)
    reliability_cost_tradeoff = {}
    for combination in reliability_results:
        reliability = reliability_results[combination]
        flops = flops_results[combination]
        reliability_cost_tradeoff[combination] = {
            'reliability': reliability,
            'flops': flops,
            'reliability_cost_tradeoff': reliability / flops
        }
    return reliability_cost_tradeoff