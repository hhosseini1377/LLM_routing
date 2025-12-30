
from transformers import AutoConfig, AutoTokenizer
import torch
import numpy as np
import pickle
import json
import os
from datasets import Dataset as HFDataset
import itertools
from tqdm import tqdm
from typing import Optional
import pandas
import numpy
model_sizes = {
    'mistral': {
        'num_layers': 32,
        'hidden_size': 4096,
        'ffn_dim': 14336,
    },
    "qwen3_8b": {
        "num_layers": 36,
        "hidden_size": 4096,
        "ffn_dim": 11008,
    },
    "qwen3_32b": {
        "num_layers": 64,
        "hidden_size": 5120,
        "ffn_dim": 13824,
    },
    'deberta': {
        'num_layers': 24,
        'hidden_size': 1024,
        'intermediate_size': 4096
    }
}

model_latencies = {
    'qwen3_8b': {
        'TTFT': 321.77,
        'TPOT': 17.97
    },
    'qwen3_32b': {
        'TTFT': 859.9,
        'TPOT': 85.64
    }
}

output_length = 400
input_length = 400
TTFT = 150
TPOT = 7.5

def analyze_output_token_lengths(output_path: str):
    with open(output_path, 'rb') as f:
        data = pickle.load(f)
    if 'responses' in data:
        response_key = 'responses'
        outputs = [item[response_key] for item in data]
    else:
        response_key = 'answer'
        outputs = [item[response_key][0] for item in data]
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
    
    if isinstance(data, pandas.DataFrame):
        if 'prompts' in data.keys():
            prompt_key = 'prompts'
        else:
            prompt_key = 'prompt'
    elif isinstance(data, dict):
        if 'prompts' in data.keys():
            prompt_key = 'prompts'
        else:
            prompt_key = 'prompt'
    # Extract prompts based on data type
    prompts = []
    if isinstance(data, pandas.DataFrame):
        data = data.to_dict(orient='list')
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

import math

def estimate_llm_flops(num_layers: int,
                       hidden_size: int,
                       ffn_dim: int,
                       seq_len_prefill: int,
                       seq_len_prompt: Optional[int] = None,
                       output_len: int = 0,
                       num_heads: Optional[int] = None,
                       return_summary: bool = False):
    """
    Estimate FLOPs for a decoder-only transformer (LLM) split into:
      - prefill (processing the entire prompt / context, O(L^2))
      - decode (autoregressive generation with KV cache, ~O(T^2))

    Uses standard engineering approximations.
    """

    L = seq_len_prefill
    T = output_len
    N = num_layers
    H = hidden_size
    F = ffn_dim

    # -----------------------------
    # Prefill FLOPs (per layer)
    # -----------------------------
    flops_per_layer_prefill = (
        4 * L * H * H +        # Q,K,V,O projections
        2 * L * L * H +        # attention weights & softmax-weighting
        8 * L * H * F          # FFN
    )
    prefill_flops = N * flops_per_layer_prefill

    # -----------------------------
    # Decode FLOPs (with KV cache)
    # Corrected quadratic term: 0.5 * H * T * (T - 1)
    # -----------------------------
    if T > 0:
        decode_flops = N * (
            T * (4 * H * H + 2 * H * L + 8 * H * F) +
            0.5 * H * T * (T - 1)     # FIXED: attention over previous tokens
        )
    else:
        decode_flops = 0

    total_flops = prefill_flops + decode_flops

    # -----------------------------
    # Result dictionary
    # -----------------------------
    result = {
        "prefill_flops": int(prefill_flops),
        "decode_flops": int(decode_flops),
        "total_flops": int(total_flops),
        "num_layers": N,
        "hidden_size": H,
        "ffn_dim": F,
        "seq_len_prefill": L,
        "output_len": T
    }

    # -----------------------------
    # Human-readable summary
    # -----------------------------
    if return_summary:
        def _fmt(n):
            return {
                "flops": int(n),
                "gflops": float(n) / 1e9,
                "tflops": float(n) / 1e12
            }

        result["prefill_readable"] = _fmt(prefill_flops)
        result["decode_readable"] = _fmt(decode_flops)
        result["total_readable"] = _fmt(total_flops)

    return result




def estimate_bert_flops(hidden_size=768,
                        intermediate_size=3072,
                        num_layers=12,
                        seq_len=128):
    H = hidden_size
    F = intermediate_size
    L = seq_len
    N = num_layers
    
    flops_per_layer = (
        4 * L * H * H +      # Q,K,V,O projections
        2 * L * L * H +      # attention scores + weighting
        8 * L * H * F        # FFN
    )
    
    total_flops = N * flops_per_layer
    return total_flops




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
    for i in range(min(len(bert_probabilities), len(cpx_probabilities))):
        if bert_probabilities[i] < small_threshold:
            n2_count += 1
        elif bert_probabilities[i] > small_threshold and cpx_probabilities[i] > large_threshold:
            n1_count += 1
        else:
            n3_count += 1
    return n1_count, n2_count, n3_count

def analyze_cost_for_different_thresholds_hierarchical_routing(input_path: str, cpx_prob_dir, bert_prob_dir, small_model, large_model):
    # Define the thresholds for the bert and cpx models
    small_thresholds = list(np.arange(0.00, 1.01, 0.01))
    large_thresholds = list(np.arange(0.00, 1.01, 0.01))

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

    if 'prompts' in bert_validation_data:
        prompt_key = 'prompts'
    else:
        prompt_key = 'prompt'
    
    # Compute the average prompts and output lengths
    average_prompts_length = input_length
    average_output_length = output_length
    small_model_size = model_sizes[small_model]
    large_model_size = model_sizes[large_model]
    deberta_model_size = model_sizes['deberta']
    m1_flops = estimate_llm_flops(seq_len_prefill=average_prompts_length, output_len=average_output_length, **small_model_size)
    m2_flops = estimate_llm_flops(seq_len_prefill=average_prompts_length, output_len=average_output_length, **large_model_size)
    deberta_flops = estimate_bert_flops(seq_len=average_prompts_length, **deberta_model_size)
    m1_decode_flops = m1_flops['decode_flops']
    m2_decode_flops = m2_flops['decode_flops']
    m1_prefill_flops = m1_flops['prefill_flops']
    m2_prefill_flops = m2_flops['prefill_flops']
    m1_TTFT = model_latencies[small_model]['TTFT']
    m2_TTFT = model_latencies[large_model]['TTFT']
    m1_TPOT = model_latencies[small_model]['TPOT']
    m2_TPOT = model_latencies[large_model]['TPOT']
    m1_decode_latency = m1_TPOT*output_length
    m2_decode_latency = m2_TPOT*output_length
    print(f'm1_decode_flops: {m1_decode_flops/1e12}, m2_decode_flops: {m2_decode_flops/1e12}, m1_prefill_flops: {m1_prefill_flops/1e12}, m2_prefill_flops: {m2_prefill_flops/1e12}, deberta_flops: {deberta_flops/1e12}')
    print(f'average input length: {average_prompts_length}, average output length: {average_output_length}')
    thresholds_results = {}
    # Compute the FLOPs for each routing scenario
    for combination in tqdm(all_combinations, desc="Computing FLOPs for different thresholds"):
        small_threshold, large_threshold = combination[0], combination[1]
        n1_count, n2_count, n3_count = compute_n1_n2_n3_count_hierarchical_routing(bert_probabilities, cpx_probabilities, small_threshold, large_threshold)
        n1_flops = n1_count * (m1_prefill_flops + m1_decode_flops)
        n2_flops = n2_count * (m2_prefill_flops + m2_decode_flops)
        n3_flops = n3_count * (m1_prefill_flops + m2_prefill_flops + m2_decode_flops)
        total_flops = n1_flops + n2_flops + n3_flops + deberta_flops*(n1_count + n2_count + n3_count)
        average_flops_per_prompt = total_flops / (n1_count + n2_count + n3_count)
        total_latency = n1_count * (m1_TTFT + m1_decode_latency) + n2_count * (m2_TTFT + m2_decode_latency) + n3_count * (m1_TTFT + m2_TTFT + m2_decode_latency)
        average_latency_per_prompt = total_latency / (n1_count + n2_count + n3_count)
        thresholds_results[combination] = {
            'n1_count': n1_count,
            'n2_count': n2_count,
            'n3_count': n3_count,
            'm1_flops': m1_flops,
            'm2_flops': m2_flops,
            'n1_flops': n1_flops,
            'n2_flops': n2_flops,
            'n3_flops': n3_flops,
            'sent_to_large_model': n2_count + n3_count,
            'total_count': n1_count + n2_count + n3_count,
            'total_flops': total_flops,
            'average_flops_per_prompt': average_flops_per_prompt,
            'average_latency_per_prompt': average_latency_per_prompt
        }
    return thresholds_results

def compute_reliability_for_threshold_hierarchical_routing(cpx_probabilities, bert_probabilities, labels, small_threshold, large_threshold):
    total_false_positives = 0
    for i in range(len(cpx_probabilities)):
        if bert_probabilities[i] > small_threshold and cpx_probabilities[i] > large_threshold and labels[i] == 0:
            total_false_positives += 1
    return 1 - (total_false_positives / len(cpx_probabilities))

def compute_reliability_for_different_thresholds_hierarchical_routing(cpx_prob_dir, bert_prob_dir):
    # Define the thresholds for the bert and cpx models
    small_thresholds = list(np.arange(0.00, 1.01, 0.01))
    large_thresholds = list(np.arange(0.00, 1.01, 0.01))

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

def reliability_cost_tradeoff_for_different_thresholds_hierarchical_routing(input_path: str, cpx_prob_dir, bert_prob_dir, small_model, large_model):
    reliability_results = compute_reliability_for_different_thresholds_hierarchical_routing(cpx_prob_dir, bert_prob_dir)
    flops_results = analyze_cost_for_different_thresholds_hierarchical_routing(input_path, cpx_prob_dir, bert_prob_dir, small_model, large_model)
    reliability_cost_tradeoff = {}
    for combination in reliability_results:
        reliability = reliability_results[combination]
        flops = flops_results[combination]
        reliability_cost_tradeoff[combination] = {
            'reliability': reliability,
            'flops': flops['average_flops_per_prompt'],
            'sent_to_large_model': flops['sent_to_large_model'],
            'total_count': flops['total_count'],
            'average_latency_per_prompt': flops['average_latency_per_prompt'],
        }
    return reliability_cost_tradeoff

def analyze_cost_for_different_thresholds_bert_routing(input_path: str, bert_prob_dir, small_model, large_model):
    # Define the thresholds for the bert and cpx models
    small_thresholds = list(np.arange(0.00, 1.01, 0.01))

    # Load the probabilities for the bert and cpx models
    with open(bert_prob_dir, 'rb') as f:
        bert_validation_data = pickle.load(f)
    bert_probabilities = bert_validation_data['probabilities']
    labels = bert_validation_data['labels']

    if 'prompts' in bert_validation_data:
        prompt_key = 'prompts'
    else:
        prompt_key = 'prompt'
    
    # Compute the average prompts and output lengths
    average_prompts_length = input_length

    thresholds_results = {}
    small_model_size = model_sizes[small_model]
    large_model_size = model_sizes[large_model]
    deberta_model_size = model_sizes['deberta']
    m1_TTFT = model_latencies[small_model]['TTFT']
    m2_TTFT = model_latencies[large_model]['TTFT']
    m1_TPOT = model_latencies[small_model]['TPOT']
    m2_TPOT = model_latencies[large_model]['TPOT']
    m1_decode_latency = m1_TPOT*output_length
    m2_decode_latency = m2_TPOT*output_length
    small_flops = estimate_llm_flops(seq_len_prefill=average_prompts_length, output_len=output_length, **small_model_size)['total_flops']
    large_flops = estimate_llm_flops(seq_len_prefill=average_prompts_length, output_len=output_length, **large_model_size)['total_flops']
    deberta_flops = estimate_bert_flops(seq_len=average_prompts_length, **deberta_model_size)
    # Compute the FLOPs for each routing scenario
    for threshold in tqdm(small_thresholds, desc="Computing FLOPs for different thresholds"):
        average_flops = deberta_flops*len(bert_probabilities)
        for i in range(len(bert_probabilities)):
            if bert_probabilities[i] > threshold:
                flops = small_flops 
            else:
                flops = large_flops
            average_flops += flops
        average_flops /= len(bert_probabilities)
        total_latency = len(bert_probabilities[bert_probabilities > threshold]) * (m1_TTFT + m1_decode_latency) + len(bert_probabilities[bert_probabilities <= threshold]) * (m2_TTFT + m2_decode_latency)
        average_latency_per_prompt = total_latency / len(bert_probabilities)
        
        thresholds_results[threshold] = {
            'average_flops': average_flops,
            'total_count': len(bert_probabilities),
            'sent_to_small_model': len(bert_probabilities) - len(bert_probabilities[bert_probabilities <= threshold]),
            'sent_to_large_model': len(bert_probabilities[bert_probabilities <= threshold]),
            'average_latency_per_prompt': average_latency_per_prompt,
            }
    return thresholds_results

def analyze_cost_for_different_thresholds_cpx_routing(input_path: str, cpx_prob_dir, small_model, large_model):
    # Define the thresholds for the bert and cpx models
    small_thresholds = list(np.arange(0.00, 1.01, 0.01))

    # Load the probabilities for the bert and cpx models
    with open(cpx_prob_dir, 'rb') as f:
        cpx_validation_data = pickle.load(f)
    cpx_probabilities = cpx_validation_data['probabilities']
    labels = cpx_validation_data['labels']
    
    prompt_key = 'prompt'
    # Compute the average prompts and output lengths
    average_prompts_length = input_length

    thresholds_results = {}
    small_model_size = model_sizes[small_model]
    large_model_size = model_sizes[large_model]
    deberta_model_size = model_sizes['deberta']
    small_flops = estimate_llm_flops(seq_len_prefill=average_prompts_length, output_len=output_length, **small_model_size)
    large_flops = estimate_llm_flops(seq_len_prefill=average_prompts_length, output_len=output_length, **large_model_size)
    small_decode_flops = small_flops['decode_flops']
    large_decode_flops = large_flops['decode_flops']
    small_prefill_flops = small_flops['prefill_flops']
    large_prefill_flops = large_flops['prefill_flops']    # Compute the FLOPs for each routing scenario
    m1_TTFT = model_latencies[small_model]['TTFT']
    m2_TTFT = model_latencies[large_model]['TTFT']
    m1_TPOT = model_latencies[small_model]['TPOT']
    m2_TPOT = model_latencies[large_model]['TPOT']
    m1_decode_latency = m1_TPOT*output_length
    m2_decode_latency = m2_TPOT*output_length
    for threshold in tqdm(small_thresholds, desc="Computing FLOPs for different thresholds"):
        average_flops = 0
        for i in range(len(cpx_probabilities)):
            if cpx_probabilities[i] > threshold:
                flops = small_prefill_flops + small_decode_flops 
            else:
                flops = large_prefill_flops + large_decode_flops + small_prefill_flops
            average_flops += flops
        average_flops /= len(cpx_probabilities)
        total_latency = len(cpx_probabilities[cpx_probabilities > threshold]) * (m1_TTFT + m1_decode_latency) + len(cpx_probabilities[cpx_probabilities <= threshold]) * (m1_TTFT + m2_TTFT + m2_decode_latency)
        average_latency_per_prompt = total_latency / len(cpx_probabilities)
        thresholds_results[threshold] = {
            'average_flops': average_flops,
            'total_count': len(cpx_probabilities),
            'sent_to_small_model': 1,
            'sent_to_large_model': len(cpx_probabilities[cpx_probabilities <= threshold]),
            'average_latency_per_prompt': average_latency_per_prompt,
        }
    return thresholds_results

def compute_reliability_for_threshold_bert_routing(bert_probabilities, labels, threshold):
    total_false_positives = 0
    for i in range(len(bert_probabilities)):
        if bert_probabilities[i] > threshold and labels[i] == 0:
            total_false_positives += 1
    return 1 - (total_false_positives / len(bert_probabilities))

def compute_reliability_for_different_thresholds_bert_routing(input_path: str, bert_prob_dir):
    # Define the thresholds for the bert and cpx models
    small_thresholds = list(np.arange(0.00, 1.01, 0.01))

    # Load the probabilities for the bert and cpx models
    with open(bert_prob_dir, 'rb') as f:
        bert_validation_data = pickle.load(f)
    bert_probabilities = bert_validation_data['probabilities']
    labels = bert_validation_data['labels']

    thresholds_results = {}
    for threshold in tqdm(small_thresholds, desc="Computing reliability for different thresholds"):
        reliability = compute_reliability_for_threshold_bert_routing(bert_probabilities, labels, threshold)
        thresholds_results[threshold] = reliability
    return thresholds_results

def compute_reliability_for_threshold_cpx_routing(cpx_probabilities, labels, threshold):
    total_false_positives = 0
    for i in range(len(cpx_probabilities)):
        if cpx_probabilities[i] > threshold and labels[i] == 0:
            total_false_positives += 1
    return 1 - (total_false_positives / len(cpx_probabilities))

def compute_reliability_for_different_thresholds_cpx_routing(input_path: str, cpx_prob_dir):
    # Define the thresholds for the bert and cpx models
    small_thresholds = list(np.arange(0.00, 1.01, 0.01))

    # Load the probabilities for the bert and cpx models
    with open(cpx_prob_dir, 'rb') as f:
        cpx_validation_data = pickle.load(f)
    cpx_probabilities = cpx_validation_data['probabilities']
    labels = cpx_validation_data['labels']
    if isinstance(labels, pandas.Series):
        labels = labels.to_list()
    thresholds_results = {}
    for threshold in tqdm(small_thresholds, desc="Computing reliability for different thresholds"):
        reliability = compute_reliability_for_threshold_cpx_routing(cpx_probabilities, labels, threshold)
        thresholds_results[threshold] = reliability
    return thresholds_results

def reliability_cost_tradeoff_for_different_thresholds_cpx_routing(input_path: str, cpx_prob_dir, small_model, large_model):
    reliability_results = compute_reliability_for_different_thresholds_cpx_routing(input_path, cpx_prob_dir)
    flops_results = analyze_cost_for_different_thresholds_cpx_routing(input_path, cpx_prob_dir, small_model, large_model)
    reliability_cost_tradeoff = {}
    for combination in reliability_results:
        reliability = reliability_results[combination]
        flops = flops_results[combination]
        reliability_cost_tradeoff[combination] = {
            'reliability': reliability,
            'flops': flops['average_flops'],
            'sent_to_large_model': flops['sent_to_large_model'],
            'total_count': flops['total_count'],
            'average_latency_per_prompt': flops['average_latency_per_prompt'],
        }
    return reliability_cost_tradeoff

def reliability_cost_tradeoff_for_different_thresholds_bert_routing(input_path: str, bert_prob_dir, small_model, large_model):
    reliability_results = compute_reliability_for_different_thresholds_bert_routing(input_path, bert_prob_dir)
    flops_results = analyze_cost_for_different_thresholds_bert_routing(input_path, bert_prob_dir, small_model, large_model)
    reliability_cost_tradeoff = {}
    for combination in reliability_results:
        reliability = reliability_results[combination]
        flops = flops_results[combination]
        reliability_cost_tradeoff[combination] = {
            'reliability': reliability,
            'flops': flops['average_flops'],
            'sent_to_large_model': flops['sent_to_large_model'],
            'total_count': flops['total_count'],
            'average_latency_per_prompt': flops['average_latency_per_prompt'],
        }
    return reliability_cost_tradeoff

def analyze_cost_for_different_thresholds_random_routing(input_path: str, labels_path: str, small_model, large_model, seed: int = 42):
    """
    Analyze cost for random routing where prompts are randomly routed to small or large model
    based on a routing probability threshold.
    
    Args:
        input_path: Path to input data (for consistency with other functions, may not be used)
        labels_path: Path to pickle file containing labels (needed to know dataset size)
        small_model: Name of small model (e.g., 'qwen3_8b')
        large_model: Name of large model (e.g., 'qwen3_32b')
        seed: Random seed for reproducibility
    
    Returns:
        dict: Dictionary mapping routing probability thresholds to cost metrics
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Define routing probability thresholds (probability of routing to small model)
    routing_probabilities = list(np.arange(0.00, 1.01, 0.01))
    
    # Load labels to determine dataset size
    with open(labels_path, 'rb') as f:
        labels_data = pickle.load(f)
    
    if isinstance(labels_data, dict):
        if 'labels' in labels_data:
            labels = labels_data['labels']
        else:
            # Try to find labels in the dict
            labels = list(labels_data.values())[0] if labels_data else []
    elif isinstance(labels_data, list):
        labels = labels_data
    else:
        raise ValueError(f"Unsupported labels format: {type(labels_data)}")
    
    if isinstance(labels, pandas.Series):
        labels = labels.to_list()
    
    num_prompts = len(labels)
    
    # Compute the average prompts and output lengths
    average_prompts_length = input_length
    average_output_length = output_length
    
    thresholds_results = {}
    small_model_size = model_sizes[small_model]
    large_model_size = model_sizes[large_model]
    
    m1_TTFT = model_latencies[small_model]['TTFT']
    m2_TTFT = model_latencies[large_model]['TTFT']
    m1_TPOT = model_latencies[small_model]['TPOT']
    m2_TPOT = model_latencies[large_model]['TPOT']
    m1_decode_latency = m1_TPOT * output_length
    m2_decode_latency = m2_TPOT * output_length
    
    small_flops = estimate_llm_flops(seq_len_prefill=average_prompts_length, output_len=output_length, **small_model_size)['total_flops']
    large_flops = estimate_llm_flops(seq_len_prefill=average_prompts_length, output_len=output_length, **large_model_size)['total_flops']
    
    # Compute the FLOPs for each routing probability
    for routing_prob in tqdm(routing_probabilities, desc="Computing FLOPs for random routing thresholds"):
        # Generate random routing decisions based on routing probability
        # routing_prob is the probability of routing to small model
        random_decisions = np.random.random(num_prompts) < routing_prob
        
        # Count how many go to each model
        sent_to_small_model = np.sum(random_decisions)
        sent_to_large_model = num_prompts - sent_to_small_model
        
        # Compute average FLOPs
        average_flops = (sent_to_small_model * small_flops + sent_to_large_model * large_flops) / num_prompts
        
        # Compute average latency
        total_latency = sent_to_small_model * (m1_TTFT + m1_decode_latency) + sent_to_large_model * (m2_TTFT + m2_decode_latency)
        average_latency_per_prompt = total_latency / num_prompts
        
        thresholds_results[routing_prob] = {
            'average_flops': average_flops,
            'total_count': num_prompts,
            'sent_to_small_model': int(sent_to_small_model),
            'sent_to_large_model': int(sent_to_large_model),
            'average_latency_per_prompt': average_latency_per_prompt,
        }
    
    return thresholds_results

def compute_reliability_for_threshold_random_routing(labels, routing_prob, seed: int = 42):
    """
    Compute reliability for random routing at a given routing probability.
    
    Reliability = 1 - (false_positives / total_prompts)
    False positive = routed to small model when label is 0 (should go to large model)
    
    Args:
        labels: List of labels (1 = small model capable, 0 = needs large model)
        routing_prob: Probability of routing to small model
        seed: Random seed for reproducibility
    
    Returns:
        float: Reliability score
    """
    np.random.seed(seed)
    
    total_false_positives = 0
    # Generate random routing decisions
    random_decisions = np.random.random(len(labels)) < routing_prob
    
    for i in range(len(labels)):
        # False positive: routed to small model (True) but label is 0 (needs large model)
        if random_decisions[i] and labels[i] == 0:
            total_false_positives += 1
    
    reliability = 1 - (total_false_positives / len(labels)) if len(labels) > 0 else 0.0
    return reliability

def compute_reliability_for_different_thresholds_random_routing(labels_path: str, seed: int = 42):
    """
    Compute reliability for different random routing probability thresholds.
    
    Args:
        labels_path: Path to pickle file containing labels
        seed: Random seed for reproducibility
    
    Returns:
        dict: Dictionary mapping routing probability thresholds to reliability scores
    """
    # Define routing probability thresholds
    routing_probabilities = list(np.arange(0.00, 1.01, 0.01))
    
    # Load labels
    with open(labels_path, 'rb') as f:
        labels_data = pickle.load(f)
    
    if isinstance(labels_data, dict):
        if 'labels' in labels_data:
            labels = labels_data['labels']
        else:
            labels = list(labels_data.values())[0] if labels_data else []
    elif isinstance(labels_data, list):
        labels = labels_data
    else:
        raise ValueError(f"Unsupported labels format: {type(labels_data)}")
    
    if isinstance(labels, pandas.Series):
        labels = labels.to_list()
    
    reliability_results = {}
    for routing_prob in tqdm(routing_probabilities, desc="Computing reliability for random routing thresholds"):
        reliability = compute_reliability_for_threshold_random_routing(labels, routing_prob, seed=seed)
        reliability_results[routing_prob] = reliability
    
    return reliability_results

def reliability_cost_tradeoff_for_different_thresholds_random_routing(labels_path: str, small_model, large_model, seed: int = 42):
    """
    Compute reliability-cost tradeoff for random routing at different probability thresholds.
    
    This function combines reliability and cost analysis for random routing, where prompts
    are randomly routed to small or large model based on a routing probability threshold.
    
    Args:
        labels_path: Path to pickle file containing labels
        small_model: Name of small model (e.g., 'qwen3_8b')
        large_model: Name of large model (e.g., 'qwen3_32b')
        seed: Random seed for reproducibility
    
    Returns:
        dict: Dictionary mapping routing probability thresholds to reliability-cost metrics
    """
    # Set random seed once for consistency
    np.random.seed(seed)
    
    # Load labels once
    with open(labels_path, 'rb') as f:
        labels_data = pickle.load(f)
    labels_data = labels_data['labels']
    if isinstance(labels_data, dict):
        if 'labels' in labels_data:
            labels = labels_data['labels']
        else:
            labels = list(labels_data.values())[0] if labels_data else []
    elif isinstance(labels_data, list):
        labels = labels_data
    elif isinstance(labels_data, pandas.Series):
        labels = labels_data.to_list()
    elif isinstance(labels_data, numpy.ndarray):
        labels = labels_data.tolist()
    else:
        raise ValueError(f"Unsupported labels format: {type(labels_data)}")
    
    if isinstance(labels, pandas.Series):
        labels = labels.to_list()
    
    num_prompts = len(labels)
    
    # Define routing probability thresholds
    routing_probabilities = list(np.arange(0.00, 1.01, 0.01))
    
    # Compute model parameters
    average_prompts_length = input_length
    average_output_length = output_length
    small_model_size = model_sizes[small_model]
    large_model_size = model_sizes[large_model]
    
    m1_TTFT = model_latencies[small_model]['TTFT']
    m2_TTFT = model_latencies[large_model]['TTFT']
    m1_TPOT = model_latencies[small_model]['TPOT']
    m2_TPOT = model_latencies[large_model]['TPOT']
    m1_decode_latency = m1_TPOT * output_length
    m2_decode_latency = m2_TPOT * output_length
    
    small_flops = estimate_llm_flops(seq_len_prefill=average_prompts_length, output_len=output_length, **small_model_size)['total_flops']
    large_flops = estimate_llm_flops(seq_len_prefill=average_prompts_length, output_len=output_length, **large_model_size)['total_flops']
    
    reliability_cost_tradeoff = {}
    
    # Compute reliability and cost for each routing probability
    for routing_prob in tqdm(routing_probabilities, desc="Computing reliability-cost tradeoff for random routing"):
        # Generate random routing decisions (same seed ensures reproducibility)
        random_decisions = np.random.random(num_prompts) < routing_prob
        
        # Count routing decisions
        sent_to_small_model = np.sum(random_decisions)
        sent_to_large_model = num_prompts - sent_to_small_model
        
        # Compute cost metrics
        average_flops = (sent_to_small_model * small_flops + sent_to_large_model * large_flops) / num_prompts
        total_latency = sent_to_small_model * (m1_TTFT + m1_decode_latency) + sent_to_large_model * (m2_TTFT + m2_decode_latency)
        average_latency_per_prompt = total_latency / num_prompts
        
        # Compute reliability (false positives: routed to small model when label is 0)
        total_false_positives = 0
        for i in range(num_prompts):
            if random_decisions[i] and labels[i] == 0:
                total_false_positives += 1
        reliability = 1 - (total_false_positives / num_prompts) if num_prompts > 0 else 0.0
        
        reliability_cost_tradeoff[routing_prob] = {
            'reliability': reliability,
            'flops': average_flops,
            'sent_to_large_model': int(sent_to_large_model),
            'total_count': num_prompts,
            'average_latency_per_prompt': average_latency_per_prompt,
        }
    
    return reliability_cost_tradeoff

def get_pareto_front(costs, reliabilities):
    """
    Finds the Pareto-efficient solutions from a set of (cost, reliability) points.

    Objectives: Minimize Cost, Maximize Reliability.
    
    Args:
        costs (list or np.ndarray): List of cost values.
        reliabilities (list or np.ndarray): List of reliability values.
        
    Returns:
        list: A list of tuples [(cost, reliability), ...] representing the Pareto Front, 
              sorted by ascending cost.
    """
    # 1. Combine and prepare data: Convert lists to a 2D NumPy array
    #    where each row is a point (Cost, Reliability)
    data = np.array([costs, reliabilities]).T
    
    # 2. Initialize mask: Assume all points are efficient initially
    is_efficient = np.ones(data.shape[0], dtype=bool)
    
    # 3. Iterate and filter (O(N^2) complexity)
    for i, point_i in enumerate(data):
        if is_efficient[i]: # Only check points that haven't been dominated yet
            
            # For Pareto dominance: minimize cost, maximize reliability
            # Point j dominates point i if:
            #   (Cost_j <= Cost_i) AND (Reliability_j >= Reliability_i)
            #   AND at least one is strict: (Cost_j < Cost_i) OR (Reliability_j > Reliability_i)
            
            # Check if cost_j <= cost_i AND reliability_j >= reliability_i
            cost_better_or_equal = data[:, 0] <= point_i[0]  # cost: lower is better
            reliability_better_or_equal = data[:, 1] >= point_i[1]  # reliability: higher is better
            domination_mask = cost_better_or_equal & reliability_better_or_equal
            
            # Check if strictly better in at least one dimension
            cost_strictly_better = data[:, 0] < point_i[0]
            reliability_strictly_better = data[:, 1] > point_i[1]
            superiority_mask = cost_strictly_better | reliability_strictly_better

            # A point 'j' dominates 'i' if it satisfies both masks (and j != i)
            # Exclude point i from dominating itself
            domination_mask[i] = False
            is_dominated_by_any = np.any(domination_mask & superiority_mask)
            
            if is_dominated_by_any:
                is_efficient[i] = False
                
    # 4. Filter, sort, and return
    pareto_points = data[is_efficient]
    
    # Sort the final points by Cost for a clean, traceable curve
    pareto_points = pareto_points[np.argsort(pareto_points[:, 0])]
    
    return [tuple(p) for p in pareto_points]