"""
Inference script for CPX models.

This script loads a trained CPX model and runs inference on a dataset to get probabilities.

Usage:
    python inference.py --model_path <path_to_model.pth> --dataset_path <path_to_dataset.pkl> --model_name <base_model_name> [options]
"""

import argparse
import torch
import pickle
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from torch.amp import autocast
from transformers import AutoModelForCausalLM
from peft import LoraConfig

from cpx_model.cpx_causal_lm import CPXCausalLM
from cpx_model.cpx_causal_tokenizer import CPXTokenizer
from cpx_model.cpx_causal_utils import TextRegressionDataset, load_pickle_data
from cpx_model.config import CPXTrainingConfig
from datasets import Dataset as HFDataset
import pandas
import time
from tqdm import tqdm


def load_config_from_json(config_path: str) -> CPXTrainingConfig:
    """
    Load training config from JSON file.
    
    Args:
        config_path: Path to the JSON config file
    
    Returns:
        CPXTrainingConfig instance
    """
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create CPXTrainingConfig from dict
    # Handle None values and empty lists
    for key, value in config_dict.items():
        if value is None:
            config_dict[key] = None
        elif isinstance(value, list) and len(value) == 0:
            config_dict[key] = []
    
    return CPXTrainingConfig(**config_dict)


def find_config_file(model_path: str) -> str:
    """
    Try to find the corresponding config file for a model checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
    
    Returns:
        Path to config file if found, None otherwise
    """
    # Extract timestamp and model name from model path
    # Format: model_{model_basename}_cpx_{timestamp}.pth
    base_dir = os.path.dirname(model_path)
    model_filename = os.path.basename(model_path)
    
    # Try to extract timestamp from model filename
    if '_cpx_' in model_filename:
        parts = model_filename.split('_cpx_')
        if len(parts) == 2:
            timestamp = parts[1].replace('.pth', '').replace('.pkl', '')
            model_basename = parts[0].replace('model_', '')
            config_filename = f"config_{model_basename}_cpx_{timestamp}.json"
            config_path = os.path.join(base_dir, config_filename)
            if os.path.exists(config_path):
                return config_path
    
    return None


def find_tokenizer_dir(model_path: str) -> str:
    """
    Try to find the corresponding tokenizer directory for a model checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
    
    Returns:
        Path to tokenizer directory if found, None otherwise
    """
    # Extract timestamp and model name from model path
    # Format: model_{model_basename}_cpx_{timestamp}.pth
    base_dir = os.path.dirname(model_path)
    model_filename = os.path.basename(model_path)
    
    # Try to extract timestamp from model filename
    if '_cpx_' in model_filename:
        parts = model_filename.split('_cpx_')
        if len(parts) == 2:
            timestamp = parts[1].replace('.pth', '').replace('.pkl', '')
            model_basename = parts[0].replace('model_', '')
            tokenizer_dir = os.path.join(base_dir, f"tokenizer_{model_basename}_cpx_{timestamp}")
            if os.path.exists(tokenizer_dir) and os.path.isdir(tokenizer_dir):
                return tokenizer_dir
    
    return None


def load_model_from_checkpoint(
    model_path: str,
    base_model_name: str,
    cpx_token_ids: list,
    tokenizer_size: int = None,
    use_lora: bool = False,
    lora_config: dict = None,
    dropout_rate: float = 0.1,
    classifier_dropout: bool = True,
    aggregation_type: str = 'attention',
    num_layers: int = None,
    mask_lora_for_non_cpx: bool = True,
    is_cpx_token_trainable: bool = True,
    freeze_LoRA_layers: bool = False,
    freeze_LoRA_start_layer_idx: int = 0,
    use_last_hidden_state_baseline: bool = False,
    device: str = 'cuda'
):
    """
    Load a CPX model from a checkpoint file.
    
    Args:
        model_path: Path to the model checkpoint (.pth or .pkl file)
        base_model_name: Name of the base model (e.g., "mistralai/Mistral-7B-Instruct-v0.3")
        cpx_token_ids: List of CPX token IDs
        use_lora: Whether LoRA was used during training
        lora_config: LoRA configuration dict
        dropout_rate: Dropout rate used during training
        classifier_dropout: Whether classifier dropout was used
        aggregation_type: Aggregation type for multiple CPX tokens
        num_layers: Number of layers (if model was sliced)
        mask_lora_for_non_cpx: Whether LoRA activations were masked for non-CPX positions during training.
                              Must match training config. If True, LoRA forward hooks will be registered.
        is_cpx_token_trainable: Whether CPX token embeddings are trainable (must match training config)
        freeze_LoRA_layers: Whether LoRA layers are frozen (must match training config)
        freeze_LoRA_start_layer_idx: Starting layer index for freezing LoRA (must match training config)
        use_last_hidden_state_baseline: If True, use last hidden state of original prompt (before CPX tokens) instead of CPX token hidden states
        device: Device to load model on
    
    Returns:
        model: Loaded CPX model
    """
    # Load model state dict
    if model_path.endswith('.pkl'):
        with open(model_path, 'rb') as f:
            state_dict = pickle.load(f)
    else:
        state_dict = torch.load(model_path, map_location=device)
    
    # Create LoRA config if needed
    if use_lora:
        if lora_config is None:
            # Default LoRA config - should match training config
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=None,  # Will be auto-detected by CPXCausalLM
                lora_dropout=0.15,
                bias="none",
                task_type="CAUSAL_LM",
            )
    
    # Initialize model with same config as training
    model = CPXCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_model_name,
        cpx_token_ids=cpx_token_ids,
        num_labels=1,
        is_cpx_token_trainable=is_cpx_token_trainable,
        tokenizer_size=tokenizer_size,
        use_lora=use_lora,
        lora_config=lora_config,
        dropout_rate=dropout_rate,
        classifier_dropout=classifier_dropout,
        aggregation_type=aggregation_type,
        num_layers=num_layers,
        mask_lora_for_non_cpx=mask_lora_for_non_cpx,
        freeze_LoRA_layers=freeze_LoRA_layers,
        freeze_LoRA_start_layer_idx=freeze_LoRA_start_layer_idx,
        use_last_hidden_state_baseline=use_last_hidden_state_baseline,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    
    # Load state dict
    # Handle DDP-wrapped state dict (remove 'module.' prefix if present)
    if any(key.startswith('module.') for key in state_dict.keys()):
        # State dict from DDP model, remove 'module.' prefix
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    
    # Move model to device BEFORE loading state dict to ensure proper dtype handling
    model.to(device)
    
    # Ensure classifier and aggregator weights match saved dtypes
    # Classifier and aggregator are float32, so convert saved weights if needed
    for key, value in state_dict.items():
        if 'classifier' in key or 'aggregator' in key:
            # Ensure classifier/aggregator weights are loaded as float32
            if value.dtype != torch.float32:
                state_dict[key] = value.to(torch.float32)
    
    # Load state dict with strict=False to handle minor mismatches
    # But log any missing or unexpected keys for debugging
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠ Warning: Missing keys when loading checkpoint: {missing_keys[:10]}...")  # Show first 10
        if len(missing_keys) > 10:
            print(f"  ... and {len(missing_keys) - 10} more missing keys")
    
    if unexpected_keys:
        print(f"⚠ Warning: Unexpected keys in checkpoint: {unexpected_keys[:10]}...")  # Show first 10
        if len(unexpected_keys) > 10:
            print(f"  ... and {len(unexpected_keys) - 10} more unexpected keys")
    
    # Verify classifier and aggregator are still float32 after loading
    if hasattr(model, 'classifier'):
        if model.classifier.weight.dtype != torch.float32:
            print(f"⚠ Warning: Classifier weight dtype is {model.classifier.weight.dtype}, converting to float32")
            model.classifier = model.classifier.float()
    
    if hasattr(model, 'aggregator') and model.aggregator is not None:
        if hasattr(model.aggregator, 'attention'):
            if model.aggregator.attention.weight.dtype != torch.float32:
                print(f"⚠ Warning: Aggregator attention weight dtype is {model.aggregator.attention.weight.dtype}, converting to float32")
                model.aggregator.attention = model.aggregator.attention.float()
    
    model.eval()
    
    return model


def load_dataset_from_pickle(dataset_path: str, cpx_tokens: list = None):
    """
    Load dataset from pickle file.
    
    Args:
        dataset_path: Path to the dataset pickle file
        cpx_tokens: Optional list of CPX tokens to append to prompts
    
    Returns:
        texts: List of text prompts
        labels: List of labels (if available)
    """
    data = load_pickle_data(dataset_path)
    
    # Handle HuggingFace Dataset objects
    if isinstance(data, HFDataset):
        # Get available column names
        column_names = data.column_names
        
        # Find text field
        if 'prompt' in column_names:
            texts = data['prompt']
        elif 'text' in column_names:
            texts = data['text']
        elif 'question' in column_names:
            texts = data['question']
        else:
            raise ValueError(f"Could not find text field in dataset. Available columns: {column_names}")
        
        # Get labels if available
        labels = None
        if 'correct' in column_names:
            labels = data['correct']
        elif 'labels' in column_names:
            labels = data['labels']
        elif 'label' in column_names:
            labels = data['label']
    
    # Handle different data formats
    elif isinstance(data, dict):
        if 'prompts' in data:
            texts = data['prompts']
        elif 'text' in data:
            texts = data['text']
        elif 'question' in data:
            texts = data['question']
        else:
            raise ValueError(f"Could not find text field in dataset. Available keys: {data.keys()}")
        
        # Get labels if available
        labels = None
        if 'correct_labels' in data:
            labels = data['correct_labels']
        elif 'labels' in data:
            labels = data['labels']
        elif 'label' in data:
            labels = data['label']
            
    elif isinstance(data, list):
        # Assume list of dicts
        texts = [item.get('prompt', item.get('text', item.get('question', ''))) for item in data]
        labels = [item.get('correct', item.get('labels', item.get('label', None))) for item in data]
        if all(l is None for l in labels):
            labels = None
    
    elif isinstance(data, pandas.DataFrame):
        texts = data['prompts']
        labels = data['correct_labels']
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")

    # Normalize labels to match training format
    # Training uses torch tensors with shape [N, 1], but we need flat arrays for inference
    # This ensures consistent label format between training evaluation and inference
    if labels is not None:
        # Convert to numpy array and flatten to ensure consistent shape
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        labels = np.asarray(labels)
        # Flatten to 1D array (handle both [N] and [N, 1] shapes)
        labels = labels.flatten()
        # Convert to int to match training evaluation format (training uses .int())
        labels = labels.astype(int)
    
    # Append CPX tokens if provided
    # IMPORTANT: Match training format exactly - space before, no space between tokens
    # Note: This assumes prompts don't already end with trailing spaces
    # If they do, you'll get double spaces (tokenizers usually normalize this)
    if cpx_tokens:
        cpx_suffix = ' ' + ''.join(cpx_tokens)  # Match training: space before, no space between
        texts = [text + cpx_suffix for text in texts]
    
    return texts, labels


def get_probabilities(
    model: CPXCausalLM,
    tokenizer: CPXTokenizer,
    texts: list,
    batch_size: int = 32,
    context_window: int = 1024,
    device: str = 'cuda'
):
    """
    Get probabilities for a list of texts.
    
    Args:
        model: CPX model
        tokenizer: CPX tokenizer
        texts: List of text prompts
        batch_size: Batch size for inference
        context_window: Maximum context window length
        device: Device to run inference on
    
    Returns:
        probabilities: Numpy array of probabilities
    """
    # Create dataset
    # Use dummy labels (will be ignored)
    dummy_labels = [0.0] * len(texts)
    dataset = TextRegressionDataset(texts, dummy_labels, tokenizer, context_window)
    # Match training: drop_last=False for inference to keep all samples
    # (Training uses drop_last=True only for distributed training)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    all_probs = []
    is_binary = None  # Will be determined from first batch
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing dataset"):
            # if batch_idx % (len(loader) // 10) == 0:
            #     print(f"Processing {batch_idx / len(loader) * 100:.2f}% of the dataset")
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Use autocast only for CUDA (mixed precision), skip for CPU
            if device == 'cuda':
                with autocast('cuda', dtype=torch.bfloat16):
                    logits, outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                logits, outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Determine if binary or multi-class based on logits shape (only on first batch)
            if is_binary is None:
                # Binary: logits shape [batch_size, 1] or [batch_size]
                # Multi-class: logits shape [batch_size, num_classes] where num_classes > 1
                is_binary = logits.dim() == 1 or (logits.dim() == 2 and logits.size(1) == 1)
            
            if is_binary:
                # Binary: use sigmoid to get probabilities
                probs = torch.sigmoid(logits)
                if probs.dim() > 1 and probs.size(1) == 1:
                    probs = probs.squeeze(1)
            else:
                # Multi-class: use softmax to get probability distribution
                probs = torch.softmax(logits, dim=1)
            
            all_probs.append(probs.cpu())
    
    # Concatenate all probabilities
    all_probs = torch.cat(all_probs, dim=0)
    # Convert to float32 before numpy conversion (numpy doesn't support bfloat16)
    if is_binary:
        # Binary: flatten to 1D array
        return all_probs.view(-1).float().numpy()
    else:
        # Multi-class: keep shape [batch_size, num_classes]
        return all_probs.float().numpy()


def compute_confusion_matrix(probabilities, labels, threshold):
    """
    Compute confusion matrix metrics (FN, FP, TP, TN) from probabilities, labels, and threshold.
    
    Args:
        probabilities: Array-like of probabilities (values between 0 and 1)
        labels: Array-like of true labels (binary: 0 or 1)
        threshold: Threshold value to convert probabilities to binary predictions
                   (probabilities >= threshold are predicted as 1, otherwise 0)
    
    Returns:
        dict: Dictionary containing:
            - 'FN': Number of False Negatives (predicted 0, actual 1)
            - 'FP': Number of False Positives (predicted 1, actual 0)
            - 'TP': Number of True Positives (predicted 1, actual 1)
            - 'TN': Number of True Negatives (predicted 0, actual 0)
    """
    # Convert to numpy arrays if needed
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    
    # Ensure labels are binary (0 or 1)
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Labels must be binary (0 or 1)")
    
    # Convert probabilities to binary predictions using threshold
    predictions = (probabilities >= threshold).astype(int)
    
    # Compute confusion matrix metrics
    TP = np.sum((predictions == 1) & (labels == 1))  # True Positive
    TN = np.sum((predictions == 0) & (labels == 0))  # True Negative
    FP = np.sum((predictions == 1) & (labels == 0))  # False Positive
    FN = np.sum((predictions == 0) & (labels == 1))  # False Negative
    
    return {
        'FN': int(FN),
        'FP': int(FP),
        'TP': int(TP),
        'TN': int(TN)
    }


def run_inference(
    model_path: str,
    dataset_path: str,
    batch_size: int = None,
    context_window: int = None,
    device: str = 'cuda',
    verbose: bool = True,
    config_path: str = None,
    training_config: CPXTrainingConfig = None
):
    """
    Run inference with CPX model and return probabilities.
    
    IMPORTANT: All model parameters MUST come from config file. The function will:
    1. Try to load config from config_path (if provided)
    2. Auto-detect config from model_path if config_path not provided
    3. Raise error if config file cannot be found
    
    Args:
        model_path: Path to the model checkpoint (.pth or .pkl file)
        dataset_path: Path to the dataset pickle file
        batch_size: Batch size for inference (overrides config if provided, otherwise uses config value)
        context_window: Maximum context window length (overrides config if provided, otherwise uses config value)
        device: Device to run inference on
        verbose: Whether to print progress messages
        config_path: Optional path to training config JSON file. If not provided, will auto-detect from model_path.
        training_config: Optional CPXTrainingConfig instance. Takes precedence over config_path.
                       (For internal use - typically not provided by users)
    
    Returns:
        dict: Dictionary containing:
            - 'probabilities': numpy array of probabilities
            - 'texts': list of text prompts
            - 'labels': labels from dataset (if available)
            - 'model': loaded model (optional, can be used for further inference)
            - 'tokenizer': loaded tokenizer (optional, can be used for further inference)
    
    Raises:
        ValueError: If config file cannot be found or loaded
    """
    # Step 1: Load config (priority: training_config > config_path > auto-detect)
    if training_config is None:
        if config_path is not None:
            if verbose:
                print(f"Loading training config from {config_path}...")
            training_config = load_config_from_json(config_path)
        else:
            # Try to auto-detect config file from model path
            auto_config_path = find_config_file(model_path)
            if auto_config_path is not None:
                if verbose:
                    print(f"Auto-detected config file: {auto_config_path}")
                training_config = load_config_from_json(auto_config_path)
    
    # Step 2: Require config file - raise error if not found
    if training_config is None:
        raise ValueError(
            f"Config file not found! Please provide --config_path or ensure config file exists "
            f"alongside model checkpoint. Expected config file name format: "
            f"config_{{model_basename}}_cpx_{{timestamp}}.json"
        )
    
    # ALL model parameters come from config
    if verbose:
        print("=" * 60)
        print("Using training config - ALL parameters from config file")
        print("=" * 60)
    
    model_name = training_config.model_name
    cpx_tokens = training_config.cpx_tokens
    use_lora = training_config.use_lora
    dropout_rate = training_config.dropout_rate
    classifier_dropout = training_config.classifier_dropout
    cpx_aggregation = training_config.cpx_aggregation
    num_layers = training_config.num_layers
    mask_lora_for_non_cpx = training_config.mask_lora_for_non_cpx
    is_cpx_token_trainable = training_config.is_cpx_token_trainable
    freeze_LoRA_layers = training_config.freeze_LoRA_layers
    freeze_LoRA_start_layer_idx = training_config.freeze_LoRA_start_layer_idx
    use_last_hidden_state_baseline = getattr(training_config, 'use_last_hidden_state_baseline', False)
    
    # Use LoRA config from training config (always set, even if use_lora is False)
    lora_r = training_config.lora_r
    lora_alpha = training_config.lora_alpha
    lora_dropout = training_config.lora_dropout
    lora_target_modules = training_config.lora_target_modules
    
    # Use context_window from config if available, otherwise use provided value
    if context_window is None:
        if hasattr(training_config, 'context_window') and training_config.context_window is not None:
            context_window = training_config.context_window
        else:
            context_window = 1024  # Default fallback
            if verbose:
                print(f"Warning: context_window not in config, using default: {context_window}")
    
    # Use batch_size from config if not provided
    if batch_size is None:
        if hasattr(training_config, 'batch_size') and training_config.batch_size is not None:
            batch_size = training_config.batch_size
        else:
            batch_size = 32  # Default fallback
            if verbose:
                print(f"Warning: batch_size not in config, using default: {batch_size}")
    
    if verbose:
        print(f"  Model: {model_name}")
        print(f"  CPX tokens: {cpx_tokens}")
        print(f"  Use LoRA: {use_lora}")
        print(f"  Mask LoRA for non-CPX: {mask_lora_for_non_cpx}")
        print(f"  CPX aggregation: {cpx_aggregation}")
        print(f"  Dropout rate: {dropout_rate}")
        print(f"  Classifier dropout: {classifier_dropout}")
        print(f"  CPX token trainable: {is_cpx_token_trainable}")
        print(f"  Use last hidden state baseline: {use_last_hidden_state_baseline}")
        print(f"  Context window: {context_window}")
        print(f"  Batch size: {batch_size}")
        if use_lora:
            print(f"  LoRA r: {lora_r}, alpha: {lora_alpha}, dropout: {lora_dropout}")
            print(f"  LoRA target modules: {lora_target_modules}")
            print(f"  Freeze LoRA layers: {freeze_LoRA_layers}, start_idx: {freeze_LoRA_start_layer_idx}")
        print("=" * 60)
    
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        if verbose:
            print("CUDA not available, using CPU")
        device = 'cpu'
    
    if verbose:
        print(f"Loading tokenizer...")
    # Try to load saved tokenizer from training (ensures exact reproducibility)
    tokenizer_dir = find_tokenizer_dir(model_path)
    if tokenizer_dir is not None:
        if verbose:
            print(f"Found saved tokenizer at {tokenizer_dir}, loading from there...")
        # Load saved tokenizer directly (it already has CPX tokens added)
        # This preserves all settings from training: padding_side, truncation_side, etc.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        if verbose:
            print(f"Loaded saved tokenizer with {len(tokenizer)} tokens")
            print(f"  Padding side: {tokenizer.padding_side}")
            print(f"  Truncation side: {tokenizer.truncation_side}")
            print(f"  Pad token: {tokenizer.pad_token}")
    else:
        if verbose:
            print(f"No saved tokenizer found, creating new tokenizer from {model_name}...")
        # Fall back to creating tokenizer from model_name
        # This will set: padding_side="right", truncation_side="left", truncation=True
        tokenizer = CPXTokenizer.from_pretrained(model_name, cpx_tokens=cpx_tokens)
        if verbose:
            print(f"Created new tokenizer with {len(tokenizer)} tokens")
            print(f"  Padding side: {tokenizer.padding_side}")
            print(f"  Truncation side: {tokenizer.truncation_side}")
            print(f"  Pad token: {tokenizer.pad_token}")
    
    # Get CPX token IDs
    # cpx_tokens should already be set from config or args above
    if cpx_tokens is None:
        raise ValueError("cpx_tokens must be set (from config or arguments)")
    
    cpx_token_ids = tokenizer.convert_tokens_to_ids(cpx_tokens)
    if verbose:
        print(f"CPX tokens: {cpx_tokens}")
        print(f"CPX token IDs: {cpx_token_ids}")
    
    # Get tokenizer size (needed for model initialization)
    tokenizer_size = len(tokenizer)
    if verbose:
        print(f"Tokenizer size: {tokenizer_size}")
    
    if verbose:
        print(f"Loading model from {model_path}...")
    # Prepare LoRA config if needed
    lora_config = None
    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    
    # Load model
    model = load_model_from_checkpoint(
        model_path=model_path,
        base_model_name=model_name,
        cpx_token_ids=cpx_token_ids,
        tokenizer_size=tokenizer_size,
        use_lora=use_lora,
        lora_config=lora_config,
        dropout_rate=dropout_rate,
        classifier_dropout=classifier_dropout,
        aggregation_type=cpx_aggregation,
        num_layers=num_layers,
        mask_lora_for_non_cpx=mask_lora_for_non_cpx,
        is_cpx_token_trainable=is_cpx_token_trainable,
        freeze_LoRA_layers=freeze_LoRA_layers,
        freeze_LoRA_start_layer_idx=freeze_LoRA_start_layer_idx,
        use_last_hidden_state_baseline=use_last_hidden_state_baseline,
        device=device
    )
    if verbose:
        print("Model loaded successfully!")
    
    if verbose:
        print(f"Loading dataset from {dataset_path}...")
    # Load dataset
    texts, labels = load_dataset_from_pickle(dataset_path, cpx_tokens=cpx_tokens)
    if verbose:
        print(f"Loaded {len(texts)} samples")
    
    if verbose:
        print("Running inference...")
    # Get probabilities
    probabilities = get_probabilities(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        batch_size=batch_size,
        context_window=context_window,
        device=device
    )
    if verbose:
        print(f"Got probabilities for {len(probabilities)} samples")
        print(f"Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
        print(f"Mean probability: {probabilities.mean():.4f}")
    
    # Return results
    results = {
        'probabilities': probabilities,
        'texts': texts,
        'labels': labels,
        'model': model,
        'tokenizer': tokenizer,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run inference with CPX model. All model parameters are read from config file.')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint (.pth or .pkl file)')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the dataset pickle file')
    
    # Config file argument (optional - will auto-detect if not provided)
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to training config JSON file. If not provided, will auto-detect from model path.')
    
    # Inference-specific arguments (can override config defaults)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for inference (overrides config if provided)')
    parser.add_argument('--context_window', type=int, default=None,
                        help='Maximum context window length (overrides config if provided)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (default: cuda)')
    
    args = parser.parse_args()
    
    # Run inference - all model parameters come from config file
    results = run_inference(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        context_window=args.context_window,
        device=args.device,
        verbose=True,
        config_path=args.config_path
    )
    
    # Extract model name for output filename from config
    model_name_for_output = None
    if args.config_path is not None:
        try:
            config = load_config_from_json(args.config_path)
            model_name_for_output = config.model_name
        except:
            pass
    if model_name_for_output is None:
        auto_config_path = find_config_file(args.model_path)
        if auto_config_path is not None:
            try:
                config = load_config_from_json(auto_config_path)
                model_name_for_output = config.model_name
            except:
                pass

    # Extract model name and timestamp from model_path
    # Format: model_{model_basename}_cpx_{timestamp}.pth
    model_filename = os.path.basename(args.model_path)
    model_basename = None
    model_timestamp = None
    
    if '_cpx_' in model_filename:
        parts = model_filename.split('_cpx_')
        if len(parts) == 2:
            model_timestamp = parts[1].replace('.pth', '').replace('.pkl', '')
            model_basename = parts[0].replace('model_', '')
    
    # If we couldn't extract model name from filename, use model_name from config/args
    if model_basename is None or model_basename == '':
        if model_name_for_output:
            # Replace '/' with '_' to make it filesystem-safe
            model_basename = model_name_for_output.replace('/', '_')
        else:
            model_basename = 'unknown_model'
    
    # If no timestamp found, use current timestamp
    if model_timestamp is None or model_timestamp == '':
        model_timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create output directory
    output_base_dir = "./cpx_model/inference_logs"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Construct output filename: model_name + timestamp
    output_filename = f"{model_basename}_{model_timestamp}.pkl"
    output_path = os.path.join(output_base_dir, output_filename)
    
    # Prepare results for saving (exclude model and tokenizer)
    save_results = {
        'probabilities': results['probabilities'],
        'texts': results['texts'],
        'labels': results['labels'],
        'model_path': args.model_path,
        'dataset_path': args.dataset_path,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(save_results, f)
    
    print(f"Results saved to {output_path}")
    
    # Print some statistics if labels are available
    if results['labels'] is not None:
        labels_array = np.array(results['labels'])
        if len(labels_array.shape) > 1:
            labels_array = labels_array.flatten()
        
        probabilities = results['probabilities']
        print("\nStatistics:")
        print(f"  Class 0 samples: {np.sum(labels_array == 0)}")
        print(f"  Class 1 samples: {np.sum(labels_array == 1)}")
        print(f"  Mean probability for class 0: {probabilities[labels_array == 0].mean():.4f}")
        print(f"  Mean probability for class 1: {probabilities[labels_array == 1].mean():.4f}")


if __name__ == "__main__":
    main()

