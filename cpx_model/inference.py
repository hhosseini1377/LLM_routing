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
        is_cpx_token_trainable=True,
        tokenizer_size=tokenizer_size,
        use_lora=use_lora,
        lora_config=lora_config,
        dropout_rate=dropout_rate,
        classifier_dropout=classifier_dropout,
        aggregation_type=aggregation_type,
        num_layers=num_layers,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_cache=False,
    )
    
    # Load state dict
    # Handle DDP-wrapped state dict (remove 'module.' prefix if present)
    if any(key.startswith('module.') for key in state_dict.keys()):
        # State dict from DDP model, remove 'module.' prefix
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)  # Use strict=False to handle minor mismatches
    model.to(device)
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
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing dataset"):
            # if batch_idx % (len(loader) // 10) == 0:
            #     print(f"Processing {batch_idx / len(loader) * 100:.2f}% of the dataset")
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with autocast('cuda', dtype=torch.bfloat16):
                logits, outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
    
    # Concatenate all probabilities
    all_probs = torch.cat(all_probs, dim=0)
    # Convert to float32 before numpy conversion (numpy doesn't support bfloat16)
    return all_probs.view(-1).float().numpy()


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
    model_name: str,
    cpx_tokens: list = None,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.15,
    lora_target_modules: list = None,
    dropout_rate: float = 0.1,
    classifier_dropout: bool = True,
    cpx_aggregation: str = 'attention',
    num_layers: int = None,
    batch_size: int = 32,
    context_window: int = 1024,
    device: str = 'cuda',
    verbose: bool = True
):
    """
    Run inference with CPX model and return probabilities.
    
    Args:
        model_path: Path to the model checkpoint (.pth or .pkl file)
        dataset_path: Path to the dataset pickle file
        model_name: Base model name (e.g., "mistralai/Mistral-7B-Instruct-v0.3")
        cpx_tokens: List of CPX tokens (default: ['[CPX]'])
        use_lora: Whether LoRA was used during training
        lora_r: LoRA rank (if LoRA was used)
        lora_alpha: LoRA alpha (if LoRA was used)
        lora_dropout: LoRA dropout (if LoRA was used)
        lora_target_modules: LoRA target modules (if LoRA was used)
        dropout_rate: Dropout rate used during training
        classifier_dropout: Whether classifier dropout was used
        cpx_aggregation: Aggregation type for multiple CPX tokens
        num_layers: Number of layers (if model was sliced during training)
        batch_size: Batch size for inference
        context_window: Maximum context window length
        device: Device to run inference on
        verbose: Whether to print progress messages
    
    Returns:
        dict: Dictionary containing:
            - 'probabilities': numpy array of probabilities
            - 'texts': list of text prompts
            - 'labels': labels from dataset (if available)
            - 'model': loaded model (optional, can be used for further inference)
            - 'tokenizer': loaded tokenizer (optional, can be used for further inference)
    """
    if cpx_tokens is None:
        cpx_tokens = ['[CPX]']
    
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        if verbose:
            print("CUDA not available, using CPU")
        device = 'cpu'
    
    if verbose:
        print(f"Loading tokenizer...")
    # Load tokenizer
    tokenizer = CPXTokenizer.from_pretrained(model_name, cpx_tokens=cpx_tokens)
    
    # Get CPX token IDs
    cpx_token_ids = tokenizer.convert_tokens_to_ids(cpx_tokens)
    if verbose:
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
    parser = argparse.ArgumentParser(description='Run inference with CPX model')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint (.pth or .pkl file)')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the dataset pickle file')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Base model name (e.g., mistralai/Mistral-7B-Instruct-v0.3)')
    
    # CPX token arguments
    parser.add_argument('--cpx_tokens', type=str, nargs='+', default=['[CPX]'],
                        help='List of CPX tokens (default: [CPX])')
    
    # Model configuration arguments
    parser.add_argument('--use_lora', type=lambda x: x.lower() == 'true', default=False,
                        help='Whether LoRA was used during training')
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank (if LoRA was used)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha (if LoRA was used)')
    parser.add_argument('--lora_dropout', type=float, default=0.15,
                        help='LoRA dropout (if LoRA was used)')
    parser.add_argument('--lora_target_modules', type=str, nargs='+', default=None,
                        help='LoRA target modules (if LoRA was used)')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='Dropout rate used during training')
    parser.add_argument('--classifier_dropout', type=lambda x: x.lower() == 'true', default=True,
                        help='Whether classifier dropout was used')
    parser.add_argument('--cpx_aggregation', type=str, default='attention',
                        choices=['mean', 'max', 'sum', 'attention', 'first'],
                        help='Aggregation type for multiple CPX tokens')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='Number of layers (if model was sliced during training)')
    
    # Inference arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--context_window', type=int, default=1024,
                        help='Maximum context window length')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save probabilities (default: dataset_path + _probabilities.pkl)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (default: cuda)')
    
    args = parser.parse_args()
    
    # Run inference using the function
    results = run_inference(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        cpx_tokens=args.cpx_tokens,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        dropout_rate=args.dropout_rate,
        classifier_dropout=args.classifier_dropout,
        cpx_aggregation=args.cpx_aggregation,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        context_window=args.context_window,
        device=args.device,
        verbose=True
    )

    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    # Save results
    output_dir = args.output_path
    output_path = f"{output_dir}/probabilities_{time_stamp}.pkl"
    if output_path is None:
        # Default: save next to dataset file
        base_path = args.dataset_path.rsplit('.', 1)[0]
        output_path = f"{base_path}_probabilities.pkl"
    
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

