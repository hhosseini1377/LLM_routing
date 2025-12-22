"""
Inference script for BERT models.

This script loads a trained BERT model and runs inference on a dataset to get probabilities.

Usage:
    python inference.py --model_path <path_to_model.pth> --dataset_path <path_to_dataset.pkl> --model_name <model_name> [options]
"""

import argparse
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
from transformers import DistilBertTokenizer, AutoTokenizer

from bert_routing.regression_models import TextRegressionDataset, TruncatedModel
from bert_routing.config import TrainingConfig
from cpx_model.cpx_causal_utils import load_pickle_data
from datasets import Dataset as HFDataset
import time
from tqdm import tqdm
import pandas

def load_model_from_checkpoint(
    model_path: str,
    model_name: str,
    pooling_strategy: str,
    num_outputs: int = 1,
    num_classes: int = 2,
    training_config: TrainingConfig = None,
    device: str = 'cuda'
):
    """
    Load a BERT model from a checkpoint file.
    
    Args:
        model_path: Path to the model checkpoint (.pth file)
        model_name: Name of the model (e.g., "deberta", "distilbert", "bert", "tinybert")
        pooling_strategy: Pooling strategy used during training (e.g., "cls", "mean", "max", "attention")
        num_outputs: Number of output dimensions (default: 1 for binary classification)
        num_classes: Number of classes (default: 2)
        training_config: TrainingConfig object (if None, creates default config)
        device: Device to load model on
    
    Returns:
        model: Loaded BERT model
    """
    # Load model state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Create default training config if not provided
    if training_config is None:
        training_config = TrainingConfig(
            model_name=model_name,
            dropout_rate=0.1,
            classifier_dropout=True,
            classifier_type="linear"
        )
    
    # Initialize model with same config as training
    model = TruncatedModel(
        num_outputs=num_outputs,
        num_classes=num_classes,
        model_name=model_name,
        pooling_strategy=pooling_strategy,
        training_config=training_config
    )
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def load_tokenizer(model_name: str, context_window: int = 512):
    """
    Load tokenizer for the specified model.
    
    Args:
        model_name: Name of the model (e.g., "deberta", "distilbert", "bert", "tinybert")
        context_window: Maximum context window length
    
    Returns:
        tokenizer: Loaded tokenizer
    """
    if model_name == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased",
            max_length=context_window,
            truncation_side="left",
            clean_up_tokenization_spaces=False
        )
    elif model_name == "deberta":
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/deberta-v3-large",
            max_length=context_window,
            truncation_side="left",
            clean_up_tokenization_spaces=False
        )
    elif model_name == "tinybert":
        tokenizer = AutoTokenizer.from_pretrained(
            "huawei-noah/TinyBERT_General_6L_768D",
            max_length=context_window,
            truncation_side="left",
            clean_up_tokenization_spaces=False
        )
    elif model_name == "bert":
        tokenizer = AutoTokenizer.from_pretrained(
            'microsoft/deberta-v3-base',
            max_length=context_window,
            truncation_side="left",
            clean_up_tokenization_spaces=False
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    return tokenizer


def load_dataset_from_pickle(dataset_path: str):
    """
    Load dataset from pickle file.
    
    Args:
        dataset_path: Path to the dataset pickle file
    
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
        data = data.to_dict(orient='list')
        texts = data['prompts']
        labels = data['correct_labels']
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")
    return texts, labels


def get_probabilities(
    model: TruncatedModel,
    tokenizer,
    texts: list,
    batch_size: int = 32,
    context_window: int = 512,
    device: str = 'cuda'
):
    """
    Get probabilities for a list of texts.
    
    Args:
        model: BERT model
        tokenizer: Tokenizer
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing dataset"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            with autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            
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
    pooling_strategy: str = 'cls',
    num_outputs: int = 1,
    num_classes: int = 2,
    dropout_rate: float = 0.1,
    classifier_dropout: bool = True,
    classifier_type: str = 'linear',
    batch_size: int = 32,
    context_window: int = 512,
    device: str = 'cuda',
    verbose: bool = True
):
    """
    Run inference with BERT model and return probabilities.
    
    Args:
        model_path: Path to the model checkpoint (.pth file)
        dataset_path: Path to the dataset pickle file
        model_name: Model name (e.g., "deberta", "distilbert", "bert", "tinybert")
        pooling_strategy: Pooling strategy used during training (default: "cls")
        num_outputs: Number of output dimensions (default: 1)
        num_classes: Number of classes (default: 2)
        dropout_rate: Dropout rate used during training (default: 0.1)
        classifier_dropout: Whether classifier dropout was used (default: True)
        classifier_type: Classifier type ("linear" or "mlp", default: "linear")
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
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        if verbose:
            print("CUDA not available, using CPU")
        device = 'cpu'
    
    if verbose:
        print(f"Loading tokenizer...")
    # Load tokenizer
    tokenizer = load_tokenizer(model_name, context_window)
    
    if verbose:
        print(f"Loading model from {model_path}...")
    # Create training config
    training_config = TrainingConfig(
        model_name=model_name,
        dropout_rate=dropout_rate,
        classifier_dropout=classifier_dropout,
        classifier_type=classifier_type
    )
    
    # Load model
    model = load_model_from_checkpoint(
        model_path=model_path,
        model_name=model_name,
        pooling_strategy=pooling_strategy,
        num_outputs=num_outputs,
        num_classes=num_classes,
        training_config=training_config,
        device=device
    )
    if verbose:
        print("Model loaded successfully!")
    
    if verbose:
        print(f"Loading dataset from {dataset_path}...")
    # Load dataset
    texts, labels = load_dataset_from_pickle(dataset_path)
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
    parser = argparse.ArgumentParser(description='Run inference with BERT model')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint (.pth file)')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the dataset pickle file')
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['deberta', 'distilbert', 'bert', 'tinybert'],
                        help='Model name (deberta, distilbert, bert, tinybert)')
    
    # Model configuration arguments
    parser.add_argument('--pooling_strategy', type=str, default='cls',
                        choices=['cls', 'mean', 'max', 'attention', 'last'],
                        help='Pooling strategy used during training (default: cls)')
    parser.add_argument('--num_outputs', type=int, default=1,
                        help='Number of output dimensions (default: 1)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes (default: 2)')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='Dropout rate used during training (default: 0.1)')
    parser.add_argument('--classifier_dropout', type=lambda x: x.lower() == 'true', default=True,
                        help='Whether classifier dropout was used (default: True)')
    parser.add_argument('--classifier_type', type=str, default='linear',
                        choices=['linear', 'mlp'],
                        help='Classifier type: linear or mlp (default: linear)')
    
    # Inference arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference (default: 32)')
    parser.add_argument('--context_window', type=int, default=512,
                        help='Maximum context window length (default: 512)')
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
        pooling_strategy=args.pooling_strategy,
        num_outputs=args.num_outputs,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        classifier_dropout=args.classifier_dropout,
        classifier_type=args.classifier_type,
        batch_size=args.batch_size,
        context_window=args.context_window,
        device=args.device,
        verbose=True
    )
    
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    # Save results
    if args.output_path is None:
        # Default: save next to dataset file
        base_path = args.dataset_path.rsplit('.', 1)[0]
        output_path = f"{base_path}_probabilities_{time_stamp}.pkl"
    else:
        output_path = f"{args.output_path}/probabilities_{time_stamp}.pkl"
    
    # Prepare results for saving (exclude model and tokenizer)
    save_results = {
        'probabilities': results['probabilities'],
        'texts': results['texts'],
        'labels': results['labels'],
        'model_path': args.model_path,
        'dataset_path': args.dataset_path,
        'model_name': args.model_name,
        'pooling_strategy': args.pooling_strategy,
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

