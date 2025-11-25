"""
Script to get hidden states from a normal Mistral 7B model (without CPX tokens).

This script loads texts, runs a forward pass with AutoModelForCausalLM,
and returns/saves the hidden states for comparison with CPX model hidden states.
"""

import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.amp import autocast

from cpx_model.inference import load_dataset_from_pickle


def get_hidden_states(model, tokenizer, texts, device='cuda', max_length=1024):
    """
    Run forward pass and get hidden states from the model.
    
    Args:
        model: AutoModelForCausalLM model
        tokenizer: Tokenizer
        texts: List of text strings
        device: Device to run on
        max_length: Maximum sequence length
    
    Returns:
        hidden_states: [batch_size, seq_len, hidden_dim] hidden states tensor
        input_ids: Tokenized input IDs
        attention_mask: Attention masks
    """
    # Tokenize texts
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    # Run forward pass
    model.eval()
    with torch.no_grad():
        with autocast('cuda', dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
    
    # Get hidden states from last layer
    hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
    
    return hidden_states, input_ids, attention_mask


def main():
    parser = argparse.ArgumentParser(description='Get hidden states from normal Mistral 7B model')
    
    # Required arguments
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the dataset pickle file')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Base model name (e.g., mistralai/Mistral-7B-Instruct-v0.3)')
    
    # Optional arguments
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum sequence length (default: 1024)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (default: cuda)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to process (default: all)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load tokenizer - match CPXTokenizer setup exactly (without CPX tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Set padding token if not already set (matches CPXTokenizer line 39-40)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Match CPXTokenizer settings exactly (lines 42-46)
    # ALWAYS set padding side to RIGHT for training
    # Left padding is for generation/inference, right padding is for training
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    tokenizer.truncation = True
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        output_hidden_states=True
    )
    model.eval()
    
    # Load dataset WITHOUT CPX tokens
    texts, labels = load_dataset_from_pickle(args.dataset_path, cpx_tokens=None)
    
    # Limit number of samples if specified
    if args.num_samples is not None:
        texts = texts[:args.num_samples]
    
    # Verify texts are loaded correctly
    print(f"Loaded {len(texts)} texts")
    print(f"Texts type: {type(texts)}")
    if len(texts) > 0:
        print(f"First text length: {len(texts[0])} characters")
    
    # Get hidden states
    hidden_states, input_ids, attention_mask = get_hidden_states(
        model, tokenizer, texts, device=args.device, max_length=args.max_length
    )
    
    # Verify shapes
    print(f"Input IDs shape: {input_ids.shape}")  # Should be [batch_size, seq_len]
    print(f"Hidden states shape: {hidden_states.shape}")  # Should be [batch_size, seq_len, hidden_dim]
    
    # Print only hidden states (convert bfloat16 to float32 for numpy compatibility)
    # hidden_states is already [batch_size, seq_len, hidden_dim], no need for [-1]
    print(hidden_states.cpu().float().numpy())


if __name__ == "__main__":
    main()
