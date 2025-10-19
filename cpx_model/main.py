"""
Example usage of the general CPX wrapper for Causal Language Models.

This example shows how to use the CPX wrapper with Mistral, but you can easily
swap "mistralai/Mistral-7B-Instruct-v0.1" with any other causal LM like:
- "meta-llama/Llama-2-7b-hf"
- "gpt2"
- "microsoft/phi-2"
- "tiiuae/falcon-7b"
- etc.
"""
import argparse
import gc
from itertools import product
import torch

from cpx_model.config import CPXTrainingConfig
from cpx_model.cpx_causal_tokenizer import CPXTokenizer
from cpx_model.cpx_causal_lm import CPXCausalLM
from cpx_model.train_cpx_causal import MistralTrainer
from cpx_model.cpx_causal_utils import load_mmlu_data_with_cpx


if __name__ == "__main__":
    # Configuration
    cpx_token = CPXTrainingConfig.cpx_token
    
    # You can change this to any Causal LM!
    # Examples:
    # - "mistralai/Mistral-7B-Instruct-v0.1"  (current)
    # - "meta-llama/Llama-2-7b-hf"
    # - "meta-llama/Meta-Llama-3-8B"
    # - "gpt2"
    # - "microsoft/phi-2"
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    
    # Load tokenizer with CPX token (works with any model)
    tokenizer = CPXTokenizer.from_pretrained(model_name, cpx_token=cpx_token)
    
    # Get the CPX token ID
    cpx_token_id = tokenizer.convert_tokens_to_ids(cpx_token)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train CPX wrapper on any Causal LM')
    parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.1',
                       help='HuggingFace model name or path (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")')
    parser.add_argument('--data_size', type=str, default='None')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--context_window', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--evaluation_size', type=str, default='None')
    
    args = parser.parse_args()
    
    # Update model_name if provided via command line
    if args.model_name:
        model_name = args.model_name
    
    # Load dataset
    train_texts, train_labels, test_texts, test_labels = load_mmlu_data_with_cpx()
    print('Dataset Loaded')

    # Filter dataset based on arguments
    if args.data_size != 'None':
        train_texts = train_texts[:int(args.data_size)]
        train_labels = train_labels[:int(args.data_size)]
        
    if args.evaluation_size != 'None':
        test_texts = test_texts[:int(args.evaluation_size)]
        test_labels = test_labels[:int(args.evaluation_size)]
    
    # Set CPX token ID in config
    CPXTrainingConfig.cpx_token_id = cpx_token_id
    
    # Grid search over hyperparameters
    dropout_rate = [0.1, 0.3]
    layers_to_freeze_options = [0, 2, 4]

    grid = product(dropout_rate, layers_to_freeze_options)
    for do_rate, layers in grid:
        
        CPXTrainingConfig.dropout_rate = do_rate
        CPXTrainingConfig.layers_to_freeze = layers

        # Create trainer (note: trainer code may need updates to use CPXCausalLM)
        trainer = MistralTrainer(
            tokenizer=tokenizer,
            train_texts=train_texts,
            train_labels=train_labels,
            test_texts=test_texts,
            test_labels=test_labels
        )
            
        # Compute the batch size per GPU
        per_gpu_batch_size = args.batch_size // torch.cuda.device_count() if torch.cuda.is_available() else args.batch_size
        trainer.run(batch_size=per_gpu_batch_size, context_window=args.context_window, num_epochs=args.num_epochs)

        # Clean up to avoid GPU memory leak
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        break
