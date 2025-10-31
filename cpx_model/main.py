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
import torch

from cpx_model.config import CPXTrainingConfig
from cpx_model.cpx_causal_tokenizer import CPXTokenizer
from cpx_model.cpx_causal_utils import load_mmlu_data_with_cpx, load_gsm8k_data_with_cpx, load_mix_data_with_cpx
from cpx_model.train_cpx_causal import CPXTrainer

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train CPX wrapper on any Causal LM')
    parser.add_argument('--data_size', type=str, default='None')
    parser.add_argument('--dataset', type=str, default='gsm8k')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--evaluation_size', type=str, default='None')
    parser.add_argument("--use_lora", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--mask_lora_for_non_cpx', type=lambda x: x.lower() == 'true', default=False)

    # Add learning rate arguments
    parser.add_argument('--classifier_lr', type=float, default=5e-4)
    parser.add_argument('--embedding_lr', type=float, default=1e-4)
    parser.add_argument('--lora_lr', type=float, default=2e-4)
    parser.add_argument('--scheduler', type=str, default='linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--classifier_dropout', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--freeze_LoRA_layers', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--freeze_LoRA_start_layer_idx', type=int, default=0)
    args = parser.parse_args()
    
    # Create configuration instance with command line arguments
    training_config = CPXTrainingConfig(
        classifier_lr=args.classifier_lr,
        embedding_lr=args.embedding_lr,
        lora_lr=args.lora_lr,
        use_lora=args.use_lora,
        scheduler=args.scheduler,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout_rate,
        classifier_dropout=args.classifier_dropout,
        mask_lora_for_non_cpx=args.mask_lora_for_non_cpx,
        max_grad_norm=args.max_grad_norm,
        dataset=args.dataset,
        freeze_LoRA_layers=args.freeze_LoRA_layers,
        freeze_LoRA_start_layer_idx=args.freeze_LoRA_start_layer_idx
    )
    
    print(f"Configuration loaded - use_lora: {training_config.use_lora}")
    
    # Load tokenizer with CPX token (works with any model)
    tokenizer = CPXTokenizer.from_pretrained(training_config.model_name, cpx_token=training_config.cpx_token)
    
    # Get the CPX token ID
    training_config.cpx_token_id = tokenizer.convert_tokens_to_ids(training_config.cpx_token)
    
    # Load dataset
    if args.dataset == 'gsm8k':
        train_texts, train_labels, validation_texts, validation_labels = load_gsm8k_data_with_cpx()
    elif args.dataset == 'mmlu':
        train_texts, train_labels, validation_texts, validation_labels = load_mmlu_data_with_cpx()
    elif args.dataset == 'mix':
        train_texts, train_labels, validation_texts, validation_labels = load_mix_data_with_cpx()
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    print('Dataset Loaded')

    # Filter dataset based on arguments
    if args.data_size != 'None':
        train_texts = train_texts[:int(args.data_size)]
        train_labels = train_labels[:int(args.data_size)]
        
    if args.evaluation_size != 'None':
        validation_texts = validation_texts[:int(args.evaluation_size)]
        validation_labels = validation_labels[:int(args.evaluation_size)]

    # Create trainer (note: trainer code may need updates to use CPXCausalLM)
    trainer = CPXTrainer(
        tokenizer=tokenizer,
        train_texts=train_texts,
        train_labels=train_labels,
        validation_texts=validation_texts,
        validation_labels=validation_labels,
        training_config=training_config
    )
        
    # Compute the batch size per GPU
    per_gpu_batch_size = args.batch_size // torch.cuda.device_count() if torch.cuda.is_available() else args.batch_size
    trainer.run(batch_size=per_gpu_batch_size, context_window=training_config.context_window, num_epochs=args.num_epochs, model_name=training_config.model_name)
