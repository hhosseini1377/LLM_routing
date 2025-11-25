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
import warnings
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
    
    # CPX-specific settings
    parser.add_argument('--is_cpx_token_trainable', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--cpx_aggregation', type=str, default='mean', choices=['mean', 'max', 'sum', 'attention', 'first'])
    
    # Training settings
    parser.add_argument("--use_lora", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--mask_lora_for_non_cpx', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--classifier_dropout', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--use_class_weights', type=lambda x: x.lower() == 'true', default=False,
                       help='Enable class weighting in loss function (BCEWithLogitsLoss pos_weight). '
                            'Increases loss contribution of minority class samples.')
    parser.add_argument('--use_weighted_sampling', type=lambda x: x.lower() == 'true', default=False,
                       help='Enable weighted sampling (DistributedWeightedSampler) to oversample minority class.')
    parser.add_argument('--weighting_strategy', type=str, default='dataset_source',
                       choices=['dataset_source', 'label', 'both'],
                       help='Strategy for weighted sampling: "dataset_source" (balance datasets), '
                            '"label" (balance success/failure), or "both" (balance dataset+label combinations).')
    parser.add_argument('--class_weight_power', type=float, default=0.5,
                       help='Power to apply to weights (1.0=standard, 0.5=sqrt=gentle, 1.5=more aggressive). '
                            'Applies to both loss weighting and weighted sampling.')
    parser.add_argument('--oversample_factor', type=float, default=1.5,
                       help='Factor to multiply dataset size when generating samples in DistributedWeightedSampler. '
                            'Higher values ensure better coverage of majority class samples (default: 1.5).')

    # Learning rate arguments
    parser.add_argument('--classifier_lr', type=float, default=3e-4)
    parser.add_argument('--aggregator_lr', type=float, default=2e-4)
    parser.add_argument('--embedding_lr', type=float, default=1e-4)
    parser.add_argument('--lora_lr', type=float, default=1.5e-4)
    
    # Weight decay
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--embedding_weight_decay', type=float, default=0.0)
    
    # Training hyperparameters
    parser.add_argument('--context_window', type=int, default=1024)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['linear', 'cosine', 'ReduceLROnPlateau'])
    parser.add_argument('--warmup_steps', type=float, default=0.05)
    parser.add_argument('--gradient_checkpointing', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--amsgrad', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    
    # LoRA settings
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.15)
    parser.add_argument('--lora_target_modules', type=str, nargs='+', default=None, help='List of target modules for LoRA, e.g., --lora_target_modules q_proj o_proj gate_proj')
    parser.add_argument('--freeze_LoRA_layers', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--freeze_LoRA_start_layer_idx', type=int, default=0)
    
    # CPX tokens
    parser.add_argument('--cpx_tokens', type=str, nargs='+', default=None, help='List of CPX tokens as strings, e.g., --cpx_tokens [CPX1] [CPX2]')
    
    # Model architecture
    parser.add_argument('--num_layers', type=int, default=None, help='Number of layers to keep (if None, keeps all layers). Permanently slices the model for faster training.')
    
    # Metric used for training
    parser.add_argument('--metric', type=str, default='f1', choices=['f1', 'accuracy', 'roc_auc'])
    args = parser.parse_args()
    
    # Create configuration instance with command line arguments
    training_config = CPXTrainingConfig(
        dataset=args.dataset,
        is_cpx_token_trainable=args.is_cpx_token_trainable,
        cpx_aggregation=args.cpx_aggregation,
        dropout_rate=args.dropout_rate,
        classifier_dropout=args.classifier_dropout,
        use_lora=args.use_lora,
        mask_lora_for_non_cpx=args.mask_lora_for_non_cpx,
        use_class_weights=args.use_class_weights,
        use_weighted_sampling=args.use_weighted_sampling,
        weighting_strategy=args.weighting_strategy,
        class_weight_power=args.class_weight_power,
        oversample_factor=args.oversample_factor,
        classifier_lr=args.classifier_lr,
        aggregator_lr=args.aggregator_lr,
        embedding_lr=args.embedding_lr,
        lora_lr=args.lora_lr,
        weight_decay=args.weight_decay,
        embedding_weight_decay=args.embedding_weight_decay,
        evaluation_size=args.evaluation_size,
        context_window=args.context_window,
        scheduler=args.scheduler,
        warmup_steps=args.warmup_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        patience=args.patience,
        amsgrad=args.amsgrad,
        label_smoothing=args.label_smoothing,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        freeze_LoRA_layers=args.freeze_LoRA_layers,
        freeze_LoRA_start_layer_idx=args.freeze_LoRA_start_layer_idx,
        cpx_tokens=args.cpx_tokens,
        num_layers=args.num_layers,
        METRIC=args.metric
    )
    
    print(f"Configuration loaded - use_lora: {training_config.use_lora}")
    
    # Load tokenizer with CPX token (works with any model)
    tokenizer = CPXTokenizer.from_pretrained(training_config.model_name, cpx_tokens=training_config.cpx_tokens)
    
    # Get the CPX token ID(s) - normalize to list first
    if isinstance(training_config.cpx_tokens, str):
        raise ValueError(f"Provide a list of CPX tokens, got {type(training_config.cpx_tokens)}")

    cpx_tokens = training_config.cpx_tokens
    
    # Convert to IDs
    training_config.cpx_token_ids = tokenizer.convert_tokens_to_ids(cpx_tokens)
    
    # Load dataset
    train_dataset_sources = None  # Will be set if loading combined dataset
    if args.dataset == 'gsm8k':
        train_texts, train_labels, validation_texts, validation_labels = load_gsm8k_data_with_cpx(cpx_tokens)
    elif args.dataset == 'mmlu':
        train_texts, train_labels, validation_texts, validation_labels = load_mmlu_data_with_cpx(cpx_tokens)
    elif args.dataset == 'mix':
        train_texts, train_labels, validation_texts, validation_labels = load_mix_data_with_cpx(cpx_tokens)
    elif args.dataset == 'imdb':
        from cpx_model.cpx_causal_utils import load_imdb_data_with_cpx
        train_texts, train_labels, validation_texts, validation_labels = load_imdb_data_with_cpx(cpx_tokens)
    elif args.dataset == 'combined':
        # Load combined dataset (MMLU + MMLU-Pro + GSM8K) with dataset sources
        from cpx_model.cpx_causal_utils import load_combined_data_with_cpx
        from routing_dataset.dataset_paths import DATA_DIR
        train_path = DATA_DIR / "final_splits" / "train.pkl"
        validation_path = DATA_DIR / "final_splits" / "val.pkl"
        train_texts, train_labels, train_dataset_sources, validation_texts, validation_labels, validation_dataset_sources = \
            load_combined_data_with_cpx(str(train_path), str(validation_path), cpx_tokens)
        print(f"Loaded combined dataset with dataset sources")
        if train_dataset_sources:
            from collections import Counter
            source_counts = Counter(train_dataset_sources)
            print(f"  Training dataset sources: {dict(source_counts)}")
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    print('Dataset Loaded')
    # Filter dataset based on arguments
    if args.data_size != 'None':
        train_texts = train_texts[:int(args.data_size)]
        train_labels = train_labels[:int(args.data_size)]
        if train_dataset_sources:
            train_dataset_sources = train_dataset_sources[:int(args.data_size)]
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
        training_config=training_config,
        train_dataset_sources=train_dataset_sources  # Pass dataset sources for weighting
    )
        
    # Compute the batch size per GPU
    per_gpu_batch_size = args.batch_size // torch.cuda.device_count() if torch.cuda.is_available() else args.batch_size
    trainer.run(batch_size=per_gpu_batch_size, context_window=args.context_window, num_epochs=args.num_epochs, model_name=training_config.model_name)
