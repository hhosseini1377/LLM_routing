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
from cpx_model.dataset_loaders import get_dataset_loader
from collections import Counter
from cpx_model.train_cpx_causal import CPXTrainer
from routing_dataset.dataset_paths import AVAILABLE_DATASET_NAMES
from cpx_model.dataset_loaders import _load_imdb
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train CPX wrapper on any Causal LM')
    parser.add_argument('--data_size', type=str, default='None')
    parser.add_argument('--dataset_name', type=str, default='auxiliary',
                       help='Dataset name (e.g., "auxiliary", "combined", "gsm8k", "mix", "imdb"). Used with --dataset_model_name to load specific dataset files.')
    parser.add_argument('--dataset_model_name', type=str, default=None,
                       help='Model name used for dataset (e.g., "qwen4", "qwen17b", "qwen34b"). Used with --dataset_name to load specific dataset files.')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                       help='HuggingFace model name or path to use for training (e.g., "Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.1")')
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
    parser.add_argument('--sampling_weight_power', type=float, default=None,
                       help='Power to apply to weights for weighted sampling (1.0=standard, 0.5=sqrt=gentle, 1.5=more aggressive). '
                            'If not specified, uses class_weight_power for backward compatibility.')
    parser.add_argument('--loss_weight_power', type=float, default=None,
                       help='Power to apply to class weights in loss function (1.0=standard, 0.5=sqrt=gentle, 1.5=more aggressive). '
                            'If not specified, uses class_weight_power for backward compatibility.')
    parser.add_argument('--class_weight_power', type=float, default=0.5,
                       help='DEPRECATED: Power to apply to weights. Use --sampling_weight_power and --loss_weight_power instead. '
                            'If specified, used as fallback for both sampling and loss weights.')
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
    # ReduceLROnPlateau specific parameters
    parser.add_argument('--lr_scheduler_patience', type=int, default=3, help='Patience for ReduceLROnPlateau scheduler (default: 3)')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5, help='Factor for LR reduction in ReduceLROnPlateau (default: 0.5)')
    parser.add_argument('--lr_scheduler_min_lr', type=float, default=1e-6, help='Minimum learning rate for ReduceLROnPlateau (default: 1e-6)')
    parser.add_argument('--lr_scheduler_cooldown', type=int, default=1, help='Cooldown period after LR reduction (default: 1)')
    
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
    
    # Model saving
    parser.add_argument('--save_model', type=lambda x: x.lower() == 'true', default=True, help='Whether to save model files after training (default: True)')
    
    args = parser.parse_args()
    
    # Handle backward compatibility: if new params not specified, use class_weight_power
    sampling_weight_power = args.sampling_weight_power if args.sampling_weight_power is not None else args.class_weight_power
    loss_weight_power = args.loss_weight_power if args.loss_weight_power is not None else args.class_weight_power
    
    # Determine dataset type from dataset_name
    dataset_name = args.dataset_name
    if (dataset_name, args.dataset_model_name) not in AVAILABLE_DATASET_NAMES:
        raise ValueError(f"Unknown dataset_name='{dataset_name}', dataset_model_name='{args.dataset_model_name}'. Supported values: {AVAILABLE_DATASET_NAMES}")
    dataset_model_name = args.dataset_model_name  # Can be None
    

    # Create configuration instance with command line arguments
    training_config = CPXTrainingConfig(
        dataset=dataset_name,  # Set dataset type for logging purposes
        model_name=args.model_name,
        is_cpx_token_trainable=args.is_cpx_token_trainable,
        cpx_aggregation=args.cpx_aggregation,
        dropout_rate=args.dropout_rate,
        classifier_dropout=args.classifier_dropout,
        use_lora=args.use_lora,
        mask_lora_for_non_cpx=args.mask_lora_for_non_cpx,
        use_class_weights=args.use_class_weights,
        use_weighted_sampling=args.use_weighted_sampling,
        weighting_strategy=args.weighting_strategy,
        sampling_weight_power=sampling_weight_power,
        loss_weight_power=loss_weight_power,
        class_weight_power=args.class_weight_power,  # Keep for backward compatibility
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
        METRIC=args.metric,
        save_model=args.save_model,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_min_lr=args.lr_scheduler_min_lr,
        lr_scheduler_cooldown=args.lr_scheduler_cooldown
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
    
    # Load dataset using the appropriate loader function
    loader_func = get_dataset_loader(dataset_name)
    if dataset_name == 'imdb':
        train_texts, train_labels, _, validation_texts, validation_labels, _ = loader_func(cpx_tokens, dataset_name, dataset_model_name)
        train_dataset_sources = None
        validation_dataset_sources = None
    else:
        train_texts, train_labels, train_dataset_sources, validation_texts, validation_labels, validation_dataset_sources = \
            loader_func(cpx_tokens, dataset_name, dataset_model_name)
    
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
