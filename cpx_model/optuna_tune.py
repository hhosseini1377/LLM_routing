"""
Optuna hyperparameter optimization for CPX model training.
"""
import argparse
import optuna
from optuna.trial import TrialState
import torch
import warnings
from cpx_model.config import CPXTrainingConfig
from cpx_model.cpx_causal_tokenizer import CPXTokenizer
from cpx_model.cpx_causal_utils import load_mmlu_data_with_cpx, load_gsm8k_data_with_cpx, load_mix_data_with_cpx
from cpx_model.train_cpx_causal import CPXTrainer
import os
import json


def objective(trial, args, train_texts, train_labels, validation_texts, validation_labels, tokenizer, cpx_tokens, model_name):
    """
    Optuna objective function for hyperparameter optimization.
    
    Returns the best validation score (F1 or negative loss depending on metric).
    """
    # Suggest hyperparameters
    if args.tune_learning_rates:
        classifier_lr = trial.suggest_float('classifier_lr', 5e-5, 3e-3, log=True)
        aggregator_lr = trial.suggest_float('aggregator_lr', 3e-5, 2e-4, log=True)
        embedding_lr = trial.suggest_float('embedding_lr', 5e-5, 1e-4, log=True)
        lora_lr = trial.suggest_float('lora_lr', 1e-5, 2e-4, log=True)
    else:
        classifier_lr = args.classifier_lr
        aggregator_lr = args.aggregator_lr
        embedding_lr = args.embedding_lr
        lora_lr = args.lora_lr
    
    if args.tune_lora:
        lora_r = trial.suggest_int('lora_r', 4, 64, step=4)
        lora_alpha = trial.suggest_int('lora_alpha', lora_r, lora_r * 4, step=4)
        lora_dropout = trial.suggest_float('lora_dropout', 0.0, 0.5)
    else:
        lora_r = args.lora_r
        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
    
    if args.tune_dropout:
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3)
        classifier_dropout = trial.suggest_categorical('classifier_dropout', [True, False])
    else:
        dropout_rate = args.dropout_rate
        classifier_dropout = args.classifier_dropout
    
    if args.tune_optimizer:
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 0.1, log=True)
        warmup_steps = trial.suggest_float('warmup_steps', 0.0, 0.2)
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.3)
    else:
        weight_decay = args.weight_decay
        warmup_steps = args.warmup_steps
        label_smoothing = args.label_smoothing
    
    if args.tune_aggregation:
        cpx_aggregation = trial.suggest_categorical('cpx_aggregation', ['mean', 'max', 'sum', 'attention', 'first'])
    else:
        cpx_aggregation = args.cpx_aggregation
    
    # Suggest LoRA target modules if tuning enabled
    if args.tune_target_modules:
        # Define common module combinations for Optuna to choose from
        # Each combination is a tuple that will be converted to a list
        module_combinations = [
            # ['q_proj', 'o_proj'],  # Attention-focused
            # ['q_proj', 'o_proj', 'gate_proj'],  # Attention + gate
            ['gate_proj', 'up_proj', 'down_proj'],  # MLP-focused
            ['q_proj', 'gate_proj', 'up_proj', 'down_proj'],  # Attention + MLP
            # ['q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],  # Full attention + MLP
            ['q_proj', 'o_proj', 'gate_proj', 'up_proj'],  # Most common
            # ['q_proj', 'gate_proj', 'up_proj'],  # Minimal effective
            # ['q_proj', 'o_proj', 'up_proj', 'down_proj'],  # Attention + MLP (no gate)
        ]
        # Convert to list of tuples for Optuna (Optuna needs hashable types)
        module_combinations_tuples = [tuple(combo) for combo in module_combinations]
        selected_modules_tuple = trial.suggest_categorical('lora_target_modules', module_combinations_tuples)
        lora_target_modules = list(selected_modules_tuple)
    else:
        lora_target_modules = args.lora_target_modules
    
    # Create training config with suggested hyperparameters
    training_config = CPXTrainingConfig(
        dataset=args.dataset,
        is_cpx_token_trainable=args.is_cpx_token_trainable,
        cpx_aggregation=cpx_aggregation,
        dropout_rate=dropout_rate,
        classifier_dropout=classifier_dropout,
        use_lora=args.use_lora,
        mask_lora_for_non_cpx=args.mask_lora_for_non_cpx,
        use_class_weights=args.use_class_weights,
        class_weight_power=args.class_weight_power,
        classifier_lr=classifier_lr,
        aggregator_lr=aggregator_lr,
        embedding_lr=embedding_lr,
        lora_lr=lora_lr,
        weight_decay=weight_decay,
        embedding_weight_decay=args.embedding_weight_decay,
        evaluation_size=args.evaluation_size,
        context_window=args.context_window,
        scheduler=args.scheduler,
        warmup_steps=warmup_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        patience=args.patience,
        amsgrad=args.amsgrad,
        label_smoothing=label_smoothing,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        freeze_LoRA_layers=args.freeze_LoRA_layers,
        freeze_LoRA_start_layer_idx=args.freeze_LoRA_start_layer_idx,
        lora_target_modules=lora_target_modules,
        cpx_tokens=args.cpx_tokens
    )
    
    # Set CPX token IDs
    training_config.cpx_token_ids = tokenizer.convert_tokens_to_ids(cpx_tokens)
    
    # Create trainer
    trainer = CPXTrainer(
        tokenizer=tokenizer,
        train_texts=train_texts,
        train_labels=train_labels,
        validation_texts=validation_texts,
        validation_labels=validation_labels,
        training_config=training_config
    )
    
    # Run training and get best score
    try:
        # Compute the batch size per GPU
        per_gpu_batch_size = args.batch_size // torch.cuda.device_count() if torch.cuda.is_available() else args.batch_size
        
        # Train the model (this will return the best validation score)
        best_score = trainer.train_with_optuna(
            batch_size=per_gpu_batch_size,
            context_window=args.context_window,
            num_epochs=args.num_epochs,
            model_name=model_name,
            trial=trial
        )
        
        # Return score based on metric (maximize F1, minimize loss)
        if training_config.METRIC == "f1":
            return best_score  # Maximize F1
        else:
            return -best_score  # Minimize loss (return negative for Optuna maximization)
            
    except Exception as e:
        print(f"Trial failed with error: {e}")
        # Return a poor score if training fails
        if training_config.METRIC == "f1":
            return 0.0
        else:
            return float('inf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optuna hyperparameter optimization for CPX model')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='gsm8k', choices=['gsm8k', 'mmlu', 'mix'])
    parser.add_argument('--data_size', type=str, default='None')
    parser.add_argument('--evaluation_size', type=str, default='None')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--context_window', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=5)
    
    # Optuna arguments
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--study_name', type=str, default='cpx_optuna_study', help='Name for Optuna study')
    parser.add_argument('--storage', type=str, default=None, help='Optuna storage URL (e.g., sqlite:///optuna.db)')
    parser.add_argument('--direction', type=str, default='auto', choices=['maximize', 'minimize', 'auto'], help='Auto uses maximize for F1, minimize for loss')
    parser.add_argument('--load_if_exists', action='store_true', help='Load existing study if it exists')
    
    # Tuning flags - control which hyperparameters to tune
    parser.add_argument('--tune_learning_rates', action='store_true', help='Tune learning rates')
    parser.add_argument('--tune_lora', action='store_true', help='Tune LoRA hyperparameters (r, alpha, dropout)')
    parser.add_argument('--tune_target_modules', action='store_true', help='Tune LoRA target modules (which modules to apply LoRA to)')
    parser.add_argument('--tune_dropout', action='store_true', help='Tune dropout rates')
    parser.add_argument('--tune_optimizer', action='store_true', help='Tune optimizer hyperparameters')
    parser.add_argument('--tune_aggregation', action='store_true', help='Tune aggregation method')
    
    # Fixed hyperparameters (used when not tuning)
    parser.add_argument('--is_cpx_token_trainable', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--cpx_aggregation', type=str, default='mean', choices=['mean', 'max', 'sum', 'attention', 'first'])
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--classifier_dropout', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--use_lora', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--mask_lora_for_non_cpx', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--use_class_weights', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--class_weight_power', type=float, default=0.5)
    parser.add_argument('--classifier_lr', type=float, default=3e-4)
    parser.add_argument('--aggregator_lr', type=float, default=2e-4)
    parser.add_argument('--embedding_lr', type=float, default=1e-4)
    parser.add_argument('--lora_lr', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--embedding_weight_decay', type=float, default=0.0)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['linear', 'cosine', 'ReduceLROnPlateau'])
    parser.add_argument('--warmup_steps', type=float, default=0.05)
    parser.add_argument('--gradient_checkpointing', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--amsgrad', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.15)
    parser.add_argument('--freeze_LoRA_layers', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--freeze_LoRA_start_layer_idx', type=int, default=0)
    parser.add_argument('--lora_target_modules', type=str, nargs='+', default=None)
    parser.add_argument('--cpx_tokens', type=str, nargs='+', default=None)
    
    args = parser.parse_args()
    
    # Load tokenizer
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # Default model
    tokenizer = CPXTokenizer.from_pretrained(model_name, cpx_tokens=args.cpx_tokens)
    
    cpx_tokens = args.cpx_tokens
    
    # Load dataset
    if args.dataset == 'gsm8k':
        train_texts, train_labels, validation_texts, validation_labels = load_gsm8k_data_with_cpx(cpx_tokens)
    elif args.dataset == 'mmlu':
        train_texts, train_labels, validation_texts, validation_labels = load_mmlu_data_with_cpx(cpx_tokens)
    elif args.dataset == 'mix':
        train_texts, train_labels, validation_texts, validation_labels = load_mix_data_with_cpx(cpx_tokens)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    # Filter dataset if needed
    if args.data_size != 'None':
        train_texts = train_texts[:int(args.data_size)]
        train_labels = train_labels[:int(args.data_size)]
    
    if args.evaluation_size != 'None':
        validation_texts = validation_texts[:int(args.evaluation_size)]
        validation_labels = validation_labels[:int(args.evaluation_size)]
    
    print(f"Dataset loaded: {len(train_texts)} train, {len(validation_texts)} validation")
    
    # Create Optuna study
    # Set metric direction based on dataset
    if args.direction == 'auto':
        # Default: maximize F1, minimize loss
        direction = 'maximize'  # F1 is typically the metric
    else:
        direction = args.direction
    
    # Optional: Customize sampler (default is TPE)
    # Uncomment to use a different sampler:
    # from optuna.samplers import RandomSampler, CmaEsSampler
    # sampler = RandomSampler()  # Random search
    # sampler = CmaEsSampler()   # CMA-ES for continuous optimization
    sampler = None  # Use default TPE sampler
    
    # Optional: Add pruning (stops unpromising trials early)
    # from optuna.pruners import MedianPruner
    # pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    pruner = None  # Use default pruner (or None to disable)
    
    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction=direction,
            load_if_exists=args.load_if_exists,
            sampler=sampler,
            pruner=pruner
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )
    
    # Run optimization
    print(f"Starting Optuna optimization with {args.n_trials} trials...")
    study.optimize(
        lambda trial: objective(trial, args, train_texts, train_labels, validation_texts, validation_labels, tokenizer, cpx_tokens, model_name),
        n_trials=args.n_trials,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*50)
    print("Optimization finished!")
    print("="*50)
    print(f"Number of finished trials: {len(study.trials)}")
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print(f"Number of pruned trials: {len(pruned_trials)}")
    print(f"Number of complete trials: {len(complete_trials)}")
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("\n  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best parameters to JSON
    # Convert tuples to lists for JSON serialization (e.g., lora_target_modules)
    best_params = {}
    for key, value in trial.params.items():
        if isinstance(value, tuple):
            best_params[key] = list(value)  # Convert tuple to list for JSON
        else:
            best_params[key] = value
    
    best_params_path = f"optuna_best_params_{args.study_name}.json"
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest parameters saved to {best_params_path}")
    
    # Save study to file if using SQLite storage
    if args.storage:
        print(f"Study saved to {args.storage}")

