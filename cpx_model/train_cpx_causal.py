from cpx_model.config import CPXTrainingConfig
import torch
from torch.utils.data import DataLoader
from cpx_model.cpx_causal_utils import TextRegressionDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, average_precision_score, brier_score_loss, log_loss
from transformers import get_scheduler
from torch.amp import autocast
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from cpx_model.distributed_weighted_sampler import (
    DistributedWeightedSampler, 
    compute_sample_weights_from_labels,
    compute_sample_weights_from_dataset_source,
    compute_sample_weights_from_combination
)
from cpx_model.cpx_causal_lm import CPXCausalLM
import time
import random
import numpy as np
from peft import LoraConfig
from tqdm import tqdm
# Optional Optuna import
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

class CPXTrainer:
    def __init__(self, tokenizer, train_texts, train_labels, validation_texts, validation_labels, training_config, train_dataset_sources=None):
        """
        Initialize CPX Trainer.
        
        Args:
            tokenizer: The tokenizer to use for text processing
            train_texts: List of training text samples
            train_labels: List of training labels
            validation_texts: List of validation text samples  
            validation_labels: List of validation labels
            training_config: CPXTrainingConfig instance with training parameters
            train_dataset_sources: Optional list of dataset source names (e.g., ["MMLU", "MMLU-Pro", "GSM8K"])
                                 If provided, enables dataset-source weighting instead of class-label weighting
        """
        self.tokenizer = tokenizer
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.validation_texts = validation_texts
        self.validation_labels = validation_labels
        self.training_config = training_config
        self.train_dataset_sources = train_dataset_sources  # Dataset source for each training sample
        self.world_size = torch.cuda.device_count()
        self.optuna_trial = None  # Will be set if using Optuna (main process only)
        self.best_score = None  # Store best score for Optuna
        self.optuna_score_file = None  # File path to save best score for Optuna
        self.optuna_intermediate_scores_file = None  # File to save intermediate scores for Optuna

    def setup(self, rank):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29503"
        
        # Set seeds BEFORE any model initialization
        torch.manual_seed(42)  # Same seed for all ranks!
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        random.seed(42)
        
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl", init_method="env://",
            world_size=self.world_size, rank=rank
        )

    def cleanup(self):
        dist.destroy_process_group()

    def preprocess_data(self, context_window, rank, batch_size, use_weighted_sampling=False):
        """
        Preprocess data and create DataLoader.
        
        Args:
            context_window: Maximum sequence length
            rank: Current process rank
            batch_size: Batch size per process
            use_weighted_sampling: If True, use DistributedWeightedSampler
                                  - If train_dataset_sources is available: weight by dataset source
                                  - Otherwise: weight by class labels
                                  (requires training_config.use_weighted_sampling to be True)
        
        Returns:
            DataLoader instance
        """
        dataset = TextRegressionDataset(self.train_texts, self.train_labels, self.tokenizer, context_window)
        
        # Collect weighted sampling info for logging
        self.sampling_info = None
        sample_weights = None
        if use_weighted_sampling:
            # Convert labels to list if tensor
            if isinstance(self.train_labels, torch.Tensor):
                labels_list = self.train_labels.squeeze().tolist()
            else:
                labels_list = self.train_labels
            
            weighting_strategy = self.training_config.weighting_strategy
            
            # Select weighting strategy based on config and available data
            if weighting_strategy == "both":
                # Weight by combination of dataset source AND label
                if self.train_dataset_sources is not None and len(self.train_dataset_sources) == len(self.train_texts):
                    sample_weights = compute_sample_weights_from_combination(
                        self.train_dataset_sources,
                        labels_list,
                        combination_weight_power=self.training_config.class_weight_power
                    )
                    
                    if rank == 0:
                        from collections import Counter
                        # Count combinations
                        combinations = [(source, label) for source, label in zip(self.train_dataset_sources, labels_list)]
                        combo_counts = Counter(combinations)
                        print(f"  ✓ Using DistributedWeightedSampler with combination weighting (dataset_source + label):")
                        # Get unique weights for each combination
                        combo_weights = {}
                        for i, combo in enumerate(combinations):
                            if combo not in combo_weights:
                                combo_weights[combo] = sample_weights[i]
                        
                        # Build info string
                        info_lines = ["Weighted sampling (combination: dataset_source + label):"]
                        for combo, count in sorted(combo_counts.items()):
                            weight = combo_weights.get(combo, 1.0)
                            source, label = combo
                            label_str = "Success" if label == 1 else "Fail"
                            print(f"    {source} {label_str}: {count} samples, weight={weight:.3f}")
                            info_lines.append(f"  {source} {label_str}: {count} samples, weight={weight:.3f}")
                        self.sampling_info = "\n".join(info_lines)
                else:
                    if rank == 0:
                        print(f"  ⚠ Warning: 'both' weighting strategy requires dataset sources, falling back to 'label'")
                    # Fallback to label weighting
                    sample_weights = compute_sample_weights_from_labels(
                        labels_list,
                        class_weight_power=self.training_config.class_weight_power
                    )
                    if rank == 0:
                        print(f"  ✓ Using DistributedWeightedSampler with class-label weighting")
                        from collections import Counter
                        label_counts = Counter(labels_list)
                        info_lines = ["Weighted sampling (label):"]
                        for label, count in sorted(label_counts.items()):
                            label_str = "Success" if label == 1 else "Fail"
                            # Get weight for this label
                            weight = sample_weights[labels_list.index(label)] if label in labels_list else 1.0
                            info_lines.append(f"  Class {label} ({label_str}): {count} samples, weight={weight:.3f}")
                        self.sampling_info = "\n".join(info_lines)
            
            elif weighting_strategy == "dataset_source":
                # Weight by dataset source (MMLU, MMLU-Pro, GSM8K)
                if self.train_dataset_sources is not None and len(self.train_dataset_sources) == len(self.train_texts):
                    sample_weights = compute_sample_weights_from_dataset_source(
                        self.train_dataset_sources,
                        dataset_weight_power=self.training_config.class_weight_power
                    )
                    
                    if rank == 0:
                        from collections import Counter
                        source_counts = Counter(self.train_dataset_sources)
                        print(f"  ✓ Using DistributedWeightedSampler with dataset-source weighting:")
                        # Get unique weights for each source
                        source_weights = {}
                        for i, source in enumerate(self.train_dataset_sources):
                            if source not in source_weights:
                                source_weights[source] = sample_weights[i]
                        
                        # Build info string
                        info_lines = ["Weighted sampling (dataset_source):"]
                        for source, count in sorted(source_counts.items()):
                            weight = source_weights.get(source, 1.0)
                            print(f"    {source}: {count} samples, weight={weight:.3f}")
                            info_lines.append(f"  {source}: {count} samples, weight={weight:.3f}")
                        self.sampling_info = "\n".join(info_lines)
                else:
                    if rank == 0:
                        print(f"  ⚠ Warning: 'dataset_source' weighting strategy requires dataset sources, falling back to 'label'")
                    # Fallback to label weighting
                    sample_weights = compute_sample_weights_from_labels(
                        labels_list,
                        class_weight_power=self.training_config.class_weight_power
                    )
                    if rank == 0:
                        print(f"  ✓ Using DistributedWeightedSampler with class-label weighting")
                        from collections import Counter
                        label_counts = Counter(labels_list)
                        info_lines = ["Weighted sampling (label):"]
                        for label, count in sorted(label_counts.items()):
                            label_str = "Success" if label == 1 else "Fail"
                            # Get weight for this label
                            weight = sample_weights[labels_list.index(label)] if label in labels_list else 1.0
                            info_lines.append(f"  Class {label} ({label_str}): {count} samples, weight={weight:.3f}")
                        self.sampling_info = "\n".join(info_lines)
            
            else:  # weighting_strategy == "label"
                # Weight by class labels
                sample_weights = compute_sample_weights_from_labels(
                    labels_list,
                    class_weight_power=self.training_config.class_weight_power
                )
                
                if rank == 0:
                    print(f"  ✓ Using DistributedWeightedSampler with class-label weighting")
                    from collections import Counter
                    label_counts = Counter(labels_list)
                    info_lines = ["Weighted sampling (label):"]
                    for label, count in sorted(label_counts.items()):
                        label_str = "Success" if label == 1 else "Fail"
                        # Get weight for this label
                        weight = sample_weights[labels_list.index(label)] if label in labels_list else 1.0
                        info_lines.append(f"  Class {label} ({label_str}): {count} samples, weight={weight:.3f}")
                    self.sampling_info = "\n".join(info_lines)
            
            # Use DistributedWeightedSampler
            sampler = DistributedWeightedSampler(
                dataset=dataset,
                weights=sample_weights,
                num_replicas=self.world_size,
                rank=rank,
                replacement=True,  # Sample with replacement for weighted sampling
                seed=42,
                drop_last=True,
                oversample_factor=self.training_config.oversample_factor
            )
        else:
            # Use standard DistributedSampler
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=rank,
                shuffle=True,
                seed=42
            )
        
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, sampler=sampler)
        return loader

    def train(self, rank, batch_size, context_window, num_epochs, model_name):
        print(f'Training on rank {rank} started')

        self.setup(rank)

        # Get CPX token and ID(s)
        cpx_tokens = self.training_config.cpx_tokens
        
        # Convert to IDs
        cpx_token_ids = self.training_config.cpx_token_ids

        # Load model with CPX wrapper
        # Check if LoRA sho[uld be used (via config)
        use_lora = self.training_config.use_lora
        mask_lora_for_non_cpx = self.training_config.mask_lora_for_non_cpx
        aggregation_type = getattr(self.training_config, 'cpx_aggregation', 'attention')

        # Create LoRA config
        lora_config = LoraConfig(
            r=self.training_config.lora_r,
            lora_alpha=self.training_config.lora_alpha,
            target_modules=self.training_config.lora_target_modules,
            lora_dropout=self.training_config.lora_dropout,
            bias=self.training_config.lora_bias,
            task_type=self.training_config.lora_task_type,
        )
        model = CPXCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            cpx_token_ids=cpx_token_ids,
            num_labels=1,
            is_cpx_token_trainable=self.training_config.is_cpx_token_trainable,
            tokenizer_size=len(self.tokenizer),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_cache=False,  # Disable cache for training
            use_lora=use_lora,
            mask_lora_for_non_cpx=mask_lora_for_non_cpx,
            dropout_rate=self.training_config.dropout_rate,
            classifier_dropout=self.training_config.classifier_dropout,
            lora_config=lora_config,
            freeze_LoRA_layers=self.training_config.freeze_LoRA_layers,
            freeze_LoRA_start_layer_idx=self.training_config.freeze_LoRA_start_layer_idx,
            aggregation_type=aggregation_type,
            num_layers=getattr(self.training_config, 'num_layers', None)
        ).to(rank)
        
        # Ensure cache is disabled (redundant but explicit)
        model.base_model.config.use_cache = False

        # Enable gradient checkpointing to reduce memory usage
        if self.training_config.gradient_checkpointing:
            # Enable on base model
            if hasattr(model.base_model, 'gradient_checkpointing_enable'):
                model.base_model.gradient_checkpointing_enable()
                print(f"Gradient checkpointing enabled on rank {rank}")
        else:
            print(f"Gradient checkpointing disabled on rank {rank}")
        
        ddp_model = DDP(model, device_ids=[rank])
        self.model = ddp_model  # Store DDP-wrapped model for consistency
        
        # Use weighted sampling if enabled (separate from loss function weighting)
        use_weighted_sampling = self.training_config.use_weighted_sampling
        loader = self.preprocess_data(context_window, rank, batch_size, use_weighted_sampling=use_weighted_sampling)
        
        # Build optimizer parameter groups with optimized learning rates
        # Based on best practices for LoRA fine-tuning and complexity classification
        param_groups = [
            # Classifier: New layer, can handle moderate LR
            {"params": model.classifier.parameters(), 
             "lr": self.training_config.classifier_lr, 
             "weight_decay": self.training_config.weight_decay},
        ]

        # Add aggregator attention parameters if using attention aggregation
        if model.aggregator is not None and hasattr(model.aggregator, 'attention'):
            param_groups.append({
                "params": model.aggregator.attention.parameters(),
                "lr": self.training_config.aggregator_lr,  # Slightly lower than classifier (upstream, controls info flow)
                "weight_decay": self.training_config.weight_decay
            })
            if rank == 0:
                print(f"  ✓ Added aggregator attention parameters to optimizer")
        
        # Add embedding parameters if trainable
        if self.training_config.is_cpx_token_trainable:
            if model.use_lora:
                embedding_layer = model.base_model.get_base_model().get_input_embeddings()
            else:
                embedding_layer = model.base_model.get_input_embeddings()
            # CPX Embedding: Single token, needs careful/slower tuning
            param_groups.append({
                "params": embedding_layer.parameters(), 
                "lr": self.training_config.embedding_lr,
                "weight_decay": self.training_config.embedding_weight_decay
            })
        
        # Add LoRA parameters if using LoRA
        if model.use_lora:
            lora_params = [p for n, p in model.base_model.named_parameters() if 'lora' in n and p.requires_grad]
            if len(lora_params) > 0:
                # LoRA: Standard LoRA fine-tuning LR
                param_groups.append({
                    "params": lora_params, 
                    "lr": self.training_config.lora_lr,
                    "weight_decay": self.training_config.weight_decay
                })
                print(f"  ✓ Optimizer includes {len(lora_params)} LoRA parameter groups")
        
        # Print learning rate configuration
        if rank == 0:
            print(f"  Learning Rates:")
            print(f"    - Classifier: {self.training_config.classifier_lr}")
            if model.aggregator is not None and hasattr(model.aggregator, 'attention'):
                print(f"    - Aggregator Attention: {self.training_config.aggregator_lr}")
            if self.training_config.is_cpx_token_trainable:
                print(f"    - CPX Embedding: {self.training_config.embedding_lr}")
            if model.use_lora:
                print(f"    - LoRA Adapters: {self.training_config.lora_lr}")
        
        optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8, amsgrad=self.training_config.amsgrad)

        if self.training_config.scheduler == "linear":
            num_training_steps = num_epochs * len(loader)
            num_warmup_steps = int(num_training_steps * self.training_config.warmup_steps)    

            scheduler = get_scheduler(
                name=self.training_config.scheduler,
                optimizer=optimizer,    
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif self.training_config.scheduler == "cosine":
            num_training_steps = num_epochs * len(loader)
            num_warmup_steps = int(num_training_steps * self.training_config.warmup_steps)
            
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif self.training_config.scheduler == "ReduceLROnPlateau":
            if self.training_config.METRIC == "f1":
                scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
            elif self.training_config.METRIC == "loss":
                scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
            elif self.training_config.METRIC == "roc_auc":
                scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
        else:
            raise ValueError(f"Unsupported scheduler: {self.training_config.scheduler}")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = f"{self.training_config.LOG_DIR}/{self.training_config.dataset}/log_cpx_{timestamp}.txt"
        
        # Compute class weights if enabled for imbalanced datasets
        weight_info = None
        if self.training_config.use_class_weights and rank == 0:
            # Convert labels to numpy for counting
            labels_array = np.array(self.train_labels)
            class_0_count = np.sum(labels_array == 0)
            class_1_count = np.sum(labels_array == 1)
            total_samples = len(labels_array)
            
            # Compute weights: inverse proportional to class frequency
            # Higher weight for minority class
            base_weight_0 = total_samples / (2.0 * class_0_count) if class_0_count > 0 else 1.0
            base_weight_1 = total_samples / (2.0 * class_1_count) if class_1_count > 0 else 1.0
            
            # Apply power to adjust weight strength
            power = getattr(self.training_config, 'class_weight_power', 1.0)
            weight_0 = np.power(base_weight_0, power)
            weight_1 = np.power(base_weight_1, power)
            
            class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float).to(rank)
            weight_info = f"Class weights computed - Class 0 (weight={weight_0:.4f}, count={class_0_count}), Class 1 (weight={weight_1:.4f}, count={class_1_count})"
            print(weight_info)
        elif rank == 0:
            class_weights = None
            weight_info = "Class weights disabled"
            print(weight_info)
        else:
            class_weights = None
        
        # Broadcast class weights to all ranks if using DDP
        if self.training_config.use_class_weights:
            if rank != 0:
                class_weights = torch.zeros(2, dtype=torch.float).to(rank)
            dist.broadcast(class_weights, src=0)
        
        # Create loss function
        # Note: Label smoothing is applied manually to targets before loss computation
        # Note: We don't use pos_weight here because weighted sampling already balances batches
        if self.training_config.use_class_weights:
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights, reduction='mean')
        else:
            criterion = nn.BCEWithLogitsLoss(reduction='mean')
        ddp_model.train()
        
        if self.training_config.METRIC == "f1":
            best_score = 0.0
        elif self.training_config.METRIC == "loss":
            best_score = float('inf')
        elif self.training_config.METRIC == "roc_auc":
            best_score = 0.0
        else:
            best_score = 0.0  # Default

        patience = self.training_config.patience
        patience_counter = 0
        best_model_state = None
        metric = self.training_config.METRIC
        # Write the setup to the log file 
        if rank == 0:
            with open(log_path, "a") as f:
                f.write(
                    f"model: {model_name}, \n"
                    f'epochs: {num_epochs}, \n'
                    f"dataset: {self.training_config.dataset}, \n"
                    f"use_lora: {use_lora}, \n"
                    f'mask_lora_for_non_cpx: {mask_lora_for_non_cpx}, \n'
                    f"metric: {self.training_config.METRIC}, \n"
                    f"batch_size: {batch_size*self.world_size}, \n"
                    f"context_window: {context_window}, \n"
                    f"train_size: {len(self.train_texts)}, \n"
                    f"gradient_checkpointing: {self.training_config.gradient_checkpointing}\n"
                    f'classifier_lr: {self.training_config.classifier_lr}, \n'
                    f'aggregator_lr: {self.training_config.aggregator_lr}, \n'
                    f'embedding_lr: {self.training_config.embedding_lr}, \n'
                    f'lora_lr: {self.training_config.lora_lr}, \n'
                    f'cpx_aggregation: {aggregation_type}, \n'
                    f'weight_decay: {self.training_config.weight_decay}, \n'
                    f'embedding_weight_decay: {self.training_config.embedding_weight_decay}, \n'
                    f'scheduler: {self.training_config.scheduler}, \n'
                    f'warmup_steps: {self.training_config.warmup_steps}, \n'
                    f'patience: {self.training_config.patience}, \n'
                    f'max_grad_norm: {self.training_config.max_grad_norm}, \n'
                    f'dropout_rate: {self.training_config.dropout_rate}, \n'
                    f'classifier_dropout: {self.training_config.classifier_dropout}, \n'
                    f'lora_r: {self.training_config.lora_r}, \n'
                    f'lora_alpha: {self.training_config.lora_alpha}, \n'
                    f'lora_dropout: {self.training_config.lora_dropout}, \n'
                    f'lora_target_modules: {self.training_config.lora_target_modules}, \n'
                    f'lora_bias: {self.training_config.lora_bias}, \n'
                    f'lora_task_type: {self.training_config.lora_task_type}, \n'
                    f'freeze_LoRA_layers: {self.training_config.freeze_LoRA_layers}, \n'
                    f'freeze_LoRA_start_layer_idx: {self.training_config.freeze_LoRA_start_layer_idx}, \n'
                    f'amsgrad: {self.training_config.amsgrad}, \n'
                    f'label_smoothing: {getattr(self.training_config, "label_smoothing", 0.0)}, \n'
                    f'cpx_count: {len(cpx_token_ids)}, \n'
                    f'num_layers: {self.training_config.num_layers}, \n'
                    f'use_class_weights: {self.training_config.use_class_weights}, \n'
                    f'class_weight_power: {self.training_config.class_weight_power}, \n'
                    f'use_weighted_sampling: {self.training_config.use_weighted_sampling}, \n'
                    f'weighting_strategy: {self.training_config.weighting_strategy}, \n'
                    f'oversample_factor: {self.training_config.oversample_factor}, \n'

                )
                if weight_info is not None:
                    f.write(f'{weight_info}\n')
                if hasattr(self, 'sampling_info') and self.sampling_info is not None:
                    f.write(f'{self.sampling_info}\n')
        ddp_model.eval()
        # Clear any existing computation graphs before evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Evaluate the model at start - comprehensive evaluation with all metrics
        per_gpu_evaluation_batch_size = self.training_config.evaluation_batch_size // self.world_size
        
        if metric == "f1":
            best_f1, accuracy, best_threshold, per_class_precision, per_class_recall, per_class_f1, brier_score, roc_auc, pr_auc, pr_auc_class_0, log_loss_score, _, _ = self.evaluate_comprehensive_distributed(ddp_model=ddp_model, rank=rank, batch_size=per_gpu_evaluation_batch_size, context_window=context_window)
            if rank == 0:
                score = best_f1  # Use F1 as the main score for model selection
                score_str = f"F1: {best_f1:.4f}, Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, PR-AUC-Class-0: {pr_auc_class_0:.4f}, Brier Score: {brier_score:.4f}, Best threshold: {best_threshold:.4f}"
                per_class_str = f"\nPer-class metrics:\n  Class 0 (Model not capabale): Precision={per_class_precision[0]:.4f}, Recall={per_class_recall[0]:.4f}, F1={per_class_f1[0]:.4f}\n  Class 1 (model capable): Precision={per_class_precision[1]:.4f}, Recall={per_class_recall[1]:.4f}, F1={per_class_f1[1]:.4f}"
            else:
                score_str = "Evaluation completed on other ranks"
                per_class_str = ""
                score = None
        elif metric == "loss":
            score = self.evaluate_flat(self.training_config.evaluation_batch_size, context_window)
            score_str = f"Avg Binary Cross Entropy Loss on the validation set: {score:.4f}"
            per_class_str = ""
        elif metric == "roc_auc":
            best_f1, accuracy, best_threshold, per_class_precision, per_class_recall, per_class_f1, brier_score, roc_auc, pr_auc, pr_auc_class_0, log_loss_score, _, _ = self.evaluate_comprehensive_distributed(ddp_model=ddp_model, rank=rank, batch_size=per_gpu_evaluation_batch_size, context_window=context_window)
            if rank == 0:
                score = roc_auc  # Use ROC-AUC as the main score for model selection
                score_str = f"F1: {best_f1:.4f}, Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, PR-AUC-Class-0: {pr_auc_class_0:.4f}, Brier Score: {brier_score:.4f}, Best threshold: {best_threshold:.4f}"
                per_class_str = f"\nPer-class metrics:\n  Class 0 (Model not capabale): Precision={per_class_precision[0]:.4f}, Recall={per_class_recall[0]:.4f}, F1={per_class_f1[0]:.4f}\n  Class 1 (model capable): Precision={per_class_precision[1]:.4f}, Recall={per_class_recall[1]:.4f}, F1={per_class_f1[1]:.4f}"
            else:
                score_str = "Evaluation completed on other ranks"
                per_class_str = ""
                score = None
        else:
            raise ValueError(f"Unsupported evaluation metric: {metric}")
            per_class_str = ""
            score = None

        # # Synchronize all processes before proceeding
        dist.barrier(device_ids=[rank])

        # Log the results
        if rank == 0:
            print(f"At start: {score_str}{per_class_str}")
            with open(log_path, "a") as f:
                f.write(
                    f"At start: {score_str}{per_class_str}\n"
                )

        ddp_model.train()

        for epoch in range(num_epochs):
            if rank == 0:
                print(f'Epoch {epoch + 1} started')
            dist.barrier(device_ids=[rank])
            total_loss = 0
            loader.sampler.set_epoch(epoch)

            for batch in tqdm(loader, desc=f"Epoch {epoch + 1}", disable=rank != 0):
                input_ids = batch['input_ids'].to(rank)
                attention_mask = batch['attention_mask'].to(rank)
                targets = batch['labels'].to(rank).float()
                
                # Apply label smoothing if enabled
                if hasattr(self.training_config, 'label_smoothing') and self.training_config.label_smoothing > 0:
                    smoothing = self.training_config.label_smoothing
                    # Smooth labels: y_smooth = y * (1 - smoothing) + smoothing * 0.5
                    # For binary: 0 -> smoothing*0.5, 1 -> 1 - smoothing*0.5
                    targets = targets * (1 - smoothing * 0.5) + (1 - targets) * (smoothing * 0.5)
                
                optimizer.zero_grad()  
            
                with autocast('cuda', dtype=torch.bfloat16):
                    logits, _ = ddp_model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(logits, targets)

                loss.backward()
                
                # Apply gradient clipping
                if self.training_config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), self.training_config.max_grad_norm)
                optimizer.step()
                total_loss += loss.item()
                if self.training_config.scheduler in ["linear", "cosine"]:
                    scheduler.step()
            loss_tensor = torch.tensor(total_loss, device=rank)
            count_tensor = torch.tensor(len(loader), device=rank, dtype=torch.float)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

            train_loss = loss_tensor.item() / count_tensor.item()

            # Evaluate the model - comprehensive evaluation with all metrics
            per_gpu_evaluation_batch_size = self.training_config.evaluation_batch_size // self.world_size
            
            if metric == "f1":
                best_f1, accuracy, best_threshold, per_class_precision, per_class_recall, per_class_f1, brier_score, roc_auc, pr_auc, pr_auc_class_0, log_loss_score, _, _ = self.evaluate_comprehensive_distributed(ddp_model=ddp_model, rank=rank, batch_size=per_gpu_evaluation_batch_size, context_window=context_window)
                if rank == 0:
                    score = best_f1  # Use F1 as the main score for model selection
                    score_str = f"F1: {best_f1:.4f}, Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, PR-AUC-Class-0: {pr_auc_class_0:.4f}, Brier Score: {brier_score:.4f}, Best threshold: {best_threshold:.4f}"
                    per_class_str = f"\nPer-class metrics:\n  Class 0 (Model not capabale): Precision={per_class_precision[0]:.4f}, Recall={per_class_recall[0]:.4f}, F1={per_class_f1[0]:.4f}\n  Class 1 (Model capable): Precision={per_class_precision[1]:.4f}, Recall={per_class_recall[1]:.4f}, F1={per_class_f1[1]:.4f}"
                else:
                    score_str = "Evaluation completed on other ranks"
                    per_class_str = ""
                    score = None
            elif metric == "loss":
                score = self.evaluate_flat(self.training_config.evaluation_batch_size, context_window)
                score_str = f"Avg Binary Cross Entropy Loss on the validation set: {score:.4f}"
                per_class_str = ""
            elif metric == "roc_auc":
                best_f1, accuracy, best_threshold, per_class_precision, per_class_recall, per_class_f1, brier_score, roc_auc, pr_auc, pr_auc_class_0, log_loss_score, _, _ = self.evaluate_comprehensive_distributed(ddp_model=ddp_model, rank=rank, batch_size=per_gpu_evaluation_batch_size, context_window=context_window)
                if rank == 0:
                    score = roc_auc  # Use ROC-AUC as the main score for model selection
                    score_str = f"F1: {best_f1:.4f}, Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, PR-AUC-Class-0: {pr_auc_class_0:.4f}, Brier Score: {brier_score:.4f}, Best threshold: {best_threshold:.4f}"
                    per_class_str = f"\nPer-class metrics:\n  Class 0 (Model not capabale): Precision={per_class_precision[0]:.4f}, Recall={per_class_recall[0]:.4f}, F1={per_class_f1[0]:.4f}\n  Class 1 (Model capable): Precision={per_class_precision[1]:.4f}, Recall={per_class_recall[1]:.4f}, F1={per_class_f1[1]:.4f}"
                else:
                    score_str = "Evaluation completed on other ranks"
                    per_class_str = ""
                    score = None
            else:
                raise ValueError(f"Unsupported evaluation metric: {metric}")
                per_class_str = ""
                score = None

            # Synchronize all processes before proceeding
            dist.barrier(device_ids=[rank])
        
            if self.training_config.scheduler == "ReduceLROnPlateau":
                if rank == 0 and score is not None:
                    scheduler.step(score)

            # Log the results
            if rank == 0:
                print(f"Epoch {epoch + 1}, {score_str}{per_class_str}")
                with open(log_path, "a") as f:
                    f.write(
                        f"Epoch {epoch + 1}, Avg Binary Cross Entropy Loss on the training set: {train_loss:.4f}, {score_str}{per_class_str}\n"
                    )

            # local flag: only rank 0 decides
            if rank == 0 and score is not None:
                if self.training_config.METRIC == "f1":
                    comparison = score > best_score
                elif self.training_config.METRIC == "loss":
                    comparison = score < best_score
                elif self.training_config.METRIC == "roc_auc":
                    comparison = score > best_score
                else:
                    raise ValueError(f"Unsupported metric: {self.training_config.METRIC}")

                if comparison:
                    best_score = score
                    best_model_state = ddp_model.module.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"No improvement. Patience: {patience_counter}/{patience}")
                
                # Note: Optuna trial reporting is done in the main process after training
                # We can't report from child processes because trial objects aren't picklable
                # Instead, we save scores to a file and report them after training completes
            else:
                comparison = False  # other ranks don't evaluate

            # rank 0 decides whether to stop
            if rank == 0 and patience_counter >= patience:
                stop_flag = torch.tensor([1.0], device=rank)
            else:
                stop_flag = torch.tensor([0.0], device=rank)

            # share stop_flag with everyone
            dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)

            # check and break
            if stop_flag.item() == 1.0:
                if rank == 0:
                    print("⏹️ Early stopping triggered!")
                break

        self.cleanup()
        if rank == 0:
            # Save best score and intermediate scores for Optuna
            # Note: We can't access optuna_trial here because it's not picklable
            # Instead, we save scores to files and report them in the main process
            if hasattr(self, 'optuna_score_file') and self.optuna_score_file is not None:
                try:
                    with open(self.optuna_score_file, 'w') as f:
                        f.write(str(best_score))
                except Exception as e:
                    print(f"Warning: Failed to write Optuna score file: {e}")
            
            # Save intermediate scores for Optuna reporting
            if hasattr(self, 'optuna_intermediate_scores_file') and self.optuna_intermediate_scores_file is not None:
                try:
                    # Append intermediate scores (epoch, score pairs)
                    # This will be read by the main process for Optuna reporting
                    pass  # We'll implement this if needed for pruning
                except Exception as e:
                    pass
            
            if best_model_state is not None:
                save_directory = self.training_config.MODEL_DIR
                os.makedirs(save_directory, exist_ok=True)
                model_basename = model_name.replace('/', '_')
                torch.save(best_model_state, f"{save_directory}/model_{model_basename}_cpx_{timestamp}.pth")
                print(f"Model saved to {save_directory}/model_{model_basename}_cpx_{timestamp}.pth")

    def run(self, batch_size, context_window, num_epochs, model_name):
        try:
            mp.spawn(self.train, args=(batch_size, context_window, num_epochs, model_name), nprocs=self.world_size)
        except Exception as e:
            print(f"Error: {e}")
            self.cleanup()
            raise e
    
    def train_with_optuna(self, batch_size, context_window, num_epochs, model_name, trial=None):
        """
        Train with Optuna integration. Returns the best validation score.
        
        Args:
            batch_size: Batch size per GPU
            context_window: Context window size
            num_epochs: Number of epochs
            model_name: Model name
            trial: Optuna trial object (optional, only used in main process)
        
        Returns:
            best_score: Best validation score (F1 or loss depending on metric)
        """
        # Note: We don't store trial on self because it can't be pickled for child processes
        # Instead, we'll report to Optuna after training completes
        
        # Create a unique file for this trial to store the best score
        import uuid
        score_file = f"/tmp/optuna_score_{os.getpid()}_{uuid.uuid4().hex[:8]}.txt"
        self.optuna_score_file = score_file
        
        # IMPORTANT: Don't store trial on self before spawning - it's not picklable!
        # The trial object will be used only after training completes in the main process
        
        try:
            # Run training using the existing run method
            # The train method will write best_score to score_file
            # Note: trial object is NOT stored on self (not picklable for child processes)
            self.run(batch_size, context_window, num_epochs, model_name)
            
            # Read best score from file (written by train method in rank 0)
            if os.path.exists(score_file):
                with open(score_file, 'r') as f:
                    best_score = float(f.read().strip())
                os.remove(score_file)
                
                # Report final score to Optuna (in main process)
                if trial is not None and OPTUNA_AVAILABLE:
                    try:
                        # Report the final best score to Optuna
                        # Note: We can't report intermediate scores from child processes
                        # For pruning, Optuna will use the final score
                        pass  # Optuna will automatically use the return value
                    except Exception as e:
                        print(f"Warning: Failed to report to Optuna: {e}")
                
                return best_score
            
            # Fallback: return a default score if file wasn't created
            if self.training_config.METRIC == "f1" or self.training_config.METRIC == "roc_auc":
                return 0.0
            else:
                return float('inf')
            
        except Exception as e:
            print(f"Error in Optuna training: {e}")
            # Clean up score file if it exists
            if os.path.exists(score_file):
                try:
                    os.remove(score_file)
                except:
                    pass
            self.cleanup()
            if self.training_config.METRIC == "f1" or self.training_config.METRIC == "roc_auc":
                return 0.0
            else:
                return float('inf')

    def load_model(self, model_path):
        state_dict = torch.load(model_path, map_location="cuda")  # or "cuda"
        self.model.load_state_dict(state_dict)

    def evaluate_flat(self, batch_size, context_window,):
        validation_dataset = TextRegressionDataset(texts=self.validation_texts, labels=self.validation_labels, tokenizer=self.tokenizer, max_length=context_window)
        loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        criterion = nn.BCEWithLogitsLoss()

        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                targets = batch['labels'].float().to(self.model.device)
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, targets)
                total_loss += loss.item()
                print(f"Loss: {loss.item()}")
        self.model.train()
        return total_loss / len(loader)

    def get_validation_probabilities_distributed(self, ddp_model, rank, batch_size, context_window):
        """Get probabilities for validation set in one forward pass - distributed version"""
        ddp_model.eval()
        dataset = TextRegressionDataset(texts=self.validation_texts, labels=self.validation_labels, tokenizer=self.tokenizer, max_length=context_window)
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank, shuffle=False)
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, sampler=sampler)

        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(rank)
                attention_mask = batch['attention_mask'].to(rank)
                targets = batch['labels'].to(rank)
                with autocast('cuda', dtype=torch.bfloat16):                
                    logits, _ = ddp_model(input_ids=input_ids, attention_mask=attention_mask)

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)
                all_probs.append(probs)
                all_targets.append(targets.int())

        ddp_model.train()
        
        # Concatenate as tensors
        all_probs_tensor = torch.cat(all_probs, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)

        # Pad to same size if dataset shards are unequal
        local_size = torch.tensor([all_probs_tensor.size(0)], device=rank)
        sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, local_size)
        max_size = max(s.item() for s in sizes)

        pad_size = max_size - all_probs_tensor.size(0)
        if pad_size > 0:
            all_probs_tensor = torch.cat([all_probs_tensor, torch.zeros(pad_size, *all_probs_tensor.shape[1:], device=all_probs_tensor.device, dtype=all_probs_tensor.dtype)])
            all_targets_tensor = torch.cat([all_targets_tensor, torch.zeros(pad_size, *all_targets_tensor.shape[1:], device=all_targets_tensor.device, dtype=all_targets_tensor.dtype)])

        # Allocate gather buffers
        gathered_probs = [torch.zeros_like(all_probs_tensor) for _ in range(dist.get_world_size())]
        gathered_targets = [torch.zeros_like(all_targets_tensor) for _ in range(dist.get_world_size())]

        # Gather from all ranks
        dist.all_gather(gathered_probs, all_probs_tensor)
        dist.all_gather(gathered_targets, all_targets_tensor)

        # Only return results on rank 0
        if rank == 0:
            # Concatenate all gathered results
            all_probs_global = torch.cat(gathered_probs, dim=0).to(torch.float32)
            all_targets_global = torch.cat(gathered_targets, dim=0)
            return all_probs_global.view(-1).cpu().numpy(), all_targets_global.view(-1).cpu().numpy()
        else:
            return None, None

    def find_best_threshold(self, probs, targets, threshold_range=(0.1, 0.9), num_thresholds=50):
        """Find the best threshold by evaluating F1 scores for different thresholds"""
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
        best_f1 = 0
        best_threshold = 0.5
        best_accuracy = 0
        
        for threshold in thresholds:
            preds = (probs > threshold).astype(int)
            f1 = f1_score(targets, preds, average='macro')
            accuracy = accuracy_score(targets, preds)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_accuracy = accuracy
        
        # Compute per-class metrics using the best threshold
        best_preds = (probs > best_threshold).astype(int)
        per_class_precision = precision_score(targets, best_preds, average=None, zero_division=0)
        per_class_recall = recall_score(targets, best_preds, average=None, zero_division=0)
        per_class_f1 = f1_score(targets, best_preds, average=None, zero_division=0)
        
        # Compute Brier Score (probability-based metric, doesn't require threshold)
        brier_score = brier_score_loss(targets, probs)
                
        return best_f1, best_accuracy, best_threshold, per_class_precision, per_class_recall, per_class_f1, brier_score

    def evaluate_accuracy_distributed(self, ddp_model, rank, batch_size, context_window, threshold=0.5):
        ddp_model.eval()
        """Distributed version of evaluate_accuracy for multi-GPU training"""
        dataset = TextRegressionDataset(texts=self.validation_texts, labels=self.validation_labels, tokenizer=self.tokenizer, max_length=context_window)
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank, shuffle=False)
        loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, sampler=sampler)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(rank)
                attention_mask = batch['attention_mask'].to(rank)
                targets = batch['labels'].to(rank)
                with autocast('cuda', dtype=torch.bfloat16):                
                    logits, _ = ddp_model(input_ids=input_ids, attention_mask=attention_mask)

                # Apply sigmoid and threshold
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).int()

                all_preds.append(preds)
                all_targets.append(targets.int())

        ddp_model.train()
        
        # Concatenate as tensors (stay on CPU unless needed on GPU)
        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)

        # Pad to same size if dataset shards are unequal
        local_size = torch.tensor([all_preds_tensor.size(0)], device=rank)
        sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, local_size)
        max_size = max(s.item() for s in sizes)

        pad_size = max_size - all_preds_tensor.size(0)
        if pad_size > 0:
            all_preds_tensor = torch.cat([all_preds_tensor, torch.zeros(pad_size, *all_preds_tensor.shape[1:], device=all_preds_tensor.device, dtype=all_preds_tensor.dtype)])
            all_targets_tensor = torch.cat([all_targets_tensor, torch.zeros(pad_size, *all_targets_tensor.shape[1:], device=all_targets_tensor.device, dtype=all_targets_tensor.dtype)])

        # Allocate gather buffers
        gathered_preds = [torch.zeros_like(all_preds_tensor) for _ in range(dist.get_world_size())]
        gathered_targets = [torch.zeros_like(all_targets_tensor) for _ in range(dist.get_world_size())]

        # Gather from all ranks
        dist.all_gather(gathered_preds, all_preds_tensor)
        dist.all_gather(gathered_targets, all_targets_tensor)

        # Only compute metrics on rank 0
        if rank == 0:
            # Concatenate all gathered results
            all_preds_global = torch.cat(gathered_preds, dim=0) 
            all_targets_global = torch.cat(gathered_targets, dim=0)
            y_pred = all_preds_global.view(-1).cpu().numpy()
            y_true = all_targets_global.view(-1).cpu().numpy()
            
            # Compute macro F1 score
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            accuracy = accuracy_score(y_true, y_pred)
            return macro_f1, accuracy
        else:
            return None, None

    def evaluate_with_optimal_threshold_distributed(self, ddp_model, rank, batch_size, context_window):
        """Evaluate using optimal threshold found by validationing multiple thresholds"""
        # Get probabilities in one forward pass
        probs, targets = self.get_validation_probabilities_distributed(ddp_model, rank, batch_size, context_window)
        
        if rank == 0 and probs is not None:
            # Find best threshold
            best_f1, best_accuracy, best_threshold, per_class_precision, per_class_recall, per_class_f1, brier_score = self.find_best_threshold(probs, targets)
            return best_f1, best_accuracy, best_threshold, per_class_precision, per_class_recall, per_class_f1, brier_score
        else:
            return None, None, None, None, None, None, None

    def evaluate_comprehensive_distributed(self, ddp_model, rank, batch_size, context_window):
        """
        Comprehensive evaluation that computes F1, Brier Score, and ROC-AUC together.
        Returns all metrics regardless of which metric is used for model selection.
        
        Returns:
            best_f1, best_accuracy, best_threshold, per_class_precision, per_class_recall, 
            per_class_f1, brier_score, roc_auc, pr_auc, pr_auc_class_0, log_loss_score, probs, targets
        """
        # Get probabilities in one forward pass
        probs, targets = self.get_validation_probabilities_distributed(ddp_model, rank, batch_size, context_window)
        
        if rank == 0 and probs is not None:
            # Compute threshold-based metrics (F1, accuracy, etc.)
            best_f1, best_accuracy, best_threshold, per_class_precision, per_class_recall, per_class_f1, brier_score = self.find_best_threshold(probs, targets)
            
            # Compute probability-based metrics (ROC-AUC, PR-AUC, Log Loss)
            try:
                roc_auc = roc_auc_score(targets, probs)
            except ValueError:
                # Handle case where only one class is present
                roc_auc = 0.0
            
            try:
                pr_auc = average_precision_score(targets, probs)  # PR-AUC for class 1
            except ValueError:
                pr_auc = 0.0
            
            # Compute PR-AUC for class 0 (invert labels and probabilities)
            try:
                pr_auc_class_0 = average_precision_score(1 - targets, 1 - probs)
            except ValueError:
                pr_auc_class_0 = 0.0
            
            log_loss_score = log_loss(targets, probs)
            
            return best_f1, best_accuracy, best_threshold, per_class_precision, per_class_recall, per_class_f1, brier_score, roc_auc, pr_auc, pr_auc_class_0, log_loss_score, probs, targets
        else:
            return None, None, None, None, None, None, None, None, None, None, None, None, None

    def get_validation_probabilities(self, batch_size, context_window):
        """Get probabilities for validation set in one forward pass - single GPU version"""
        validation_dataset = TextRegressionDataset(texts=self.validation_texts, labels=self.validation_labels, tokenizer=self.tokenizer, max_length=context_window)
        loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        self.model.eval()

        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                targets = batch['labels'].to(self.model.device)
                with autocast('cuda', dtype=torch.bfloat16):
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu())
                all_targets.append(targets.int().cpu())

        self.model.train()
        
        # Concatenate all probabilities and targets
        all_probs = torch.cat(all_probs, dim=0).to(torch.float32).cpu().numpy()
        all_targets = torch.cat(all_targets, dim=0).cpu().numpy()
        return all_probs.flatten(), all_targets.flatten()

    def evaluate_with_optimal_threshold(self, batch_size, context_window):
        """Evaluate using optimal threshold found by validationing multiple thresholds - single GPU version"""
        # Get probabilities in one forward pass
        probs, targets = self.get_validation_probabilities(batch_size, context_window)
        
        # Find best threshold
        best_f1, best_accuracy, best_threshold, per_class_precision, per_class_recall, per_class_f1, brier_score = self.find_best_threshold(probs, targets)
        return best_f1, best_accuracy, best_threshold, per_class_precision, per_class_recall, per_class_f1, brier_score

    def evaluate_accuracy(self, batch_size, context_window, threshold=0.5):
        """Single GPU version - kept for backward compatibility"""
        validation_dataset = TextRegressionDataset(texts=self.validation_texts, labels=self.validation_labels, tokenizer=self.tokenizer, max_length=context_window)
        loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                targets = batch['labels'].to(self.model.device)
                with autocast('cuda', dtype=torch.bfloat16):
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Apply sigmoid and threshold
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).int()

                all_preds.append(preds.cpu())
                all_targets.append(targets.int().cpu())

        self.model.train()
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        # Compute macro F1 score
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        accuracy = accuracy_score(all_targets.flatten(), all_preds.flatten())
        return macro_f1, accuracy

    def evaluate_with_confidence_distributed(self, ddp_model, rank, batch_size, context_window):
        """
        Evaluate using confidence scores (probabilities) instead of binary predictions.
        Returns probability-based metrics that don't require thresholding.
        
        Metrics computed:
        - ROC-AUC: Area under the ROC curve (measures separability)
        - PR-AUC: Area under the Precision-Recall curve (good for imbalanced data)
        - Brier Score: Measures calibration (lower is better, 0 is perfect)
        - Log Loss: Negative log-likelihood (lower is better)
        
        Returns:
            roc_auc, pr_auc, brier_score, log_loss, probs, targets
        """
        # Get probabilities in one forward pass
        probs, targets = self.get_validation_probabilities_distributed(ddp_model, rank, batch_size, context_window)
        
        if rank == 0 and probs is not None:
            # Compute probability-based metrics
            try:
                roc_auc = roc_auc_score(targets, probs)
            except ValueError:
                # Handle case where only one class is present
                roc_auc = 0.0
            
            try:
                pr_auc = average_precision_score(targets, probs)
            except ValueError:
                pr_auc = 0.0
            
            brier_score = brier_score_loss(targets, probs)
            log_loss_score = log_loss(targets, probs)
            
            return roc_auc, pr_auc, brier_score, log_loss_score, probs, targets
        else:
            return None, None, None, None, None, None

    def evaluate_with_confidence(self, batch_size, context_window):
        """
        Evaluate using confidence scores (probabilities) - single GPU version.
        Returns probability-based metrics that don't require thresholding.
        
        Metrics computed:
        - ROC-AUC: Area under the ROC curve (measures separability)
        - PR-AUC: Area under the Precision-Recall curve (good for imbalanced data)
        - Brier Score: Measures calibration (lower is better, 0 is perfect)
        - Log Loss: Negative log-likelihood (lower is better)
        
        Returns:
            roc_auc, pr_auc, brier_score, log_loss, probs, targets
        """
        # Get probabilities in one forward pass
        probs, targets = self.get_validation_probabilities(batch_size, context_window)
        
        # Compute probability-based metrics
        try:
            roc_auc = roc_auc_score(targets, probs)
        except ValueError:
            # Handle case where only one class is present
            roc_auc = 0.0
        
        try:
            pr_auc = average_precision_score(targets, probs)
        except ValueError:
            pr_auc = 0.0
        
        brier_score = brier_score_loss(targets, probs)
        log_loss_score = log_loss(targets, probs)
        
        return roc_auc, pr_auc, brier_score, log_loss_score, probs, targets