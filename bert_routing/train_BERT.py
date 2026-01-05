from transformers import DistilBertTokenizer, DebertaTokenizer, AutoTokenizer, get_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from bert_routing.regression_models import TextRegressionDataset, TruncatedModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score, brier_score_loss, log_loss, precision_score, recall_score
from bert_routing.config import TrainingConfig
from transformers import BertTokenizer
from torch.amp import autocast, GradScaler
import time
import numpy as np
import gc
from typing import List, Optional

from tqdm import tqdm

class  ModelTrainer:

    def __init__(self, model_name, num_outputs, num_classes, pooling_strategy, train_texts, train_labels, test_texts, test_labels, training_config, train_dataset_sources: Optional[List[str]] = None):
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.num_classes = num_classes
        self.num_outputs = num_outputs
        self.training_config = training_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataset_sources = train_dataset_sources

        if self.model_name == "distilbert":
            # Load and left truncate the tokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", max_length=self.training_config.context_window, truncation_side="left", clean_up_tokenization_spaces=False)
            self.model = TruncatedModel(num_outputs=num_outputs, num_classes=num_classes, model_name=model_name, pooling_strategy=pooling_strategy, training_config=self.training_config)
        elif self.model_name == "deberta":
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large", max_length=self.training_config.context_window, truncation_side="left", clean_up_tokenization_spaces=False)
            self.model = TruncatedModel(num_outputs=num_outputs, num_classes=num_classes, model_name=model_name, pooling_strategy=pooling_strategy, training_config=self.training_config)
        elif self.model_name == "tinybert":
            self.tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_6L_768D", max_length=self.training_config.context_window, truncation_side="left", clean_up_tokenization_spaces=False)
            self.model = TruncatedModel(num_outputs=num_outputs, num_classes=num_classes, model_name=model_name, pooling_strategy=pooling_strategy, training_config=self.training_config)
        elif self.model_name == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base', max_length=self.training_config.context_window, truncation_side="left", clean_up_tokenization_spaces=False)
            self.model = TruncatedModel(num_outputs=num_outputs, num_classes=num_classes, model_name=model_name, pooling_strategy=pooling_strategy, training_config=self.training_config)

        self.model.to(self.device)
        # Enable gradient checkpointing to save memory
        # Note: Disable when freezing layers to avoid graph conflicts with frozen layers
        if hasattr(self.model.transformer, 'gradient_checkpointing_enable'):
            if self.training_config.freeze_layers:
                # Don't enable checkpointing when freezing layers (can cause graph issues)
                print(f"Gradient checkpointing disabled (using layer freezing instead: {self.training_config.layers_to_freeze} layers frozen)")
            elif self.model_name == "deberta":
                # Enable checkpointing for DeBERTa-large when not freezing layers
                # self.model.transformer.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled for DeBERTa-large (memory optimization)")
            else:
                pass
                # Enable checkpointing for other models if not freezing layers
                # self.model.transformer.gradient_checkpointing_enable()
                # print("Gradient checkpointing enabled")

        if train_texts is not None and train_labels is not None:
            self.train_texts = train_texts
            self.train_labels = train_labels
        if test_texts is not None and test_labels is not None:
            self.test_texts = test_texts
            self.test_labels = test_labels

    def _compute_sample_weights(self) -> List[float]:
        """
        Compute sample weights for weighted random sampling.
        
        Priority:
        1. If train_dataset_sources is available, use dataset source weighting
        2. Otherwise, use class-based weighting from labels
        
        Returns:
            List of weights, one per sample
        """
        from collections import Counter
        
        if self.train_dataset_sources is not None:
            # Use dataset source weighting
            dataset_counts = Counter(self.train_dataset_sources)
            n_samples = len(self.train_dataset_sources)
            n_datasets = len(dataset_counts)
            
            # Compute inverse frequency weights
            dataset_weights = {}
            # Use sampling_weight_power for consistency (same as class-label weighting)
            # Falls back to dataset_weight_power or class_weight_power for backward compatibility
            sampling_power = self._get_sampling_weight_power()
            for dataset in dataset_counts:
                weight = n_samples / (n_datasets * dataset_counts[dataset])
                # Apply power transformation
                weight = weight ** sampling_power
                dataset_weights[dataset] = weight
            
            # Assign weight to each sample
            weights = [dataset_weights[source] for source in self.train_dataset_sources]
            
            # Print weight info
            weight_info = "Dataset source weights computed:\n"
            for dataset, weight in sorted(dataset_weights.items()):
                count = dataset_counts[dataset]
                weight_info += f"  {dataset}: weight={weight:.4f}, count={count}\n"
            print(weight_info)
            
        else:
            # Use class-based weighting from labels
            # Convert labels to numpy for counting
            labels_array = np.array([label.item() if isinstance(label, torch.Tensor) else label for label in self.train_labels])
            class_0_count = np.sum(labels_array == 0)
            class_1_count = np.sum(labels_array == 1)
            total_samples = len(labels_array)
            
            # Compute weights: inverse proportional to class frequency
            base_weight_0 = total_samples / (2.0 * class_0_count) if class_0_count > 0 else 1.0
            base_weight_1 = total_samples / (2.0 * class_1_count) if class_1_count > 0 else 1.0
            
            # Apply power transformation (use sampling_weight_power for weighted sampling)
            sampling_power = self._get_sampling_weight_power()
            weight_0 = np.power(base_weight_0, sampling_power)
            weight_1 = np.power(base_weight_1, sampling_power)
            
            # Assign weight to each sample based on its class
            weights = [weight_0 if label == 0 else weight_1 for label in labels_array]
            
            # Print weight info
            weight_info = f"Class weights computed - Class 0 (weight={weight_0:.4f}, count={class_0_count}), Class 1 (weight={weight_1:.4f}, count={class_1_count})"
            print(weight_info)
        
        return weights
    
    def _get_sampling_weight_power(self):
        """Get sampling weight power with backward compatibility."""
        # Priority: sampling_weight_power > dataset_weight_power > class_weight_power > default
        if hasattr(self.training_config, 'sampling_weight_power') and self.training_config.sampling_weight_power is not None:
            return self.training_config.sampling_weight_power
        elif hasattr(self.training_config, 'dataset_weight_power') and self.training_config.dataset_weight_power is not None:
            return self.training_config.dataset_weight_power
        elif hasattr(self.training_config, 'class_weight_power') and self.training_config.class_weight_power is not None:
            return self.training_config.class_weight_power
        else:
            return 1.0  # Default
    
    def _get_loss_weight_power(self):
        """Get loss weight power with backward compatibility."""
        if hasattr(self.training_config, 'loss_weight_power') and self.training_config.loss_weight_power is not None:
            return self.training_config.loss_weight_power
        elif hasattr(self.training_config, 'class_weight_power') and self.training_config.class_weight_power is not None:
            return self.training_config.class_weight_power
        else:
            return 1.0  # Default
    
    def _is_binary_classification(self):
        """Check if this is binary classification (num_outputs == 1) or multi-class (num_outputs > 1)."""
        return self.num_outputs == 1
    
    def _compute_class_weights_for_loss(self):
        """
        Compute class weights for loss function.
        - Binary: pos_weight for BCEWithLogitsLoss
        - Multi-class: per-class weights for CrossEntropyLoss
        
        Returns:
            torch.Tensor: Class weights (shape [1] for binary pos_weight, or [num_classes] for multi-class) or None
        """
        if not getattr(self.training_config, 'use_class_weights', False):
            return None
        
        is_binary = self._is_binary_classification()
        
        # Convert labels to numpy for counting
        labels_array = np.array([label.item() if isinstance(label, torch.Tensor) else label for label in self.train_labels])
        
        if is_binary:
            # Binary classification: compute pos_weight for BCEWithLogitsLoss
            class_0_count = np.sum(labels_array == 0)
            class_1_count = np.sum(labels_array == 1)
            total_samples = len(labels_array)
            
            # Compute weights: inverse proportional to class frequency
            base_weight_0 = total_samples / (2.0 * class_0_count) if class_0_count > 0 else 1.0
            base_weight_1 = total_samples / (2.0 * class_1_count) if class_1_count > 0 else 1.0
            
            # Apply power transformation (use loss_weight_power for loss function)
            loss_power = self._get_loss_weight_power()
            weight_0 = np.power(base_weight_0, loss_power)
            weight_1 = np.power(base_weight_1, loss_power)
            
            # For BCEWithLogitsLoss pos_weight, we use weight_1/weight_0 ratio
            # pos_weight multiplies the positive class (class 1) loss
            pos_weight_value = weight_1 / weight_0 if weight_0 > 0 else 1.0
            
            weight_info = f"Loss class weights computed (binary) - Class 0 (weight={weight_0:.4f}, count={class_0_count}), Class 1 (weight={weight_1:.4f}, count={class_1_count}), pos_weight={pos_weight_value:.4f}"
            print(weight_info)
            
            return torch.tensor([pos_weight_value], device=self.device, dtype=torch.float)
        else:
            # Multi-class classification: compute per-class weights for CrossEntropyLoss
            num_classes = self.num_classes
            class_counts = np.array([np.sum(labels_array == i) for i in range(num_classes)])
            total_samples = len(labels_array)
            
            # Compute inverse frequency weights: weight[i] = total_samples / (num_classes * count[i])
            class_weights_array = total_samples / (num_classes * class_counts + 1e-6)  # Add small epsilon to avoid division by zero
            
            # Apply power transformation if loss_weight_power is configured
            loss_power = self._get_loss_weight_power()
            class_weights_array = np.power(class_weights_array, loss_power)
            
            # Normalize weights so they sum to num_classes (standard practice)
            class_weights_array = class_weights_array / class_weights_array.mean() * num_classes
            
            class_weights = torch.tensor(class_weights_array, dtype=torch.float, device=self.device)
            
            class_counts_str = ", ".join([f"Class {i}: {count}" for i, count in enumerate(class_counts)])
            weight_info = f"Loss class weights computed (multi-class, {num_classes} classes) - Counts: {class_counts_str}, Weights: {class_weights_array}"
            print(weight_info)
            
            return class_weights

    def train(self, batch_size, context_window, num_epochs):

        dataset = TextRegressionDataset(self.train_texts, self.train_labels, self.tokenizer, context_window)
        
        # Setup weighted sampling if enabled
        sampler = None
        shuffle = True
        if self.training_config.use_weighted_sampling:
            # Compute sample weights
            weights = self._compute_sample_weights()
            # Convert to tensor for WeightedRandomSampler
            weights_tensor = torch.tensor(weights, dtype=torch.float)
            # Create weighted sampler
            sampler = WeightedRandomSampler(
                weights=weights_tensor,
                num_samples=len(weights),
                replacement=True  # Allow replacement for oversampling minority classes
            )
            shuffle = False  # Don't shuffle when using a sampler
            print(f"✓ Weighted random sampling enabled (num_samples={len(weights)})")
        
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            sampler=sampler,
            drop_last=True, 
            num_workers=0, 
            pin_memory=False
        )
        
        # Use mixed precision training to save memory
        # Note: GradScaler is not needed for bfloat16 (it has same exponent range as float32)
        # Only use scaler for float16, not bfloat16
        use_bfloat16 = True  # We're using bfloat16 for DeBERTa
        if use_bfloat16:
            scaler = None  # No scaler needed for bfloat16
            print("Using bfloat16 mixed precision (no gradient scaling needed)")
        else:
            scaler = GradScaler(enabled=(self.device.type == 'cuda'))
        use_amp = self.device.type == 'cuda'

        # Create parameter groups for the optimizer
        param_groups = [
            {"params": self.model.classifier.parameters(), "lr": self.training_config.classifier_lr}
        ]
        
        # Only add embeddings if they're trainable
        if not self.training_config.freeze_embedding:
            if self.model_name == "distilbert":
                param_groups.append({"params": self.model.transformer.transformer.embeddings.parameters(), "lr": self.training_config.embedding_lr})
            else:
                param_groups.append({"params": self.model.transformer.embeddings.parameters(), "lr": self.training_config.embedding_lr})

        # Add encoder layers based on freeze configuration
        if self.training_config.freeze_layers:
            # Collect parameters from unfrozen layers explicitly
            unfrozen_params = []
            if self.model_name == "distilbert":
                for layer in self.model.transformer.transformer.layer[self.training_config.layers_to_freeze:]:
                    unfrozen_params.extend(layer.parameters())
            else:
                for layer in self.model.transformer.encoder.layer[self.training_config.layers_to_freeze:]:
                    unfrozen_params.extend(layer.parameters())
            if unfrozen_params:
                param_groups.append({"params": unfrozen_params, "lr": self.training_config.model_lr})
        else:
            if self.model_name == "distilbert":
                param_groups.append({"params": self.model.transformer.transformer.layer.parameters(), "lr": self.training_config.model_lr})
            else:
                param_groups.append({"params": self.model.transformer.encoder.layer.parameters(), "lr": self.training_config.model_lr})
        # Improved optimizer configuration with better hyperparameters
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.training_config.weight_decay, betas=self.training_config.betas, eps=self.training_config.eps, amsgrad=self.training_config.amsgrad)
        
        # Verify we have trainable parameters
        total_params = 0
        trainable_params = 0
        for group in param_groups:
            group_total = sum(p.numel() for p in group["params"])
            group_trainable = sum(p.numel() for p in group["params"] if p.requires_grad)
            total_params += group_total
            trainable_params += group_trainable
        
        if trainable_params == 0:
            raise ValueError("ERROR: No trainable parameters found! Check freeze_layers, freeze_embedding, and layers_to_freeze settings.")
        
        print(f"Total parameters in optimizer: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")

        timestamp=time.strftime("%Y%m%d-%H%M%S")

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
            scheduler = get_scheduler(
                name=self.training_config.scheduler,
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

        # Compute class weights for loss function if enabled
        class_weights = self._compute_class_weights_for_loss()
        is_binary = self._is_binary_classification()
        
        # Create loss function based on classification type
        if is_binary:
            # Binary classification: use BCEWithLogitsLoss
            if class_weights is None:
                pos_weight = torch.tensor([1.0], device=self.device)
            else:
                pos_weight = class_weights
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
        else:
            # Multi-class classification: use CrossEntropyLoss
            criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        log_path = f"{self.training_config.LOG_DIR}/{self.training_config.dataset_name}/log_{self.model_name}_{self.pooling_strategy}_{timestamp}.txt"
        self.model.train()
        
        if self.training_config.METRIC == "f1":
            best_score = 0
        elif self.training_config.METRIC == "loss":
            best_score = float('inf')
        elif self.training_config.METRIC == "roc_auc":
            best_score = 0.0

        patience = 3
        patience_counter = 0
        best_model_state = None  # Initialize before loop
        metric = self.training_config.METRIC

        # Write the setup to the log file incudling 
        with open(log_path, "a") as f:
            f.write(f"Setup: model: {self.model_name}, \n"
                    f"dataset: {self.training_config.dataset_name}, \n"
                    f"dataset_model_name: {self.training_config.dataset_model_name}, \n"
                   f"pooling: {self.pooling_strategy}, \n"
                   f"metric: {self.training_config.METRIC}, \n"
                   f"batch_size: {batch_size}, \n"
                   f"context_window: {context_window}, \n"
                   f"train_size: {len(self.train_texts)}, \n"
                   f"dropout: {self.training_config.dropout_rate}, \n"
                   f"layers_to_freeze: {self.training_config.layers_to_freeze}, \n"
                   f"freeze_layers: {self.training_config.freeze_layers}, \n"
                   f"classifier_dropout: {self.training_config.classifier_dropout}, \n"
                   f"embedding_lr: {self.training_config.embedding_lr}, \n"
                   f"classifier_lr: {self.training_config.classifier_lr}, \n"
                   f"model_lr: {self.training_config.model_lr}, \n"
                   f"weight_decay: {self.training_config.weight_decay}, \n"
                   f"betas: {self.training_config.betas}, \n"
                   f"eps: {self.training_config.eps}, \n"   
                   f"amsgrad: {self.training_config.amsgrad}, \n"
                   f"freeze_embedding: {self.training_config.freeze_embedding}, \n"
                   f"use_weighted_sampling: {self.training_config.use_weighted_sampling}, \n"
                   f"sampling_weight_power: {self._get_sampling_weight_power()}, \n"
                   f"loss_weight_power: {self._get_loss_weight_power()}, \n"
                   f"class_weight_power: {getattr(self.training_config, 'class_weight_power', None)}, \n"
                   f"use_class_weights: {getattr(self.training_config, 'use_class_weights', False)}, \n"
                   f"num_classes: {self.num_classes}, \n"
                   f"num_outputs: {self.num_outputs}, \n"
                   f"is_binary: {is_binary}, \n"
                   f"class_weights: {class_weights.item() if is_binary and class_weights is not None else (class_weights.tolist() if class_weights is not None else None)}, \n")

        # Ensure model is in eval mode for initial evaluation
        self.model.eval()
        # Clear any existing computation graphs before evaluation
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Evaluate the model at start - use smaller batch size for evaluation
        # Reduce evaluation batch size for DeBERTa-large to save memory
        eval_batch_size = min(self.training_config.evaluation_batch_size, 16 if self.model_name == "deberta" else 32)
        if metric == "f1":
            f1_val, acc_val, best_threshold = self.evaluate_with_optimal_threshold(eval_batch_size, context_window)
            is_binary = self._is_binary_classification()
            if is_binary:
                # Binary: only show F1 and accuracy
                score_str = f"F1 at start: {f1_val:.4f}, Accuracy: {acc_val:.4f}"
            else:
                # Multi-class: only show F1 and accuracy
                score_str = f"F1 at start: {f1_val:.4f}, Accuracy: {acc_val:.4f}"
        elif metric == "loss":
            loss_val = self.evaluate_flat(eval_batch_size, context_window)
            score_str = f"loss at start: {loss_val:.4f}"
        elif metric == "roc_auc":
            roc_auc, pr_auc_class_0, pr_auc_class_1, brier_score, log_loss_score, macro_f1, \
                precision_0, recall_0, f1_0, precision_1, recall_1, f1_1, _, _ = self.evaluate_with_confidence(eval_batch_size, context_window)
            is_binary = self._is_binary_classification()
            if is_binary:
                # Binary: only show F1 and accuracy (ROC-AUC still used internally for model selection)
                _, acc_val, _ = self.evaluate_with_optimal_threshold(eval_batch_size, context_window)
                score_str = f"F1 at start: {macro_f1:.4f}, Accuracy: {acc_val:.4f}"
            else:
                # Multi-class: only show F1
                score_str = f"F1 at start: {macro_f1:.4f}"
        else:
            raise ValueError(f"Unsupported evaluation metric: {metric}")
        
        # Ensure model is back in training mode before training loop
        self.model.train()
        print(score_str)
        with open(log_path, "a") as f:
            f.write(f"{score_str}\n")
        
        # Clear cache before training
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in tqdm(loader, desc="Training", total=len(loader)):
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                targets = batch['labels'].to(self.device, non_blocking=True)
                
                # Ensure model is in training mode (important for gradient checkpointing)
                self.model.train()
                
                # Zero gradients at the start of each iteration
                optimizer.zero_grad(set_to_none=True)
                
                # Use mixed precision training
                with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    # Handle binary vs multi-class loss computation
                    is_binary = self._is_binary_classification()
                    if is_binary:
                        # Binary: targets are float (0.0 or 1.0), outputs shape [batch_size, 1]
                        targets_float = targets.float()
                        if outputs.dim() > 1 and outputs.size(1) == 1:
                            outputs = outputs.squeeze(1)
                        loss = criterion(outputs, targets_float)
                    else:
                        # Multi-class: targets are integers (class indices), outputs shape [batch_size, num_classes]
                        targets_long = targets.long()
                        loss = criterion(outputs, targets_long)
                    
                # Extract loss value before backward (to avoid any graph issues)
                loss_value = loss.item()
                
                # For bfloat16, use regular backward (no scaler needed)
                if scaler is None:
                    # Backward pass
                    loss.backward()
                    # Apply gradient clipping for training stability
                    if self.training_config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                    optimizer.step()
                else:
                    # For float16, use scaler
                    scaler.scale(loss).backward()
                    # Apply gradient clipping for training stability
                    if self.training_config.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                
                total_loss += loss_value

                if self.training_config.scheduler == "linear" or self.training_config.scheduler == "cosine":
                    scheduler.step()

            train_loss = total_loss / len(loader)

            # Clear cache before evaluation
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            
            # Evaluate the model - use smaller batch size for evaluation
            # Reduce evaluation batch size for DeBERTa-large to save memory
            eval_batch_size = min(self.training_config.evaluation_batch_size, 16 if self.model_name == "deberta" else 32)
            if metric == "f1":
                score, acc_val, best_threshold = self.evaluate_with_optimal_threshold(eval_batch_size, context_window)
                is_binary = self._is_binary_classification()
                if is_binary:
                    # Binary: only show F1 and accuracy
                    score_str = f"F1: {score:.4f}, Accuracy: {acc_val:.4f}"
                else:
                    # Multi-class: only show F1 and accuracy
                    score_str = f"F1: {score:.4f}, Accuracy: {acc_val:.4f}"
            elif metric == "loss":
                score = self.evaluate_flat(eval_batch_size, context_window)
                score_str = f"Avg Loss on the test set: {score:.4f}"
            elif metric == "roc_auc":
                roc_auc, pr_auc_class_0, pr_auc_class_1, brier_score, log_loss_score, macro_f1, \
                    precision_0, recall_0, f1_0, precision_1, recall_1, f1_1, _, _ = self.evaluate_with_confidence(eval_batch_size, context_window)
                is_binary = self._is_binary_classification()
                if is_binary:
                    score = roc_auc  # Use ROC-AUC as the main score for model selection (binary only)
                    # Binary: only show F1 and accuracy (ROC-AUC still used internally for model selection)
                    _, acc_val, _ = self.evaluate_with_optimal_threshold(eval_batch_size, context_window)
                    score_str = f"F1: {macro_f1:.4f}, Accuracy: {acc_val:.4f}"
                else:
                    # Multi-class: ROC-AUC not supported, use F1 instead
                    score = macro_f1  # Fallback to F1 for multi-class
                    score_str = f"F1: {macro_f1:.4f}"
            else:
                raise ValueError(f"Unsupported evaluation metric: {metric}")

            if self.training_config.scheduler == "ReduceLROnPlateau":
                scheduler.step(score)

            # Log the results
            print(f"Epoch {epoch + 1}, {score_str}")
            with open(log_path, "a") as f:
                f.write(
                    f"Epoch {epoch + 1}, Avg Loss on the training set: {train_loss:.4f}, {score_str}\n"
                )

            # Select metric and direction
            if self.training_config.METRIC == "f1":
                comparison = score > best_score
            elif self.training_config.METRIC == "loss":
                comparison = score < best_score
            elif self.training_config.METRIC == "roc_auc":
                comparison = score > best_score
            else:
                raise ValueError(f"Unsupported metric: {self.training_config.METRIC}")

            # Early stopping logic
            if comparison:
                best_score = score
                # Clear previous best model state to save memory
                if best_model_state is not None:
                    del best_model_state
                    torch.cuda.empty_cache()
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("⏹️ Early stopping triggered!")
                    break

        save_directory = self.training_config.MODEL_DIR
        # Save best model and clear memory
        if best_model_state is not None:
            torch.save(best_model_state, f"{save_directory}/model_{self.model_name}_{timestamp}.pth")
            del best_model_state
            torch.cuda.empty_cache()
        else:
            # Save final model if no best model was saved
            torch.save(self.model.state_dict(), f"{save_directory}/model_{self.model_name}_{timestamp}.pth")
    
    def load_model(self, model_path):
        state_dict = torch.load(model_path, map_location="cuda")  # or "cuda"
        self.model.load_state_dict(state_dict)

    def evaluate_flat(self, batch_size, context_window,):
        test_dataset = TextRegressionDataset(self.test_texts, self.test_labels, self.tokenizer, context_window)
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)

        is_binary = self._is_binary_classification()
        if is_binary:
            criterion = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            criterion = nn.CrossEntropyLoss(reduction='mean')
        
        use_amp = self.device.type == 'cuda'

        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                targets = batch['labels'].to(self.device, non_blocking=True)
                
                with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    if is_binary:
                        targets_float = targets.float()
                        if outputs.dim() > 1 and outputs.size(1) == 1:
                            outputs = outputs.squeeze(1)
                        loss = criterion(outputs, targets_float)
                    else:
                        targets_long = targets.long()
                        loss = criterion(outputs, targets_long)
                        
                total_loss += loss.item()
                print(f"Loss: {loss.item()}")
        self.model.train()
        return total_loss / len(loader)

    def get_test_probabilities(self, batch_size, context_window):
        test_dataset = TextRegressionDataset(self.test_texts, self.test_labels, self.tokenizer, context_window)
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)

        self.model.eval()
        use_amp = self.device.type == 'cuda'

        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                targets = batch['labels'].to(self.device, non_blocking=True)

                # Use mixed precision for evaluation too
                with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Apply activation based on classification type
                is_binary = self._is_binary_classification()
                if is_binary:
                    # Binary: use sigmoid to get probabilities
                    probs = torch.sigmoid(outputs)
                    if probs.dim() > 1 and probs.size(1) == 1:
                        probs = probs.squeeze(1)
                    all_probs.append(probs.detach().to(torch.float32).cpu())
                    all_targets.append(targets.int().cpu())
                else:
                    # Multi-class: use softmax to get probability distribution
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.append(probs.detach().to(torch.float32).cpu())
                    all_targets.append(targets.long().cpu())
                
                # Clear cache periodically during evaluation
                if batch_idx % 50 == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        self.model.train()
        
        # Concatenate and convert to numpy, then clear GPU memory
        is_binary = self._is_binary_classification()
        all_probs_tensor = torch.cat(all_probs, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        
        if is_binary:
            # Binary: flatten to 1D array [batch_size]
            all_probs = all_probs_tensor.numpy().flatten()
            all_targets = all_targets_tensor.numpy().flatten()
        else:
            # Multi-class: keep shape [batch_size, num_classes] for probs, flatten targets
            all_probs = all_probs_tensor.numpy()
            all_targets = all_targets_tensor.numpy().flatten()
        
        # Clear GPU cache after evaluation
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
        
        return all_probs, all_targets

    def find_best_threshold(self, probs, targets, threshold_range=(0.1, 0.9), num_thresholds=50):
        """Find the best threshold by evaluating F1 scores for different thresholds (binary) or use argmax (multi-class)"""
        is_binary = self._is_binary_classification()
        
        if is_binary:
            # Binary classification: find best threshold
            thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
            best_f1 = 0.0
            best_threshold = 0.5
            best_accuracy = 0.0
            for thr in thresholds:
                preds = (probs > thr).astype(int)
                f1 = f1_score(targets, preds, average='macro')
                acc = accuracy_score(targets, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = thr
                    best_accuracy = acc
            return best_f1, best_accuracy, best_threshold
        else:
            # Multi-class classification: use argmax (no threshold needed)
            preds = np.argmax(probs, axis=1)
            best_f1 = f1_score(targets, preds, average='macro')
            best_accuracy = accuracy_score(targets, preds)
            best_threshold = 0.5  # Not used for multi-class, but kept for compatibility
            return best_f1, best_accuracy, best_threshold

    def evaluate_with_optimal_threshold(self, batch_size, context_window):
        probs, targets = self.get_test_probabilities(batch_size, context_window)
        return self.find_best_threshold(probs, targets)

    def evaluate_accuracy(self, batch_size, context_window, threshold=0.5):
        test_dataset = TextRegressionDataset(self.test_texts, self.test_labels, self.tokenizer, context_window)
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)

        self.model.eval()
        use_amp = self.device.type == 'cuda'

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                targets = batch['labels'].to(self.device, non_blocking=True)

                with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Apply activation and get predictions
                is_binary = self._is_binary_classification()
                if is_binary:
                    # Binary: use sigmoid and threshold
                    probs = torch.sigmoid(outputs)
                    if probs.dim() > 1 and probs.size(1) == 1:
                        probs = probs.squeeze(1)
                    preds = (probs > threshold).int()
                    all_preds.append(preds.cpu())
                    all_targets.append(targets.int().cpu())
                else:
                    # Multi-class: use softmax and argmax
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1).int()
                    all_preds.append(preds.cpu())
                    all_targets.append(targets.long().cpu())

        self.model.train()
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        # Compute macro F1 score
        macro_f1 = f1_score(all_targets, all_preds, average='macro')
        accuracy = accuracy_score(all_targets.flatten(), all_preds.flatten())
        return macro_f1, accuracy

    def evaluate_with_confidence(self, batch_size, context_window):
        """
        Evaluate using confidence scores (probabilities) instead of binary predictions.
        Returns probability-based metrics that don't require thresholding.
        
        For binary classification:
        - ROC-AUC, PR-AUC, Brier Score, Log Loss, Macro F1, Per-class metrics
        
        For multi-class classification:
        - Only F1 and Accuracy (skip probability-based metrics)
        
        Returns:
            roc_auc, pr_auc_class_0, pr_auc_class_1, brier_score, log_loss, macro_f1,
            precision_0, recall_0, f1_0, precision_1, recall_1, f1_1, probs, targets
        """
        # Get probabilities in one forward pass
        probs, targets = self.get_test_probabilities(batch_size, context_window)
        
        is_binary = self._is_binary_classification()
        
        if is_binary:
            # Binary classification metrics
            # Compute probability-based metrics
            try:
                roc_auc = roc_auc_score(targets, probs)
            except ValueError:
                # Handle case where only one class is present
                roc_auc = 0.0
            
            # Compute PR-AUC for class 0 (invert labels and probabilities)
            try:
                pr_auc_class_0 = average_precision_score(1 - targets, 1 - probs)
            except ValueError:
                pr_auc_class_0 = 0.0
            
            # Compute PR-AUC for class 1 (positive class)
            try:
                pr_auc_class_1 = average_precision_score(targets, probs)
            except ValueError:
                pr_auc_class_1 = 0.0
            
            brier_score = brier_score_loss(targets, probs)
            log_loss_score = log_loss(targets, probs)
            
            # Compute macro F1 at optimal threshold and get the threshold
            macro_f1, _, best_threshold = self.find_best_threshold(probs, targets)
            
            # Compute per-class metrics at optimal threshold
            preds = (probs > best_threshold).astype(int)
            
            # Compute per-class precision, recall, and F1
            try:
                precision_0 = precision_score(targets, preds, pos_label=0, zero_division=0)
                recall_0 = recall_score(targets, preds, pos_label=0, zero_division=0)
                f1_0 = f1_score(targets, preds, pos_label=0, zero_division=0)
            except ValueError:
                precision_0 = recall_0 = f1_0 = 0.0
            
            try:
                precision_1 = precision_score(targets, preds, pos_label=1, zero_division=0)
                recall_1 = recall_score(targets, preds, pos_label=1, zero_division=0)
                f1_1 = f1_score(targets, preds, pos_label=1, zero_division=0)
            except ValueError:
                precision_1 = recall_1 = f1_1 = 0.0
            
            return roc_auc, pr_auc_class_0, pr_auc_class_1, brier_score, log_loss_score, macro_f1, \
                   precision_0, recall_0, f1_0, precision_1, recall_1, f1_1, probs, targets
        else:
            # Multi-class classification: skip probability-based metrics (only F1 and accuracy)
            # Find best predictions using argmax
            preds = np.argmax(probs, axis=1)
            macro_f1 = f1_score(targets, preds, average='macro')
            
            # Compute per-class metrics
            per_class_precision = precision_score(targets, preds, average=None, zero_division=0)
            per_class_recall = recall_score(targets, preds, average=None, zero_division=0)
            per_class_f1 = f1_score(targets, preds, average=None, zero_division=0)
            
            # Return None for probability-based metrics (not computed for multi-class)
            # For compatibility, return precision/recall/f1 for first two classes (or zeros if fewer classes)
            num_classes = self.num_classes
            precision_0 = per_class_precision[0] if num_classes > 0 else 0.0
            recall_0 = per_class_recall[0] if num_classes > 0 else 0.0
            f1_0 = per_class_f1[0] if num_classes > 0 else 0.0
            precision_1 = per_class_precision[1] if num_classes > 1 else 0.0
            recall_1 = per_class_recall[1] if num_classes > 1 else 0.0
            f1_1 = per_class_f1[1] if num_classes > 1 else 0.0
            
            return None, None, None, None, None, macro_f1, \
                   precision_0, recall_0, f1_0, precision_1, recall_1, f1_1, probs, targets