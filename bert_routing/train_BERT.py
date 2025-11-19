from transformers import DistilBertTokenizer, DebertaTokenizer, AutoTokenizer, get_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from bert_routing.regression_models import TextRegressionDataset, TruncatedModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score, brier_score_loss, log_loss
from bert_routing.config import TrainingConfig
from transformers import BertTokenizer
from torch.amp import autocast, GradScaler
import time
import numpy as np
import gc

class  ModelTrainer:

    def __init__(self, model_name, num_outputs, num_classes, pooling_strategy, train_texts, train_labels, test_texts, test_labels, training_config):
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.num_classes = num_classes
        self.num_outputs = num_outputs
        self.training_config = training_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def train(self, batch_size, context_window, num_epochs):

        dataset = TextRegressionDataset(self.train_texts, self.train_labels, self.tokenizer, context_window)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)
        
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

        pos_weight = torch.tensor([2], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        log_path = f"{self.training_config.LOG_DIR}/{self.training_config.dataset}/log_{self.model_name}_{self.pooling_strategy}_{timestamp}.txt"
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
                    f"dataset: {self.training_config.dataset}, \n"
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
                   f"freeze_embedding: {self.training_config.freeze_embedding}, \n")

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
            score_str = f"f1 score at start: {f1_val:.4f}, accuracy at start: {acc_val:.4f}, best threshold: {best_threshold:.4f}"
        elif metric == "loss":
            loss_val = self.evaluate_flat(eval_batch_size, context_window)
            score_str = f"loss at start: {loss_val:.4f}"
        elif metric == "roc_auc":
            roc_auc, pr_auc, pr_auc_class_0, brier_score, log_loss_score, macro_f1, _, _ = self.evaluate_with_confidence(eval_batch_size, context_window)
            score_str = f"ROC-AUC at start: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, PR-AUC-Class-0: {pr_auc_class_0:.4f}, Brier Score: {brier_score:.4f}, Log Loss: {log_loss_score:.4f}, Macro F1: {macro_f1:.4f}"
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
            for iter, batch in enumerate(loader):
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
                    loss = criterion(outputs, targets)
                    
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

                # write the loss every 10% of the dataset
                if iter % (len(loader) // 10) == 0:
                    print(f"Loaded {(iter / len(loader))*100:.2f}%: Loss: {loss_value}")
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
                score_str = f"Avg F1 score on the test set: {score:.4f}, Avg Accuracy on the test set: {acc_val:.4f}, Best threshold: {best_threshold:.4f}"
            elif metric == "loss":
                score = self.evaluate_flat(eval_batch_size, context_window)
                score_str = f"Avg Loss on the test set: {score:.4f}"
            elif metric == "roc_auc":
                roc_auc, pr_auc, pr_auc_class_0, brier_score, log_loss_score, macro_f1, _, _ = self.evaluate_with_confidence(eval_batch_size, context_window)
                score = roc_auc  # Use ROC-AUC as the main score for model selection
                score_str = f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, PR-AUC-Class-0: {pr_auc_class_0:.4f}, Brier Score: {brier_score:.4f}, Log Loss: {log_loss_score:.4f}, Macro F1: {macro_f1:.4f}"
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

        criterion = nn.BCEWithLogitsLoss()
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
                    loss = criterion(outputs, targets)
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

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)
                all_probs.append(probs.detach().to(torch.float32).cpu())
                all_targets.append(targets.int().cpu())
                
                # Clear cache periodically during evaluation
                if batch_idx % 50 == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        self.model.train()
        
        # Concatenate and convert to numpy, then clear GPU memory
        all_probs = torch.cat(all_probs, dim=0).numpy().flatten()
        all_targets = torch.cat(all_targets, dim=0).numpy().flatten()
        
        # Clear GPU cache after evaluation
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
        
        return all_probs, all_targets

    def find_best_threshold(self, probs, targets, threshold_range=(0.1, 0.9), num_thresholds=50):
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

                # Apply sigmoid and threshold
                probs = torch.sigmoid(outputs)
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

    def evaluate_with_confidence(self, batch_size, context_window):
        """
        Evaluate using confidence scores (probabilities) instead of binary predictions.
        Returns probability-based metrics that don't require thresholding.
        
        Metrics computed:
        - ROC-AUC: Area under the ROC curve (measures separability)
        - PR-AUC: Area under the Precision-Recall curve (good for imbalanced data)
        - PR-AUC-Class-0: PR-AUC for class 0 (inverted labels and probabilities)
        - Brier Score: Measures calibration (lower is better, 0 is perfect)
        - Log Loss: Negative log-likelihood (lower is better)
        - Macro F1: F1 score computed at optimal threshold (macro average)
        
        Returns:
            roc_auc, pr_auc, pr_auc_class_0, brier_score, log_loss, macro_f1, probs, targets
        """
        # Get probabilities in one forward pass
        probs, targets = self.get_test_probabilities(batch_size, context_window)
        
        # Compute probability-based metrics
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
        
        brier_score = brier_score_loss(targets, probs)
        log_loss_score = log_loss(targets, probs)
        
        # Compute macro F1 at optimal threshold
        macro_f1, _, _ = self.find_best_threshold(probs, targets)
        
        return roc_auc, pr_auc, pr_auc_class_0, brier_score, log_loss_score, macro_f1, probs, targets