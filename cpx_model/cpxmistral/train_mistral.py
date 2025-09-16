from generate_dataset.model_loader import ModelLoader
from datasets import load_dataset
from bert_routing.train_BERT import ModelTrainer
import pickle
import argparse
import random
import os
from cpx_model.cpxmistral.config import CPXMistralDatasetConfig, MistralTrainingConfig
from itertools import product
import torch
import gc
from torch.utils.data import DataLoader
from cpx_model.cpxmistral.config import MistralTrainingConfig
from cpx_model.cpxmistral.cpx_mistral import MyMistral
from cpx_model.cpxmistral.utils import TextRegressionDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from transformers import get_scheduler
from torch.amp import GradScaler, autocast


class MistralTrainer:
    def __init__(self, model, tokenizer, train_texts=None, train_labels=None, test_texts=None, test_labels=None):
        self.model = model
        self.tokenizer = tokenizer
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.test_texts = test_texts
        self.test_labels = test_labels


    def train(self, batch_size, context_window, num_epochs):

        dataset = TextRegressionDataset(self.train_texts, self.train_labels, self.tokenizer, context_window)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        learning_rate = 5e-7  # Extremely small learning rate to prevent NaN
        weight_decay = MistralTrainingConfig.weight_decay
        optimizer = torch.optim.AdamW(self.model.params_to_train, lr=learning_rate, weight_decay=weight_decay)

        if MistralTrainingConfig.scheduler == "linear":
            num_training_steps = num_epochs * len(loader)
            num_warmup_steps = int(num_training_steps * MistralTrainingConfig.warmup_steps)    

            scheduler = get_scheduler(
                name=MistralTrainingConfig.scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif MistralTrainingConfig.scheduler == "ReduceLROnPlateau":
            if MistralTrainingConfig.METRIC == "f1":
                scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
            elif MistralTrainingConfig.METRIC == "loss":
                scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
        else:
            raise ValueError(f"Unsupported scheduler: {MistralTrainingConfig.scheduler}")

        criterion = nn.BCEWithLogitsLoss()
        log_path = f"{MistralTrainingConfig.LOG_DIR}/log_mistral.txt"
        
        self.model.train()
        
        if MistralTrainingConfig.METRIC == "f1":
            best_score = 0
        elif MistralTrainingConfig.METRIC == "loss":
            best_score = float('inf')

        patience = 3
        patience_counter = 0
        best_model_state = None
        metric = MistralTrainingConfig.METRIC

        # Write the setup to the log file incudling 
        with open(log_path, "a") as f:
            f.write(f"metric: {MistralTrainingConfig.METRIC}, "
                   f"batch_size: {batch_size}, "
                   f"context_window: {context_window}, "
                   f"train_size: {len(self.train_texts)}, "
                   f"dropout: {MistralTrainingConfig.dropout_rate}, "
                   f"layers_to_freeze: {MistralTrainingConfig.layers_to_freeze}, "
                   f"freeze_layers: {MistralTrainingConfig.freeze_layers}, "
                   f"classifier_dropout: {MistralTrainingConfig.classifier_dropout}, "
                   f"learning_rate: {MistralTrainingConfig.learning_rate}"
                   f"weight_decay: {MistralTrainingConfig.weight_decay}\n")

        scaler = GradScaler(device="cuda")

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in loader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                targets = batch['labels'].to(self.model.device)

                optimizer.zero_grad()  
            
                with autocast('cuda', dtype=torch.bfloat16):
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(logits, targets)
                
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                print(f"Loss: {loss.item()}")

                if MistralTrainingConfig.scheduler == "linear":
                    scheduler.step()

            train_loss = total_loss / len(loader)
            
            # Check for NaN weights after each epoch
            has_nan_weights = False
            for name, param in self.model.named_parameters():
                if param.requires_grad and (torch.isnan(param).any() or torch.isinf(param).any()):
                    print(f"Warning: NaN/Inf detected in {name} after epoch {epoch + 1}")
                    has_nan_weights = True
            
            if has_nan_weights:
                print("Stopping training due to NaN weights")
                break

            # Evaluate the model
            if metric == "f1":
                score, accuracy_score = self.evaluate_accuracy(MistralTrainingConfig.evaluation_batch_size, context_window)
                score_str = f"Avg F1 score on the test set: {score:.4f}, Avg Accuracy on the test set: {accuracy_score:.4f}"
            elif metric == "loss":
                score = self.evaluate_flat(MistralTrainingConfig.evaluation_batch_size, context_window)
                score_str = f"Avg Loss on the test set: {score:.4f}"
            else:
                raise ValueError(f"Unsupported evaluation metric: {metric}")

            if MistralTrainingConfig.scheduler == "ReduceLROnPlateau":
                scheduler.step(score)

            # Log the results
            print(f"Epoch {epoch + 1}, {score_str}")
            with open(log_path, "a") as f:
                f.write(
                    f"Epoch {epoch + 1}, Avg Loss on the training set: {train_loss:.4f}, {score_str}\n"
                )

            # Select metric and direction
            if MistralTrainingConfig.METRIC == "f1":
                comparison = score > best_score
            elif MistralTrainingConfig.METRIC == "loss":
                comparison = score < best_score
            else:
                raise ValueError(f"Unsupported metric: {MistralTrainingConfig.METRIC}")

            # Early stopping logic
            if comparison:
                best_score = score
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("⏹️ Early stopping triggered!")
                    break

        save_directory = MistralTrainingConfig.MODEL_DIR
        torch.save(best_model_state, f"{save_directory}/model_{self.model_name}_{self.pooling_strategy}.pth")
    
    def load_model(self, model_path):
        state_dict = torch.load(model_path, map_location="cuda")  # or "cuda"
        self.model.load_state_dict(state_dict)

    def evaluate_flat(self, batch_size, context_window,):
        test_dataset = TextRegressionDataset(self.test_texts, self.test_labels, self.tokenizer, context_window)
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

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

    def evaluate_accuracy(self, batch_size, context_window,):
        test_dataset = TextRegressionDataset(self.test_texts, self.test_labels, self.tokenizer, context_window)
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                targets = batch['labels'].to(self.model.device)

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Apply sigmoid and threshold at 0.5
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()

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
        
    

