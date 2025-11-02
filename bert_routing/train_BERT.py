from transformers import DistilBertTokenizer, DebertaTokenizer, AutoTokenizer, get_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from bert_routing.regression_models import TextRegressionDataset, TruncatedModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score
from bert_routing.config import TrainingConfig
from transformers import BertTokenizer
import time
import numpy as np

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
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", max_length=self.training_config.context_window, truncation_side="left", clean_up_tokenization_spaces=False)
            self.model = TruncatedModel(num_outputs=num_outputs, num_classes=num_classes, model_name=model_name, pooling_strategy=pooling_strategy, training_config=self.training_config)
        elif self.model_name == "tinybert":
            self.tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_6L_768D", max_length=self.training_config.context_window, truncation_side="left", clean_up_tokenization_spaces=False)
            self.model = TruncatedModel(num_outputs=num_outputs, num_classes=num_classes, model_name=model_name, pooling_strategy=pooling_strategy, training_config=self.training_config)
        elif self.model_name == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base', max_length=self.training_config.context_window, truncation_side="left", clean_up_tokenization_spaces=False)
            self.model = TruncatedModel(num_outputs=num_outputs, num_classes=num_classes, model_name=model_name, pooling_strategy=pooling_strategy, training_config=self.training_config)

        self.model.to(self.device)

        if train_texts is not None and train_labels is not None:
            self.train_texts = train_texts
            self.train_labels = train_labels
        if test_texts is not None and test_labels is not None:
            self.test_texts = test_texts
            self.test_labels = test_labels


    def train(self, batch_size, context_window, num_epochs):

        dataset = TextRegressionDataset(self.train_texts, self.train_labels, self.tokenizer, context_window)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        

        # Create parameter groups for the optimizer
        param_groups = [
            {"params": self.model.transformer.embeddings.parameters(), "lr": self.training_config.embedding_lr},
            {"params": self.model.classifier.parameters(), "lr": self.training_config.classifier_lr}
        ]

        if self.training_config.freeze_layers:
            param_groups.append({"params": self.model.transformer.encoder.layer[self.training_config.layers_to_freeze-1:].parameters(), "lr": self.training_config.model_lr})
        else:
            param_groups.append({"params": self.model.transformer.encoder.layer.parameters(), "lr": self.training_config.model_lr})
        # Improved optimizer configuration with better hyperparameters
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.training_config.weight_decay, betas=self.training_config.betas, eps=self.training_config.eps, amsgrad=self.training_config.amsgrad)

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
        else:
            raise ValueError(f"Unsupported scheduler: {self.training_config.scheduler}")

        criterion = nn.BCEWithLogitsLoss()
        log_path = f"{self.training_config.LOG_DIR}/log_{self.model_name}_{self.pooling_strategy}_{timestamp}.txt"
        self.model.train()

        # Evaluate the model at start (optimal threshold)
        f1_val, acc_val, best_threshold = self.evaluate_with_optimal_threshold(self.training_config.evaluation_batch_size, context_window)
        print(f'f1 score at start: {f1_val}, accuracy at start: {acc_val}, best threshold: {best_threshold:.4f}')
        with open(log_path, "a") as f:
            f.write(f"f1 score at start: {f1_val:.4f}, accuracy at start: {acc_val:.4f}, best threshold: {best_threshold:.4f}\n")
        
        if self.training_config.METRIC == "f1":
            best_score = 0
        elif self.training_config.METRIC == "loss":
            best_score = float('inf')

        patience = 3
        patience_counter = 0
        best_model_state = None
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
                   f"amsgrad: {self.training_config.amsgrad}, \n")

        for epoch in range(num_epochs):
            total_loss = 0
            for iter, batch in enumerate(loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['labels'].to(self.device)
                optimizer.zero_grad()  
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Apply gradient clipping for training stability
                if self.training_config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                
                optimizer.step()
                total_loss += loss.item()

                # write the loss every 10% of the dataset
                if iter % (len(loader) // 10) == 0:
                    print(f"Loaded {(iter / len(loader))*100:.2f}%: Loss: {loss.item()}")
                if self.training_config.scheduler == "linear" or self.training_config.scheduler == "cosine":
                    scheduler.step()

            train_loss = total_loss / len(loader)

            # Evaluate the model
            if metric == "f1":
                score, acc_val, best_threshold = self.evaluate_with_optimal_threshold(self.training_config.evaluation_batch_size, context_window)
                score_str = f"Avg F1 score on the test set: {score:.4f}, Avg Accuracy on the test set: {acc_val:.4f}, Best threshold: {best_threshold:.4f}"
            elif metric == "loss":
                score = self.evaluate_flat(self.training_config.evaluation_batch_size, context_window)
                score_str = f"Avg Loss on the test set: {score:.4f}"
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
            else:
                raise ValueError(f"Unsupported metric: {self.training_config.METRIC}")

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

        save_directory = self.training_config.MODEL_DIR
        torch.save(best_model_state, f"{save_directory}/model_{self.model_name}_{timestamp}.pth")
    
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
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['labels'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                print(f"Loss: {loss.item()}")
        self.model.train()
        return total_loss / len(loader)

    def get_test_probabilities(self, batch_size, context_window):
        test_dataset = TextRegressionDataset(self.test_texts, self.test_labels, self.tokenizer, context_window)
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        self.model.eval()

        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)
                all_probs.append(probs.detach().to(torch.float32).cpu())
                all_targets.append(targets.int().cpu())

        self.model.train()
        
        all_probs = torch.cat(all_probs, dim=0).numpy().flatten()
        all_targets = torch.cat(all_targets, dim=0).numpy().flatten()
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
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['labels'].to(self.device)

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