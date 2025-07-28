from transformers import DistilBertTokenizer, DebertaTokenizer, AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from regression_models import TextRegressionDataset, TruncatedModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score
from config import TrainingConfig
from transformers import BertTokenizer
class  ModelTrainer:

    def __init__(self, model_name, num_outputs, num_classes, pooling_strategy, train_texts=None, train_labels=None, test_texts=None, test_labels=None):
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.num_classes = num_classes
        self.num_outputs = num_outputs
        
        if self.model_name == "distilbert":
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = TruncatedModel(num_outputs=num_outputs, num_classes=num_classes, model_name=model_name, pooling_strategy=pooling_strategy)
        elif self.model_name == "deberta":
            self.tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
            self.model = TruncatedModel(num_outputs=num_outputs, num_classes=num_classes, model_name=model_name, pooling_strategy=pooling_strategy)
        elif self.model_name == "tinybert":
            self.tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_6L_768D")
            self.model = TruncatedModel(num_outputs=num_outputs, num_classes=num_classes, model_name=model_name, pooling_strategy=pooling_strategy)
        elif self.model_name == "bert":
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = TruncatedModel(num_outputs=num_outputs, num_classes=num_classes, model_name=model_name, pooling_strategy=pooling_strategy)

        if train_texts is not None and train_labels is not None:
            self.train_texts = train_texts
            self.train_labels = train_labels
        if test_texts is not None and test_labels is not None:
            self.test_texts = test_texts
            self.test_labels = test_labels


    def train(self, batch_size, context_window, num_epochs):

        dataset = TextRegressionDataset(self.train_texts, self.train_labels, self.tokenizer, context_window)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        learning_rate = TrainingConfig.learning_rate
        weight_decay = TrainingConfig.weight_decay
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if TrainingConfig.METRIC == "f1":
            scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
        elif TrainingConfig.METRIC == "loss":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        f1_score, accuracy = self.evaluate_accuracy(TrainingConfig.evaluation_batch_size, context_window)
        print(f'f1 score at start: {f1_score}, accuracy at start: {accuracy}')
        if TrainingConfig.METRIC == "f1":
            best_score = 0
        elif TrainingConfig.METRIC == "loss":
            best_score = float('inf')

        patience = 3
        patience_counter = 0
        best_model_state = None
        metric = TrainingConfig.METRIC
        log_path = f"results_logs/log_{self.model_name}_{self.pooling_strategy}.txt"

        # Write the setup to the log file incudling 
        with open(log_path, "a") as f:
            f.write(f"Setup: model: {self.model_name}, pooling: {self.pooling_strategy}, metric: {TrainingConfig.METRIC}, batch_size: {batch_size}, context_window: {context_window}, train_size: {len(self.train_texts)}\n")

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                targets = batch['labels']

                optimizer.zero_grad()  
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Log the training loss
            print(f"Epoch {epoch+1}, Avg Loss on the training set: {total_loss / len(loader):.4f}")

            train_loss = total_loss / len(loader)

            # Evaluate the model
            if metric == "f1":
                score, accuracy_score = self.evaluate_accuracy(TrainingConfig.evaluation_batch_size, context_window)
                score_str = f"Avg F1 score on the test set: {score:.4f}, Avg Accuracy on the test set: {accuracy_score:.4f}"
            elif metric == "loss":
                score = self.evaluate_flat(TrainingConfig.evaluation_batch_size, context_window)
                score_str = f"Avg Loss on the test set: {score:.4f}"
            else:
                raise ValueError(f"Unsupported evaluation metric: {metric}")

            # Update the learning rate
            scheduler.step(score)

            # Log the results
            print(f"Epoch {epoch + 1}, {score_str}")
            with open(log_path, "a") as f:
                f.write(
                    f"Epoch {epoch + 1}, Avg Loss on the training set: {train_loss:.4f}, {score_str}\n"
                )

            # Select metric and direction
            if TrainingConfig.METRIC == "f1":
                comparison = score > best_score
            elif TrainingConfig.METRIC == "loss":
                comparison = score < best_score
            else:
                raise ValueError(f"Unsupported metric: {TrainingConfig.METRIC}")

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

        save_directory = f"./finetuned_models/"
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
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                targets = batch['labels']
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
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
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                targets = batch['labels']

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Apply sigmoid and threshold at 0.5
                probs = torch.sigmoid(outputs)
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