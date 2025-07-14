from transformers import DistilBertTokenizer
from transformers import DebertaTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from regression_models import TextRegressionDataset, TrunkatedModel

class  ModelTrainer:

    def __init__(self, model_name, num_outputs, pooling_strategy):
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        if self.model_name == "distilbert":
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = TrunkatedModel(num_outputs=num_outputs, model_name=model_name, pooling_strategy=pooling_strategy, is_backbone_trainable=True)
        elif self.model_name == "deberta":
            self.tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
            self.model = TrunkatedModel(num_outputs=num_outputs, model_name=model_name, pooling_strategy=pooling_strategy, is_backbone_trainable=True)

    def train(self, batch_size, context_window, num_epochs, texts, labels):

        dataset = TextRegressionDataset(texts, labels, self.tokenizer, context_window)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        optimizer = optim.AdamW(self.model.parameters(), lr=2e-5)
        criterion = nn.MSELoss()

        self.model.train()
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
                print(f'loss: {loss.item():.4f}')
            print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(loader):.4f}")
            with open(f"results_logs/log_{self.model_name}_{self.pooling_strategy}.txt", "a") as f:
                f.write(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(loader):.4f}\n")

        save_directory = f"./finetuned_models/"

        torch.save(self.model.state_dict(), f"{save_directory}/model_{self.model_name}_{self.pooling_strategy}.pth")
    
    def load_model(self, model_path):
        state_dict = torch.load(model_path, map_location="cuda")  # or "cuda"
        self.model.load_state_dict(state_dict)

    def evaluate(self, batch_size, context_window, test_texts, test_labels):

        test_dataset = TextRegressionDataset(test_texts, test_labels, self.tokenizer, context_window)
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()

        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                print('ali')
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                targets = batch['labels']
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(loader)