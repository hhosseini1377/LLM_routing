from transformers import DistilBertModel, DistilBertTokenizer
from transformers import DebertaTokenizer, DebertaModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from regression_models import DebertaForMultiLabelRegression, TextRegressionDataset, DistilBertForMultiLabelRegression
import argparse

with open('./datasets/cleaned_routerbench_0shot.pkl', 'rb') as f:
    data = pickle.load(f)

data_size = 1000
texts = [sample['text'] for sample in data]
labels = [sample['labels'] for sample in data]
if data_size is not None:
    texts = texts[:data_size]
    labels = labels[:data_size]

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="distilbert")
args = parser.parse_args()
model_name = args.model

if model_name == "distilbert":
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForMultiLabelRegression(num_outputs=len(labels[0]))
elif model_name == "deberta":
    tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
    model = DebertaForMultiLabelRegression(num_outputs=len(labels[0]))

batch_size = 32
context_window = 512
dataset = TextRegressionDataset(texts, labels, tokenizer, context_window)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.MSELoss()

num_epochs = 3

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        targets = batch['labels']

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"Current Avg Loss: {loss.item():.4f}")
    print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(loader):.4f}")
    with open(f"log_{model_name}.txt", "a") as f:
        f.write(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(loader):.4f}\n")

