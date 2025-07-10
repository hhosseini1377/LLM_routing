import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import DistilBertModel, DebertaModel

class DistilBertForMultiLabelRegression(nn.Module,):
    def __init__(self, num_outputs, is_backbone_trainable=True):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        if is_backbone_trainable:
            for param in self.bert.parameters():
                param.requires_grad = True
        else:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.regressor = nn.Linear(self.bert.config.hidden_size, num_outputs)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden)
        cls_embedding = last_hidden_state[:, 0]        # Use [CLS] token representation
        raw_output = self.regressor(cls_embedding)
        return torch.sigmoid(raw_output)

class DebertaForMultiLabelRegression(nn.Module):
    def __init__(self, num_outputs, is_backbone_trainable=True):
        super().__init__()
        self.deberta = DebertaModel.from_pretrained("microsoft/deberta-base")
        if is_backbone_trainable:
            for param in self.deberta.parameters():
                param.requires_grad = True
        else:
            for param in self.deberta.parameters():
                param.requires_grad = False
        self.regressor = nn.Linear(self.deberta.config.hidden_size, num_outputs)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden)
        cls_embedding = last_hidden_state[:, 0]        # Use [CLS] token representation
        raw_output = self.regressor(cls_embedding)
        return torch.sigmoid(raw_output)

class TextRegressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Remove batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.float)
        return item