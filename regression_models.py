import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import DistilBertModel, DebertaModel, AutoModel

class TruncatedModel(nn.Module):
    def __init__(self, num_outputs, num_classes, model_name, pooling_strategy, is_backbone_trainable=True):
        self.pooling_strategy = pooling_strategy
        super().__init__()
        if model_name == "deberta":
            self.transformer = DebertaModel.from_pretrained("microsoft/deberta-base")
        elif model_name == "distilbert":
            self.transformer = DistilBertModel.from_pretrained("distilbert-base-uncased")
        elif model_name == "tinybert":
            self.transformer = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_6L_768D")
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        if is_backbone_trainable:
            for param in self.transformer.parameters():
                param.requires_grad = True
        else:
            for param in self.transformer.parameters():
                param.requires_grad = False
        if self.pooling_strategy == "attention":
            self.attention_vector= nn.Parameter(torch.randn(self.transformer.config.hidden_size))
        self.regressor = nn.Linear(self.transformer.config.hidden_size, num_outputs*num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state     # (batch_size, seq_len, hidden)
        if self.pooling_strategy == "cls":
            cls_embedding = last_hidden_state[:, 0]       # Use [CLS] token representation
        elif self.pooling_strategy == "last":
            cls_embedding = last_hidden_state[:, -1]      # Use [CLS] token representation
        elif self.pooling_strategy == "mean":
            masked_hidden_state  = last_hidden_state * torch.unsqueeze(attention_mask, -1)
            cls_embedding = masked_hidden_state.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True) 
        elif self.pooling_strategy == "max":
            cls_embedding = last_hidden_state.max(dim=1).values  
        elif self.pooling_strategy == "attention":
            attention_weights = torch.matmul(last_hidden_state, self.attention_vector)
            attention_weights = F.softmax(attention_weights, dim=1)
            cls_embedding = torch.sum(attention_weights.unsqueeze(2) * last_hidden_state, dim=1)
        else:
            raise ValueError(f"Invalid pooling strategy: {self.pooling_strategy}")
        raw_output = self.regressor(cls_embedding)
        # return torch.sigmoid(raw_output)
        return raw_output

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