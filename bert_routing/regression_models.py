import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import DistilBertModel, AutoModel

class TruncatedModel(nn.Module):
    def __init__(self, num_outputs, num_classes, model_name, pooling_strategy, training_config):
        self.pooling_strategy = pooling_strategy
        self.training_config = training_config

        super().__init__()
        # Load model with bfloat16 precision to save memory (H100 supports BF16 natively)
        if model_name == "deberta":
            self.transformer = AutoModel.from_pretrained("microsoft/deberta-v3-large", torch_dtype=torch.bfloat16)
        elif model_name == "distilbert":
            self.transformer = DistilBertModel.from_pretrained("distilbert-base-uncased", torch_dtype=torch.bfloat16)
        elif model_name == "tinybert":
            self.transformer = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_6L_768D", torch_dtype=torch.bfloat16)
        elif model_name == "bert":
            self.transformer = AutoModel.from_pretrained("microsoft/deberta-v3-base", torch_dtype=torch.bfloat16)
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        # Freeze the layers
        if self.training_config.freeze_layers and model_name != "distilbert":    
            for i, layer in enumerate(self.transformer.encoder.layer):
                if i < self.training_config.layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    # Explicitly ensure unfrozen layers have requires_grad=True
                    for param in layer.parameters():
                        param.requires_grad = True

        elif self.training_config.freeze_layers and model_name == "distilbert":
            for i, layer in enumerate(self.transformer.transformer.layer):
                if i < self.training_config.layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    # Explicitly ensure unfrozen layers have requires_grad=True
                    for param in layer.parameters():
                        param.requires_grad = True

        
        # Freeze the embedding layer 
        if self.training_config.freeze_embedding:
            if model_name == "deberta":
                embedding_module = self.transformer.embeddings
            elif model_name == "distilbert":
                embedding_module = self.transformer.transformer.embeddings
            elif model_name == "tinybert":
                embedding_module = self.transformer.embeddings
            elif model_name == "bert":
                embedding_module = self.transformer.embeddings
            for param in embedding_module.parameters():
                param.requires_grad = False

        if self.pooling_strategy == "attention":
            self.attention_vector= nn.Parameter(torch.randn(self.transformer.config.hidden_size))
        
        # Create classifier based on configuration
        if self.training_config.classifier_type == "linear":
            self.classifier = nn.Linear(self.transformer.config.hidden_size, num_outputs)
        elif self.training_config.classifier_type == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(self.transformer.config.hidden_size, self.training_config.mlp_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.training_config.dropout_rate),
                nn.Linear(self.training_config.mlp_hidden_size, num_outputs)
            )
        else:
            raise ValueError(f"Invalid classifier_type: {self.training_config.classifier_type}. Must be 'linear' or 'mlp'")
            
        if self.training_config.classifier_dropout and self.training_config.classifier_type == "linear":
            self.dropout = nn.Dropout(self.training_config.dropout_rate)
        else:
            self.dropout = None
        
        # Ensure classifier is in float32 for numerical stability
        self.classifier = self.classifier.float()
        
        # Explicitly ensure classifier parameters are trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

    
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
        if self.dropout is not None:    
            cls_embedding = self.dropout(cls_embedding)
        # Convert to float32 for classifier (more stable for final layer)
        cls_embedding = cls_embedding.float()
        raw_output = self.classifier(cls_embedding)
        
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
        # Check if label is a tensor
        if isinstance(label, torch.Tensor):
            item['labels'] = label
        else:
            item['labels'] = torch.tensor(label, dtype=torch.float)
        return item