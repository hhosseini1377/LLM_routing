import torch
from torch.utils.data import Dataset
import pickle
import os
from cpx_model.config import CPXDatasetConfig, CPXTrainingConfig

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
        
        # Ensure CPX token is always present
        if CPXTrainingConfig.cpx_token not in text:
            print('CPX token not in text')
            text = text.strip() + ' ' + CPXTrainingConfig.cpx_token

        # Sequence is short enough, tokenize normally
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

def load_pickle_data(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_mmlu_data():
    # Load train data
    train_path = os.path.join(CPXDatasetConfig.MMLU_DATA_DIR, CPXDatasetConfig.MMLU_TRAIN_FILE)
    train_data = load_pickle_data(train_path)
    train_texts = train_data['prompt']
    train_labels = torch.tensor(train_data['correct'], dtype=torch.float).unsqueeze(1)
    # Load validation data
    validation_path = os.path.join(CPXDatasetConfig.MMLU_DATA_DIR, CPXDatasetConfig.MMLU_VALIDATION_FILE)
    validation_data = load_pickle_data(validation_path)
    validation_texts = validation_data['prompt']
    validation_labels = torch.tensor(validation_data['correct'], dtype=torch.float).unsqueeze(1)
    return train_texts, train_labels, validation_texts, validation_labels

def load_mmlu_data_with_cpx():
    train_texts, train_labels, validation_texts, validation_labels = load_mmlu_data()
    train_texts = [text + ' ' + CPXTrainingConfig.cpx_token for text in train_texts]
    validation_texts = [text + ' ' + CPXTrainingConfig.cpx_token for text in validation_texts]
    return train_texts, train_labels, validation_texts, validation_labels