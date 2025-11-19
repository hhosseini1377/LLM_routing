import torch
from torch.utils.data import Dataset
from datasets import Dataset as DS
import pickle
import os
from cpx_model.config import CPXDatasetConfig, CPXTrainingConfig
from datasets import load_dataset
class TextRegressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        

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

def load_mmlu_data_with_cpx(cpx_tokens=None):
    train_texts, train_labels, validation_texts, validation_labels = load_mmlu_data()
    # Append all the cpx tokens to each text
    if cpx_tokens:
        cpx_suffix = ' ' + ''.join(cpx_tokens)
        train_texts = [text + cpx_suffix for text in train_texts]
        validation_texts = [text + cpx_suffix for text in validation_texts]
    else:
        # Default: use single [CPX] token for backward compatibility
        train_texts = [text + ' [CPX]' for text in train_texts]
        validation_texts = [text + ' [CPX]' for text in validation_texts]
    return train_texts, train_labels, validation_texts, validation_labels

def load_gsm8k_data():
    train_path = os.path.join(CPXDatasetConfig.GSM8K_DATA_DIR, CPXDatasetConfig.GSM8K_TRAIN_FILE)
    validation_path = os.path.join(CPXDatasetConfig.GSM8K_DATA_DIR, CPXDatasetConfig.GSM8K_TEST_FILE)
    train_data = load_pickle_data(train_path)
    validation_data = load_pickle_data(validation_path)
    if not isinstance(train_data, DS):
        train_data = DS.from_list(train_data)
    if not isinstance(validation_data, DS):
        validation_data = DS.from_list(validation_data)
    train_texts = train_data['question']
    train_labels = torch.tensor(train_data['correct'], dtype=torch.float).unsqueeze(1)
    validation_texts = validation_data['question']
    validation_labels = torch.tensor(validation_data['correct'], dtype=torch.float).unsqueeze(1)
    return train_texts, train_labels, validation_texts, validation_labels

def load_gsm8k_data_with_cpx(cpx_tokens=None):
    train_texts, train_labels, validation_texts, validation_labels = load_gsm8k_data()
    # Append all the cpx tokens to each text
    if cpx_tokens:
        cpx_suffix = ' ' + ''.join(cpx_tokens)
        train_texts = [text + cpx_suffix for text in train_texts]
        validation_texts = [text + cpx_suffix for text in validation_texts]
    else:
        # Default: use single [CPX] token for backward compatibility
        train_texts = [text + ' [CPX]' for text in train_texts]
        validation_texts = [text + ' [CPX]' for text in validation_texts]
    return train_texts, train_labels, validation_texts, validation_labels

def load_mix_data():
    train_path = os.path.join(CPXDatasetConfig.MIX_DATA_DIR, CPXDatasetConfig.MIX_TRAIN_FILE)
    validation_path = os.path.join(CPXDatasetConfig.MIX_DATA_DIR, CPXDatasetConfig.MIX_VALIDATION_FILE)
    train_data = load_pickle_data(train_path)
    validation_data = load_pickle_data(validation_path)
    if not isinstance(train_data, DS):
        train_data = DS.from_list(train_data)
    if not isinstance(validation_data, DS):
        validation_data = DS.from_list(validation_data)
    train_texts = train_data['prompt']
    train_labels = torch.tensor(train_data['correct'], dtype=torch.float).unsqueeze(1)
    validation_texts = validation_data['prompt']
    validation_labels = torch.tensor(validation_data['correct'], dtype=torch.float).unsqueeze(1)
    return train_texts, train_labels, validation_texts, validation_labels

def load_mix_data_with_cpx(cpx_tokens=None):
    train_texts, train_labels, validation_texts, validation_labels = load_mix_data()
    # Append all the cpx tokens to each text
    if cpx_tokens:
        cpx_suffix = ' ' + ''.join(cpx_tokens)
        train_texts = [text + cpx_suffix for text in train_texts]
        validation_texts = [text + cpx_suffix for text in validation_texts]
    else:
        # Default: use single [CPX] token for backward compatibility
        train_texts = [text + ' [CPX]' for text in train_texts]
        validation_texts = [text + ' [CPX]' for text in validation_texts]
    return train_texts, train_labels, validation_texts, validation_labels

def load_imdb_data():
    """
    Loads the IMDb sentiment analysis dataset from the Hugging Face Hub.

    Returns:
        tuple: (train_texts, train_labels, validation_texts, validation_labels)
               - texts: List of strings (movie reviews).
               - labels: torch.Tensor of shape [N, 1] (0.0 for negative, 1.0 for positive).
    """
    print("Loading IMDb dataset from Hugging Face Hub...")
    try:
        # Load the dataset (this downloads the data if not cached)
        imdb_dataset_dict = load_dataset("imdb")
    except Exception as e:
        print(f"Error loading IMDb dataset: {e}")
        return [], None, [], None

    # Use 'train' for training and 'test' for validation
    train_data = imdb_dataset_dict['train']
    validation_data = imdb_dataset_dict['test']
    
    print(f"Train samples: {len(train_data)}, Validation samples: {len(validation_data)}")

    # Extract texts (column name is 'text')
    train_texts = train_data['text']
    validation_texts = validation_data['text']

    # Extract labels (column name is 'label') and convert to PyTorch float tensor
    # .unsqueeze(1) is applied to ensure the shape is [N, 1] as per the GSM8K example
    train_labels = torch.tensor(train_data['label'], dtype=torch.float).unsqueeze(1)
    validation_labels = torch.tensor(validation_data['label'], dtype=torch.float).unsqueeze(1)

    return train_texts, train_labels, validation_texts, validation_labels


def load_imdb_data_with_cpx(cpx_tokens=None):
    """
    Loads IMDb data and appends complexity tokens (CPX) to the text field.

    Args:
        cpx_tokens (list, optional): A list of complexity tokens to append. 
                                     Defaults to using a single '[CPX]' token.

    Returns:
        tuple: (train_texts, train_labels, validation_texts, validation_labels)
    """
    # Load the base IMDb data
    train_texts, train_labels, validation_texts, validation_labels = load_imdb_data()

    # Append all the cpx tokens to each text (same complexity logic as original)
    if cpx_tokens and train_texts:
        cpx_suffix = ' ' + ''.join(cpx_tokens)
        train_texts = [text + cpx_suffix for text in train_texts]
        validation_texts = [text + cpx_suffix for text in validation_texts]
    elif train_texts:
        # Default: use single [CPX] token for backward compatibility
        cpx_suffix = ' [CPX]'
        train_texts = [text + cpx_suffix for text in train_texts]
        validation_texts = [text + cpx_suffix for text in validation_texts]
    
    return train_texts, train_labels, validation_texts, validation_labels