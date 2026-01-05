import torch
from torch.utils.data import Dataset
from datasets import Dataset as DS
import pickle
import os
import numpy as np
from cpx_model.config import CPXDatasetConfig, CPXTrainingConfig
from datasets import load_dataset
from routing_dataset.dataset_paths import *
import pandas
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
            # Convert to tensor
            # The training code will handle conversion based on num_labels:
            # - Binary (num_labels=1): expects float (0.0 or 1.0)
            # - Multi-class (num_labels>1): expects long (0, 1, 2, ...)
            # For now, preserve the original type: if it's an integer, keep as int/long
            # If it's float or 0/1, use float for backward compatibility
            if isinstance(label, (int, np.integer)):
                # Integer label: use long (works for both binary 0/1 and multi-class 0,1,2,...)
                # Training code will convert to float if needed for binary
                item['labels'] = torch.tensor(label, dtype=torch.long)
            else:
                # Float label: use float (for binary classification)
                item['labels'] = torch.tensor(label, dtype=torch.float)
        return item

def load_pickle_data(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def load_mmlu_data(dataset_name='auxiliary', dataset_model_name=None):
    """
    Load MMLU dataset based on dataset name and model name.
    
    Args:
        dataset_name: Name of the dataset split (e.g., 'auxiliary', 'combined')
        dataset_model_name: Name of the model used (e.g., 'qwen4', 'qwen17b', 'qwen34b', None)
    
    Returns:
        tuple: (train_texts, train_labels, validation_texts, validation_labels)
    """
    from routing_dataset.dataset_paths import get_dataset_files
    
    # Get file paths based on dataset name and model name
    train_path, validation_path, _ = get_dataset_files(dataset_name, dataset_model_name)
    
    # Load train data
    train_data = load_pickle_data(str(train_path))
    if isinstance(train_data, pandas.DataFrame):
        train_data = train_data.to_dict(orient='list')
    train_texts = train_data['prompts']
    train_labels = torch.tensor(train_data['correct_labels'], dtype=torch.float).unsqueeze(1)
    
    # Load validation data
    validation_data = load_pickle_data(str(validation_path))
    if isinstance(validation_data, pandas.DataFrame):
        validation_data = validation_data.to_dict(orient='list')
    validation_texts = validation_data['prompts']
    validation_labels = torch.tensor(validation_data['correct_labels'], dtype=torch.float).unsqueeze(1)
    
    return train_texts, train_labels, validation_texts, validation_labels

def load_mmlu_data_with_cpx(cpx_tokens=None, dataset_name='auxiliary', dataset_model_name=None):
    """
    Load MMLU dataset with CPX tokens appended.
    
    Args:
        cpx_tokens: List of CPX tokens to append
        dataset_name: Name of the dataset split (e.g., 'auxiliary', 'combined')
        dataset_model_name: Name of the model used (e.g., 'qwen4', 'qwen17b', 'qwen34b', None)
    
    Returns:
        tuple: (train_texts, train_labels, validation_texts, validation_labels)
    """
    train_texts, train_labels, validation_texts, validation_labels = load_mmlu_data(dataset_name, dataset_model_name)
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
        train_data = DS.from_dict(train_data)
    if not isinstance(validation_data, DS):
        validation_data = DS.from_dict(validation_data)
    
    train_texts = train_data['prompts']
    train_labels = torch.tensor(train_data['correct_labels'], dtype=torch.float).unsqueeze(1)
    validation_texts = validation_data['prompts']
    validation_labels = torch.tensor(validation_data['correct_labels'], dtype=torch.float).unsqueeze(1)
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

def load_combined_data(train_path, validation_path):
    """
    Load combined dataset (from create_final_splits.py) with dataset sources.
    
    The combined dataset files contain:
    - 'prompts': List of prompts
    - 'ground_truths': List of ground truth answers
    - 'correct_labels': List of 0/1 labels
    - 'dataset_source': List of dataset names ("MMLU", "MMLU-Pro", "GSM8K")
    
    Args:
        train_path: Path to training pickle file
        validation_path: Path to validation pickle file
    
    Returns:
        tuple: (train_texts, train_labels, train_dataset_sources, 
                validation_texts, validation_labels, validation_dataset_sources)
    """
    train_data = load_pickle_data(train_path)
    validation_data = load_pickle_data(validation_path)
    if isinstance(train_data, pandas.DataFrame):
        train_data = train_data.to_dict(orient='list')
    if isinstance(validation_data, pandas.DataFrame):
        validation_data = validation_data.to_dict(orient='list')
    train_texts = train_data['prompts']
    train_labels = torch.tensor(train_data['correct_labels'], dtype=torch.float).unsqueeze(1)
    train_dataset_sources = train_data.get('dataset_source', None)  # Extract dataset sources
    validation_texts = validation_data['prompts']
    validation_labels = torch.tensor(validation_data['correct_labels'], dtype=torch.float).unsqueeze(1)
    validation_dataset_sources = validation_data.get('dataset_source', None)
    
    return (train_texts, train_labels, train_dataset_sources,
            validation_texts, validation_labels, validation_dataset_sources)

def load_combined_data_with_cpx(train_path, validation_path, cpx_tokens=None):
    """
    Load combined dataset with CPX tokens appended.
    
    Args:
        train_path: Path to training pickle file
        validation_path: Path to validation pickle file
        cpx_tokens: Optional list of CPX tokens to append
    
    Returns:
        tuple: (train_texts, train_labels, train_dataset_sources,
                validation_texts, validation_labels, validation_dataset_sources)
    """
    train_texts, train_labels, train_dataset_sources, validation_texts, validation_labels, validation_dataset_sources = \
        load_combined_data(train_path, validation_path)
    
    # Append CPX tokens if provided
    if cpx_tokens:
        cpx_suffix = ' ' + ''.join(cpx_tokens)
        train_texts = [text + cpx_suffix for text in train_texts]
        validation_texts = [text + cpx_suffix for text in validation_texts]
    else:
        # Default: use single [CPX] token for backward compatibility
        train_texts = [text + ' [CPX]' for text in train_texts]
        validation_texts = [text + ' [CPX]' for text in validation_texts]
    
    return (train_texts, train_labels, train_dataset_sources,
            validation_texts, validation_labels, validation_dataset_sources)

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


def load_mnli_data(train_file_path=None, validation_split=0.1):
    """
    Loads MNLI dataset from parquet file.
    
    Args:
        train_file_path: Path to the training parquet file. If None, uses default path.
        validation_split: Fraction of data to use for validation (default: 0.1)
    
    Returns:
        tuple: (train_texts, train_labels, validation_texts, validation_labels)
               - texts: List of formatted strings with Premise and Hypothesis
               - labels: torch.Tensor of shape [N] with integer labels (0=entailment, 1=neutral, 2=contradiction)
    """
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    
    if train_file_path is None:
        train_file_path = Path("./routing_dataset/datasets/train-00000-of-00001.parquet")
    else:
        train_file_path = Path(train_file_path)
    
    print(f"Loading MNLI dataset from: {train_file_path}")
    
    # Load parquet file
    df = pd.read_parquet(train_file_path)
    
    print(f"Loaded {len(df)} samples")
    print(f"Label distribution: {df['label'].value_counts().sort_index().to_dict()}")
    
    # Format prompts: "Premise: {premise}\n\nHypothesis: {hypothesis}"
    texts = []
    for _, row in df.iterrows():
        prompt = f"Premise:\n{row['premise']}\n\nHypothesis:\n{row['hypothesis']}"
        texts.append(prompt)
    
    # Extract labels (0=entailment, 1=neutral, 2=contradiction)
    labels = df['label'].values.astype(int)
    
    # Split into train and validation
    train_texts, validation_texts, train_labels, validation_labels = train_test_split(
        texts, labels, test_size=validation_split, random_state=42, stratify=labels
    )
    
    # Convert labels to tensors (long for multi-class classification)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    validation_labels = torch.tensor(validation_labels, dtype=torch.long)
    
    print(f"Train samples: {len(train_texts)}, Validation samples: {len(validation_texts)}")
    
    return train_texts, train_labels, validation_texts, validation_labels


def load_mnli_data_with_cpx(cpx_tokens=None, train_file_path=None, validation_split=0.1):
    """
    Loads MNLI data and appends complexity tokens (CPX) to the formatted prompts.
    
    Args:
        cpx_tokens (list, optional): A list of complexity tokens to append. 
                                     Defaults to using a single '[CPX]' token.
        train_file_path: Path to the training parquet file. If None, uses default path.
        validation_split: Fraction of data to use for validation (default: 0.1)
    
    Returns:
        tuple: (train_texts, train_labels, validation_texts, validation_labels)
    """
    # Load the base MNLI data
    train_texts, train_labels, validation_texts, validation_labels = load_mnli_data(
        train_file_path=train_file_path, validation_split=validation_split
    )
    
    # Append CPX tokens to each text
    if cpx_tokens and train_texts:
        cpx_suffix = '\n\n' + ''.join(cpx_tokens)
        train_texts = [text + cpx_suffix for text in train_texts]
        validation_texts = [text + cpx_suffix for text in validation_texts]
    elif train_texts:
        # Default: use single [CPX] token for backward compatibility
        cpx_suffix = '\n\n[CPX]'
        train_texts = [text + cpx_suffix for text in train_texts]
        validation_texts = [text + cpx_suffix for text in validation_texts]
    
    return train_texts, train_labels, validation_texts, validation_labels


def load_anli_data(train_file_path=None, validation_split=0.1):
    """
    Loads ANLI dataset from parquet file.
    
    Args:
        train_file_path: Path to the training parquet file. If None, uses default path.
        validation_split: Fraction of data to use for validation (default: 0.1)
    
    Returns:
        tuple: (train_texts, train_labels, validation_texts, validation_labels)
               - texts: List of formatted strings with Premise and Hypothesis
               - labels: torch.Tensor of shape [N] with integer labels (0=entailment, 1=neutral, 2=contradiction)
    """
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    
    if train_file_path is None:
        train_file_path = Path("./routing_dataset/datasets/train_r3-00000-of-00001.parquet")
    else:
        train_file_path = Path(train_file_path)
    
    print(f"Loading ANLI dataset from: {train_file_path}")
    
    # Load parquet file
    df = pd.read_parquet(train_file_path)
    
    print(f"Loaded {len(df)} samples")
    print(f"Label distribution: {df['label'].value_counts().sort_index().to_dict()}")
    
    # Format prompts: "Premise: {premise}\n\nHypothesis: {hypothesis}"
    texts = []
    for _, row in df.iterrows():
        prompt = f"Premise:\n{row['premise']}\n\nHypothesis:\n{row['hypothesis']}"
        texts.append(prompt)
    
    # Extract labels (0=entailment, 1=neutral, 2=contradiction)
    labels = df['label'].values.astype(int)
    
    # Split into train and validation
    train_texts, validation_texts, train_labels, validation_labels = train_test_split(
        texts, labels, test_size=validation_split, random_state=42, stratify=labels
    )
    
    # Convert labels to tensors (long for multi-class classification)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    validation_labels = torch.tensor(validation_labels, dtype=torch.long)
    
    print(f"Train samples: {len(train_texts)}, Validation samples: {len(validation_texts)}")
    
    return train_texts, train_labels, validation_texts, validation_labels


def load_anli_data_with_cpx(cpx_tokens=None, train_file_path=None, validation_split=0.1):
    """
    Loads ANLI data and appends complexity tokens (CPX) to the formatted prompts.
    
    Args:
        cpx_tokens (list, optional): A list of complexity tokens to append. 
                                     Defaults to using a single '[CPX]' token.
        train_file_path: Path to the training parquet file. If None, uses default path.
        validation_split: Fraction of data to use for validation (default: 0.1)
    
    Returns:
        tuple: (train_texts, train_labels, validation_texts, validation_labels)
    """
    # Load the base ANLI data
    train_texts, train_labels, validation_texts, validation_labels = load_anli_data(
        train_file_path=train_file_path, validation_split=validation_split
    )
    
    # Append CPX tokens to each text
    if cpx_tokens and train_texts:
        cpx_suffix = '\n\n' + ''.join(cpx_tokens)
        train_texts = [text + cpx_suffix for text in train_texts]
        validation_texts = [text + cpx_suffix for text in validation_texts]
    elif train_texts:
        # Default: use single [CPX] token for backward compatibility
        cpx_suffix = '\n\n[CPX]'
        train_texts = [text + cpx_suffix for text in train_texts]
        validation_texts = [text + cpx_suffix for text in validation_texts]
    
    return train_texts, train_labels, validation_texts, validation_labels