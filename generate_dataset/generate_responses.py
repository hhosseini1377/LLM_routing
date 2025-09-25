from vllm import LLM, SamplingParams
from datasets import load_dataset, get_dataset_config_names
import pickle
from datasets import Dataset
from generation_config import generation_config
import random
from datasets import concatenate_datasets


def format_mistral_mmlu_prompt(question: str, choices: list[str]) -> str:
    """
    Formats a multiple-choice question and choices into a prompt
    that is optimized for the Mistral-7B-Instruct model.

    Args:
        question (str): The text of the question.
        choices (list[str]): A list of the answer choices.
                             It is assumed there are 4 choices.

    Returns:
        str: The correctly formatted prompt string.
    """
    # Define the labels for the multiple-choice options
    labels = ["A", "B", "C", "D"]

    # Construct the question and choices part of the prompt
    prompt_body = f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt_body += f"{labels[i]}. {choice}\n"
    
    # Add a final instruction for the model to generate the answer
    prompt_body += "\nPlease give the letter of the correct answer, e.g., 'Answer: A'."
    
    # Wrap the entire prompt in the Mistral-7B-Instruct format
    formatted_prompt = f"[INST] {prompt_body} [/INST]"
    
    return formatted_prompt

class GenerateResponses:
    def __init__(self, model_name, prompts):
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        max_tokens = generation_config.max_tokens
        model_name = generation_config.model_name
        max_num_sequences = generation_config.max_num_sequences
        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        self.llm = LLM(model=model_name, max_num_seqs=max_num_sequences)
        self.prompts = prompts

    def generate_responses(self):
        outputs = self.llm.generate(self.prompts, self.sampling_params)
        return outputs

def downsample_dataset(dataset):

    # Let's say you have an imbalanced dataset where label 1 is the majority
    # We can filter the dataset into majority and minority classes
    majority_class = dataset.filter(lambda example: example["correct"] == 1)
    minority_class = dataset.filter(lambda example: example["correct"] == 0)

    # Determine the size of the minority class
    minority_size = len(minority_class)

    # Downsample the majority class to match the minority class size
    # The `shuffle` and `select` functions are useful here
    # We randomly sample indices from the majority class
    majority_indices = list(range(len(majority_class)))
    random.shuffle(majority_indices)
    downsampled_majority_indices = majority_indices[:minority_size]

    downsampled_majority = majority_class.select(downsampled_majority_indices)

    # Concatenate the downsampled majority class with the minority class
    balanced_dataset = concatenate_datasets([downsampled_majority, minority_class])

    # Optional: Shuffle the final balanced dataset to mix the samples
    balanced_dataset = balanced_dataset.shuffle(seed=42)

    return balanced_dataset

class DatasetLoader:
    incorrect_num = 0

    @classmethod
    def load_dataset_configs(self, dataset_name):
        configs = get_dataset_config_names(dataset_name)
        return configs

    @classmethod
    def load_auxiliary_train(self, dataset_name, config, dist_file = None):
        dataset = load_dataset(dataset_name, config)['train']
        train_prompts = []
        train_labels = []
        for row in dataset:
            row = row['train']
            formatted_prompt = format_mistral_mmlu_prompt(row['question'], row['choices'])
            train_prompts.append(formatted_prompt)
            train_labels.append(row['answer'])
        dataset_dict = {
            'prompt': train_prompts,
            'label': train_labels
        }
        cleaned_dataset = Dataset.from_dict(dataset_dict)
        if dist_file:
            with open(dist_file, 'wb') as f:
                pickle.dump(cleaned_dataset, f)
        return cleaned_dataset

    @classmethod
    def load_dataset(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    @classmethod
    def load_MMLU_dataset(self, dataset_name):
        test_prompts = []
        test_labels = []
        train_prompts = []
        train_labels = []
        validation_prompts = []
        validation_labels = []
        # load the configs  
        configs = self.load_dataset_configs(dataset_name)
        for config in configs:
            dataset = load_dataset(dataset_name, config)
            print(dataset, config)
            for row in dataset['test']:
                formatted_prompt = format_mistral_mmlu_prompt(row['question'], row['choices'])
                test_prompts.append(formatted_prompt)
                test_labels.append(row['answer'])
            for row in dataset['validation']:
                formatted_prompt = format_mistral_mmlu_prompt(row['question'], row['choices'])
                train_prompts.append(formatted_prompt)
                train_labels.append(row['answer'])
            for row in dataset['dev']:
                formatted_prompt = format_mistral_mmlu_prompt(row['question'], row['choices'])
                validation_prompts.append(formatted_prompt)
                validation_labels.append(row['answer'])
        
        test_dict = {
            'prompts': test_prompts,
            'labels': test_labels
        }
        train_dict = {
            'prompts': train_prompts,
            'labels': train_labels
        }
        validation_dict = {
            'prompts': validation_prompts,
            'labels': validation_labels
        }
        test_dataset = Dataset.from_dict(test_dict)
        train_dataset = Dataset.from_dict(train_dict)
        validation_dataset = Dataset.from_dict(validation_dict)
        # Save the datasets to a pickle file
        with open('./dataset/datasets/mmlu_dataset.pkl', 'wb') as f:
            pickle.dump((test_dataset, train_dataset, validation_dataset), f)
        return test_dataset, train_dataset, validation_dataset
    
    @classmethod
    def concat_responses(self, raw_dataset, raw_responses):
        responses = []
        for row in raw_responses:
            responses.append(row.outputs[0].text)
        
        dataset = raw_dataset.add_column('answer', responses)
        return dataset
    
    @classmethod
    def format_labels(self, dataset):
        new_labels = []
        for row in dataset:
            if row['label'] == 0:
                new_labels.append('A')
            elif row['label'] == 1:
                new_labels.append('B')
            elif row['label'] == 2:
                new_labels.append('C')
            elif row['label'] == 3:
                new_labels.append('D')
            else:
                raise ValueError(f'Found an invalid lable')

        # Remove the original label column
        dataset = dataset.remove_columns('label')
        # Add the new label column
        dataset = dataset.add_column('label', new_labels)
        return dataset

def add_correct_label(example):
    answer = example['answer']
    label = example['label']

    try:
        if 'Answer' not in answer:
            checked_label = answer[1]
        else:
            checked_label = answer[9]
    except:
        example['correct'] = 0
        DatasetLoader.incorrect_num += 1
        return example
    if checked_label not in ['A', 'B', 'C', 'D']:
        DatasetLoader.incorrect_num += 1
        example['correct'] = 0
    elif checked_label == label:
        example['correct'] = 1
    else:
        example['correct'] = 0
    return example

def split_dataset(dataset):
    # Split into train, test, and validation
    train_dataset = dataset.select(range(0, int(len(dataset) * 0.8)))
    test_dataset = dataset.select(range(int(len(dataset) * 0.8), int(len(dataset) * 0.9)))
    validation_dataset = dataset.select(range(int(len(dataset) * 0.9), len(dataset)))
    return train_dataset, test_dataset, validation_dataset

def create_clean_auxiliary_dataset():
    model_name = generation_config.model_name
    formatted_mmlu_Mistral_file = generation_config.mmlu_dataset_folder + '/mmlu_auxiliary_formatted.pkl'
    mmlu_auxiliary_with_responses_file = generation_config.mmlu_dataset_folder + '/mmlu_auxiliary_with_responses.pkl'
    dataset_name = "cais/mmlu"
    
    raw_dataset = DatasetLoader.load_auxiliary_train(dataset_name, 'auxiliary_train', formatted_mmlu_Mistral_file)
    prompts = raw_dataset['prompt']

    generate_responses = GenerateResponses(model_name, prompts)
    raw_responses = generate_responses.generate_responses()

    dataset = DatasetLoader.concat_responses(raw_dataset, raw_responses)
    dataset = DatasetLoader.format_labels(dataset)
    dataset = dataset.map(add_correct_label)
    dataset = dataset.shuffle(seed=42)
    with open(mmlu_auxiliary_with_responses_file, 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":

    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    formatted_mmlu_Mistral_file = generation_config.mmlu_dataset_folder + '/mmlu_auxiliary_formatted.pkl'
    mmlu_Mistral_responses_file = generation_config.mmlu_dataset_folder + '/mmlu_auxiliary_Mistral_responses.pkl'
    mmlu_auxiliary_with_responses_file = generation_config.mmlu_dataset_folder + '/mmlu_auxiliary_with_responses.pkl'
    label_file = generation_config.mmlu_dataset_folder + '/mmlu_auxiliary_labeled.pkl'
    downsampled_label_file = generation_config.mmlu_dataset_folder + '/mmlu_auxiliary_labeled_downsampled.pkl'
    train_dataset_file = generation_config.mmlu_dataset_folder + '/mmlu_auxiliary_train.pkl'
    test_dataset_file = generation_config.mmlu_dataset_folder + '/mmlu_auxiliary_test.pkl'
    validation_dataset_file = generation_config.mmlu_dataset_folder + '/mmlu_auxiliary_validation.pkl'
    dataset_name = "cais/mmlu"
    
    # Read the responses from the pickle file
    with open(downsampled_label_file, 'rb') as f:
        dataset = pickle.load(f)

    # Split the dataset into train, test, and validation
    train_dataset, test_dataset, validation_dataset = split_dataset(dataset)

    # Save the datasets to a pickle file
    with open(train_dataset_file, 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(test_dataset_file, 'wb') as f:
        pickle.dump(test_dataset, f)
    with open(validation_dataset_file, 'wb') as f:
        pickle.dump(validation_dataset, f)