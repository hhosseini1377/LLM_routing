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
        dtype = generation_config.dtype
        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        self.llm = LLM(model=model_name, max_num_seqs=max_num_sequences, dtype=dtype)
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
    def load_dataset_configs(cls, dataset_name):
        configs = get_dataset_config_names(dataset_name)
        return configs

    @classmethod
    def load_auxiliary_train(cls, dataset_name, config, dist_file = None):
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

def add_correct_label_with_threshold(example, num_runs, threshold):
    """
    Checks correctness across multiple runs and applies threshold for labeling.
    
    Args:
        example: Dataset example containing 'answers' (list of responses) and 'label'
        num_runs: Number of times the prompt was run
        threshold: Minimum fraction of correct answers (e.g., 0.6 for 60%)
    
    Returns:
        Updated example with 'correct' label
    """
    answers = example['answers']
    label = example['label']
    correct_count = 0
    
    for answer in answers:
        try:
            if 'Answer' not in answer:
                checked_label = answer[1]
            else:
                checked_label = answer[9]
        except:
            continue
            
        if checked_label in ['A', 'B', 'C', 'D'] and checked_label == label:
            correct_count += 1
    
    # Apply threshold: label as 1 if correct_count/num_runs >= threshold
    if correct_count / num_runs >= threshold:
        example['correct'] = 1
    else:
        example['correct'] = 0
    
    example['correct_count'] = correct_count
    example['total_runs'] = num_runs
    
    return example

def create_clean_auxiliary_dataset_with_multiple_runs(num_runs=5, threshold=0.6):
    """
    Creates a labeled dataset by running each prompt multiple times.
    
    Args:
        num_runs: Number of times to run each prompt (default: 5)
        threshold: Minimum fraction of correct answers to label as 1 (default: 0.6)
    """
    model_name = generation_config.model_name
    formatted_mmlu_Mistral_file = generation_config.mmlu_dataset_folder + '/mmlu_auxiliary_formatted.pkl'
    mmlu_auxiliary_with_responses_file = generation_config.mmlu_dataset_folder + f'/mmlu_auxiliary_with_responses_n{num_runs}_t{threshold}.pkl'
    dataset_name = "cais/mmlu"
    
    # Load the dataset
    raw_dataset = DatasetLoader.load_auxiliary_train(dataset_name, 'auxiliary_train', formatted_mmlu_Mistral_file)
    prompts = raw_dataset['prompt']
    
    # Collect all responses for each prompt
    all_answers = []
    
    print(f"Running each prompt {num_runs} times...")
    for run_idx in range(num_runs):
        print(f"Run {run_idx + 1}/{num_runs}")
        generate_responses = GenerateResponses(model_name, prompts)
        raw_responses = generate_responses.generate_responses()
        
        # Extract responses for this run
        run_answers = [output.outputs[0].text for output in raw_responses]
        all_answers.append(run_answers)
    
    # Transpose: from [num_runs x num_prompts] to [num_prompts x num_runs]
    answers_per_prompt = list(zip(*all_answers))
    
    # Add answers to dataset
    dataset = raw_dataset.add_column('answers', [list(answers) for answers in answers_per_prompt])
    dataset = DatasetLoader.format_labels(dataset)
    
    # Apply threshold-based labeling
    dataset = dataset.map(lambda x: add_correct_label_with_threshold(x, num_runs, threshold))
    dataset = dataset.shuffle(seed=42)
    
    print(f"Dataset created with {len(dataset)} examples")
    print(f"Correct (label=1): {sum(1 for x in dataset if x['correct'] == 1)}")
    print(f"Incorrect (label=0): {sum(1 for x in dataset if x['correct'] == 0)}")
    
    with open(mmlu_auxiliary_with_responses_file, 'wb') as f:
        pickle.dump(dataset, f)
    
    return dataset

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

    # Generate dataset with multiple runs: 5 runs, require at least 4 correct (0.8 threshold)
    print("Creating dataset with multiple runs (n=5, threshold=0.8)...")
    dataset = create_clean_auxiliary_dataset_with_multiple_runs(num_runs=5, threshold=0.8)
    
    # Downsample the dataset to balance classes
    print("\nDownsampling dataset to balance classes...")
    downsampled_dataset = downsample_dataset(dataset)
    
    # Save downsampled dataset
    downsampled_label_file = generation_config.mmlu_dataset_folder + '/mmlu_auxiliary_labeled_downsampled_n5_t0.8.pkl'
    with open(downsampled_label_file, 'wb') as f:
        pickle.dump(downsampled_dataset, f)
    print(f"Downsampled dataset saved to {downsampled_label_file}")
    
    # Split the dataset into train, test, and validation
    print("\nSplitting dataset into train/test/validation...")
    train_dataset, test_dataset, validation_dataset = split_dataset(downsampled_dataset)
    
    # Save the datasets to pickle files
    train_dataset_file = generation_config.mmlu_dataset_folder + '/mmlu_auxiliary_train_n5_t0.8.pkl'
    test_dataset_file = generation_config.mmlu_dataset_folder + '/mmlu_auxiliary_test_n5_t0.8.pkl'
    validation_dataset_file = generation_config.mmlu_dataset_folder + '/mmlu_auxiliary_validation_n5_t0.8.pkl'
    
    with open(train_dataset_file, 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(test_dataset_file, 'wb') as f:
        pickle.dump(test_dataset, f)
    with open(validation_dataset_file, 'wb') as f:
        pickle.dump(validation_dataset, f)
    
    print(f"\nTrain dataset saved to {train_dataset_file} ({len(train_dataset)} examples)")
    print(f"Test dataset saved to {test_dataset_file} ({len(test_dataset)} examples)")
    print(f"Validation dataset saved to {validation_dataset_file} ({len(validation_dataset)} examples)")
    print("\nDataset creation complete!")