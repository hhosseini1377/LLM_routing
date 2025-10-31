from vllm import LLM, SamplingParams
from datasets import load_dataset, get_dataset_config_names
import pickle
from datasets import Dataset
from generate_dataset.generation_config import generation_config
import random
from datasets import concatenate_datasets
import argparse


def format_mistral_mmlu_prompt(question: str, choices: list[str]) -> str:
    """
    Formats a multiple-choice question and choices into a prompt
    optimized for the Mistral-7B-Instruct model.

    The prompt explicitly instructs the model to output only the
    letter corresponding to the correct answer (A, B, C, or D)
    as the very first character of its response.

    Args:
        question (str): The text of the question.
        choices (list[str]): A list of the answer choices.
                             Assumes there are 4 choices.

    Returns:
        str: The correctly formatted prompt string.
    """
    labels = ["A", "B", "C", "D"]

    # Build the multiple-choice question body
    prompt_body = f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt_body += f"{labels[i]}. {choice}\n"

    # Strong, explicit instruction for deterministic output
    instruction = (
        "\nAnswer the question by writing ONLY the letter of the correct choice "
        "(A, B, C, or D) as the first character of your response. "
        "Do not include any explanation or additional text.\n"
        "Example format:\nAnswer: A\n\n"
        "Now provide your answer:\nAnswer:"
    )

    formatted_prompt = f"[INST] {prompt_body}{instruction} [/INST]"
    return formatted_prompt

class GenerateResponses:
    def __init__(self, model_name, prompts):
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        max_tokens = generation_config.max_tokens
        model_name = generation_config.model_name
        max_num_sequences = generation_config.max_num_sequences
        dtype = generation_config.dtype
        gpu_memory_utilization = generation_config.gpu_memory_utilization
        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        self.llm = LLM(
            model=model_name, 
            dtype=dtype, 
            max_num_seqs=max_num_sequences, 
            gpu_memory_utilization=gpu_memory_utilization, 
            tokenizer_mode="mistral"
        )
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
    def load_mmlu_dataset(cls, dataset_name, config, splits, dist_file = None):
        prompts = []
        labels = []
        for split in splits:
            dataset = load_dataset(dataset_name, config)[split]
    
            for row in dataset:
                formatted_prompt = format_mistral_mmlu_prompt(row['question'], row['choices'])
                prompts.append(formatted_prompt)
                labels.append(row['answer'])
        dataset_dict = {
            'prompt': prompts,
            'label': labels
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
                raise ValueError(f'Found an invalid label')

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
    print(DatasetLoader.incorrect_num )
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

def find_valid_indexes(data) -> list[int]:

    valid_indexes = []
    for index, example in enumerate(data):
        correct_flag = True
        for answer in example['answers']:
            try:
                if 'Answer' not in answer:
                    checked_label = answer[1]
                else:
                    checked_label = answer[9]
            except:
                correct_flag = False
                break

            if checked_label not in ['A', 'B', 'C', 'D']:
                correct_flag = False
                break
        if correct_flag:
            valid_indexes.append(index)

    return valid_indexes

def add_correct_counts(data):
    valid_indexes = find_valid_indexes(data)
    valid_dataset = data.select(valid_indexes)
    valid_dataset = DatasetLoader.format_labels(valid_dataset)
    correct_counts = []
    for index, example in enumerate(valid_dataset):
        answers = example['answers']
        ground_truth = example['label']
        correct_count = 0
        for answer in answers:
            if 'Answer' not in answer:
                checked_label = answer[1]
            else:
                checked_label = answer[9]
            if checked_label not in ['A', 'B', 'C', 'D']:
                print('bad answer')
            if checked_label == ground_truth:
                correct_count += 1
        correct_counts.append(correct_count)

    # Append correct counts to the valid dataset
    valid_dataset = valid_dataset.add_column('correct_count', correct_counts)

    all_correct = []

    for count in correct_counts:
        if count >= 4:
            all_correct.append(1)
        else:
            all_correct.append(0)

    valid_dataset = valid_dataset.add_column('correct', all_correct)
    return valid_dataset

def create_clean_mmlu_dataset_with_multiple_runs(dataset_name, config, splits, num_runs=5):
    """
    Creates a labeled dataset by running each prompt multiple times.
    
    Args:
        num_runs: Number of times to run each prompt (default: 5)
        threshold: Minimum fraction of correct answers to label as 1 (default: 0.6)
    """
    model_name = generation_config.model_name
    formatted_mmlu_Mistral_file = generation_config.mmlu_dataset_folder + f'/mmlu_{config}_{splits}_formatted.pkl'
    mmlu_with_answers_file = generation_config.mmlu_dataset_folder + f'/mmlu_{config}_{splits}_with_answers_n{num_runs}.pkl'
    mmlu_with_correct_counts_file = generation_config.mmlu_dataset_folder + f'/mmlu_{config}_{splits}_with_correct_counts_n{num_runs}.pkl'
    # Load the dataset
    raw_dataset = DatasetLoader.load_mmlu_dataset(dataset_name, config, splits, formatted_mmlu_Mistral_file)

    prompts = raw_dataset['prompt']
    
    # Collect all responses for each prompt
    all_answers = []
    
    # Create the model once and reuse it
    print(f"Loading model for {num_runs} runs...")
    generate_responses = GenerateResponses(model_name, prompts)
    
    print(f"Running each prompt {num_runs} times...")
    for run_idx in range(num_runs):
        print(f"Run {run_idx + 1}/{num_runs}")
        raw_responses = generate_responses.generate_responses()
        
        # Extract responses for this run
        run_answers = [output.outputs[0].text for output in raw_responses]
        all_answers.append(run_answers)
    
    # Transpose: from [num_runs x num_prompts] to [num_prompts x num_runs]
    answers_per_prompt = list(zip(*all_answers))
    
    # Add answers to dataset
    dataset = raw_dataset.add_column('answers', [list(answers) for answers in answers_per_prompt])
    dataset = DatasetLoader.format_labels(dataset)
    with open(mmlu_with_answers_file, 'wb') as f:
        pickle.dump(dataset, f)  
    print(f'dataset with answers saved to {mmlu_with_answers_file}')
    
    dataset_with_correct_counts = add_correct_counts(dataset)
    with open(mmlu_with_correct_counts_file, 'wb') as f:
        pickle.dump(dataset_with_correct_counts, f)
    print(f'dataset with correct counts saved to {mmlu_with_correct_counts_file}')
    return dataset_with_correct_counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="all")
    parser.add_argument("--splits", type=str, nargs='+', default=["test"])
    parser.add_argument("--num_runs", type=int, default=5)
    args = parser.parse_args()
    dataset_name = "cais/mmlu"
    config = args.config
    splits = args.splits
    num_runs = args.num_runs
    # Generate dataset with multiple runs: 5 runs, require at least 4 correct (0.8 threshold)
    dataset_with_correct_counts = create_clean_mmlu_dataset_with_multiple_runs(dataset_name, config, splits, num_runs)
    
    