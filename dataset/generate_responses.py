from vllm import LLM, SamplingParams
from datasets import load_dataset, get_dataset_config_names
import pickle
from datasets import Dataset
from generation_config import generation_config

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
        outputs = self.llm.generate(prompts, self.sampling_params)
        return outputs

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

def add_correctness(example):
    answer = example['outputs']
    label = example['label']
    try:
        checked_label = answer[9]
    except:
        print(f'Found an invalid lable {answer}')
        example['correctness'] = 0
        DatasetLoader.incorrect_num += 1
        return example
    if checked_label not in ['A', 'B', 'C', 'D']:
        print(f'Found an invalid lable')
        DatasetLoader.incorrect_num += 1
        example['correctness'] = 0
    elif checked_label == label:
        example['correctness'] = 1
    else:
        example['correctness'] = 0
    return example

if __name__ == "__main__":

    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    formatted_mmlu_Mistral_file = generation_config.mmlu_dataset_folder + '/mmlu_auxiliary_formatted.pkl'
    mmlu_Mistral_responses_file = generation_config.mmlu_dataset_folder + '/mmlu_auxiliary_Mistral_responses.pkl'
    dataset_name = "cais/mmlu"
    
    # formatted_mmlu = DatasetLoader.load_auxiliary_train(dataset_name, 'auxiliary_train', formatted_mmlu_Mistral_file)
    # prompts = formatted_mmlu['prompt']

    # generate_responses = GenerateResponses(model_name, prompts)
    # outputs = generate_responses.generate_responses()

    # # save the outputs to a pickle file
    # with open(mmlu_Mistral_responses_file, 'wb') as f:
    #     pickle.dump(outputs, f)
    # print(outputs)

    # read from the pickle file
    with open(mmlu_Mistral_responses_file, 'rb') as f:
        outputs = pickle.load(f)
    print(outputs[0])