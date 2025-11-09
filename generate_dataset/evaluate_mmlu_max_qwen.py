from vllm import LLM, SamplingParams
from datasets import load_dataset, get_dataset_config_names, Dataset
import pickle
import argparse
import re
import os
from typing import List


def format_qwen_mmlu_prompt(question: str, choices: List[str]) -> str:
    """
    Formats a multiple-choice question and choices into a prompt
    optimized for Qwen3 models using their chat template.
    Supports up to 10 choices (A-J).
    
    Args:
        question (str): The text of the question.
        choices (List[str]): A list of the answer choices (up to 10 choices).
    
    Returns:
        str: The correctly formatted prompt string for Qwen.
    """
    # Generate labels dynamically based on number of choices (A-J for up to 10 choices)
    num_choices = len(choices)
    if num_choices > 10:
        raise ValueError(f"Too many choices: {num_choices}. Maximum is 10.")
    
    labels = [chr(65 + i) for i in range(num_choices)]  # A, B, C, ..., J
    
    # Build the multiple-choice question body
    prompt_body = f"Question: {question}\n\n"
    for i, choice in enumerate(choices):
        prompt_body += f"{labels[i]}. {choice}\n"
    
    # Create label list string for instruction
    if num_choices <= 4:
        label_list = ", ".join(labels)
    else:
        label_list = f"{', '.join(labels[:-1])}, or {labels[-1]}"
    
    # Instruction for Qwen to output answer
    instruction = (
        f"\nPlease answer the question by providing only the letter of the correct choice "
        f"({label_list}). Output your answer in the following format: "
        '{"answer": "X"} where X is the letter of your choice.'
    )
    
    # Use Qwen's chat template format
    # Qwen3 uses <|im_start|> and <|im_end|> tokens
    formatted_prompt = (
        "<|im_start|>user\n"
        f"{prompt_body}{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    return formatted_prompt


def extract_predicted_label(response: str) -> str:
    """
    Extracts the predicted label from the model's response.
    Tries multiple extraction methods to be robust.
    Supports labels A-J (up to 10 choices).
    
    Args:
        response (str): The model's response text.
    
    Returns:
        str: The extracted label (A-J), or empty string if not found.
    """
    # Method 1: Look for JSON format {"answer": "X"} where X is A-J
    json_match = re.search(r'["\']answer["\']\s*:\s*["\']([A-J])["\']', response, re.IGNORECASE)
    if json_match:
        return json_match.group(1).upper()
    
    # Method 2: Look for "answer": "X" without quotes
    json_match2 = re.search(r'answer\s*:\s*["\']?([A-J])["\']?', response, re.IGNORECASE)
    if json_match2:
        return json_match2.group(1).upper()
    
    # Method 3: Look for standalone letter A-J at the start or after newline
    letter_match = re.search(r'\b([A-J])\b', response, re.IGNORECASE)
    if letter_match:
        return letter_match.group(1).upper()
    
    # Method 4: Look for the first occurrence of A-J in the response
    first_letter = re.search(r'[A-J]', response, re.IGNORECASE)
    if first_letter:
        return first_letter.group(0).upper()
    
    return ""


class QwenMMLUEvaluator:
    def __init__(self, model_name: str, temperature: float = 0.0, top_p: float = 1.0, 
                 max_tokens: int = 128, gpu_memory_utilization: float = 0.8):
        """
        Initialize the Qwen MMLU evaluator.
        
        Args:
            model_name: Name/path of the Qwen model (e.g., "Qwen/Qwen3-32B-AWQ")
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            gpu_memory_utilization: GPU memory utilization ratio
        """
        self.model_name = model_name
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        
        print(f"Loading model: {model_name}")
        self.llm = LLM(
            model=model_name,
            quantization="awq",
            dtype="half",
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True
        )
        print("Model loaded successfully!")
    
    def generate_responses(self, prompts: List[str]):
        """Generate responses for a list of prompts."""
        outputs = self.llm.generate(prompts, self.sampling_params)
        return outputs


def load_mmlu_dataset(dataset_name: str, splits: List[str]):
    """
    Load MMLU dataset and format prompts.
    
    Args:
        dataset_name: Name of the dataset (e.g., "cais/mmlu")
        splits: List of splits to load (e.g., ["test"])
    
    Returns:
        Dataset with prompts and labels
    """
    prompts = []
    labels = []
    
    for split in splits:
        print(f"Loading {split} split...")
        dataset = load_dataset(dataset_name)[split]
        
        for row in dataset:
            formatted_prompt = format_qwen_mmlu_prompt(row['question'], row['options'])
            prompts.append(formatted_prompt)
            labels.append(row['answer_index'])
    
    # Convert numeric labels to letters (0-9 -> A-J)
    label_letters = []
    for label in labels:
        if 0 <= label <= 9:
            label_letters.append(chr(65 + label))  # 0->A, 1->B, ..., 9->J
        else:
            raise ValueError(f'Invalid label: {label}. Must be between 0 and 9.')
    
    dataset_dict = {
        'prompt': prompts,
        'label': label_letters
    }
    cleaned_dataset = Dataset.from_dict(dataset_dict)
    
    print(f"Loaded {len(cleaned_dataset)} examples")
    return cleaned_dataset


def evaluate_and_label(dataset: Dataset, evaluator: QwenMMLUEvaluator, 
                       output_file: str = None):
    """
    Evaluate the dataset, extract predicted labels, and add correctness labels.
    
    Args:
        dataset: Dataset with prompts and labels
        evaluator: QwenMMLUEvaluator instance
        output_file: Optional path to save the results
    
    Returns:
        Dataset with predicted labels and correctness
    """
    prompts = dataset['prompt']
    
    print("Generating responses...")
    outputs = evaluator.generate_responses(prompts)
    
    # Extract responses
    responses = [output.outputs[0].text for output in outputs]
    
    # Extract predicted labels
    predicted_labels = [extract_predicted_label(response) for response in responses]
    
    # Add columns to dataset
    dataset = dataset.add_column('response', responses)
    dataset = dataset.add_column('predicted_label', predicted_labels)
    
    # Calculate correctness
    correct_labels = []
    for i, example in enumerate(dataset):
        predicted = example['predicted_label']
        ground_truth = example['label']
        
        if predicted == ground_truth:
            correct_labels.append(1)
        else:
            correct_labels.append(0)
    
    dataset = dataset.add_column('correct', correct_labels)
    
    # Calculate and print statistics
    total = len(dataset)
    correct_count = sum(correct_labels)
    accuracy = correct_count / total if total > 0 else 0.0
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"{'='*60}")
    print(f"Total examples: {total}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"{'='*60}\n")
    
    # Count invalid predictions
    invalid_count = sum(1 for pred in predicted_labels if pred == "")
    if invalid_count > 0:
        print(f"Warning: {invalid_count} predictions could not be extracted")
    
    # Save if output file is provided
    if output_file:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Results saved to {output_file}")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-32B-AWQ on MMLU-max dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-32B-AWQ",
                       help="Model name or path")
    parser.add_argument("--splits", type=str, nargs='+', default=["test"],
                       help="Dataset splits to evaluate (default: ['test'])")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Path to save results (pickle format)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature (default: 0.0 for deterministic)")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="Top-p sampling parameter (default: 1.0)")
    parser.add_argument("--max_tokens", type=int, default=128,
                       help="Maximum tokens to generate (default: 128)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                       help="GPU memory utilization (default: 0.8)")
    
    args = parser.parse_args()
    
    # Load dataset
    dataset_name = "TIGER-Lab/MMLU-Pro"
    dataset = load_mmlu_dataset(dataset_name, args.splits)
    
    # Initialize evaluator
    evaluator = QwenMMLUEvaluator(
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization
    )
    
    # Evaluate and label
    if args.output_file is None:
        # Default output file name
        splits_str = "_".join(args.splits)
        args.output_file = f"./generate_dataset/datasets/MMLU/mmlu_{splits_str}_qwen_results.pkl"
    
    results = evaluate_and_label(dataset, evaluator, args.output_file)
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()

