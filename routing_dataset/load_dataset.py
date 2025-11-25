import pickle
import logging
import re
from pathlib import Path
from typing import Optional, Union, Dict, List
from datasets import load_dataset
from routing_dataset.prompt_formats import (
    create_non_thinking_mmlu_prompt, 
    create_non_thinking_mmlu_pro_prompt,
    create_gsm8k_prompt,
    ModelType
)
from routing_dataset.dataset_types import (
    MMLUSplit, 
    MMLU_SPLIT_TO_OUTPUT_FILE, 
    MMLUProSplit, 
    MMLU_PRO_SPLIT_TO_OUTPUT_FILE,
    GSMSplit,
    GSM_SPLIT_TO_OUTPUT_FILE
)

logger = logging.getLogger(__name__)

def load_mmlu_pro_dataset(
    split: MMLUProSplit,
    output_file: Optional[Union[str, Path]] = None,
    model_type: ModelType = "qwen"
) -> Dict[str, List[str]]:
    """
    Load a split of MMLU-Pro dataset, format prompts using create_non_thinking_mmlu_pro_prompt,
    and save prompts and ground truths to a file.
    
    MMLU-Pro can have up to 10 choices (A through J), unlike regular MMLU which has 4 choices.
    
    Args:
        split: Dataset split to load. Must be one of: 'test', 'validation'
        output_file: Path to save the output file. 
                    If None, automatically determined from split name.
        model_type: Model type for prompt formatting. Options: 'qwen' or 'mistral'.
                   Defaults to 'qwen'.
    
    Returns:
        Dictionary containing 'prompts' and 'ground_truths' lists, both as List[str]
    
    Raises:
        ValueError: If split is not one of the supported values or dataset structure is invalid.
        FileNotFoundError: If dataset cannot be loaded from HuggingFace.
        IOError: If file cannot be written.
    """
    # Load MMLU-Pro split from HuggingFace
    logger.info(f"Loading MMLU-Pro {split} split...")
    try:
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", trust_remote_code=True)[split]
    except Exception as e:
        raise FileNotFoundError(f"Failed to load MMLU-Pro dataset split '{split}': {e}") from e
    
    # Validate dataset structure
    if len(dataset) == 0:
        raise ValueError(f"Dataset split '{split}' is empty")
    
    # Check required fields in first example
    first_example = dataset[0]
    required_fields = ['question', 'options', 'answer_index']
    missing_fields = [field for field in required_fields if field not in first_example]
    if missing_fields:
        raise ValueError(f"Dataset missing required fields: {missing_fields}")
    
    prompts = []
    ground_truths = []
    
    # Process each example in the dataset
    for idx, example in enumerate(dataset):
        try:
            # Extract question and choices
            question = example['question']
            choices = example['options']  # List of choices (can be 2-10 items)
            
            # Validate choices structure
            if not isinstance(choices, list) or len(choices) < 2 or len(choices) > 10:
                logger.warning(
                    f"Example {idx} has invalid choices structure "
                    f"(expected 2-10 items, got {len(choices) if isinstance(choices, list) else type(choices)}). Skipping."
                )
                continue
            
            # Convert choices list to options dictionary (A, B, C, ..., J)
            options = {}
            for i, choice_text in enumerate(choices):
                letter = chr(ord('A') + i)  # A, B, C, ..., J
                options[letter] = choice_text
            
            # Extract subject name from dataset if available, otherwise use generic
            subject_name = example.get('subject', "General Knowledge")
            
            # Create formatted prompt using create_non_thinking_mmlu_pro_prompt
            prompt = create_non_thinking_mmlu_pro_prompt(subject_name, question, options, model_type=model_type)
            
            # Convert answer (0-based index) to letter ('A'-'J')
            answer_idx = example['answer_index']
            if not isinstance(answer_idx, int) or answer_idx < 0 or answer_idx >= len(choices):
                logger.warning(
                    f"Example {idx} has invalid answer index {answer_idx} "
                    f"(expected 0-{len(choices)-1}). Skipping."
                )
                continue
            
            ground_truth = chr(ord('A') + answer_idx)  # 0->'A', 1->'B', ..., 9->'J'
            
            prompts.append(prompt)
            ground_truths.append(ground_truth)
        except KeyError as e:
            logger.warning(f"Example {idx} missing required field: {e}. Skipping.")
            continue
        except Exception as e:
            logger.warning(f"Error processing example {idx}: {e}. Skipping.")
            continue
    
    if len(prompts) == 0:
        raise ValueError(f"No valid examples found in dataset split '{split}'")
    
    # Create output dictionary
    output_data = {
        'prompts': prompts,
        'ground_truths': ground_truths
    }
    
    # Determine output file path
    if output_file is None:
        if split not in MMLU_PRO_SPLIT_TO_OUTPUT_FILE:
            raise ValueError(
                f"Unknown split '{split}'. Supported splits: {list(MMLU_PRO_SPLIT_TO_OUTPUT_FILE.keys())}. "
                "Please specify output_file explicitly."
            )
        output_file = MMLU_PRO_SPLIT_TO_OUTPUT_FILE[split]
    
    # Ensure output directory exists
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file
    logger.info(f"Saving {len(prompts)} prompts and ground truths to {output_file}...")
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)
        logger.info(f"Successfully saved prompts and ground truths to {output_file}")
    except IOError as e:
        raise IOError(f"Failed to write output file '{output_file}': {e}") from e
    
    return output_data

def load_mmlu_split(
    split: MMLUSplit,
    output_file: Optional[Union[str, Path]] = None,
    model_type: ModelType = "qwen"
) -> Dict[str, List[str]]:
    """
    Load a split of MMLU dataset, format prompts using create_non_thinking_mmlu_prompt,
    and save prompts and ground truths to a file.
    
    Args:
        split: Dataset split to load. Must be one of: 'auxiliary', 'test', 'validation', 'dev'
        output_file: Path to save the output file. 
                    If None, automatically determined from split name.
        model_type: Model type for prompt formatting. Options: 'qwen' or 'mistral'.
                   Defaults to 'qwen'.
    
    Returns:
        Dictionary containing 'prompts' and 'ground_truths' lists, both as List[str]
    
    Raises:
        ValueError: If split is not one of the supported values or dataset structure is invalid.
        FileNotFoundError: If dataset cannot be loaded from HuggingFace.
        IOError: If file cannot be written.
    """
    # Load MMLU split from HuggingFace
    logger.info(f"Loading MMLU {split} split...")
    try:
        dataset = load_dataset("cais/mmlu", "all", split=split)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load MMLU dataset split '{split}': {e}") from e
    
    # Validate dataset structure
    if len(dataset) == 0:
        raise ValueError(f"Dataset split '{split}' is empty")
    
    # Check required fields in first example
    first_example = dataset[0]
    required_fields = ['question', 'choices', 'answer']
    missing_fields = [field for field in required_fields if field not in first_example]
    if missing_fields:
        raise ValueError(f"Dataset missing required fields: {missing_fields}")
    
    prompts = []
    ground_truths = []
    
    # Process each example in the dataset
    for idx, example in enumerate(dataset):
        try:
            # Extract question and choices
            question = example['question']
            choices = example['choices']  # List of 4 choices
            
            # Validate choices structure
            if not isinstance(choices, list) or len(choices) != 4:
                logger.warning(f"Example {idx} has invalid choices structure (expected 4 items, got {len(choices) if isinstance(choices, list) else type(choices)}). Skipping.")
                continue
            
            # Convert choices list to options dictionary
            options = {
                'A': choices[0],
                'B': choices[1],
                'C': choices[2],
                'D': choices[3]
            }
            
            # Extract subject name from dataset if available, otherwise use generic
            subject_name = example.get('subject', "General Knowledge")
            
            # Create formatted prompt using create_non_thinking_mmlu_prompt
            prompt = create_non_thinking_mmlu_prompt(subject_name, question, options, model_type=model_type)
            
            # Convert answer (0-3) to letter ('A'-'D')
            answer_idx = example['answer']
            if not isinstance(answer_idx, int) or answer_idx < 0 or answer_idx > 3:
                logger.warning(f"Example {idx} has invalid answer index {answer_idx}. Skipping.")
                continue
            
            ground_truth = chr(ord('A') + answer_idx)  # 0->'A', 1->'B', 2->'C', 3->'D'
            
            prompts.append(prompt)
            ground_truths.append(ground_truth)
        except KeyError as e:
            logger.warning(f"Example {idx} missing required field: {e}. Skipping.")
            continue
        except Exception as e:
            logger.warning(f"Error processing example {idx}: {e}. Skipping.")
            continue
    
    if len(prompts) == 0:
        raise ValueError(f"No valid examples found in dataset split '{split}'")
    
    # Create output dictionary
    output_data = {
        'prompts': prompts,
        'ground_truths': ground_truths
    }
    
    # Determine output file path
    if output_file is None:
        if split not in MMLU_SPLIT_TO_OUTPUT_FILE:
            raise ValueError(
                f"Unknown split '{split}'. Supported splits: {list(MMLU_SPLIT_TO_OUTPUT_FILE.keys())}. "
                "Please specify output_file explicitly."
            )
        output_file = MMLU_SPLIT_TO_OUTPUT_FILE[split]
    
    # Ensure output directory exists
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file
    logger.info(f"Saving {len(prompts)} prompts and ground truths to {output_file}...")
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)
        logger.info(f"Successfully saved prompts and ground truths to {output_file}")
    except IOError as e:
        raise IOError(f"Failed to write output file '{output_file}': {e}") from e
    
    return output_data


def load_gsm8k_split(
    split: GSMSplit,
    output_file: Optional[Union[str, Path]] = None,
    model_type: ModelType = "qwen"
) -> Dict[str, List[str]]:
    """
    Load a split of GSM8K dataset, format prompts using create_gsm8k_prompt,
    and save prompts and ground truths to a file.
    
    GSM8K is a dataset of grade school math word problems. Unlike MMLU which is
    multiple choice, GSM8K requires free-form numerical answers.
    
    Args:
        split: Dataset split to load. Must be 'train' or 'test'
        output_file: Path to save the output file. 
                    If None, automatically determined from split name.
        model_type: Model type for prompt formatting. Options: 'qwen' or 'mistral'.
                   Defaults to 'qwen'.
    
    Returns:
        Dictionary containing 'prompts' and 'ground_truths' lists, both as List[str]
        Ground truths are the numerical answers (as strings).
    
    Raises:
        ValueError: If split is not one of the supported values or dataset structure is invalid.
        FileNotFoundError: If dataset cannot be loaded from HuggingFace.
        IOError: If file cannot be written.
    """
    # Load GSM8K split from HuggingFace
    logger.info(f"Loading GSM8K {split} split...")
    try:
        dataset = load_dataset("gsm8k", "main")[split]
    except Exception as e:
        raise FileNotFoundError(f"Failed to load GSM8K dataset split '{split}': {e}") from e
    
    # Validate dataset structure
    if len(dataset) == 0:
        raise ValueError(f"Dataset split '{split}' is empty")
    
    # Check required fields in first example
    first_example = dataset[0]
    required_fields = ['question', 'answer']
    missing_fields = [field for field in required_fields if field not in first_example]
    if missing_fields:
        raise ValueError(f"Dataset missing required fields: {missing_fields}")
    
    prompts = []
    ground_truths = []
    
    # Process each example in the dataset
    for idx, example in enumerate(dataset):
        try:
            # Extract question
            question = example['question']
            
            # Validate question
            if not isinstance(question, str) or len(question.strip()) == 0:
                logger.warning(f"Example {idx} has invalid question. Skipping.")
                continue
            
            # Extract answer - GSM8K answer can be just a number or include reasoning
            # The answer field typically contains the final numerical answer
            answer = example['answer']
            
            # Extract numerical answer from answer field
            # GSM8K answers are typically formatted as "#### <number>" or just "<number>"
            if isinstance(answer, str):
                # Try to extract number after ####
                match = re.search(r'####\s*([0-9,.\-]+)', answer)
                if match:
                    ground_truth = match.group(1).replace(',', '')  # Remove commas
                else:
                    # Try to extract any number at the end
                    match = re.search(r'([0-9,.\-]+)\s*$', answer.strip())
                    if match:
                        ground_truth = match.group(1).replace(',', '')
                    else:
                        # Use the whole answer string if no number found
                        ground_truth = answer.strip()
            elif isinstance(answer, (int, float)):
                ground_truth = str(answer)
            else:
                logger.warning(f"Example {idx} has invalid answer type {type(answer)}. Skipping.")
                continue
            
            # Create formatted prompt using create_gsm8k_prompt
            prompt = create_gsm8k_prompt(question, model_type=model_type)
            
            prompts.append(prompt)
            ground_truths.append(ground_truth)
        except KeyError as e:
            logger.warning(f"Example {idx} missing required field: {e}. Skipping.")
            continue
        except Exception as e:
            logger.warning(f"Error processing example {idx}: {e}. Skipping.")
            continue
    
    if len(prompts) == 0:
        raise ValueError(f"No valid examples found in dataset split '{split}'")
    
    # Create output dictionary
    output_data = {
        'prompts': prompts,
        'ground_truths': ground_truths
    }
    
    # Determine output file path
    if output_file is None:
        if split not in GSM_SPLIT_TO_OUTPUT_FILE:
            raise ValueError(
                f"Unknown split '{split}'. Supported splits: {list(GSM_SPLIT_TO_OUTPUT_FILE.keys())}. "
                "Please specify output_file explicitly."
            )
        output_file = GSM_SPLIT_TO_OUTPUT_FILE[split]
    
    # Ensure output directory exists
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file
    logger.info(f"Saving {len(prompts)} prompts and ground truths to {output_file}...")
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)
        logger.info(f"Successfully saved prompts and ground truths to {output_file}")
    except IOError as e:
        raise IOError(f"Failed to write output file '{output_file}': {e}") from e
    
    return output_data


