from typing import Literal

ModelType = Literal["qwen", "mistral"]

def create_non_thinking_mmlu_prompt(
    subject_name: str, 
    question: str, 
    options: dict,
    model_type: str = "qwen"
) -> str:
    """
    Generates a zero-shot MMLU prompt that forces Qwen3/DeepSeek-R1 
    to bypass 'Thinking Mode' and output only the answer letter.
    """

    # 1. System Instruction: The "/no_think" command is CRITICAL for Qwen 3.
    #    It acts as a 'soft switch' to disable the internal monologue.
    system_instruction = (
        f"/no_think You are an expert academic assistant. "
        f"Answer the following multiple-choice question by giving ONLY the correct "
        f"option letter (A, B, C, or D). Do not explain your reasoning."
    )

    # 2. Build the Question Block
    question_block = f"Q: {question}\n"
    for letter, text in options.items():
        question_block += f"{letter}) {text}\n"
    
    # 3. The Pre-fill: Ends with a SPACE to prevent tokenization glue issues.
    #    e.g., prevents the model from trying to merge ":" and "A" into one token.
    final_instruction = "Answer: "

    if model_type == "mistral":
        # Mistral / Llama format (Instruction inside [INST])
        # Note: We place 'Answer: ' AFTER the [/INST] to force pre-fill behavior.
        user_content = (
            f"{system_instruction}\n\n"
            f"The following is a multiple-choice question about {subject_name}.\n\n"
            f"{question_block.strip()}"
        )
        prompt = f"<s>[INST] {user_content} [/INST] {final_instruction}"
        
    else:
        # Qwen / ChatML Format
        prompt = (
            f"<|im_start|>system\n"
            f"{system_instruction}\n"  # Includes /no_think
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"The following is a multiple-choice question about {subject_name}.\n\n"
            f"{question_block.strip()}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{final_instruction}"      # <--- CRITICAL: Ends prompt with "Answer: "
        )
    
    return prompt


def create_non_thinking_mmlu_pro_prompt(
    subject_name: str, 
    question: str, 
    options: dict,
    model_type: str = "qwen"
) -> str:
    """
    Generates a zero-shot MMLU-Pro prompt that forces Qwen3/DeepSeek-R1 
    to bypass 'Thinking Mode' and output only the answer letter.
    
    MMLU-Pro can have up to 10 choices (A through J).
    
    Args:
        subject_name: The name of the MMLU-Pro subject
        question: The question text
        options: Dictionary mapping letters to option text, e.g., {'A': 'Option A', 'B': 'Option B', ..., 'J': 'Option J'}
        model_type: Model type ('qwen' or 'mistral')
    
    Returns:
        Formatted prompt string
    """
    # Determine the valid letter range based on number of options
    num_options = len(options)
    if num_options > 10:
        raise ValueError(f"MMLU-Pro supports up to 10 choices, got {num_options}")
    
    # Create letter range string (e.g., "A through J" or "A through D")
    if num_options <= 4:
        letter_range = "A, B, C, or D"
    elif num_options <= 10:
        last_letter = chr(ord('A') + num_options - 1)
        letter_range = f"A through {last_letter}"
    else:
        letter_range = "A through J"
    
    # 1. System Instruction: The "/no_think" command is CRITICAL for Qwen 3.
    system_instruction = (
        f"/no_think You are an expert academic assistant. "
        f"Answer the following multiple-choice question by giving ONLY the correct "
        f"option letter ({letter_range}). Do not explain your reasoning."
    )

    # 2. Build the Question Block
    question_block = f"Q: {question}\n"
    for letter, text in options.items():
        question_block += f"{letter}) {text}\n"
    
    # 3. The Pre-fill: Ends with a SPACE to prevent tokenization glue issues.
    final_instruction = "Answer: "

    if model_type == "mistral":
        # Mistral / Llama format
        user_content = (
            f"{system_instruction}\n\n"
            f"The following is a multiple-choice question about {subject_name}.\n\n"
            f"{question_block.strip()}"
        )
        prompt = f"<s>[INST] {user_content} [/INST] {final_instruction}"
        
    else:
        # Qwen / ChatML Format
        prompt = (
            f"<|im_start|>system\n"
            f"{system_instruction}\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"The following is a multiple-choice question about {subject_name}.\n\n"
            f"{question_block.strip()}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
            f"{final_instruction}"
        )
    
    return prompt


def create_gsm8k_prompt(
    question: str,
    model_type: str = "qwen"
) -> str:
    """
    Generates a GSM8K math word problem prompt.
    
    GSM8K is a dataset of grade school math word problems that require step-by-step
    reasoning to solve. The model should solve the problem and provide a numerical answer.
    
    Args:
        question: The math word problem text
        model_type: Model type ('qwen' or 'mistral')
    
    Returns:
        Formatted prompt string
    """
    # System instruction for math problem solving
    # Note: We don't use /no_think here because GSM8K requires step-by-step reasoning
    system_instruction = (
        "You are a helpful math assistant. "
        "Solve the following math word problem step by step. "
        "After your reasoning, provide the final numerical answer."
    )
    
    if model_type == "mistral":
        # Mistral / Llama format
        user_content = (
            f"{system_instruction}\n\n"
            f"Question: {question.strip()}"
        )
        prompt = f"<s>[INST] {user_content} [/INST]"
        
    else:
        # Qwen / ChatML Format
        prompt = (
            f"<|im_start|>system\n"
            f"{system_instruction}\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Question: {question.strip()}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    
    return prompt