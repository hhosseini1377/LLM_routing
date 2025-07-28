PROMPT_TEMPLATES = {
    "deepseek-ai/deepseek-llm-7b-chat": lambda ctx, inp: (
        "<|system|>\nYou are a helpful assistant.\n"
        "<|user|>\n"
        f"### Context:\n{ctx}\n\n### Question:\n{inp}\n"
        "<|assistant|>\n"
    ),
    "llama3-instruct": lambda ctx, inp: (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a helpful assistant.<|eot|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{ctx}\n\n{inp}<|eot|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    ),
    "mistral-instruct": lambda ctx, inp: (
        f"[INST] Context:\n{ctx}\n\nInstruction:\n{inp} [/INST]"
    ),
}
