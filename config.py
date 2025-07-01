from transformers import GenerationConfig

generator_config = {
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.2,
    "do_sample": True,
}
