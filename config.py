generator_config_with_sampling = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "repetition_penalty": 1.2,
    "do_sample": True,
}

generator_config_without_sampling = {
    "max_new_tokens": 1024,
    "temperature": 0.0,
    "top_p": 1.0,
    "do_sample": False,
}
