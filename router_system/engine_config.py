from dataclasses import dataclass

@dataclass
class EngineConfig:
    """
    Configuration for the LLM engine.
    """
    model: str = "mistralai/Mistral-7B-Instruct-v0.1"
    max_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.95
    top_k: int = 40
    num_gpus: int = 1
    num_cpus: int = 4
    max_batch_size: int = 8