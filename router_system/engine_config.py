from dataclasses import dataclass
from typing import Optional
@dataclass
class EngineConfig:
    """
    Configuration for the LLM engine.
    """
    model: str = "mistralai/Mistral-7B-Instruct-v0.1"
    quantization : Optional[str] = None
    dtype: str = "float16"
    swap_space: int = 3
    enforce_eager: bool = True
    max_model_len: int = 1024
    kv_cache_dtype: str = "fp8_e5m2"
    max_num_seqs: int = 8
    memory_utilization: float = 0.5
    max_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.95
    top_k: int = 40
    num_gpus: int = 1
    num_cpus: int = 4
    max_batch_size: int = 8
    max_tokens_per_request: int = 512