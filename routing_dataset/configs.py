from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class SamplerConfig:
    """
    Configuration for sampler.
    """
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 128
    max_num_seqs: int = 256
    tokenizer_mode: Optional[str] = None
