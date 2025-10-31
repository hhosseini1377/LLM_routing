from transformers import AutoTokenizer
from cpx_model.config import CPXTrainingConfig


class CPXTokenizer:
    """
    General tokenizer wrapper that adds CPX token to any pretrained tokenizer.
    Works with any model supported by transformers AutoTokenizer.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cpx_token, **kwargs):
        """
        Load a pretrained tokenizer and add the CPX special token.
        
        Args:
            pretrained_model_name_or_path: Model name or path (e.g., "mistralai/Mistral-7B-Instruct-v0.1", 
                                          "meta-llama/Llama-2-7b-hf", "gpt2", etc.)
            cpx_token: The complexity token to add (default: '[CPX]')
            **kwargs: Additional arguments passed to AutoTokenizer.from_pretrained
        
        Returns:
            tokenizer: Tokenizer with CPX token added
        """
        # Load base tokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Add CPX special token
        tokenizer.add_special_tokens({'additional_special_tokens': [cpx_token]})
        
        # Set padding token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # ALWAYS set padding side to RIGHT for training
        # Left padding is for generation/inference, right padding is for training
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "left"
        tokenizer.truncation = True
        return tokenizer

