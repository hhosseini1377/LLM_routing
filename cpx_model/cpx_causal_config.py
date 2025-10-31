from transformers import AutoConfig
from cpx_model.config import CPXTrainingConfig


class CPXConfig:
    """
    General configuration wrapper for CPX models.
    Works with any model configuration from transformers.
    """
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load a pretrained model config and add CPX-specific parameters.
        
        Args:
            pretrained_model_name_or_path: Model name or path
            **kwargs: Additional config parameters including:
                - num_labels: Number of classification labels (default: 1)
                - cpx_token: The CPX token string (default: '[CPX]')
                - cpx_token_id: The CPX token ID (must be set after tokenizer)
        
        Returns:
            config: Model config with CPX parameters added
        """
        # Load base config using AutoConfig
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        # Add CPX-specific parameters
        config.num_labels = kwargs.get("num_labels", 1)
        config.cpx_token = kwargs.get("cpx_token", '[CPX]')
        
        # cpx_token_id should be set after tokenizer initialization
        if 'cpx_token_id' in kwargs:
            config.cpx_token_id = kwargs['cpx_token_id']
        
        return config

