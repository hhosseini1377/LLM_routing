from transformers import MistralConfig
import warnings 
from cpx_model.cpxmistral.config import MistralTrainingConfig
class CPXMistralConfig(MistralConfig):
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        config.num_labels = kwargs.get("num_labels", 1)
        config.cpx_token = kwargs.get("cpx_token", MistralTrainingConfig.cpx_token)
        return config
        