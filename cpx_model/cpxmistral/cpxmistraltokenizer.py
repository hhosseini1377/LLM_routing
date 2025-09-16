from transformers import AutoTokenizer
from cpx_model.cpxmistral.config import MistralTrainingConfig

class CPXMistralTokenizer(AutoTokenizer):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        cpx_token = kwargs.get("cpx_token", MistralTrainingConfig.cpx_token)
        tokenizer.add_special_tokens({'additional_special_tokens': [cpx_token]})
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer