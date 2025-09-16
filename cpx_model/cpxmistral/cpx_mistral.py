from transformers import MistralForCausalLM, MistralConfig, AutoTokenizer
import torch.nn as nn
from cpx_model.cpxmistral.cpxmistralconfig import CPXMistralConfig
from typing import Tuple
import torch
from cpx_model.cpxmistral.config import MistralTrainingConfig
from warnings import warn
class MyMistral(MistralForCausalLM):
    def __init__(self, config):
        
        super().__init__(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        if hasattr(config, 'cpx_token_id'):
            self.cpx_token_id = config.cpx_token_id
        else:
            warn('cpx_token not found in config, using default')
            self.cpx_token_id = MistralTrainingConfig.cpx_token_id

        
    def mask_gradients(grad, cpx_id):
        mask = torch.zeros_like(grad)
        mask[cpx_id] = 1.0
        return grad * mask

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config, *model_args, **kwargs):
        # Load the base model using the parent class method
        model = super().from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path, config=config, *model_args, **kwargs)
        model.resize_token_embeddings(config.tokenizer_size)

        emb = model.get_input_embeddings().weight
        with torch.no_grad():
            model.get_input_embeddings().weight[model.cpx_token_id] = emb.mean(dim=0)

        
        # First, freeze ALL parameters
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the classifier
        for param in model.classifier.parameters():
            param.requires_grad = True
        embedding_layer = model.get_input_embeddings()

        # Freeze embedding weight
        for param in embedding_layer.parameters():
            param.requires_grad = True
        embedding_layer.weight.register_hook(lambda grad: model.mask_gradients(grad, model.cpx_token_id))
        
        # Reinitialize classifier with smaller weights to prevent explosion
        torch.nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
        if model.classifier.bias is not None:
            torch.nn.init.zeros_(model.classifier.bias)
        
        print("Classifier reinitialized with smaller weights")
        model.params_to_train = list(model.classifier.parameters()) + list(embedding_layer.parameters())
        
        return model

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        past_key_values = kwargs.get("past_key_values", None)
        seq_length = input_ids.shape[1]
        is_prefill = (past_key_values is None) or (seq_length != 1)
        phase = "prefilling" if is_prefill else "decoding"
        if phase == "prefilling":    
            classifier_logits, outputs =  self.prefill_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            # Return logits for training, outputs for generation
            if self.training:
                return classifier_logits
            else:
                return outputs
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, Padding= True, **kwargs)
    
    def generate_prefill(self, inputs=None, generation_config=None, logits_processor=None, stopping_criteria=None, prefix_allowed_tokens_fn=None, synced_gpus=None, assistant_model=None, streamer=None, negative_prompt_ids=None, negative_prompt_attention_mask=None, use_model_defaults=None, custom_generate=None, **kwargs):
        # Detect prefilling phase (e.g., by checking input shape or a flag)
        input_ids = inputs["input_ids"]
        if input_ids is not None:
            seq_length = len(input_ids[0])
            past_key_values = kwargs.get("past_key_values", None)
            is_prefill = (past_key_values is None) or (seq_length != 1)
            if is_prefill:
                # Run prefilling logic and return
                logits, outputs = self.prefill_forward(input_ids=input_ids, attention_mask=inputs.get("attention_mask") if isinstance(inputs, dict) else None, **kwargs)
                return logits, outputs  # or logits, outputs as needed

        # Otherwise, proceed with normal generation
        return super().generate(inputs, generation_config, logits_processor, stopping_criteria, prefix_allowed_tokens_fn, synced_gpus, assistant_model, streamer, negative_prompt_ids, negative_prompt_attention_mask, use_model_defaults, custom_generate, **kwargs)
    
    def prefill_forward(self, input_ids=None, attention_mask=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:

        # Check if config has cpx_token_id attribute
        if hasattr(self.config, 'cpx_token_id'):
            cpx_token_id = self.config.cpx_token_id
        else:
            raise ValueError("config must have cpx_token_id attribute set before using the model.")
        # Check if all the batch samples contain the cpx_token_id
        has_cpx_token = (input_ids == cpx_token_id).any(dim=1)
        batch_size = input_ids.size(0)
        if not has_cpx_token.all():
            print('there are samples without the cpx token')
            missing_indices = torch.where(~has_cpx_token)[0]
            warn(f"Samples at indices {missing_indices.tolist()} do not contain the special token ID {cpx_token_id}.")
            # For samples without CPX token, use the last token position instead
            cpx_positions = torch.zeros(input_ids.size(0), dtype=torch.long, device=input_ids.device)
            for i in range(input_ids.size(0)):
                if has_cpx_token[i]:
                    # Find the first occurrence of cpx_token_id in this sample
                    cpx_positions[i] = (input_ids[i] == cpx_token_id).nonzero(as_tuple=True)[0][0]
                else:
                    # Use the last non-padding token position
                    if attention_mask is not None:
                        # Debug: Check attention mask
                        mask_sum = attention_mask[i].sum().item()
                        print(f"Sample {i}: attention mask sum = {mask_sum}")
                        if mask_sum > 0:
                            cpx_positions[i] = mask_sum - 1
                        else:
                            cpx_positions[i] = 0  # Fallback to first position
                    else:
                        cpx_positions[i] = input_ids.size(1) - 1
        else:
            positions = (input_ids == self.cpx_token_id).nonzero(as_tuple=False)
            # positions is [ [batch_idx, position], ... ]
            # To get only positions per sequence:
            cpx_positions = torch.full((batch_size,), -1, device=input_ids.device)  # -1 for sequences without token
            cpx_positions[positions[:,0]] = positions[:,1]
        # Run the forward function of the base model to get hidden states
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, **kwargs)
        hidden_states = outputs.hidden_states[-1]  # Get the last layer hidden states
        
        # Check if positions are valid
        max_valid_pos = input_ids.size(1) - 1
        if (cpx_positions > max_valid_pos).any():
            print(f"ERROR: Invalid CPX positions detected! Max valid position: {max_valid_pos}")
            print(f"Invalid positions: {cpx_positions[cpx_positions > max_valid_pos]}")
            cpx_positions = torch.clamp(cpx_positions, 0, max_valid_pos)
        
        cpx_hidden_states = hidden_states[range(batch_size), cpx_positions]
        
        # Check for extreme values before classifier application
        if torch.isnan(cpx_hidden_states).any() or torch.isinf(cpx_hidden_states).any():
            print("Warning: NaN/Inf in CPX hidden states before classifier")
            cpx_hidden_states = torch.nan_to_num(cpx_hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(self.classifier.weight).any() or torch.isinf(self.classifier.weight).any():
            print("Warning: NaN/Inf in classifier weights before forward pass")
            # torch.nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
            # if self.classifier.bias is not None:
            #     torch.nn.init.zeros_(self.classifier.bias)
        
        logits = self.classifier(cpx_hidden_states)
        
        # Check for NaN/Inf in logits immediately after computation
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Warning: NaN/Inf in logits immediately after classifier")
            print(f"Hidden states stats: min={cpx_hidden_states.min().item():.6f}, max={cpx_hidden_states.max().item():.6f}")
            print(f"Classifier weight stats: min={self.classifier.weight.min().item():.6f}, max={self.classifier.weight.max().item():.6f}")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
        # Return logits and hidden states as a tuple
        return logits, outputs