from transformers import AutoModelForCausalLM, AutoConfig
import torch.nn as nn
from typing import Tuple, Optional, Union
import torch
from warnings import warn


class MaskGradHook:
    """Hook class that can be pickled for multi-GPU training"""
    def __init__(self, token_id):
        self.token_id = token_id
    
    def __call__(self, grad):
        mask = torch.zeros_like(grad)
        mask[self.token_id] = 1.0
        return grad * mask


def mask_gradients(grad, cpx_id):
    mask = torch.zeros_like(grad)
    mask[cpx_id] = 1.0
    return grad * mask


class CPXCausalLM(nn.Module):
    """
    General wrapper for any Causal Language Model that adds complexity classification.
    
    This wrapper:
    - Adds a [CPX] token to the vocabulary
    - Extracts hidden states at the CPX token position
    - Uses a classifier head to predict prompt complexity
    - Works with any AutoModelForCausalLM-compatible model
    """
    
    def __init__(self, base_model, config, cpx_token_id):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.cpx_token_id = cpx_token_id
        
        # Add classifier head
        hidden_size = self._get_hidden_size()
        num_labels = getattr(config, 'num_labels', 1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def _get_hidden_size(self):
        """Get hidden size from the base model config"""
        config = self.base_model.config
        # Try common attribute names for hidden size
        for attr in ['hidden_size', 'd_model', 'n_embd', 'dim']:
            if hasattr(config, attr):
                return getattr(config, attr)
        raise ValueError(f"Could not determine hidden size from model config: {type(config)}")

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: str,
        cpx_token_id: int,
        num_labels: int = 1,
        is_cpx_token_trainable: bool = True,
        *model_args, 
        **kwargs
    ):
        """
        Load a pretrained causal LM and wrap it with CPX complexity classification.
        
        Args:
            pretrained_model_name_or_path: Model name or path (e.g., "mistralai/Mistral-7B-Instruct-v0.1")
            cpx_token_id: Token ID for the [CPX] special token
            num_labels: Number of labels for classification (default: 1 for binary)
            is_cpx_token_trainable: Whether to train the CPX token embedding
            **kwargs: Additional arguments passed to AutoModelForCausalLM.from_pretrained
        """
        # Extract CPX-specific parameters before passing to base model
        tokenizer_size = kwargs.pop('tokenizer_size', None)
        use_cache = kwargs.pop('use_cache', None)
        
        # Set use_cache in kwargs if provided (for base model)
        if use_cache is not None:
            kwargs['use_cache'] = use_cache
        
        # Load the base model and config
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, 
            *model_args, 
            **kwargs
        )
        
        config = base_model.config
        config.num_labels = num_labels
        config.cpx_token_id = cpx_token_id
        
        # Resize token embeddings if new vocabulary size provided
        if tokenizer_size is not None:
            base_model.resize_token_embeddings(tokenizer_size)
        
        # Initialize CPX token embedding as mean of existing embeddings
        emb = base_model.get_input_embeddings().weight
        with torch.no_grad():
            base_model.get_input_embeddings().weight[cpx_token_id] = emb.mean(dim=0)
        
        # Create the wrapper model
        model = cls(base_model, config, cpx_token_id)
        
        # Freeze ALL base model parameters
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Unfreeze the classifier (it's new, so needs training)
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        # Handle CPX token embedding training
        embedding_layer = base_model.get_input_embeddings()
        if is_cpx_token_trainable:
            for param in embedding_layer.parameters():
                param.requires_grad = True
            # Create gradient mask to only train CPX token embedding
            mask_hook = MaskGradHook(cpx_token_id)
            embedding_layer.weight.register_hook(mask_hook)
        else:
            for param in embedding_layer.parameters():
                param.requires_grad = False
        
        # Track trainable parameters
        model.params_to_train = list(model.classifier.parameters()) + list(embedding_layer.parameters())
        
        return model

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        Forward pass - delegates to base model or prefill_forward depending on phase.
        """
        past_key_values = kwargs.get("past_key_values", None)
        seq_length = input_ids.shape[1]
        is_prefill = (past_key_values is None) or (seq_length != 1)
        
        if is_prefill:   
            # During prefilling, extract CPX hidden states and classify
            classifier_logits, outputs = self.prefill_forward(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                **kwargs
            )
            return classifier_logits, outputs
        
        # During decoding, just use base model
        return self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            **kwargs
        )
    
    def generate(self, inputs=None, **kwargs):
        """
        Generate method that wraps base model's generate.
        """
        # For generation, we may want to run prefill first to get complexity scores
        # Then delegate to base model for actual generation
        return self.base_model.generate(inputs, **kwargs)
    
    def parameters(self, recurse: bool = True):
        """Override to return all parameters (base model + classifier)"""
        return nn.Module.parameters(self, recurse=recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Override to return all named parameters"""
        return nn.Module.named_parameters(self, prefix=prefix, recurse=recurse)
    
    def prefill_forward(self, input_ids=None, attention_mask=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass during prefill phase - extracts hidden states at CPX token position.
        
        Returns:
            Tuple of (classifier_logits, base_model_outputs)
        """
        cpx_token_id = self.cpx_token_id
            
        # Check if all batch samples contain the cpx_token_id
        has_cpx_token = (input_ids == cpx_token_id).any(dim=1)
        batch_size = input_ids.size(0)
        
        if not has_cpx_token.all():
            print('Warning: some samples do not contain the CPX token')
            missing_indices = torch.where(~has_cpx_token)[0]
            warn(f"Samples at indices {missing_indices.tolist()} do not contain the special token ID {cpx_token_id}.")
            
            # For samples without CPX token, use the last token position instead
            cpx_positions = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
            for i in range(batch_size):
                if has_cpx_token[i]:
                    # Find the first occurrence of cpx_token_id in this sample
                    cpx_positions[i] = (input_ids[i] == cpx_token_id).nonzero(as_tuple=True)[0][0]
                else:
                    # Use the last non-padding token position
                    if attention_mask is not None:
                        mask_sum = attention_mask[i].sum().item()
                        if mask_sum > 0:
                            cpx_positions[i] = mask_sum - 1
                        else:
                            cpx_positions[i] = 0  # Fallback to first position
                    else:
                        cpx_positions[i] = input_ids.size(1) - 1
        else:
            # All samples have CPX token - extract positions efficiently
            positions = (input_ids == cpx_token_id).nonzero(as_tuple=False)
            cpx_positions = torch.full((batch_size,), -1, device=input_ids.device)
            cpx_positions[positions[:, 0]] = positions[:, 1]
            
        # Run the base model forward to get hidden states
        outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True, 
            **kwargs
        )
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