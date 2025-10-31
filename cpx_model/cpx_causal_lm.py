from transformers import AutoModelForCausalLM, AutoConfig
import torch.nn as nn
from typing import Tuple, Optional, Union, Dict, List
import torch
from warnings import warn
from cpx_model.config import CPXTrainingConfig
# Optional PEFT/LoRA support
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. Install with: pip install peft")


class MaskGradHook:
    """Hook class that can be pickled for multi-GPU training"""
    def __init__(self, token_id):
        self.token_id = token_id
    
    def __call__(self, grad):
        mask = torch.zeros_like(grad)
        mask[self.token_id] = 1.0
        return grad * mask


def mask_lora_activations(output: torch.Tensor, cpx_positions: torch.Tensor) -> torch.Tensor:
    """
    Mask LoRA activations to only keep values at CPX token positions.
    
    By masking activations in forward pass, gradients are automatically masked
    in backward pass through autograd (simpler than manually masking gradients).
    
    Args:
        output: Activation tensor of shape [batch_size, seq_len, hidden_dim]
        cpx_positions: Tensor of shape [batch_size] with CPX token positions
        
    Returns:
        Masked activation tensor (zeros except at CPX positions)
    """
    if cpx_positions is None or len(cpx_positions) == 0:
        return output
    
    # Create mask: only allow activations at CPX positions
    batch_size = output.shape[0]
    seq_len = output.shape[1]

    # Create zero mask
    mask = torch.zeros_like(output)
    
    # Vectorized masking using advanced indexing (no for loop!)
    # Create batch indices [0, 1, 2, ..., batch_size-1]
    batch_indices = torch.arange(batch_size, device=output.device, dtype=torch.long)
    # Filter valid positions (within sequence length)
    valid_mask = (cpx_positions >= 0) & (cpx_positions < seq_len)
    valid_batch_indices = batch_indices[valid_mask]
    valid_positions = cpx_positions[valid_mask]
    # Set mask to 1.0 at valid CPX positions (vectorized!)
    if len(valid_positions) > 0:
        mask[valid_batch_indices, valid_positions, :] = 1.0
    # Mask activations - autograd will automatically handle backward!
    return output * mask

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
    
    def __init__(self, base_model, model_config, cpx_token_id, use_lora, freeze_LoRA_layers, freeze_LoRA_start_layer_idx):
        super().__init__()
        self.base_model = base_model
        self.config = model_config
        self.cpx_token_id = cpx_token_id
        self.use_lora = use_lora
        self.freeze_LoRA_layers = freeze_LoRA_layers
        # Add classifier head with optional dropout
        hidden_size = self._get_hidden_size()
        num_labels = getattr(model_config, 'num_labels', 1)
        dropout_rate = getattr(model_config, 'dropout_rate', 0.0)
        classifier_dropout = getattr(model_config, 'classifier_dropout', False)
        
        # Create classifier with optional dropout
        self.classifier = nn.Linear(hidden_size, num_labels)
        if classifier_dropout and dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        
        # Storage for gradient hooks (for LoRA masking)
        self.lora_hooks = []
        self.current_cpx_positions = None
        
    def _get_hidden_size(self):
        """Get hidden size from the base model config"""
        model_config = self.base_model.config
        # Try common attribute names for hidden size
        for attr in ['hidden_size', 'd_model', 'n_embd', 'dim']:
            if hasattr(model_config, attr):
                return getattr(model_config, attr)
        raise ValueError(f"Could not determine hidden size from model config: {type(model_config)}")

    @staticmethod
    def get_target_modules_for_deep_layers(
        base_model,
        candidate_target_modules,
        start_layer_idx: int,
        max_layers: int,
    ):
        import re
        start_layer_idx = max(0, int(start_layer_idx))
        max_layers = max(start_layer_idx, int(max_layers))

        # Regex to catch common layer containers across architectures
        layer_key_regex = re.compile(r"\.(model\.)?(layers|h|blocks|gpt_neox\.layers)\.(\d+)\.")

        targets = []
        for name, _ in base_model.named_modules():
            m = layer_key_regex.search(name)
            if not m:
                continue
            layer_idx = int(m.group(3))
            if layer_idx < start_layer_idx or layer_idx >= max_layers:
                continue

            # Match by terminal segment to avoid substring false-positives
            terminal = name.split(".")[-1]
            if terminal in candidate_target_modules or any(name.endswith(f".{t}") or f".{t}." in name for t in candidate_target_modules):
                targets.append(name)

        return sorted(set(targets))

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: str,
        cpx_token_id: int,
        num_labels: int = 1,
        is_cpx_token_trainable: bool = True,
        use_lora: bool = False,
        lora_config: Optional[Dict] = None,
        dropout_rate: float = 0.0,
        classifier_dropout: bool = False,
        freeze_LoRA_layers: bool = False,
        freeze_LoRA_start_layer_idx: int = 0,
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
            use_lora: Whether to add LoRA adapters to the base model
            lora_config: LoRA configuration dict (r, alpha, dropout, target_modules, etc.)
            **kwargs: Additional arguments passed to AutoModelForCausalLM.from_pretrained
        """
        # Extract CPX-specific parameters before passing to base model
        tokenizer_size = kwargs.pop('tokenizer_size', None)
        use_cache = kwargs.pop('use_cache', None)
        mask_lora_for_non_cpx = kwargs.pop('mask_lora_for_non_cpx', True)
        
        # Set use_cache in kwargs if provided (for base model)
        if use_cache is not None:
            kwargs['use_cache'] = use_cache
        
        # Load the base model and config
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, 
            *model_args, 
            **kwargs
        )

        if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
            num_layers = len(base_model.model.layers)
        elif hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
            # for GPT2-style models
            num_layers = len(base_model.transformer.h)
        else:
            raise ValueError("Couldn't find layer container in this model.")

        # Clamp start index into valid range
        clamped_start_idx = max(0, min(int(freeze_LoRA_start_layer_idx), max(0, num_layers - 1)))

        
        model_config = base_model.config
        model_config.num_labels = num_labels
        model_config.cpx_token_id = cpx_token_id
        
        model_config.dropout_rate = dropout_rate
        model_config.classifier_dropout = classifier_dropout
        
        # Resize token embeddings if new vocabulary size provided
        if tokenizer_size is not None:
            base_model.resize_token_embeddings(tokenizer_size)
        
        # Initialize CPX token embedding as mean of existing embeddings
        emb = base_model.get_input_embeddings().weight
        with torch.no_grad():
            base_model.get_input_embeddings().weight[cpx_token_id] = emb.mean(dim=0)        

        # Normalize lora_config to a LoraConfig instance early (if provided)
        if use_lora and lora_config is not None and isinstance(lora_config, dict):
            lora_config = LoraConfig(**lora_config)

        # Find target modules for deep layers if requested to freeze LoRA layers
        if freeze_LoRA_layers and lora_config is not None:
            candidate = getattr(lora_config, "target_modules", None)
            # Only resolve if we have candidate targets; otherwise keep default behavior
            if candidate:
                target_modules = cls.get_target_modules_for_deep_layers(
                    base_model,
                    candidate,
                    clamped_start_idx,
                    num_layers,
                )
                lora_config.target_modules = target_modules
        
        # Apply LoRA if requested
        if use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError("PEFT library required for LoRA. Install with: pip install peft")
            
            # Apply LoRA to base model
            base_model = get_peft_model(base_model, lora_config)
            print(f"✓ Applied LoRA to base model")
        
        # Create the wrapper model
        model = cls(base_model, model_config, cpx_token_id, use_lora=use_lora, freeze_LoRA_layers=freeze_LoRA_layers, freeze_LoRA_start_layer_idx=freeze_LoRA_start_layer_idx)
        model.mask_lora_for_non_cpx = mask_lora_for_non_cpx
        
        # Freeze ALL base model parameters first
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Unfreeze the classifier (it's new, so needs training)
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        # Get embedding layer (handle PEFT wrapper)
        if use_lora:
            embedding_layer = base_model.get_base_model().get_input_embeddings()
        else:
            embedding_layer = base_model.get_input_embeddings()
        
        # Handle CPX token embedding training
        if is_cpx_token_trainable:
            for param in embedding_layer.parameters():
                param.requires_grad = True
            # Create gradient mask to only train CPX token embedding
            mask_hook = MaskGradHook(cpx_token_id)
            embedding_layer.weight.register_hook(mask_hook)
        else:
            for param in embedding_layer.parameters():
                param.requires_grad = False
        
        # Unfreeze LoRA parameters if using LoRA
        if use_lora:
            for name, param in base_model.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
        
        # Track trainable parameters
        trainable_params = list(model.classifier.parameters())
        if is_cpx_token_trainable:
            trainable_params += list(embedding_layer.parameters())
        if use_lora:
            trainable_params += [p for n, p in base_model.named_parameters() if 'lora_' in n]
        
        model.params_to_train = trainable_params
        
        # Register LoRA hooks ONCE if using LoRA with position masking
        if use_lora and mask_lora_for_non_cpx:
            model._register_lora_hooks()
        
        return model
    
    def _register_lora_hooks(self):
        """
        Register LoRA activation masking hooks ONCE during initialization.
        
        These hooks mask LoRA activations at non-CPX positions in the forward pass.
        PyTorch autograd automatically handles gradient masking in backward pass.
        
        This is simpler and more efficient than manually masking gradients!
        """
        if not self.use_lora or not self.mask_lora_for_non_cpx:
            return
        
        if len(self.lora_hooks) > 0:
            # Already registered
            return
        
        print("  ✓ Registering LoRA activation masking hooks (one-time setup)")
 
        # Apply hooks to LoRA modules
        for name, module in self.base_model.named_modules():
            if name.endswith("lora_B"):
                # This is a LoRA module - mask its output activations
                def create_hook(model_ref):
                    def hook_fn(module, input, output):
                        
                        # Get current CPX positions
                        cpx_positions = model_ref.current_cpx_positions
                        if cpx_positions is None:
                            return output
                        
                        # Mask activations at non-CPX positions
                        # Autograd will automatically propagate this to gradients!
                        return mask_lora_activations(output, cpx_positions)
                    
                    return hook_fn
                
                hook = module.register_forward_hook(create_hook(self))
                self.lora_hooks.append(hook)

        print(f"  ✓ Registered {len(self.lora_hooks)} LoRA activation masking hooks")
    
    def _update_cpx_positions(self, cpx_positions: torch.Tensor):
        """
        Update CPX positions for current batch.
        This is much more efficient than re-registering hooks!
        
        Args:
            cpx_positions: Tensor of shape [batch_size] with CPX token positions
        """
        self.current_cpx_positions = cpx_positions
    
    def _remove_lora_hooks(self):
        """Remove all LoRA gradient masking hooks"""
        for hook in self.lora_hooks:
            hook.remove()
        self.lora_hooks = []
    
    def _get_lora_parameters(self) -> List[nn.Parameter]:
        """Get all LoRA parameters for the optimizer"""
        if not self.use_lora:
            return []
        return [p for n, p in self.base_model.named_parameters() if 'lora_' in n and p.requires_grad]

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
        
        # Update CPX positions for LoRA gradient masking (if using LoRA)
        # Hooks were registered once during init, now just update positions
        if self.use_lora and self.mask_lora_for_non_cpx and self.training:
            self._update_cpx_positions(cpx_positions)
            
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
        
        # Apply dropout if enabled
        if self.dropout is not None:
            cpx_hidden_states = self.dropout(cpx_hidden_states)
        
        logits = self.classifier(cpx_hidden_states)
        
        # Check for NaN/Inf in logits immediately after computation
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Warning: NaN/Inf in logits immediately after classifier")
            print(f"Hidden states stats: min={cpx_hidden_states.min().item():.6f}, max={cpx_hidden_states.max().item():.6f}")
            print(f"Classifier weight stats: min={self.classifier.weight.min().item():.6f}, max={self.classifier.weight.max().item():.6f}")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
        # Return logits and hidden states as a tuple
        return logits, outputs