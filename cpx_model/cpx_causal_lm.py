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
    def __init__(self, token_ids):
        self.token_ids = token_ids
    
    def __call__(self, grad):
        mask = torch.zeros_like(grad)
        mask[self.token_ids] = 1.0
        return grad * mask


def mask_lora_activations(output: torch.Tensor, cpx_positions: torch.Tensor) -> torch.Tensor:
    """
    Mask all positions NOT in cpx_positions.
    
    Args:
        outputs: [B, seq_len, hidden_dim]
        cpx_positions: [B, cpx_count] - positions to KEEP
    
    Returns:
        masked_outputs: [B, seq_len, hidden_dim] - non-cpx positions set to 0
    """
    batch_size, seq_len, hidden_dim = output.shape
    cpx_count = cpx_positions.shape[1]
    
    # Create mask: True for positions to KEEP (cpx positions)
    # Start with all False
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=output.device)
    
    # Set cpx positions to True
    batch_indices = torch.arange(batch_size, device=output.device).unsqueeze(1).expand(-1, cpx_count)
    mask[batch_indices, cpx_positions] = True
    
    # Expand mask to match hidden_dim
    mask_expanded = mask.unsqueeze(-1).expand(-1, -1, hidden_dim)
    
    # Apply mask (set non-cpx to 0)
    # Preserve output dtype (bfloat16) when masking
    mask_expanded = mask_expanded.to(dtype=output.dtype)
    masked_outputs = output * mask_expanded
    return masked_outputs

def mask_gradients(grad, cpx_id):
    mask = torch.zeros_like(grad)
    mask[cpx_id] = 1.0
    return grad * mask

class CPXAggregator(nn.Module):
    """
    Aggregates hidden states from multiple CPX tokens.
    Supports multiple strategies including learnable attention.
    """
    def __init__(self, hidden_size: int, aggregation_type: str = 'attention'):
        super().__init__()
        self.aggregation_type = aggregation_type
        
        if aggregation_type == 'attention':
            # Learnable attention weights
            self.attention = nn.Linear(hidden_size, 1)
        elif aggregation_type in ['mean', 'max', 'sum', 'first']:
            # No learnable parameters for these
            pass
        else:
            raise ValueError(f"Unknown aggregation type: {aggregation_type}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, num_tokens, hidden_size]
        Returns:
            aggregated: [batch_size, hidden_size]
        """
        if self.aggregation_type == 'mean':
            return hidden_states.mean(dim=1)
        elif self.aggregation_type == 'max':
            return hidden_states.max(dim=1)[0]
        elif self.aggregation_type == 'sum':
            return hidden_states.sum(dim=1)
        elif self.aggregation_type == 'first':
            return hidden_states[:, 0, :]
        elif self.aggregation_type == 'attention':
            # Compute attention weights
            attn_weights = self.attention(hidden_states)  # [batch, num_tokens, 1]
            attn_weights = torch.softmax(attn_weights, dim=1)
            # Weighted sum
            aggregated = (hidden_states * attn_weights).sum(dim=1)
            return aggregated
        else:
            raise ValueError(f"Unknown aggregation type: {self.aggregation_type}")

class CPXCausalLM(nn.Module):
    """
    General wrapper for any Causal Language Model that adds complexity classification.
    
    This wrapper:
    - Adds a [CPX] token to the vocabulary
    - Extracts hidden states at the CPX token position
    - Uses a classifier head to predict prompt complexity
    - Works with any AutoModelForCausalLM-compatible model
    """
    
    def __init__(self, base_model, model_config, cpx_token_ids, use_lora, freeze_LoRA_layers, freeze_LoRA_start_layer_idx, aggregation_type='first'):
        super().__init__()
        self.base_model = base_model
        self.config = model_config
        self.cpx_token_ids = cpx_token_ids
        self.use_lora = use_lora
        self.freeze_LoRA_layers = freeze_LoRA_layers
        self.cpx_tokens_count = len(cpx_token_ids)
        
        # Determine if multiple tokens
        self.is_multi_token = self.cpx_tokens_count > 1
        # Add classifier head with optional dropout
        hidden_size = self._get_hidden_size()
        num_labels = model_config.num_labels
        dropout_rate = model_config.dropout_rate
        classifier_dropout = model_config.classifier_dropout
        
        # Create aggregator for multiple tokens
        if self.is_multi_token:
            self.aggregator = CPXAggregator(hidden_size, aggregation_type)
            # Ensure aggregator is in float32 for numerical stability (especially if it has learnable params)
            if hasattr(self.aggregator, 'attention'):
                self.aggregator.attention = self.aggregator.attention.float()
            # For operations without learnable params (mean, max, sum, first), dtype conversion happens in forward
        else:
            self.aggregator = None
        
        # Create classifier with optional dropout
        self.classifier = nn.Linear(hidden_size, num_labels)
        # Ensure classifier is in float32 for numerical stability
        self.classifier = self.classifier.float()
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
        max_layers = int(max_layers)
        
        # Ensure max_layers is valid
        if max_layers <= start_layer_idx:
            raise ValueError(f"max_layers ({max_layers}) must be greater than start_layer_idx ({start_layer_idx})")

        # Regex to catch common layer containers across architectures
        # Matches: .model.layers.N. or .layers.N. or .h.N. etc.
        # Also handles cases where layer number might be at the end of the name
        layer_key_regex = re.compile(r"\.(model\.)?(layers|h|blocks|gpt_neox\.layers)\.(\d+)(\.|$)")

        # Convert candidate_target_modules to set for faster lookup
        candidate_set = set(candidate_target_modules)
        
        targets = []
        for name, _ in base_model.named_modules():
            m = layer_key_regex.search(name)
            if not m:
                continue
            layer_idx = int(m.group(3))
            if layer_idx < start_layer_idx or layer_idx >= max_layers:
                continue

            # Match by terminal segment (last component of the module name)
            # This ensures we match the actual module, not its submodules
            terminal = name.split(".")[-1]
            if terminal in candidate_set:
                targets.append(name)
                
        print(f"Found {len(targets)} target modules for layers {start_layer_idx}-{max_layers-1}")
        return sorted(set(targets))

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: str,
        cpx_token_ids: List[int],
        num_labels: int = 1,
        is_cpx_token_trainable: bool = True,
        use_lora: bool = False,
        lora_config: Optional[Dict] = None,
        dropout_rate: float = 0.0,
        classifier_dropout: bool = False,
        freeze_LoRA_layers: bool = False,
        freeze_LoRA_start_layer_idx: int = 0,
        aggregation_type: str = 'attention',
        *model_args, 
        **kwargs
    ):
        """
        Load a pretrained causal LM and wrap it with CPX complexity classification.
        
        Args:
            pretrained_model_name_or_path: Model name or path (e.g., "mistralai/Mistral-7B-Instruct-v0.1")
            cpx_token_id: Token ID(s) for the [CPX] special token(s) - int or list[int]
            num_labels: Number of labels for classification (default: 1 for binary)
            is_cpx_token_trainable: Whether to train the CPX token embedding
            use_lora: Whether to add LoRA adapters to the base model
            lora_config: LoRA configuration dict (r, alpha, dropout, target_modules, etc.)
            aggregation_type: How to aggregate multiple CPX tokens ('mean', 'max', 'sum', 'attention', 'first')
            **kwargs: Additional arguments passed to AutoModelForCausalLM.from_pretrained
        """
        # Extract CPX-specific parameters before passing to base model
        tokenizer_size = kwargs.pop('tokenizer_size', None)
        use_cache = kwargs.pop('use_cache', None)
        mask_lora_for_non_cpx = kwargs.pop('mask_lora_for_non_cpx', True)
        
        # Set use_cache in kwargs if provided (for base model)
        if use_cache is not None:
            kwargs['use_cache'] = use_cache
        
        is_multi_token = len(cpx_token_ids) > 1
        torch_dtype = kwargs.pop('torch_dtype', torch.bfloat16)
        # Load the base model and config
        base_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, 
            torch_dtype=torch_dtype,
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
        model_config.cpx_token_ids = cpx_token_ids
        
        model_config.dropout_rate = dropout_rate
        model_config.classifier_dropout = classifier_dropout
        
        # Resize token embeddings if new vocabulary size provided
        if tokenizer_size is not None:
            base_model.resize_token_embeddings(tokenizer_size)
        
        # Initialize CPX token embeddings as mean of existing embeddings
        # emb = base_model.get_input_embeddings().weight
        # with torch.no_grad():
        #     for tid in cpx_token_ids:
        #         base_model.get_input_embeddings().weight[tid] = emb.mean(dim=0)        

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
        model = cls(base_model, model_config, cpx_token_ids, use_lora=use_lora, 
                    freeze_LoRA_layers=freeze_LoRA_layers, freeze_LoRA_start_layer_idx=freeze_LoRA_start_layer_idx,
                    aggregation_type=aggregation_type)
        model.mask_lora_for_non_cpx = mask_lora_for_non_cpx
        
        # Freeze ALL base model parameters first
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Unfreeze the classifier 
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        # Unfreeze aggregator if it has parameters (attention-based)
        if model.aggregator is not None and hasattr(model.aggregator, 'attention'):
            for param in model.aggregator.attention.parameters():
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
            # Create gradient mask to only train CPX token embeddings
            mask_hook = MaskGradHook(cpx_token_ids)
            embedding_layer.weight.register_hook(mask_hook)
        else:
            for param in embedding_layer.parameters():
                param.requires_grad = False
        
        # Unfreeze LoRA parameters if using LoRA
        if use_lora:
            for name, param in base_model.named_parameters():
                if 'lora_' in name.lower():
                    param.requires_grad = True
        
        # Track trainable parameters
        trainable_params = list(model.classifier.parameters())
        if model.aggregator is not None and hasattr(model.aggregator, 'attention'):
            trainable_params += list(model.aggregator.attention.parameters())
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
            if 'lora_B' in name or 'lora_embedding_B' in name:
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
    
    def prefill_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass during prefill phase - extracts hidden states at CPX token position(s).
        
        Returns:
            Tuple of (classifier_logits, base_model_outputs)
        """
        cpx_token_ids = self.cpx_token_ids
        # Find CPX token positions [Batch_size, CPX_tokens_count]
        cpx_positions = self.find_token_indices_vectorized(input_ids, cpx_token_ids)
        self._update_cpx_positions(cpx_positions)
            
        # Run base model forward
        outputs = self.base_model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True, 
            **kwargs
        )
        hidden_states = outputs.hidden_states[-1]
        cpx_positions_expanded = cpx_positions.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
        cpx_hidden_states = torch.gather(hidden_states, dim=1, index=cpx_positions_expanded)

        # Check for extreme values
        if torch.isnan(cpx_hidden_states).any() or torch.isinf(cpx_hidden_states).any():
            print("Warning: NaN/Inf in CPX hidden states before classifier")
            cpx_hidden_states = torch.nan_to_num(cpx_hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(self.classifier.weight).any() or torch.isinf(self.classifier.weight).any():
            print("Warning: NaN/Inf in classifier weights before forward pass")
        
        # Convert to float32 for classifier and aggregator (more stable for final layers)
        cpx_hidden_states = cpx_hidden_states.float()
        
        # Aggregate CPX hidden states
        if self.is_multi_token:
            cpx_hidden_states = self.aggregator(cpx_hidden_states)
        else:
            cpx_hidden_states = cpx_hidden_states[:, 0, :]
        # Apply dropout if enabled
        if self.dropout is not None:
            cpx_hidden_states = self.dropout(cpx_hidden_states)
        
        logits = self.classifier(cpx_hidden_states)
        
        # Check for NaN/Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Warning: NaN/Inf in logits immediately after classifier")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return logits, outputs
        
    @staticmethod
    def find_token_indices_vectorized(input_ids: torch.Tensor, cpx_token_ids: List[int]) -> torch.Tensor:
        """
        Find first occurrence of each token in cpx_token_ids for each batch.
        
        Args:
            input_ids: [batch_size, seq_len]
            cpx_token_ids: list or tensor of token IDs to find
        
        Returns:
            indices: [batch_size, len(cpx_token_ids)] 
                    Contains position of each token, -1 if not found
        """
        batch_size, seq_len = input_ids.shape
        
        # Convert to tensor if list
        if isinstance(cpx_token_ids, list):
            cpx_token_ids = torch.tensor(cpx_token_ids, device=input_ids.device)
        
        num_tokens = len(cpx_token_ids)
        
        # Reshape for broadcasting: [batch_size, seq_len, 1]
        input_ids_expanded = input_ids.unsqueeze(2)
        
        # Reshape cpx_token_ids: [1, 1, num_tokens]
        cpx_tokens_expanded = cpx_token_ids.view(1, 1, num_tokens)
        
        # Compare: [batch_size, seq_len, num_tokens]
        matches = (input_ids_expanded == cpx_tokens_expanded)
        
        # Find first occurrence for each token
        # Add sentinel column to handle not-found cases
        matches_with_sentinel = torch.cat([
            matches,
            torch.ones(batch_size, 1, num_tokens, dtype=torch.bool, device=matches.device)
        ], dim=1)
        
        # Get first match position: [batch_size, num_tokens]
        indices = matches_with_sentinel.long().argmax(dim=1)
        
        # Mark not-found as -1
        not_found = ~matches.any(dim=1)
        indices[not_found] = -1
        
        return indices