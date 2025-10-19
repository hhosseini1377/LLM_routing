# CPX Wrapper - Universal Complexity Classifier for Causal Language Models

A general-purpose wrapper that adds prompt complexity classification to any Causal Language Model from HuggingFace Transformers.

## Overview

The CPX (Complexity) wrapper adds a special `[CPX]` token to any causal language model and uses a lightweight classifier to assess prompt complexity. This is useful for:

- **Adaptive model routing**: Route simple prompts to smaller models, complex prompts to larger models
- **Prompt analysis**: Understand which prompts require more computational resources
- **Dynamic inference**: Adjust generation parameters based on prompt complexity

## Key Features

✅ **Model-Agnostic**: Works with any `AutoModelForCausalLM`-compatible model  
✅ **Minimal Training**: Only trains the classifier head + CPX token embedding  
✅ **Base Model Frozen**: Preserves pretrained knowledge while adding new capabilities  
✅ **Easy Integration**: Drop-in replacement with simple API  
✅ **Flexible**: Supports multi-GPU training, gradient checkpointing, and more  

## Supported Models

The wrapper works with any causal language model, including:

- **Mistral**: `mistralai/Mistral-7B-Instruct-v0.1`
- **Llama**: `meta-llama/Llama-2-7b-hf`, `meta-llama/Meta-Llama-3-8B`
- **GPT-2/GPT-Neo/GPT-J**: `gpt2`, `EleutherAI/gpt-neo-2.7B`, `EleutherAI/gpt-j-6B`
- **Phi**: `microsoft/phi-2`, `microsoft/phi-1_5`
- **Falcon**: `tiiuae/falcon-7b`
- **Qwen**: `Qwen/Qwen-7B`
- And many more!

## Quick Start

### Basic Usage

```python
from cpx_model.cpxmistral.cpxmistraltokenizer import CPXTokenizer
from cpx_model.cpxmistral.cpx_mistral import CPXCausalLM

# Choose any causal LM
model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # or "gpt2", "meta-llama/Llama-2-7b-hf", etc.

# Load tokenizer with CPX token
tokenizer = CPXTokenizer.from_pretrained(model_name, cpx_token='[CPX]')
cpx_token_id = tokenizer.convert_tokens_to_ids('[CPX]')

# Load model with CPX wrapper
model = CPXCausalLM.from_pretrained(
    model_name,
    cpx_token_id=cpx_token_id,
    num_labels=1,
    is_cpx_token_trainable=True,
    tokenizer_size=len(tokenizer),
    device_map="auto"
)

# Use the model
text = "Explain quantum computing [CPX]"
inputs = tokenizer(text, return_tensors="pt")

# Get complexity score
classifier_logits, outputs = model(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask']
)

import torch
complexity_score = torch.sigmoid(classifier_logits).item()
print(f"Complexity score: {complexity_score:.4f}")
```

### Training

```python
# Your training loop
for batch in dataloader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    
    # Forward pass
    classifier_logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Compute loss
    loss = loss_fn(classifier_logits.squeeze(), labels.float())
    
    # Backward pass (only trains classifier + CPX embedding)
    loss.backward()
    optimizer.step()
```

See `main.py` for a complete training example.

## Architecture

### How It Works

1. **Token Addition**: Adds `[CPX]` special token to the vocabulary
2. **Prefill Phase**: During forward pass, extracts hidden states at CPX token position
3. **Classification**: Passes hidden states through a linear classifier
4. **Training**: Only trains the classifier head and CPX token embedding (base model frozen)

### Components

- **`CPXCausalLM`**: Main wrapper class that adds complexity classification
- **`CPXTokenizer`**: Helper to add CPX token to any tokenizer
- **`CPXConfig`**: Configuration utilities for CPX models
- **`CPXTrainingConfig`**: Training hyperparameters and settings

## Files

```
cpx_model/cpxmistral/
├── cpx_mistral.py           # Main CPX wrapper (model-agnostic)
├── cpxmistraltokenizer.py   # Tokenizer wrapper
├── cpxmistralconfig.py      # Config utilities
├── config.py                # Training configurations
├── main.py                  # Training script example
├── usage_example.py         # Simple usage examples
└── README.md               # This file
```

## Configuration

Key configuration options in `CPXTrainingConfig`:

```python
@dataclass
class CPXTrainingConfig:
    # CPX-specific
    cpx_token = '[CPX]'
    is_cpx_token_trainable = True
    num_labels = 1
    
    # Training
    learning_rate = 1e-5
    weight_decay = 0.01
    num_epochs = 200
    
    # Memory optimization
    gradient_checkpointing = True
```

## Command Line Usage

```bash
# Train with Mistral (default)
python main.py --batch_size 32 --num_epochs 4

# Train with a different model
python main.py --model_name gpt2 --batch_size 16

# Train with Llama
python main.py --model_name meta-llama/Llama-2-7b-hf --batch_size 8

# With custom dataset size
python main.py --data_size 10000 --evaluation_size 1000
```

## Examples

See `usage_example.py` for comprehensive examples:

```bash
python usage_example.py
```

This will demonstrate:
- Using CPX wrapper with different models (GPT-2, Mistral, Llama, Phi)
- Comparing model characteristics
- Basic inference with complexity scoring

## Backward Compatibility

For backward compatibility with existing code, the following aliases are provided:

- `CPXMistralTokenizer` → `CPXTokenizer`
- `CPXMistralConfig` → `CPXConfig`
- `MistralTrainingConfig` → `CPXTrainingConfig`
- `CPXMistralDatasetConfig` → `CPXDatasetConfig`

Existing code using the old names will continue to work.

## Requirements

```
torch
transformers
datasets
```

## Technical Details

### Parameter Efficiency

The CPX wrapper is extremely parameter-efficient:

- **Base Model**: Frozen (no training)
- **Classifier**: ~4K-16K parameters (depends on hidden size)
- **CPX Embedding**: 1 token embedding (e.g., 4096 dims for Mistral)

Example for Mistral-7B:
- Total parameters: ~7B
- Trainable parameters: ~4K (classifier) + ~4K (CPX embedding) = ~8K
- **Trainable %: 0.0001%**

### Gradient Masking

The wrapper uses gradient masking to ensure only the CPX token embedding is updated:

```python
class MaskGradHook:
    def __init__(self, token_id):
        self.token_id = token_id
    
    def __call__(self, grad):
        mask = torch.zeros_like(grad)
        mask[self.token_id] = 1.0
        return grad * mask
```

This allows efficient training without modifying the entire embedding matrix.

### Hidden State Extraction

The wrapper automatically detects the hidden size attribute name (varies by model):

```python
def _get_hidden_size(self):
    for attr in ['hidden_size', 'd_model', 'n_embd', 'dim']:
        if hasattr(config, attr):
            return getattr(config, attr)
```

This ensures compatibility with different model architectures.

## Migration from Mistral-Specific Code

If you have existing code using the Mistral-specific implementation:

**Before:**
```python
from transformers import MistralForCausalLM
from cpx_model.cpxmistral.cpx_mistral import MyMistral

model = MyMistral.from_pretrained(...)
```

**After:**
```python
from cpx_model.cpxmistral.cpx_mistral import CPXCausalLM

model = CPXCausalLM.from_pretrained(...)  # Works with ANY causal LM!
```

The API is similar, but now works with any model.

## Future Improvements

- [ ] Add support for decoder-only vs encoder-decoder models
- [ ] Multi-label classification (complexity dimensions)
- [ ] LoRA/QLoRA integration for even more efficiency
- [ ] Automatic model architecture detection
- [ ] Pre-computed complexity embeddings

## Citation

If you use this wrapper in your research, please cite:

```bibtex
@software{cpx_wrapper,
  title={CPX Wrapper: Universal Complexity Classifier for Causal Language Models},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/yourrepo}
}
```

## License

[Your License Here]

## Contributing

Contributions are welcome! Please open an issue or pull request.

