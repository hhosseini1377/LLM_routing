# Migration Guide: Mistral-Specific → Universal CPX Wrapper

This guide explains the changes made to transform the Mistral-specific CPX implementation into a universal wrapper that works with any Causal Language Model.

## Summary of Changes

### Before (Mistral-Specific)
- ❌ Hardcoded inheritance from `MistralForCausalLM`
- ❌ Only worked with Mistral models
- ❌ Used `super()` calls specific to Mistral
- ❌ Config and tokenizer classes were Mistral-only

### After (Universal)
- ✅ Uses composition pattern with `AutoModelForCausalLM`
- ✅ Works with ANY causal language model
- ✅ Delegates to wrapped base model
- ✅ Config and tokenizer work with any model

## Code Changes

### 1. Main Model Class

**Before:**
```python
from transformers import MistralForCausalLM

class MyMistral(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
```

**After:**
```python
from transformers import AutoModelForCausalLM

class CPXCausalLM(nn.Module):
    def __init__(self, base_model, config, cpx_token_id):
        super().__init__()
        self.base_model = base_model  # Composition instead of inheritance
        self.config = config
        self.cpx_token_id = cpx_token_id
        
        hidden_size = self._get_hidden_size()  # Auto-detect hidden size
        self.classifier = nn.Linear(hidden_size, num_labels)
```

### 2. Loading Models

**Before:**
```python
model = MyMistral.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    config=config
)
```

**After:**
```python
# Works with ANY model!
model = CPXCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",  # or "gpt2", "meta-llama/Llama-2-7b-hf", etc.
    cpx_token_id=cpx_token_id,
    num_labels=1,
    tokenizer_size=len(tokenizer)
)
```

### 3. Forward Pass

**Before:**
```python
def forward(self, input_ids=None, attention_mask=None, **kwargs):
    # ...
    return super().forward(...)  # Calls MistralForCausalLM.forward
```

**After:**
```python
def forward(self, input_ids=None, attention_mask=None, **kwargs):
    # ...
    return self.base_model(...)  # Calls wrapped model's forward
```

### 4. Tokenizer

**Before:**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer.add_special_tokens({'additional_special_tokens': ['[CPX]']})
tokenizer.pad_token = tokenizer.eos_token
```

**After:**
```python
from cpx_model.cpxmistral.cpxmistraltokenizer import CPXTokenizer

# Works with any model - handles everything automatically
tokenizer = CPXTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",  # or any other model
    cpx_token='[CPX]'
)
```

### 5. Configuration

**Before:**
```python
from dataclasses import dataclass

@dataclass
class MistralTrainingConfig:
    cpx_token = '[CPX]'
    learning_rate = 1e-5
    # ... Mistral-specific settings
```

**After:**
```python
@dataclass
class CPXTrainingConfig:  # Model-agnostic name
    cpx_token = '[CPX]'
    learning_rate = 1e-5
    # ... Universal settings

# Backward compatibility
MistralTrainingConfig = CPXTrainingConfig
```

## Key Architectural Changes

### 1. Composition Over Inheritance

**Motivation**: Inheritance from `MistralForCausalLM` locked us into Mistral-specific behavior.

**Solution**: Use composition pattern where we wrap any `AutoModelForCausalLM` instance:

```python
class CPXCausalLM(nn.Module):
    def __init__(self, base_model, config, cpx_token_id):
        super().__init__()
        self.base_model = base_model  # Any causal LM
        # ...
```

### 2. Automatic Hidden Size Detection

**Motivation**: Different models use different attribute names for hidden size.

**Solution**: Check multiple common attribute names:

```python
def _get_hidden_size(self):
    config = self.base_model.config
    # Try common attribute names
    for attr in ['hidden_size', 'd_model', 'n_embd', 'dim']:
        if hasattr(config, attr):
            return getattr(config, attr)
    raise ValueError(f"Could not determine hidden size")
```

### 3. Model-Agnostic API

**Motivation**: Make it easy to switch between different models.

**Solution**: Use `AutoModelForCausalLM` and accept model name as string:

```python
base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path,  # Any model name/path
    *model_args,
    **kwargs
)
```

## Backward Compatibility

All old code continues to work via aliases:

```python
# Old imports still work
from cpx_model.cpxmistral.config import MistralTrainingConfig
from cpx_model.cpxmistral.cpxmistraltokenizer import CPXMistralTokenizer

# These are now aliases to the universal versions
MistralTrainingConfig → CPXTrainingConfig
CPXMistralTokenizer → CPXTokenizer
CPXMistralConfig → CPXConfig
```

## Migration Examples

### Example 1: Basic Model Loading

**Before:**
```python
from cpx_model.cpxmistral.cpx_mistral import MyMistral
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer.add_special_tokens({'additional_special_tokens': ['[CPX]']})

model = MyMistral.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    config=config
)
```

**After:**
```python
from cpx_model.cpxmistral.cpx_mistral import CPXCausalLM
from cpx_model.cpxmistral.cpxmistraltokenizer import CPXTokenizer

tokenizer = CPXTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
cpx_token_id = tokenizer.convert_tokens_to_ids('[CPX]')

model = CPXCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    cpx_token_id=cpx_token_id,
    tokenizer_size=len(tokenizer)
)
```

### Example 2: Training Loop

**Before & After:** No changes needed! The training loop remains the same:

```python
for batch in dataloader:
    classifier_logits, outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask']
    )
    loss = criterion(classifier_logits, batch['labels'])
    loss.backward()
    optimizer.step()
```

### Example 3: Using Different Models

**Before:** Not possible without rewriting code

**After:**
```python
# Easy to switch models!
models = [
    "mistralai/Mistral-7B-Instruct-v0.1",
    "meta-llama/Llama-2-7b-hf",
    "gpt2",
    "microsoft/phi-2"
]

for model_name in models:
    tokenizer = CPXTokenizer.from_pretrained(model_name)
    cpx_token_id = tokenizer.convert_tokens_to_ids('[CPX]')
    
    model = CPXCausalLM.from_pretrained(
        model_name,
        cpx_token_id=cpx_token_id,
        tokenizer_size=len(tokenizer)
    )
    # Same training code for all models!
```

## Benefits of Refactoring

### 1. Flexibility
- ✅ Works with any causal LM from HuggingFace
- ✅ Easy to experiment with different base models
- ✅ No code changes needed to switch models

### 2. Maintainability
- ✅ Cleaner separation of concerns
- ✅ Less coupled to specific model implementations
- ✅ Easier to test with smaller models (e.g., GPT-2)

### 3. Extensibility
- ✅ Easy to add support for new models
- ✅ Can extend to other model families in future
- ✅ Compatible with future transformers updates

### 4. Research Velocity
- ✅ Quickly compare complexity classification across models
- ✅ Test hypotheses on multiple architectures
- ✅ Leverage pretrained models from the community

## Testing the Migration

Test with a small model first:

```python
# Quick test with GPT-2
from cpx_model.cpxmistral.cpx_mistral import CPXCausalLM
from cpx_model.cpxmistral.cpxmistraltokenizer import CPXTokenizer

tokenizer = CPXTokenizer.from_pretrained("gpt2")
cpx_token_id = tokenizer.convert_tokens_to_ids('[CPX]')

model = CPXCausalLM.from_pretrained(
    "gpt2",
    cpx_token_id=cpx_token_id,
    tokenizer_size=len(tokenizer)
)

# Test inference
text = "Hello world [CPX]"
inputs = tokenizer(text, return_tensors="pt")
logits, outputs = model(**inputs)

print(f"✅ Successfully loaded GPT-2 with CPX wrapper!")
print(f"Classifier output: {logits.shape}")
```

## Troubleshooting

### Issue: "Could not determine hidden size"

**Cause**: Model uses a non-standard attribute name for hidden size.

**Solution**: Add the attribute name to `_get_hidden_size()`:

```python
def _get_hidden_size(self):
    config = self.base_model.config
    for attr in ['hidden_size', 'd_model', 'n_embd', 'dim', 'your_custom_attr']:
        if hasattr(config, attr):
            return getattr(config, attr)
```

### Issue: Model doesn't have `output_hidden_states`

**Cause**: Some models don't support this by default.

**Solution**: The wrapper automatically handles this. If you still have issues, check model documentation.

### Issue: Tokenizer doesn't have `pad_token`

**Cause**: Some tokenizers don't define a padding token.

**Solution**: `CPXTokenizer` automatically sets `pad_token = eos_token` if not present.

## Questions?

For questions or issues:
1. Check `usage_example.py` for working examples
2. See `README.md` for comprehensive documentation
3. Open an issue on GitHub

## Summary

The refactoring transforms a Mistral-specific implementation into a universal wrapper with:
- 🎯 Same functionality
- 🚀 Works with any causal LM
- 🔄 Backward compatible
- 📚 Better documented
- 🧪 Easier to test

All existing code continues to work, but you now have the flexibility to use any base model!

