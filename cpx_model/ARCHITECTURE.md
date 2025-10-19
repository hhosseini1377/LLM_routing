# CPX Wrapper Architecture

## High-Level Overview

The CPX (Complexity) wrapper adds a lightweight complexity classifier to any causal language model by:
1. Adding a special `[CPX]` token to the vocabulary
2. Extracting hidden states at the CPX token position during inference
3. Passing those hidden states through a linear classifier
4. Training only the classifier and CPX token embedding (base model frozen)

## Architecture Comparison

### Before: Mistral-Specific

```
┌─────────────────────────────────────────────┐
│           MyMistral (Inheritance)           │
│                                             │
│  ┌───────────────────────────────────────┐ │
│  │    MistralForCausalLM (Base)         │ │
│  │  ┌─────────────────────────────────┐ │ │
│  │  │  - Mistral-specific forward()   │ │ │
│  │  │  - Mistral-specific config      │ │ │
│  │  │  - Mistral architecture         │ │ │
│  │  └─────────────────────────────────┘ │ │
│  └───────────────────────────────────────┘ │
│                                             │
│  + classifier: Linear(4096, 1)              │
│  + cpx_token_id: int                        │
│                                             │
└─────────────────────────────────────────────┘

❌ Problem: Tightly coupled to Mistral
❌ Can't use with other models
❌ Inheritance makes it rigid
```

### After: Universal Wrapper

```
┌──────────────────────────────────────────────────┐
│         CPXCausalLM (Composition)                │
│                                                  │
│  ┌────────────────────────────────────────────┐ │
│  │  base_model: ANY AutoModelForCausalLM     │ │
│  │  ┌──────────────────────────────────────┐ │ │
│  │  │  - Could be Mistral                  │ │ │
│  │  │  - Could be Llama                    │ │ │
│  │  │  - Could be GPT-2                    │ │ │
│  │  │  - Could be Phi                      │ │ │
│  │  │  - Could be Falcon                   │ │ │
│  │  │  - Could be ANY causal LM!           │ │ │
│  │  └──────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
│  + classifier: Linear(hidden_size, num_labels)  │
│  + cpx_token_id: int                            │
│  + config: base model config                    │
│                                                  │
└──────────────────────────────────────────────────┘

✅ Solution: Wraps any model
✅ Works with entire HF ecosystem
✅ Composition provides flexibility
```

## Detailed Component Architecture

### 1. CPXCausalLM (Main Wrapper)

```python
class CPXCausalLM(nn.Module):
    """
    ┌─────────────────────────────────────────┐
    │           CPXCausalLM                   │
    ├─────────────────────────────────────────┤
    │                                         │
    │  Components:                            │
    │  ─────────────────────────────────────  │
    │                                         │
    │  1. base_model: AutoModelForCausalLM   │
    │     └─ Any causal LM from HuggingFace  │
    │                                         │
    │  2. classifier: nn.Linear               │
    │     └─ Input: hidden_size               │
    │     └─ Output: num_labels (default: 1)  │
    │                                         │
    │  3. config: model config                │
    │     └─ From base_model                  │
    │     └─ + CPX-specific params            │
    │                                         │
    │  4. cpx_token_id: int                   │
    │     └─ Position of [CPX] token          │
    │                                         │
    ├─────────────────────────────────────────┤
    │                                         │
    │  Key Methods:                           │
    │  ─────────────────────────────────────  │
    │                                         │
    │  • from_pretrained()                    │
    │    └─ Load any model + add wrapper      │
    │                                         │
    │  • forward()                            │
    │    ├─ Prefill: extract CPX hidden states│
    │    └─ Decode: delegate to base_model    │
    │                                         │
    │  • prefill_forward()                    │
    │    ├─ Find CPX token positions          │
    │    ├─ Run base_model forward            │
    │    ├─ Extract hidden states @ CPX       │
    │    └─ Pass through classifier           │
    │                                         │
    │  • _get_hidden_size()                   │
    │    └─ Auto-detect from various attrs    │
    │                                         │
    └─────────────────────────────────────────┘
    """
```

### 2. Data Flow During Training

```
Input Text: "Explain quantum computing [CPX]"
         │
         ↓
    Tokenizer
         │
         ↓
    Token IDs: [101, 5492, 9875, ..., 50257]
                                        ↑
                                    CPX token
         │
         ↓
    ┌──────────────────────────────────────┐
    │         CPXCausalLM.forward()        │
    │                                      │
    │  1. Detect prefill phase             │
    │     └─ seq_length > 1 or no cache    │
    │                                      │
    │  2. Call prefill_forward()           │
    │     │                                │
    │     ├─ Find CPX token position       │
    │     │  (position = 8 in this case)   │
    │     │                                │
    │     ├─ base_model.forward()          │
    │     │  ├─ Through all transformer    │
    │     │  │  layers                     │
    │     │  └─ output_hidden_states=True  │
    │     │                                │
    │     ├─ Extract hidden state @ pos 8  │
    │     │  └─ shape: [batch, hidden_size]│
    │     │                                │
    │     └─ classifier(hidden_state)      │
    │        └─ output: [batch, 1]         │
    │                                      │
    └──────────────────────────────────────┘
         │
         ↓
    Complexity Logit: 0.73
         │
         ↓
    Loss = BCE(logit, label)
         │
         ↓
    Backward Pass
    ├─ Updates classifier weights ✓
    ├─ Updates CPX embedding ✓
    └─ Skips base model (frozen) ✗
```

### 3. Gradient Flow

```
┌────────────────────────────────────────────────┐
│              Gradient Flow                     │
├────────────────────────────────────────────────┤
│                                                │
│  Loss (BCE)                                    │
│     │                                          │
│     ↓                                          │
│  ┌──────────────┐                              │
│  │  Classifier  │  ← Gradients flow ✅         │
│  │  (trainable) │     Updates weights          │
│  └──────────────┘                              │
│     │                                          │
│     ↓                                          │
│  ┌─────────────────────────────────────────┐  │
│  │     Hidden States @ CPX position        │  │
│  └─────────────────────────────────────────┘  │
│     │                                          │
│     ↓                                          │
│  ┌─────────────────────────────────────────┐  │
│  │        Base Model Transformer           │  │
│  │          (ALL FROZEN ❄️)                │  │
│  │  ┌─────────────────────────────────┐   │  │
│  │  │  Layer N  (frozen)              │   │  │
│  │  │  Layer N-1 (frozen)             │   │  │
│  │  │  ...                            │   │  │
│  │  │  Layer 1 (frozen)               │   │  │
│  │  └─────────────────────────────────┘   │  │
│  └─────────────────────────────────────────┘  │
│     │                                          │
│     ↓                                          │
│  ┌─────────────────────────────────────────┐  │
│  │      Embedding Layer                    │  │
│  │  ┌───────────────────────────────────┐ │  │
│  │  │ Token 0: frozen ❄️                │ │  │
│  │  │ Token 1: frozen ❄️                │ │  │
│  │  │ ...                               │ │  │
│  │  │ Token 50257 (CPX): trainable ✅   │ │← Gradient mask
│  │  │ ...                               │ │  │
│  │  │ Token 32000: frozen ❄️            │ │  │
│  │  └───────────────────────────────────┘ │  │
│  └─────────────────────────────────────────┘  │
│                                                │
│  Trainable Parameters:                        │
│  • Classifier: ~4K-16K params                 │
│  • CPX embedding: ~4K-8K params               │
│  • Total: ~0.0001% of model                   │
│                                                │
└────────────────────────────────────────────────┘
```

### 4. Gradient Masking Mechanism

```python
class MaskGradHook:
    """
    Ensures only CPX token embedding gets gradients
    
    Before hook:
    ┌──────────────────────────────┐
    │  Embedding Gradients         │
    │  ────────────────────        │
    │  Token 0:     [0.01, ...]    │  All tokens
    │  Token 1:     [0.02, ...]    │  receive
    │  ...                         │  gradients
    │  Token 50257: [0.03, ...]    │  ← CPX
    │  ...                         │
    └──────────────────────────────┘
    
    After hook:
    ┌──────────────────────────────┐
    │  Masked Gradients            │
    │  ────────────────────        │
    │  Token 0:     [0.0, ...]     │  Zeroed out
    │  Token 1:     [0.0, ...]     │  Zeroed out
    │  ...                         │
    │  Token 50257: [0.03, ...]    │  ← CPX (kept!)
    │  ...                         │
    │  Token N:     [0.0, ...]     │  Zeroed out
    └──────────────────────────────┘
    """
    def __init__(self, token_id):
        self.token_id = token_id
    
    def __call__(self, grad):
        mask = torch.zeros_like(grad)
        mask[self.token_id] = 1.0
        return grad * mask  # Element-wise multiplication
```

### 5. Inference Flow

```
                    Input with CPX token
                           │
                           ↓
                ┌──────────────────────┐
                │  CPXCausalLM         │
                ├──────────────────────┤
                │                      │
      ┌─────────┤  forward()           │
      │         │                      │
      │         └──────────────────────┘
      │                  │
      │                  ↓
      │         ┌─────────────────┐
      │         │ Is prefill?     │
      │         └─────────────────┘
      │            │            │
      │           Yes          No
      │            │            │
      │            ↓            ↓
      │  ┌──────────────┐   ┌──────────────┐
      │  │ prefill_     │   │ base_model   │
      │  │ forward()    │   │ forward()    │
      │  └──────────────┘   └──────────────┘
      │         │                  │
      │         ↓                  │
      │  ┌──────────────┐          │
      │  │ Extract CPX  │          │
      │  │ hidden state │          │
      │  └──────────────┘          │
      │         │                  │
      │         ↓                  │
      │  ┌──────────────┐          │
      │  │ Classifier   │          │
      │  └──────────────┘          │
      │         │                  │
      │         ↓                  │
      │  ┌──────────────┐          │
      └─►│ Complexity   │          │
         │ Score        │          │
         └──────────────┘          │
                                   │
                                   ↓
                            ┌──────────────┐
                            │ Next Token   │
                            │ Logits       │
                            └──────────────┘
```

## Model Compatibility

The wrapper automatically adapts to different model architectures:

```
┌─────────────────────────────────────────────────────┐
│           Hidden Size Detection                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  def _get_hidden_size(self):                        │
│      for attr in ['hidden_size',  # ← Most models  │
│                    'd_model',      # ← T5, BART     │
│                    'n_embd',       # ← GPT-2        │
│                    'dim']:         # ← Some others  │
│          if hasattr(config, attr):                  │
│              return getattr(config, attr)           │
│                                                     │
│  Examples:                                          │
│  ────────────────────────────────────────────       │
│  • Mistral: config.hidden_size = 4096               │
│  • GPT-2:   config.n_embd = 768                     │
│  • Llama:   config.hidden_size = 4096               │
│  • Phi:     config.hidden_size = 2560               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Memory Efficiency

```
┌────────────────────────────────────────────────┐
│         Parameter Distribution                 │
│         (Example: Mistral-7B)                  │
├────────────────────────────────────────────────┤
│                                                │
│  Base Model (Frozen ❄️)                       │
│  ═══════════════════════════════════          │
│  Parameters: 7,241,748,480                    │
│  Trainable:  NO                               │
│  Memory:     ~14 GB (float16)                 │
│                                                │
│  ─────────────────────────────────────────    │
│                                                │
│  CPX Token Embedding (Trainable ✅)           │
│  ═══════════════════════════════              │
│  Parameters: 4,096 (1 token × hidden_size)    │
│  Trainable:  YES                              │
│  Memory:     ~16 KB                           │
│                                                │
│  ─────────────────────────────────────────    │
│                                                │
│  Classifier (Trainable ✅)                    │
│  ═══════════════════════════════              │
│  Parameters: 4,097 (4096 × 1 + bias)          │
│  Trainable:  YES                              │
│  Memory:     ~16 KB                           │
│                                                │
│  ─────────────────────────────────────────    │
│                                                │
│  Total Trainable: 8,193 parameters            │
│  Percentage:      0.0001%                     │
│  Memory Overhead: ~32 KB                      │
│                                                │
│  ═══════════════════════════════════════      │
│  Conclusion: Extremely parameter-efficient!   │
│  ═══════════════════════════════════════      │
│                                                │
└────────────────────────────────────────────────┘
```

## Distributed Training Architecture

```
┌─────────────────────────────────────────────────────┐
│           Multi-GPU Training Setup                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  GPU 0                  GPU 1                       │
│  ┌──────────────┐      ┌──────────────┐            │
│  │ CPXCausalLM  │      │ CPXCausalLM  │            │
│  │ (DDP wrapped)│      │ (DDP wrapped)│            │
│  └──────────────┘      └──────────────┘            │
│         │                     │                     │
│         ↓                     ↓                     │
│  ┌──────────────┐      ┌──────────────┐            │
│  │  Batch 0-15  │      │  Batch 16-31 │            │
│  └──────────────┘      └──────────────┘            │
│         │                     │                     │
│         ↓                     ↓                     │
│  ┌──────────────┐      ┌──────────────┐            │
│  │  Forward     │      │  Forward     │            │
│  │  + Backward  │      │  + Backward  │            │
│  └──────────────┘      └──────────────┘            │
│         │                     │                     │
│         └──────────┬──────────┘                     │
│                    ↓                                │
│         ┌─────────────────────┐                     │
│         │  Gradient AllReduce │                     │
│         └─────────────────────┘                     │
│                    │                                │
│                    ↓                                │
│         ┌─────────────────────┐                     │
│         │  Optimizer Step     │                     │
│         │  (synchronized)     │                     │
│         └─────────────────────┘                     │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Summary

The CPX wrapper architecture is designed for:

✅ **Flexibility**: Works with any causal LM  
✅ **Efficiency**: Only trains ~0.0001% of parameters  
✅ **Simplicity**: Clean composition-based design  
✅ **Scalability**: Supports distributed training  
✅ **Maintainability**: Model-agnostic implementation  

The key innovation is using composition instead of inheritance, allowing the wrapper to adapt to any model architecture automatically while maintaining a simple, efficient interface.

