# Environment Setup Guide

## Hugging Face Authentication Token

This repository requires a Hugging Face authentication token to access gated models. **Never commit your token to the repository.**

### Setting Up Your Token

#### Option 1: Environment Variable (Recommended)

Set the token in your shell session before running scripts:

```bash
export HF_AUTH_TOKEN="your_token_here"
# Or use one of these alternative names:
# export HF_TOKEN="your_token_here"
# export HUGGINGFACE_TOKEN="your_token_here"
```

To make it persistent, add it to your `~/.bashrc` or `~/.bash_profile`:

```bash
echo 'export HF_AUTH_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

#### Option 2: Local Config File (Not Committed)

Create a local file `.env.local` (this file is gitignored):

```bash
echo 'export HF_AUTH_TOKEN="your_token_here"' > .env.local
source .env.local
```

#### Option 3: For Jupyter Notebooks

In your notebook, set it programmatically:

```python
import os
os.environ['HF_AUTH_TOKEN'] = "your_token_here"
```

### Getting Your Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Read" access
3. Copy the token (starts with `hf_`)

### Security Notes

- **Never commit tokens to git**
- If you accidentally committed a token, revoke it immediately on Hugging Face and create a new one
- The scripts will warn you if no token is found
- Tokens are automatically detected from environment variables: `HF_AUTH_TOKEN`, `HUGGINGFACE_TOKEN`, or `HF_TOKEN`

