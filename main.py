from model_loader import ModelLoader
from datasets import load_dataset
from evaluator import Evaluator
import gc
import torch
import json

MODEL_REGISTRY = {
    "llama3_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "deepseek_7b": "deepseek-ai/deepseek-llm-7b-chat",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "openchat_3.5": "openchat/openchat-3.5-1210",
    "phi_2": "microsoft/phi-2",
    "ultralm_13b": "openbmb/UltraLM-13B",
    'qwen-3b': "Qwen/Qwen2.5-VL-3B-Instruct"
}

if __name__ == "__main__":
    #sharedgpt
    # Load the dataset
    dataset = load_dataset("databricks/databricks-dolly-15k")
    # Load the model
    model_id = MODEL_REGISTRY['deepseek_7b']
    evaluator_model_id = "openbmb/UltraLM-13B"
    Llama3_3B_loaded = ModelLoader(model_id)
    print('The model has been loaded')
    train_dataset = dataset['train']
    Llama3_3B_loaded.create_data_set(train_dataset, model_id.split('/')[-1] + 'generated_data.json', max_samples=500, use_sampling=True)