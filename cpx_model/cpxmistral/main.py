from transformers import MistralForCausalLM, MistralConfig, AutoTokenizer
import torch.nn as nn
from cpx_model.cpxmistral.cpxmistralconfig import CPXMistralConfig
from cpx_model.cpxmistral.cpx_mistral import MyMistral
import argparse
import gc
from itertools import product
from cpx_model.cpxmistral.config import MistralTrainingConfig
from cpx_model.cpxmistral.train_mistral import MistralTrainer
import torch
from cpx_model.cpxmistral.utils import load_mmlu_data_with_cpx
from cpx_model.cpxmistral.train_mistral import MistralTrainer

if __name__ == "__main__":
    cpx_token = MistralTrainingConfig.cpx_token
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    mistral_config = CPXMistralConfig.from_pretrained(pretrained_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1", num_labels=1, cpx_token=cpx_token)

    # Add the cpx token to the tokenizer
    tokenizer.add_special_tokens({'additional_special_tokens': [cpx_token]})
    mistral_config.tokenizer_size = len(tokenizer)
    
    # Get the cpx token id
    cpx_token_id = tokenizer.convert_tokens_to_ids(cpx_token)
    mistral_config.cpx_token_id = cpx_token_id

    # load model with pretrained weights and custom config
    # Use device_map="auto" for better memory management
    model = MyMistral.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        config=mistral_config,
        torch_dtype=torch.bfloat16,  # Use bfloat16
        device_map="auto",          # Automatic device placement
        low_cpu_mem_usage=True      # Reduce CPU memory usage during loading
    )

    # get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', type=str, default='1000')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--context_window', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=4)
    args = parser.parse_args()
    train_texts, train_labels, test_texts, test_labels = load_mmlu_data_with_cpx()
    if args.data_size != 'None':
        train_texts = train_texts[:int(args.data_size)]
        train_labels = train_labels[:int(args.data_size)]
    
    dropout_rate = [0.1, 0.3]
    layers_to_freeze_options = [0, 2, 4]

    grid = product(dropout_rate, layers_to_freeze_options)
    for do_rate, layers in grid:
        
        MistralTrainingConfig.dropout_rate = do_rate
        MistralTrainingConfig.layers_to_freeze = layers

        trainer = MistralTrainer(
            model=model,
            tokenizer=tokenizer,
            train_texts=train_texts,
            train_labels=train_labels,
            test_texts=test_texts,
            test_labels=test_labels)

        trainer.train(
            batch_size=args.batch_size, 
            context_window=args.context_window, 
            num_epochs=args.num_epochs,)

        # Clean up to avoid GPU memory leak
        del trainer
        torch.cuda.empty_cache()
        gc.collect()

