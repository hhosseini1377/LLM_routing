from transformers import AutoTokenizer
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
    # Add the cpx token to the tokenizer
    tokenizer.add_special_tokens({'additional_special_tokens': [cpx_token]})
    
    # Get the cpx token id
    cpx_token_id = tokenizer.convert_tokens_to_ids(cpx_token)

    # load model with pretrained weights and custom config
    # Use device_map="auto" for better memory management

    # get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', type=str, default='1000')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--context_window', type=int, default=512)
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--evaluation_size', type=int, default=1000)
    
    args = parser.parse_args()
    train_texts, train_labels, test_texts, test_labels = load_mmlu_data_with_cpx()
    print('Dataset Loaded')

    if args.data_size != 'None':
        train_texts = train_texts[:int(args.data_size)]
        train_labels = train_labels[:int(args.data_size)]
        
    if args.evaluation_size != 'None':
        test_texts = test_texts[:int(args.evaluation_size)]
        test_labels = test_labels[:int(args.evaluation_size)]
        
    MistralTrainingConfig.cpx_token_id = cpx_token_id
    dropout_rate = [0.1, 0.3]
    layers_to_freeze_options = [0, 2, 4]

    grid = product(dropout_rate, layers_to_freeze_options)
    for do_rate, layers in grid:
        
        MistralTrainingConfig.dropout_rate = do_rate
        MistralTrainingConfig.layers_to_freeze = layers

        trainer = MistralTrainer(
            tokenizer=tokenizer,
            train_texts=train_texts,
            train_labels=train_labels,
            test_texts=test_texts,
            test_labels=test_labels)
            
        # Compute the batch size per GPU
        per_gpu_batch_size = args.batch_size // torch.cuda.device_count()
        trainer.run(batch_size=per_gpu_batch_size, context_window=args.context_window, num_epochs=args.num_epochs)

        # Clean up to avoid GPU memory leak
        del trainer
        torch.cuda.empty_cache()
        gc.collect()

