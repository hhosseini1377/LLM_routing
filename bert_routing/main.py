# from generate_dataset.model_loader import ModelLoader
from datasets import load_dataset
from bert_routing.train_BERT import ModelTrainer
import pickle
import argparse
import random
import os
from bert_routing.config import DatasetConfig, MODEL_REGISTRY, TrainingConfig
from cpx_model.cpx_causal_utils import load_mix_data_with_cpx, load_gsm8k_data_with_cpx, load_mmlu_data_with_cpx
from itertools import product
import torch
import gc

def load_pickle_data(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    task = 'train'
    if task == 'train':
        # get the arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', type=str, default='distilbert')
        parser.add_argument('--data_size', type=str, default='1000')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--context_window', type=int, default=512)
        parser.add_argument('--num_epochs', type=int, default=4)
        parser.add_argument('--strategy', type=str, default=6)
        parser.add_argument('--dataset', type=str, default='gsm8k')
        args = parser.parse_args()
        index = 0
        if args.dataset == 'gsm8k':
            train_texts, train_labels, test_texts, test_labels = load_gsm8k_data_with_cpx()
        elif args.dataset == 'mmlu':
            train_texts, train_labels, test_texts, test_labels = load_mmlu_data_with_cpx()
        elif args.dataset == 'mix':
            train_texts, train_labels, test_texts, test_labels = load_mix_data_with_cpx()
        else:
            raise ValueError(f"Invalid dataset: {args.dataset}")

        if args.data_size != 'None':
            train_texts = train_texts[:int(args.data_size)]
            train_labels = train_labels[:int(args.data_size)]
        num_classes = 2

        print('dataset loaded')
        
        training_config = TrainingConfig()
        training_config.model_name = args.model_name
        training_config.data_size = args.data_size
        training_config.dataset = args.dataset
        trainer = ModelTrainer(model_name=args.model_name,
            num_outputs=len(train_labels[0]),
            num_classes=num_classes,
            pooling_strategy=args.strategy, 
            train_texts=train_texts,
            train_labels=train_labels,
            test_texts=test_texts,
            test_labels=test_labels,
            training_config=training_config)

        trainer.train(batch_size=args.batch_size, 
            context_window=args.context_window, 
            num_epochs=args.num_epochs,)


    elif task == 'evaluate':
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', type=str, default='distilbert')
        parser.add_argument('--data_size', type=int, default=1000)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--context_window', type=int, default=512)
        args = parser.parse_args()
        with open('./datasets/test_routerbench_0shot.pkl', 'rb') as f:
            data = pickle.load(f)
        test_texts = [sample['text'] for sample in data]
        test_labels = [sample['labels'] for sample in data]
        if args.data_size is not None:
            texts = test_texts[:args.data_size]
            labels = test_labels[:args.data_size]
        trainer = ModelTrainer(model_name=args.model_name, num_outputs=len(labels[0]), pooling_strategy=args.strategy)
        trainer.load_model('./finetuned_models/model_'+args.model_name+'_'+args.strategy+'.pth')
        loss = trainer.evaluate(batch_size=args.batch_size, 
            context_window=args.context_window, 
            test_texts=texts, 
            test_labels=labels)
        print(loss)
        
    elif task == 'create_dataset':
        # Load the dataset
        dataset = load_dataset("databricks/databricks-dolly-15k")
        # Load the model
        model_id = MODEL_REGISTRY['deepseek_7b']
        evaluator_model_id = "openbmb/UltraLM-13B"
        Llama3_3B_loaded = ModelLoader(model_id)
        print('The model has been loaded')
        train_dataset = dataset['train']
        Llama3_3B_loaded.create_data_set(train_dataset, model_id.split('/')[-1] + 'generated_data.json', max_samples=500, use_sampling=True)