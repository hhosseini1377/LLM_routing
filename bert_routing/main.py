# from generate_dataset.model_loader import ModelLoader
from datasets import load_dataset
from bert_routing.train_BERT import ModelTrainer
import pickle
import argparse
import random
import os
from bert_routing.config import DatasetConfig, MODEL_REGISTRY, TrainingConfig
from cpx_model.dataset_loaders import get_dataset_loader
from itertools import product
import torch
import gc

def load_pickle_data(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def str_to_bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    task = 'train'
    if task == 'train':
        # get the arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', type=str, default='deberta')
        parser.add_argument('--data_size', type=str, default='None')
        parser.add_argument('--evaluation_size', type=str, default='None')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--context_window', type=int, default=512)
        parser.add_argument('--num_epochs', type=int, default=10)
        parser.add_argument('--strategy', type=str, default=6)
        parser.add_argument('--dataset_name', type=str, default='auxiliary',
                           help='Dataset name (e.g., "auxiliary", "combined", "gsm8k", "mix", "imdb"). Used with --dataset_model_name to load specific dataset files.')
        parser.add_argument('--dataset_model_name', type=str, default=None,
                           help='Model name used for dataset (e.g., "qwen4", "qwen17b", "qwen34b"). Used with --dataset_name to load specific dataset files.')
        parser.add_argument('--dropout_rate', type=float, default=0.1)
        parser.add_argument('--classifier_dropout', type=str_to_bool, default=True)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--layers_to_freeze', type=int, default=4)
        parser.add_argument('--freeze_layers', type=str_to_bool, default=False)
        parser.add_argument('--scheduler', type=str, default='cosine')
        parser.add_argument('--warmup_steps', type=float, default=0.1)
        parser.add_argument('--max_grad_norm', type=float, default=1.0)
        parser.add_argument('--embedding_lr', type=float, default=1e-5)
        parser.add_argument('--classifier_lr', type=float, default=1e-4)
        parser.add_argument('--model_lr', type=float, default=2e-5)
        parser.add_argument('--freeze_embedding', type=str_to_bool, default=False)
        parser.add_argument('--cpx_tokens', type=str, nargs='+', default=None, help='List of CPX tokens as strings, e.g., --cpx_tokens [CPX1] [CPX2]')
        parser.add_argument('--metric', type=str, default='f1')
        parser.add_argument('--use_weighted_sampling', type=str_to_bool, default=False, help='Enable weighted random sampling')
        parser.add_argument('--dataset_weight_power', type=float, default=1.0, help='Power to apply to dataset source weights')
        parser.add_argument('--sampling_weight_power', type=float, default=None,
                           help='Power to apply to weights for weighted sampling. If not specified, uses class_weight_power for backward compatibility.')
        parser.add_argument('--loss_weight_power', type=float, default=None,
                           help='Power to apply to class weights in loss function. If not specified, uses class_weight_power for backward compatibility.')
        parser.add_argument('--class_weight_power', type=float, default=1.0,
                           help='DEPRECATED: Power to apply to weights. Use --sampling_weight_power and --loss_weight_power instead.')
        parser.add_argument('--use_class_weights', type=str_to_bool, default=False,
                           help='Enable class weighting in loss function (BCEWithLogitsLoss pos_weight)')
        args = parser.parse_args()
        index = 0
        
        # Load dataset using the appropriate loader function
        # Pass cpx_tokens=False to load without CPX tokens (BERT routing doesn't use CPX tokens)
        loader_func = get_dataset_loader(args.dataset_name)
        train_texts, train_labels, train_dataset_sources, test_texts, test_labels, _ = \
            loader_func(cpx_tokens=False, dataset_name=args.dataset_name, dataset_model_name=args.dataset_model_name)

        if args.data_size != 'None':
            train_texts = train_texts[:int(args.data_size)]
            train_labels = train_labels[:int(args.data_size)]
            if train_dataset_sources is not None:
                train_dataset_sources = train_dataset_sources[:int(args.data_size)]

        if args.evaluation_size != 'None':
            test_texts = test_texts[:int(args.evaluation_size)]
            test_labels = test_labels[:int(args.evaluation_size)]

        num_classes = 2

        print('dataset loaded')
        
        training_config = TrainingConfig(
            model_name=args.model_name,
            data_size=args.data_size,
            evaluation_size=args.evaluation_size,
            dataset_name=args.dataset_name,
            dropout_rate=args.dropout_rate,
            classifier_dropout=args.classifier_dropout,
            weight_decay=args.weight_decay,
            layers_to_freeze=args.layers_to_freeze,
            freeze_layers=args.freeze_layers,
            num_epochs=args.num_epochs,
            scheduler=args.scheduler,
            warmup_steps=args.warmup_steps,
            max_grad_norm=args.max_grad_norm,
            embedding_lr=args.embedding_lr,
            classifier_lr=args.classifier_lr,
            model_lr=args.model_lr,
            freeze_embedding=args.freeze_embedding,
            METRIC=args.metric,
            use_weighted_sampling=args.use_weighted_sampling,
            dataset_weight_power=args.dataset_weight_power,
            sampling_weight_power=args.sampling_weight_power if args.sampling_weight_power is not None else args.class_weight_power,
            loss_weight_power=args.loss_weight_power if args.loss_weight_power is not None else args.class_weight_power,
            class_weight_power=args.class_weight_power,  # Keep for backward compatibility
            use_class_weights=args.use_class_weights
        )
        trainer = ModelTrainer(model_name=args.model_name,
            num_outputs=len(train_labels[0]),
            num_classes=num_classes,
            pooling_strategy=args.strategy, 
            train_texts=train_texts,
            train_labels=train_labels,
            test_texts=test_texts,
            test_labels=test_labels,
            training_config=training_config,
            train_dataset_sources=train_dataset_sources)

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