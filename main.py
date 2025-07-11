from model_loader import ModelLoader
from datasets import load_dataset
from train_BERT import RegressionModel
import pickle
import argparse
import random
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
    task = 'train'
    if task == 'train':
        # get the arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', type=str, default='distilbert')
        parser.add_argument('--data_size', type=str, default='1000')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--context_window', type=int, default=512)
        parser.add_argument('--num_epochs', type=int, default=6)
        args = parser.parse_args()

        # load the data
        with open('./datasets/train_routerbench_0shot.pkl', 'rb') as f:
            data = pickle.load(f)
        random.shuffle(data)

        texts = [sample['text'] for sample in data]
        labels = [sample['labels'] for sample in data]

        if args.data_size != 'None':
            texts = texts[:int(args.data_size)]
            labels = labels[:int(args.data_size)]
        regressor = RegressionModel(args.model_name, num_outputs=len(labels[0]))
        regressor.train(batch_size=args.batch_size, 
            context_window=args.context_window, 
            num_epochs=args.num_epochs,
            texts=texts,
            labels=labels)

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
        regressor = RegressionModel(args.model_name, num_outputs=len(labels[0]))
        regressor.load_model('./finetuned_models/model_'+args.model_name+'.pth')
        loss = regressor.evaluate(batch_size=args.batch_size, 
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