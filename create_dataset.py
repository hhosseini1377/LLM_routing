import pickle
import re
available_models = ['WizardLM/WizardLM-13B-V1.2',                                                                      
'claude-instant-v1',                                                                                
'claude-v1',                                                                                       
'claude-v2',                                                                                         
'gpt-3.5-turbo-1106',                                                                              
'gpt-4-1106-preview',                                                                               
'meta/code-llama-instruct-34b-chat',                                                                
'meta/llama-2-70b-chat',                                                                             
'mistralai/mistral-7b-chat',                                                                         
'mistralai/mixtral-8x7b-chat',                                                                      
'zero-one-ai/Yi-34B-Chat']

def process_row(row, max_length):
    cleaned_row = {}
    prompt = row['prompt']
    if bool(re.search(r'[\u4e00-\u9fff]', prompt)):
        return None
    if len(prompt) > max_length:
        return None
    if max_length is not None:
        cleaned_row['text'] = prompt[-max_length:]
    else:
        cleaned_row['text'] = prompt
    cleaned_row['labels'] = [row.get(model, 0) for model in available_models]
    return cleaned_row

def create_dataset(source_file, dest_file, max_samples=None, max_length=None):
    with open(source_file, 'rb') as f:
        data = pickle.load(f)
    if max_samples is not None:
        data = data.iloc[:max_samples]
    
    # Use ProcessPoolExecutor to parallelize row processing
    # with ProcessPoolExecutor() as executor:
    #     func = partial(process_row, max_length=max_length)
    #     cleaned_data = list(executor.map(func, [row for _, row in data.iterrows()]))
    cleaned_data = []
    for i, row in data.iterrows():
        cleaned_row = process_row(row, max_length)
        if cleaned_row is None:
            continue
        cleaned_data.append(cleaned_row)
        if i % 100 == 0:
            print(f"Processed {i} rows")
    with open(dest_file, 'wb') as f:
        pickle.dump(cleaned_data, f)

    return data

def split_dataset(source_file, dest_train_file, dest_test_file):
    with open(source_file, 'rb') as f:
        data = pickle.load(f)
    train_data = data[:int(len(data)*0.8)]
    test_data = data[int(len(data)*0.8):]
    with open(dest_train_file, 'wb') as f:
        pickle.dump(train_data, f)
    with open(dest_test_file, 'wb') as f:
        pickle.dump(test_data, f)

max_length = 512

create_dataset('./datasets/routerbench_0shot.pkl', './datasets/cleaned_routerbench_0shot_truncated.pkl', max_length=512)
split_dataset('./datasets/cleaned_routerbench_0shot_truncated.pkl', './datasets/train_routerbench_0shot_truncated.pkl', './datasets/test_routerbench_0shot_truncated.pkl')