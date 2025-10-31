import pickle
import re
from generate_dataset.evaluate_MMLU import DatasetLoader
import matplotlib.pyplot as plt
from datasets import concatenate_datasets, Dataset as DS, Value, ClassLabel
answers_file = '/data/gpfs/projects/punim2662/LLM_routing/LLM_routing/generate_dataset/datasets/MMLU/mmlu_all_with_answers_n5.pkl'
dist_file = '/data/gpfs/projects/punim2662/LLM_routing/LLM_routing/generate_dataset/datasets/MMLU/mmlu_all_with_correct_counts_n5.pkl'

def find_valid_indexes() -> list[int]:
    with open(answers_file, 'rb') as f:
        data = pickle.load(f)

    valid_indexes = []
    for index, example in enumerate(data):
        correct_flag = True
        for answer in example['answers']:
            try:
                if 'Answer' not in answer:
                    checked_label = answer[1]
                else:
                    checked_label = answer[9]
            except:
                correct_flag = False
                break

            if checked_label not in ['A', 'B', 'C', 'D']:
                correct_flag = False
                break
        if correct_flag:
            valid_indexes.append(index)

    return valid_indexes

def create_file_with_correct_counts():
    valid_indexes = find_valid_indexes()
    with open(answers_file, 'rb') as f:
        data = pickle.load(f)

    # Select the valid indexes from the data
    valid_dataset = data.select(valid_indexes)

    # valid_dataset = DatasetLoader.format_labels(valid_dataset)

    correct_counts = []
    for index, example in enumerate(valid_dataset):
        answers = example['answers']
        ground_truth = example['label']
        correct_count = 0
        for answer in answers:
            if 'Answer' not in answer:
                checked_label = answer[1]
            else:
                checked_label = answer[9]
            if checked_label not in ['A', 'B', 'C', 'D']:
                print('bad answer')
            if checked_label == ground_truth:
                correct_count += 1
        correct_counts.append(correct_count)

    # Append correct counts to the valid dataset
    valid_dataset = valid_dataset.add_column('correct_count', correct_counts)

    all_correct = []

    for count in correct_counts:
        if count >= 4:
            all_correct.append(1)
        else:
            all_correct.append(0)

    valid_dataset = valid_dataset.add_column('correct', all_correct)
    with open(dist_file, 'wb') as f:
        pickle.dump(valid_dataset, f)
    print(f'files saved to {dist_file}')

def concat_auxiliary_and_all():
    auxiliary_file = '/data/gpfs/projects/punim2662/LLM_routing/LLM_routing/generate_dataset/datasets/MMLU/mmlu_auxiliary_with_correct_counts_n5.pkl'
    all_file = '/data/gpfs/projects/punim2662/LLM_routing/LLM_routing/generate_dataset/datasets/MMLU/mmlu_all_with_correct_counts_n5.pkl'
    dest_file = '/data/gpfs/projects/punim2662/LLM_routing/LLM_routing/generate_dataset/datasets/MMLU/mmlu_auxiliary_and_all_with_correct_counts_n5.pkl'
    auxiliary_data = pickle.load(open(auxiliary_file, 'rb'))
    all_data = pickle.load(open(all_file, 'rb'))
    concatenated_data = concatenate_datasets([auxiliary_data, all_data])
    pickle.dump(concatenated_data, open(dest_file, 'wb'))
    print(f'files saved to {dest_file}')

def concat_gsm8k_generated_data():
    train_file = '/data/gpfs/projects/punim2662/LLM_routing/LLM_routing/generate_dataset/datasets/GSM8K/gsm8k_generated_data_train.pkl'
    test_file = '/data/gpfs/projects/punim2662/LLM_routing/LLM_routing/generate_dataset/datasets/GSM8K/gsm8k_generated_data_test.pkl'
    dest_file = '/data/gpfs/projects/punim2662/LLM_routing/LLM_routing/generate_dataset/datasets/GSM8K/gsm8k_generated_data.pkl'
    train_data = pickle.load(open(train_file, 'rb'))
    test_data = pickle.load(open(test_file, 'rb'))
    concatenated_data = concatenate_datasets([DS.from_list(train_data), DS.from_list(test_data) ])
    pickle.dump(concatenated_data, open(dest_file, 'wb'))
    print(f'files saved to {dest_file}')

def rename_gsm8k_generated_data():
    file = '/data/gpfs/projects/punim2662/LLM_routing/LLM_routing/generate_dataset/datasets/GSM8K/gsm8k_generated_data.pkl'
    with open(file, 'rb') as f:
        data = pickle.load(f)
    data = data.rename_column('question', 'prompt')
    pickle.dump(data, open(file, 'wb'))
    print(f'files saved to {file}')

def concat_mmlu_and_gsm8k_with_task_name():
    mmlu_file = '/data/gpfs/projects/punim2662/LLM_routing/LLM_routing/generate_dataset/datasets/MMLU/mmlu_all_with_correct_counts_n5.pkl'
    gsm8k_file = '/data/gpfs/projects/punim2662/LLM_routing/LLM_routing/generate_dataset/datasets/GSM8K/gsm8k_generated_data.pkl'
    dest_file = '/data/gpfs/projects/punim2662/LLM_routing/LLM_routing/generate_dataset/datasets/mix/mmlu_and_gsm8k_with_correct.pkl'
    mmlu_data = pickle.load(open(mmlu_file, 'rb')).select_columns(['prompt', 'correct'])
    gsm8k_data = pickle.load(open(gsm8k_file, 'rb')).select_columns(['prompt', 'correct'])
    gsm8k_data = gsm8k_data.cast_column("correct", Value("bool"))
    mmlu_data = mmlu_data.cast_column("correct", Value("bool"))
    mmlu_data = mmlu_data.add_column('task_name', ['mmlu'] * len(mmlu_data))
    gsm8k_data = gsm8k_data.add_column('task_name', ['gsm8k'] * len(gsm8k_data))
    concatenated_data = concatenate_datasets([mmlu_data, gsm8k_data])
    pickle.dump(concatenated_data, open(dest_file, 'wb'))
    print(f'files saved to {dest_file}')

def add_strat_key(example):
    return {"strat_key": f"{example['task_name']}_{int(example['correct'])}"}

def stratified_split_dataset():
    mix_file = '/data/gpfs/projects/punim2662/LLM_routing/LLM_routing/generate_dataset/datasets/mix/mmlu_and_gsm8k_with_correct.pkl'
    dest_train_file = '/data/gpfs/projects/punim2662/LLM_routing/LLM_routing/generate_dataset/datasets/mix/mmlu_and_gsm8k_with_correct_train.pkl'
    dest_val_file = '/data/gpfs/projects/punim2662/LLM_routing/LLM_routing/generate_dataset/datasets/mix/mmlu_and_gsm8k_with_correct_val.pkl'
    with open(mix_file, 'rb') as f:
        combined = pickle.load(f)
    combined = combined.map(add_strat_key)
    # 1️⃣ Get all possible unique keys
    unique_keys = list(set(combined["strat_key"]))

    # 2️⃣ Create a ClassLabel feature
    class_label = ClassLabel(names=unique_keys)

    # 3️⃣ Cast the column
    combined = combined.cast_column("strat_key", class_label)
    split = combined.train_test_split(
    test_size=0.1,
    stratify_by_column="strat_key", 
    seed=42
    )
    train_dataset = split["train"]
    val_dataset = split["test"]
    pickle.dump(train_dataset, open(dest_train_file, 'wb'))
    pickle.dump(val_dataset, open(dest_val_file, 'wb'))
    print(f'files saved to {dest_train_file} and {dest_val_file}')
    file = '/data/gpfs/projects/punim2662/LLM_routing/LLM_routing/generate_dataset/datasets/mix/mmlu_and_gsm8k_with_correct.pkl'
if __name__ == '__main__':
    stratified_split_dataset()