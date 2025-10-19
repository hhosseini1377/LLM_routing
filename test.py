# Oprn datasets/MMLU/mmlu_auxiliary_formatted.pkl
import pickle

with open('generate_dataset/datasets/GSM8K/gsm8k_generated_data.pkl', 'rb') as f:
    data = pickle.load(f)


total_correct = 0
total_runs = 0

data_point = data[10]

print(data_point.keys())
print(f'Question: {data_point["question"]}')
print(f'Ground Truth: {data_point["ground_truth"]}')
print(f'Predicted: {data_point["predicted"]}')
print(f'Correct: {data_point["correct"]}')
print(f'Model Output: {data_point["model_output"]}')