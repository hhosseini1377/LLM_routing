import pickle
from datasets import Dataset as DS
with open("generate_dataset/datasets/MMLU/mmlu_auxiliary_and_all_with_correct_counts_n5_train.pkl", "rb") as f:
    data = pickle.load(f)

total_correct = 0
for item in data:
    if item['correct'] == 1:
        total_correct += 1
print(total_correct)
print(len(data))
print(total_correct / len(data))