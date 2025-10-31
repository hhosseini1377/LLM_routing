import pickle
from datasets import Dataset as DS
with open("generate_dataset/datasets/mix/mmlu_and_gsm8k_with_correct_val.pkl", "rb") as f:
    data = pickle.load(f)

total_correct = 0

print(int(True))
print(data[0])