from datasets import load_dataset
import pickle
with open('./generate_dataset/datasets/mmlu_max/mmlu_max_test_qwen_results.pkl', 'rb') as f:
    ds = pickle.load(f)

print(ds[0].keys())
correct = 0
for data in ds:
    if data['correct'] == 1:
        correct += 1

for data in ds:
    if data['correct'] != 1:
        print(len(data['response'].split(' ')))
        break
print(f"Correct: {correct}")
print(f"Total: {len(ds)}")
print(f"Accuracy: {correct / len(ds)}")