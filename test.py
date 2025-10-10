# Oprn datasets/MMLU/mmlu_auxiliary_formatted.pkl
import pickle

with open('generate_dataset/datasets/MMLU/mmlu_auxiliary_train_n5_t0.8.pkl', 'rb') as f:
    data = pickle.load(f)


total_correct = 0
total_runs = 0
for data_point in data:
    # for answer in data_point['answers']:
    #     if answer[1] not in ['A', 'B', 'C', 'D']:
    #         print(answer)
    total_correct += data_point['correct']
    total_runs += data_point['total_runs']


print(f"Total correct: {total_correct}")
print(f"Total runs: {total_runs}")
print(f"Accuracy: {total_correct / len(data)}")

# data_point = data[110]
# for key, value in data_point.items():
#     print(key, value)

