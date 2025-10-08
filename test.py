# Oprn datasets/MMLU/mmlu_auxiliary_formatted.pkl
import pickle

with open('generate_dataset/datasets/MMLU/mmlu_auxiliary_formatted.pkl', 'rb') as f:
    data = pickle.load(f)

print(len(data))
# data_point = data[110]
# for key, value in data_point.items():
#     print(key, value)

