from datasets import load_dataset

ds = load_dataset("TIGER-Lab/MMLU-Pro")

# print(len(ds['test'][120]['options']))

a= [0] * 10

for data in ds['test']:
    a[len(data['options'])-1] += 1

print(a)