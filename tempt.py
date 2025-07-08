from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle

with open('./datasets/cleaned_routerbench_0shot.pkl', 'rb') as f:
    data = pickle.load(f)

# Now `data` contains the deserialized Python object
print(data[0])