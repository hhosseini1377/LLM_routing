from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn


class CPXModel(AutoModelForCausalLM):
    def __init__(self, model_name):
        print('salam ahmad')
        super().__init__(model_name)

    def generate(self, input_ids, attention_mask):
        print('salam ahmad')
        outputs = super().generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=100)
        return outputs

cpx_token_name = "<cpx>"
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Add new tokens to the tokenizer
num_added = tokenizer.add_tokens([cpx_token_name])
print(f"Added {num_added} new tokens to the tokenizer")
# Update the model to use the new tokenizer
model = CPXModel.from_pretrained(model_name)
print(type(model))
# Resize model embeddings if new tokens were added
if num_added > 0:
    model.resize_token_embeddings(len(tokenizer))
    print(f"Model embeddings resized to {len(tokenizer)} tokens")

# Encode prompt
prompt = "Write a short poem about the moon."
# append the cpx token to the prompt
prompt = prompt + cpx_token_name

# Fix: Properly extract input_ids and attention_mask from the dictionary
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Generate response
# Use generate() method for text generation, not forward()
outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))




        

