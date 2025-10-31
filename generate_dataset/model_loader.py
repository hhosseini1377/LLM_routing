from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from config import generator_config_with_sampling, generator_config_without_sampling
import json
from generate_dataset.prompt_formats import PROMPT_TEMPLATES

class ModelLoader:
    def __init__(self, model_id, use_vllm=False, is_quantized=False, use_sampling=True, **kwargs):
        self.use_sampling = use_sampling
        if use_vllm:
            from vllm import LLM, SamplingParams
            self.llm = LLM(
                model=model_id,
                tokenizer=model_id,
                quantization="awq",
                special_token_policy="ignore",
                **kwargs
            )  
        else:  
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            if is_quantized:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype="float16",  # or bfloat16 if supported
                bnb_4bit_quant_type="nf4"
                )
            self.model_id = model_id
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, clean_up_tokenization_spaces=False)    
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype="auto",      
                device_map="auto",
                trust_remote_code=True,
                **kwargs)

    def generate(self, prompt):
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs,
            eos_token_id=self.tokenizer.eos_token_id, 
            pad_token_id=self.tokenizer.eos_token_id,
            **generator_config_with_sampling if self.use_sampling else generator_config_without_sampling)
        input_size = inputs['input_ids'].shape[1]
        return outputs, input_size


    def create_data_set(self, used_dataset, file_name, max_samples=None, **kwargs):
        dataset = []
        for i, example in enumerate(used_dataset):
            if max_samples is not None and i >= max_samples:
                print(f"Reached maximum sample limit: {max_samples}")
                break
            prompt = PROMPT_TEMPLATES[self.model_id](example['context'], example['instruction'])
            if 'answers' in example:
                answer = example['answers']
            elif 'response' in example:
                answer = example['response']
            else:
                answer = example['answer']
            try:
                with torch.no_grad():
                    outputs, input_size = self.generate(prompt)
                    response = self.tokenizer.decode(outputs[0][input_size:], skip_special_tokens=True)
            except Exception as e:
                print(f'Error: {e}')
                continue
            dataset.append({
                'prompt': prompt,
                'response': response,
                'answer': answer
            })
            print(f'Sample {i} has been generated')
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=4)
        print(f'Data has been saved to {file_name}')
