from config import generator_config_with_sampling, generator_config_without_sampling
from transformers import BitsAndBytesConfig
""""
This class is used to evaluate the response of the model.
It can use vllm or transformers to generate the response.
"""
class Evaluator:
    def __init__(self, evaluator_model_id, use_vllm=False, is_quantized=False, **kwargs):
        self.use_vllm = use_vllm
        self.evaluator_model_id = evaluator_model_id
        
        if is_quantized:
            kwargs["quantization"] = "awq"
        if use_vllm:
            from vllm import LLM, SamplingParams
            self.llm = LLM(
                model=evaluator_model_id,
                tokenizer=evaluator_model_id,
                dtype="half",
                **kwargs
            )
            self.sampling_params = SamplingParams(
                temperature=0.7, top_p=0.95, max_tokens=512
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
            import torch
            if is_quantized:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype="float16",  # or bfloat16 if supported
                bnb_4bit_quant_type="nf4"
                )
            self.evaluator_model_id = evaluator_model_id
            self.tokenizer = AutoTokenizer.from_pretrained(self.evaluator_model_id, use_fast=True, clean_up_tokenization_spaces=False)
            self.evaluator_model = AutoModelForCausalLM.from_pretrained(
                self.evaluator_model_id,
                torch_dtype="auto",      
                device_map="auto",
                **kwargs)

            self.generation_config = GenerationConfig(
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.2,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
    """
    This function is used to generate the response of the model.
    args:
        prompt: the prompt to generate the response.
        reponse: the reponse to evaluate.
    return:
        the response of the model.
    """
    def generate_response(self, prompt, reponse):
        eval_prompt = f"<|system|>\nYou are an expert judge for evaluating AI responses.\n<|user|>\nEvaluate the helpfulness of the following response:\n\nQuestion: {prompt}\nAnswer: {reponse}\n\nIs this answer correct and helpful? Justify your answer and give a score from 1 to 5.\n<|assistant|>\n"

        if self.use_vllm:
            outputs = self.llm.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text.strip()
        else:
            inputs = self.tokenizer(eval_prompt, return_tensors="pt").to(self.evaluator_model.device)
            outputs = self.evaluator_model.generate(**inputs,
                generation_config=self.generation_config)
            input_size = inputs['input_ids'].shape[1]
            return self.tokenizer.decode(outputs[0][input_size:], skip_special_tokens=True)