
from config import generator_config

""""
This class is used to evaluate the response of the model.
It can use vllm or transformers to generate the response.
"""
class Evaluator:
    def __init__(self, evaluator_model_id, use_vllm=False, **kwargs):
        self.use_vllm = use_vllm
        self.evaluator_model_id = evaluator_model_id
        if use_vllm:
            from vllm import LLM, SamplingParams
            self.llm = LLM(
                model=evaluator_model_id,
                tokenizer=evaluator_model_id,
                **kwargs
            )
            self.sampling_params = SamplingParams(
                temperature=0.7, top_p=0.95, max_tokens=512
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
            import torch

            self.generation_config = GenerationConfig(
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.2,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            self.evaluator_model_id = evaluator_model_id
            self.tokenizer = AutoTokenizer.from_pretrained(self.evaluator_model_id, use_fast=True)
            self.evaluator_model = AutoModelForCausalLM.from_pretrained(
                self.evaluator_model_id,
                torch_dtype="auto",      
            device_map="auto")
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