from model_loader import ModelLoader
from datasets import load_dataset
from evaluator import Evaluator
import httpx

if __name__ == "__main__":

    # Load the dataset
    dataset = load_dataset("THUDM/LongBench", "2wikimqa")
    # Load the model
    model_id = "./models/llama3-3b-local"
    evaluator_model_id = "./models/llama3-3b-local"
    Llama3_3B_loaded = ModelLoader(model_id)
    evaluator = Evaluator(evaluator_model_id, use_vllm=True)
    data_set = []
    # Create the data set
    for example in dataset['test']:
        prompt = f"### Context: \n{example['context']}\n\n###Question: {example['input']}\n### Instruction:\nUsing the context above, provide a clear, concise, and well-reasoned answer. Justify your response when appropriate.\n\n### Answer:"
        answer = example['answers']
        outputs, input_size = Llama3_3B_loaded.generate(prompt)
        response = Llama3_3B_loaded.tokenizer.decode(outputs[0][input_size:], skip_special_tokens=True)
        Llama3_3B_loaded.model.to("cpu")
        print(f"Output generated: {response}, expected: {answer}")
        eval_result = evaluator.generate_response(prompt, response,)
        print(f"Evaluation result: {eval_result}")
        break