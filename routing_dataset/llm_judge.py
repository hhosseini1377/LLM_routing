import json
import os
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from huggingface_hub import login
from routing_dataset.dataset_paths import HOTPOTQA_QWEN17B_CORRECT_LABELS_FILE
#############################################
# 1. File paths (EDIT THESE)
#############################################

def judge_hotpotqa_qwen8b():
    INPUT_FILE = HOTPOTQA_QWEN17B_CORRECT_LABELS_FILE          # your input file
    OUTPUT_FILE = HOTPOTQA_QWEN17B_CORRECT_LABELS_FILE # output file name


    #############################################
    # 2. Load DataFrame (.pkl)
    #############################################

    print(f"Loading dataframe from {INPUT_FILE}...")
    df = pd.read_pickle(INPUT_FILE)

    if "pred" not in df.columns or "answer" not in df.columns:
        raise ValueError("DataFrame must contain 'pred' and 'answer' columns.")


    #############################################
    # 3. Load Llama-3-8B-Instruct as Judge with vLLM
    #############################################

    # Authenticate with Hugging Face
    hf_token = os.environ.get("HF_AUTH_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token:
        print("Authenticating with Hugging Face...")
        login(token=hf_token)
    else:
        print("Warning: No Hugging Face token found in environment variables. Attempting to use cached credentials...")

    judge_model = "meta-llama/Llama-3.1-8B-Instruct"

    print("Loading tokenizer for chat template formatting...")
    tokenizer = AutoTokenizer.from_pretrained(judge_model, trust_remote_code=True)
    
    print("Loading Llama-3-8B-Instruct judge model with vLLM (2 GPUs)...")
    judge = LLM(
        model=judge_model,
        tensor_parallel_size=1,  # Use 2 GPUs
        dtype="bfloat16",
        max_model_len=4096,  # Adjust based on your needs
        trust_remote_code=True,
    )


    #############################################
    # 4. Judge Prompt (Llama-3.1 Chat Format)
    #############################################

    def build_judge_prompt(question, pred, gt):
        # Use chat template format for Llama-3.1
        system_message = (
            "You are evaluating whether a model's answer to a question is correct. "
            "Respond only in JSON format with the fields 'is_correct' (true/false) and 'reason' (short explanation)."
        )
        
        user_message = f"""You are evaluating whether a model's answer to a question is correct.

Question:
{question}

Ground-truth answer:
{gt}

Model's answer:
{pred}

Treat the model's answer as correct if it refers to the same entity or value as the ground-truth
answer, allowing for minor formatting differences (e.g., missing middle names, added punctuation, or
phrases like "the answer is ..."). However, if the answers refer to different entities, persons,
values, or concepts, it is incorrect.

Respond only in JSON with the following fields:
{{
"is_correct": true or false,
"reason": "very short explanation"
}}"""
        
        # Format using tokenizer's chat template
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt


    #############################################
    # 5. Build All Prompts
    #############################################

    print("Building prompts for all rows...")
    prompts = []
    row_metadata = []  # Store (idx, pred, gt) for logging

    for idx, row in df.iterrows():
        question = row["question"] if "question" in df.columns else ""
        pred = str(row["pred"])
        gt = str(row["answer"])
        
        prompt = build_judge_prompt(question, pred, gt)
        prompts.append(prompt)
        row_metadata.append((idx, pred, gt))

    print(f"Built {len(prompts)} prompts. Generating responses with vLLM...")

    #############################################
    # 6. Batch Generate All Prompts
    #############################################

    sampling_params = SamplingParams(
        temperature=0.0,  # deterministic judge
        max_tokens=128,
    )
    
    outputs = judge.generate(prompts, sampling_params)
    print(f"Generated {len(outputs)} responses. Processing results...")

    #############################################
    # 7. Post-process All Results
    #############################################

    def extract_judgment(generated_text):
        """Extract JSON judgment from generated text."""
        # Clean up the text - remove any markdown code blocks
        text = generated_text.strip()
        
        # Remove markdown code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        try:
            # Find JSON object
            start_idx = text.index("{")
            end_idx = text.rindex("}") + 1
            json_str = text[start_idx:end_idx]
            result = json.loads(json_str)
            return result["is_correct"], result.get("reason", "")
        except (ValueError, json.JSONDecodeError) as e:
            # Debug: print first few failed cases
            return False, f"JSON parsing failed: {str(e)[:50]}"

    is_correct_list = []
    reason_list = []

    # Debug: print first few outputs to see what we're getting
    print("\nDebugging first 3 outputs:")
    for i in range(min(3, len(outputs))):
        generated_text = outputs[i].outputs[0].text
        print(f"\n--- Output {i} ---")
        print(f"Generated text (first 200 chars): {generated_text[:200]}")
        print(f"Full generated text: {repr(generated_text)}")
    print("\n" + "="*50 + "\n")

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        correct, reason = extract_judgment(generated_text)
        
        is_correct_list.append(correct)
        reason_list.append(reason)
        
        idx, pred, gt = row_metadata[i]
        print(f"[{idx}] pred='{pred}' | gt='{gt}' → correct={correct}")
        
        # Show first few failures for debugging
        if not correct and i < 5:
            print(f"  Generated text: {generated_text[:200]}")
            print(f"  Reason: {reason}")


    df["is_correct"] = is_correct_list
    df["judge_reason"] = reason_list


    #############################################
    # 8. Save Output (.pkl)
    #############################################

    df.to_pickle(OUTPUT_FILE)
    print(f"\nDone! Saved judged dataframe to {OUTPUT_FILE}")

if __name__ == "__main__":
    judge_hotpotqa_qwen8b()