"""
Generate HotpotQA dataset with Mixtral-8x7B-Instruct using vLLM.

Creates a dataset with each prompt and the generated response.
"""
import re
from typing import Dict, Any, List

import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams

from routing_dataset.dataset_paths import (
    HOTPOTQA_MIXTRAL_VALIDATION_FILE,
    HOTPOTQA_MIXTRAL_TRAIN_FILE,
)

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------

MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
HOT_POT_CONFIG = "distractor"
SPLIT = "validation"  # or "train" for training set

OUT_PKL = HOTPOTQA_MIXTRAL_VALIDATION_FILE  # Change to HOTPOTQA_MIXTRAL_TRAIN_FILE for train

# Model / decoding settings
MAX_MODEL_LEN = 8192
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0
PROGRESS_INTERVAL = 320

# Mixtral 8x7B typically needs tensor parallelism (2+ GPUs)
TENSOR_PARALLEL_SIZE = 2


# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------


def format_mistral_hotpot_prompt(context: str, question: str) -> str:
    """Format a HotpotQA example into Mistral/Mixtral [INST] chat format."""
    system_instruction = (
        "You are given context passages and a question. "
        "Answer the question using ONLY the provided context. "
        "Use information from multiple passages if needed. "
        "Return only the short final answer word or phrase, with no explanation."
    )
    user_content = (
        f"{system_instruction}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )
    return f"<s>[INST] {user_content} [/INST]"


def build_context(ex: Dict[str, Any]) -> str:
    """
    Build a flat text context from a HotpotQA example.

    In HF hotpot_qa, ex["context"] is a dict with:
        {"title": [title1, title2, ...], "sentences": [[sent1, sent2, ...], ...]}
    """
    parts: List[str] = []
    titles = ex["context"]["title"]
    sentences_list = ex["context"]["sentences"]
    for title, sents in zip(titles, sentences_list):
        parts.append(f"{title}: " + " ".join(sents))
    return "\n".join(parts)


def normalize(s: str) -> str:
    """Lowercase and remove non-word characters for EM comparison."""
    return re.sub(r"\W+", " ", (s or "").lower()).strip()


def postprocess_pred(pred: str) -> str:
    """Strip common answer prefixes that the model might add."""
    pred = pred.strip()
    lower = pred.lower()
    prefixes = ["answer:", "the answer is", "the final answer is"]
    for p in prefixes:
        if lower.startswith(p):
            pred = pred[len(p) :].strip()
            break
    return pred


def exact_match(pred: str, gold: str) -> int:
    pred_norm = normalize(postprocess_pred(pred))
    gold_norm = normalize(gold)
    return int(pred_norm == gold_norm)


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------


def main():
    out_file = OUT_PKL
    if SPLIT == "train":
        out_file = HOTPOTQA_MIXTRAL_TRAIN_FILE

    # Load HotpotQA
    ds = load_dataset("hotpot_qa", HOT_POT_CONFIG)[SPLIT]
    print(f"Loaded hotpot_qa/{HOT_POT_CONFIG}/{SPLIT}: {len(ds)} examples")

    # Initialize vLLM LLM
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    )

    params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
        top_p=1.0,
        stop=["</s>", "[/INST]"],  # Mixtral EOS and instruction end
    )

    # Pre-build prompts and metadata
    items = []
    prompts = []

    for ex in ds:
        context = build_context(ex)
        prompt = format_mistral_hotpot_prompt(context=context, question=ex["question"])
        prompts.append(prompt)
        items.append(
            {
                "id": ex["id"],
                "question": ex["question"],
                "context": context,
                "answer": ex["answer"],
                "prompt": prompt,
            }
        )

    n = len(items)
    print(f"Generating responses for {n} examples with {MODEL_ID}...")

    # vLLM handles batching internally
    outputs = llm.generate(prompts, params)

    # Collect results: prompt + generated response
    rows = []
    correct = 0

    for i, (item, out) in enumerate(zip(items, outputs)):
        response = out.outputs[0].text.strip()
        em = exact_match(response, item["answer"])
        correct += em

        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "context": item["context"],
                "answer": item["answer"],
                "prompt": item["prompt"],
                "response": response,
                "em": em,
            }
        )

        if (i + 1) % PROGRESS_INTERVAL == 0:
            print(f"Processed {i + 1}/{n} | running EM={correct / (i + 1):.4f}")

    # Save dataset with prompt and generated response
    df = pd.DataFrame(rows)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_file)

    print(f"Done. Saved to {out_file}")
    print(f"Final EM (exact match): {correct / n:.4f}")
    print(f"Dataset columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
