import re
from typing import Dict, Any, List

import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from routing_dataset.dataset_paths import (
    HOTPOTQA_QWEN34B_VALIDATION_CORRECT_LABELS_FILE,
    HOTPOTQA_QWEN8B_VALIDATION_CORRECT_LABELS_FILE,
)

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-1.7B"

HOT_POT_CONFIG = "distractor"
SPLIT = "validation"  # HF's dev split

# NOTE: This still uses your existing constant.
# You may want to create QWEN8B-specific paths instead.
OUT_PKL = HOTPOTQA_QWEN34B_VALIDATION_CORRECT_LABELS_FILE

# Model / decoding settings
MAX_MODEL_LEN = 8192          # adjust to your GPU / model context limit for Qwen3-8B
MAX_NEW_TOKENS = 64           # HotpotQA answers are short; no need for 4k tokens
TEMPERATURE = 0.0
PROGRESS_INTERVAL = 320       # print progress every N examples


# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------

def format_qwen_hotpot_prompt(context: str, question: str) -> str:
    """Format a single HotpotQA example into a Qwen-style chat prompt."""
    system_instruction = (
        "You are given context passages and a question. "
        "Answer the question using ONLY the provided context. "
        "Use information from multiple passages if needed. "
        "Return only the short final answer word or phrase, with no explanation."
    )

    user_content = (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )

    prompt = (
        "<|im_start|>system\n"
        f"{system_instruction}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_content}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Answer:"
    )

    return prompt


def build_context(ex: Dict[str, Any]) -> str:
    """
    Build a flat text context from a HotpotQA example.

    In HF hotpot_qa, ex["context"] is a dict with:
        {"title": [title1, title2, ...], "sentences": [[sent1, sent2, ...], [sent3, sent4, ...], ...]}
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
            pred = pred[len(p):].strip()
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
    # Load HotpotQA
    ds = load_dataset("hotpot_qa", HOT_POT_CONFIG)[SPLIT]
    print(f"Loaded hotpot_qa/{HOT_POT_CONFIG}/{SPLIT}: {len(ds)} examples")

    # Initialize vLLM LLM
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
    )

    params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
        top_p=1.0,                # deterministic with temperature=0
        stop=["<|im_end|>"],      # stop at end-of-assistant message if generated
    )

    # Pre-build prompts and metadata
    items = []
    prompts = []

    for ex in ds:
        context = build_context(ex)
        prompt = format_qwen_hotpot_prompt(context=context, question=ex["question"])
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
    print(f"Generating responses for {n} examples...")

    # Let vLLM handle batching internally
    outputs = llm.generate(prompts, params)

    # Collect all results
    rows = []
    correct = 0

    # vLLM returns outputs in the same order as prompts
    for i, (item, out) in enumerate(zip(items, outputs)):
        pred = out.outputs[0].text.strip()
        em = exact_match(pred, item["answer"])
        correct += em

        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "context": item["context"],
                "answer": item["answer"],
                "prompt": item["prompt"],
                "pred": pred,
                "em": em,
            }
        )

        if (i + 1) % PROGRESS_INTERVAL == 0:
            print(f"Processed {i + 1}/{n} | running EM={correct / (i + 1):.4f}")

    # Convert to DataFrame and save
    df = pd.DataFrame(rows)
    df.to_pickle(OUT_PKL)

    print(f"Done. Saved: {OUT_PKL}")
    print(f"Final EM (exact match): {correct / n:.4f}")


if __name__ == "__main__":
    main()
