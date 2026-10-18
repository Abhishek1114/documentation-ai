"""Phase 3: turn one (input, target) pair into training tensors with loss masking.

Design rules (be ready to explain each in the interview):
1. The prompt comes from src.eval.prompts.build_messages, the SAME function the baselines used,
   so the fine-tune sees exactly the zero-shot prompt (shots=(), tables=False).
2. Only the assistant answer is trained on. Prompt positions get label -100.
3. Over-long examples raise an error. Truncating would cut the JSON (or its end token)
   and teach the model to emit broken output.
"""
import json
import sys

from src.eval.prompts import build_messages

IGNORE_INDEX = -100  # PyTorch cross-entropy skips targets equal to this value


def load_jsonl(path: str) -> list:
    """Read a .jsonl file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_example(pair: dict, tok, max_len: int) -> dict:
    """Return {"input_ids": [...], "attention_mask": [...], "labels": [...]} for one pair."""
    # 1. Build the two conversations with the shared prompt builder (never by hand).
    prompt_msgs = build_messages(pair, shots=(), tables=False, with_answer=False)
    full_msgs = build_messages(pair, shots=(), tables=False, with_answer=True)

    # 2. Render to text with the model's own chat template.
    #    prompt_text ends with "<|im_start|>assistant\n" (the generation prompt).
    #    full_text is the same text + answer + "<|im_end|>\n".
    prompt_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(full_msgs, tokenize=False)

    # The masking logic assumes full_text begins with prompt_text. Fail loudly if it does not.
    if not full_text.startswith(prompt_text):
        raise ValueError(f"{pair.get('id')}: prompt text is not a prefix of the full text")
    answer_text = full_text[len(prompt_text):]

    # 3. Tokenize the two parts separately and join the ids. add_special_tokens=False because
    #    the chat template already wrote every special token as text (<|im_start|>, <|im_end|>).
    #    Doing it this way makes the prompt/answer boundary exact by construction, and the
    #    prompt tokens are identical to what the model sees at inference time.
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tok(answer_text, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_len:
        raise ValueError(
            f"{pair.get('id')}: {len(input_ids)} tokens exceeds max_len={max_len}; "
            "refusing to truncate"
        )

    # 4. Labels: -100 over the prompt, real token ids over the answer.
    #    HF causal LMs shift labels by one internally, so labels is a plain copy of input_ids
    #    (with the prompt masked). Do NOT shift it yourself.
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def tokenize_split(pairs: list, tok, max_len: int) -> list:
    """Tokenize every pair in a split. Raises if any example is too long."""
    return [tokenize_example(p, tok, max_len) for p in pairs]


if __name__ == "__main__":
    # Sanity report: python -m src.training.data [max_len]
    from transformers import AutoTokenizer

    max_len = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    for split in ("train", "val", "test"):
        pairs = load_jsonl(f"data/pairs/{split}.jsonl")
        exs = tokenize_split(pairs, tok, max_len)
        lens = sorted(len(e["input_ids"]) for e in exs)
        trained = [sum(1 for x in e["labels"] if x != IGNORE_INDEX) for e in exs]
        print(
            f"{split}: n={len(exs)} total_tokens p50={lens[len(lens)//2]} max={lens[-1]} | "
            f"trained tokens/example avg={sum(trained)/len(trained):.0f}"
        )