"""Phase 3 prep: token lengths using the REAL baseline prompt (src/eval/prompts.build_messages).

Also checks the property loss masking depends on: the tokenized prompt (with generation prompt)
must be an exact prefix of the tokenized full conversation (prompt + assistant answer).
"""
import json

from transformers import AutoTokenizer

from src.eval.prompts import SYSTEM_PROMPT, build_messages

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FILES = {
    "train": "data/pairs/train.jsonl",
    "val": "data/pairs/val.jsonl",
    "test": "data/pairs/test.jsonl",
}
THRESHOLDS = [1024, 1280, 1536, 2048]


def pct(sorted_vals, p):
    idx = min(len(sorted_vals) - 1, int(round(p / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def summarize(name, vals):
    v = sorted(vals)
    print(
        f"  {name:<7} n={len(v):>5}  p50={pct(v, 50):>5}  p90={pct(v, 90):>5}  "
        f"p99={pct(v, 99):>5}  max={v[-1]:>5}"
    )


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    print("system prompt tokens:", len(tok(SYSTEM_PROMPT)["input_ids"]))

    prefix_failures = 0
    for split, path in FILES.items():
        by_kind = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                pair = json.loads(line)
                prompt_msgs = build_messages(pair, shots=(), tables=False, with_answer=False)
                full_msgs = build_messages(pair, shots=(), tables=False, with_answer=True)
                prompt_ids = tok.apply_chat_template(
                    prompt_msgs, tokenize=True, add_generation_prompt=True
                )
                full_ids = tok.apply_chat_template(full_msgs, tokenize=True)
                if hasattr(prompt_ids, "input_ids"):  # some versions return a BatchEncoding
                    prompt_ids, full_ids = prompt_ids["input_ids"], full_ids["input_ids"]
                if full_ids[: len(prompt_ids)] != prompt_ids:
                    prefix_failures += 1
                d = by_kind.setdefault(pair["kind"], {"prompt": [], "target": [], "total": []})
                d["prompt"].append(len(prompt_ids))
                d["target"].append(len(full_ids) - len(prompt_ids))
                d["total"].append(len(full_ids))
        print(f"\n[{split}]")
        for kind, d in sorted(by_kind.items()):
            print(f" {kind}")
            for k in ("prompt", "target", "total"):
                summarize(k, d[k])
            print("  total > threshold:", {t: sum(1 for x in d["total"] if x > t) for t in THRESHOLDS})

    print("\nprefix property failures:", prefix_failures, "(must be 0)")


if __name__ == "__main__":
    main()