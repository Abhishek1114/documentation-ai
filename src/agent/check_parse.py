"""Does parse(pdf)['text'] equal the pair's input_text? If not, the agent's first try will not match ft_q4."""
import argparse
import json

from src.agent.paths import find_pdf
from src.parsing.parse import parse

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--split", default="val")
a = ap.parse_args()

pairs = [json.loads(l) for l in open(f"data/pairs/{a.split}.jsonl", encoding="utf-8") if l.strip()]
for kind in ("digital", "scan"):
    sub = [p for p in pairs if p["kind"] == kind][: a.n]
    same = 0
    for p in sub:
        live = parse(find_pdf(p["id"], kind))["text"]
        if live == p["input_text"]:
            same += 1
        elif same == 0 or True:
            k = next((i for i, (x, y) in enumerate(zip(live, p["input_text"])) if x != y), min(len(live), len(p["input_text"])))
            print(f"  DIFF {p['id']} at char {k}: live={live[k-20:k+40]!r} pair={p['input_text'][k-20:k+40]!r}")
    print(f"{kind}: {same}/{len(sub)} identical")