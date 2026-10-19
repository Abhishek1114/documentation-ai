"""Run a GGUF model through Ollama on the SAME docs as ft_run1 and write preds in the same format,
so extraction_metrics can score fp16 vs Q8_0 vs Q4_K_M side by side. Also records latency."""
import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

from src.eval.prompts import build_messages

OLLAMA_URL = "http://localhost:11434/api/chat"


def chat(model, messages, num_predict, num_ctx, timeout):
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0,        # greedy, as in the HF baselines
            "repeat_penalty": 1.0,   # Ollama's default is 1.1; baselines used 1.0
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "seed": 42,
        },
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, time.perf_counter() - t0


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ollama model name, e.g. gmp-q4")
    ap.add_argument("--tag", required=True, help="short label, e.g. q4 -> mode 'ft_q4'")
    ap.add_argument("--split", default="val")
    ap.add_argument("--kinds", nargs="+", default=["digital", "scan"])
    ap.add_argument("--n", type=int, default=100, help="docs per kind (first n of the ft_run1 file)")
    ap.add_argument("--ref-mode", default="ft_run1")
    ap.add_argument("--num-predict", type=int, default=768)
    ap.add_argument("--num-ctx", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()

    mode = f"ft_{a.tag}"
    pairs = load_jsonl(f"data/pairs/{a.split}.jsonl")
    try:
        index = {(p["id"], p["kind"]): p for p in pairs}
    except KeyError:
        print("pair keys are:", list(pairs[0].keys()))
        raise SystemExit("ASSUMPTION FAILED: pairs need 'id' and 'kind' keys. Paste the line above.")

    for kind in a.kinds:
        ref_rows = load_jsonl(f"results/preds_{a.ref_mode}_{a.split}_{kind}.jsonl")[: a.n]
        out_path = Path(f"results/preds_{mode}_{a.split}_{kind}.jsonl")
        done = {r["id"] for r in load_jsonl(out_path)} if out_path.exists() else set()
        wall, gen_tps, pe_s = [], [], []
        with open(out_path, "a", encoding="utf-8") as fout:
            for i, ref in enumerate(ref_rows):
                if ref["id"] in done:
                    continue
                pair = index[(ref["id"], kind)]
                messages = build_messages(pair, shots=(), tables=False)
                out, secs = chat(a.model, messages, a.num_predict, a.num_ctx, a.timeout)
                n_out = out.get("eval_count", 0)
                eval_s = out.get("eval_duration", 0) / 1e9
                pe = out.get("prompt_eval_duration", 0) / 1e9
                row = {
                    "id": ref["id"], "layout": ref["layout"], "split": ref["split"], "kind": kind,
                    "target": ref["target"], "mode": mode,
                    "raw": out["message"]["content"],
                    "n_in_tokens": out.get("prompt_eval_count"),  # can undercount: Ollama reuses cached prompt prefix
                    "n_out_tokens": n_out,
                    "truncated": n_out >= a.num_predict,
                    "batch_seconds": round(secs, 2), "batch_size": 1,
                    "tables_in_prompt": False,
                    "model": f"{a.model} (gguf via ollama)",
                    "prompt_eval_s": round(pe, 3), "eval_s": round(eval_s, 3),
                    "load_s": round(out.get("load_duration", 0) / 1e9, 3),
                }
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                wall.append(secs)
                pe_s.append(pe)
                if eval_s > 0:
                    gen_tps.append(n_out / eval_s)
                print(f"[{mode} {kind}] {i + 1}/{len(ref_rows)} {ref['id']} {secs:.1f}s", flush=True)
        if wall:
            print(
                f"\n== {mode} {kind}: {len(wall)} new docs | median {statistics.median(wall):.1f} s/doc | "
                f"mean gen {statistics.mean(gen_tps):.1f} tok/s | median prompt-eval {statistics.median(pe_s):.2f} s\n"
            )


if __name__ == "__main__":
    main()