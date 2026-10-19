"""Run the agent over the same docs as ft_run1. Writes preds in the extraction_metrics format twice:
  agent1_<tag>: the FIRST model output (before any repair)
  agent_<tag>:  the FINAL model output (after repairs)
plus per-doc outcomes and a summary."""
import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from src.agent.graph import build_graph, ollama_llm
from src.agent.paths import find_pdf
from src.agent.tracing import Tracer


def load_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pred_row(ref, kind, mode, call, model):
    return {
        "id": ref["id"], "layout": ref["layout"], "split": ref["split"], "kind": kind,
        "target": ref["target"], "mode": mode, "raw": call["content"],
        "n_in_tokens": call["n_in"], "n_out_tokens": call["n_out"], "truncated": call["truncated"],
        "batch_seconds": call["secs"], "batch_size": 1, "tables_in_prompt": False,
        "model": f"{model} (gguf via ollama, agent)",
    }


def summarize(docs):
    n = len(docs)
    resc = Counter(d["rescued_by"] for d in docs if d["rescued_by"])
    errs = Counter(d["error_kind"] for d in docs if d["error_kind"])
    return {
        "n": n,
        "ok_first_try": round(sum(d["ok_first_try"] for d in docs) / n, 3),
        "ok_final": round(sum(d["ok_final"] for d in docs) / n, 3),
        "mean_repairs": round(sum(d["attempts"] for d in docs) / n, 3),
        "median_secs": round(statistics.median(d["secs"] for d in docs), 1),
        "kind_detected_wrong": sum(d["kind"] != d["kind_detected"] for d in docs),
        "rescued_by": dict(resc), "final_failure_kinds": dict(errs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gmp-q4")
    ap.add_argument("--tag", required=True, help="e.g. q4 or q4_stress")
    ap.add_argument("--split", default="val")
    ap.add_argument("--kinds", nargs="+", default=["digital", "scan"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--ocr-psm", type=int, default=None, help="first-attempt Tesseract psm (stress test: 11)")
    ap.add_argument("--ref-mode", default="ft_run1")
    a = ap.parse_args()

    llm = ollama_llm(a.model)
    summary = {}
    for kind in a.kinds:
        refs = load_jsonl(f"results/preds_{a.ref_mode}_{a.split}_{kind}.jsonl")[: a.n]
        p_final = Path(f"results/preds_agent_{a.tag}_{a.split}_{kind}.jsonl")
        p_first = Path(f"results/preds_agent1_{a.tag}_{a.split}_{kind}.jsonl")
        p_docs = Path(f"results/agent_docs_{a.tag}_{a.split}_{kind}.jsonl")
        done = {d["id"] for d in load_jsonl(p_docs)}
        for i, ref in enumerate(refs):
            if ref["id"] in done:
                continue
            path = find_pdf(ref["id"], kind)
            if i == 0:
                print("first file used:", path)
            tracer = Tracer(f"results/traces/{a.tag}/{ref['id']}_{kind}.jsonl")
            graph = build_graph(llm, tracer)
            t0 = time.perf_counter()
            fin = graph.invoke({
                "pdf_path": path, "doc_id": ref["id"],
                "ocr_kw": {"psm": a.ocr_psm} if a.ocr_psm else {},
                "max_retries": a.max_retries,
            })
            secs = time.perf_counter() - t0
            ok = fin["status"] == "ok"
            rescued = None
            if ok and fin["attempt"] > 0:
                step = fin["ladder"][fin["attempt"] - 1]
                rescued = step["action"] + (f"_psm{step['ocr']['psm']}" if "ocr" in step else "")
            doc = {
                "id": ref["id"], "kind": kind, "kind_detected": fin["kind"],
                "ok_first_try": bool(fin.get("ok_first_try")), "ok_final": ok,
                "attempts": fin["attempt"], "n_llm_calls": len(fin["calls"]),
                "rescued_by": rescued,
                "error_kind": None if ok else fin["errors"][0].split(":")[0],
                "secs": round(secs, 1),
            }
            with open(p_first, "a", encoding="utf-8") as f:
                f.write(json.dumps(pred_row(ref, kind, f"agent1_{a.tag}", fin["calls"][0], a.model)) + "\n")
            with open(p_final, "a", encoding="utf-8") as f:
                f.write(json.dumps(pred_row(ref, kind, f"agent_{a.tag}", fin["calls"][-1], a.model)) + "\n")
            with open(p_docs, "a", encoding="utf-8") as f:
                f.write(json.dumps(doc) + "\n")
            print(f"[{a.tag} {kind}] {i + 1}/{len(refs)} {ref['id']} ok_first={doc['ok_first_try']} "
                  f"ok_final={ok} repairs={fin['attempt']} {secs:.1f}s", flush=True)
        docs = load_jsonl(p_docs)
        if docs:
            summary[kind] = summarize(docs)
            print(f"\n== {a.tag} {kind}: {summary[kind]}\n")
    Path(f"results/agent_summary_{a.tag}_{a.split}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()