import argparse
import json
from pathlib import Path

from src.agent.graph import build_graph, ollama_llm
from src.agent.tracing import Tracer

ap = argparse.ArgumentParser()
ap.add_argument("--pdf", required=True)
ap.add_argument("--model", default="gmp-q4")
ap.add_argument("--max-retries", type=int, default=2)
ap.add_argument("--ocr-psm", type=int, default=None)
ap.add_argument("--out-dir", default="results/traces/demo")
a = ap.parse_args()

p = Path(a.pdf)
trace_path = Path(a.out_dir) / f"{p.parent.name}_{p.stem}.jsonl"
graph = build_graph(ollama_llm(a.model), Tracer(trace_path))
final = graph.invoke({
    "pdf_path": str(p), "doc_id": p.stem,
    "ocr_kw": {"psm": a.ocr_psm} if a.ocr_psm else {},
    "max_retries": a.max_retries,
})
print(f"status={final['status']} kind={final['kind']} repairs={final['attempt']} "
      f"ok_first_try={final.get('ok_first_try')} llm_calls={len(final['calls'])}")
print("errors:", final["errors"])
print(final["calls"][-1]["content"])
print("trace:", trace_path)