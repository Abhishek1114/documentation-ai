"""LangGraph agent: classify -> parse_doc -> extract -> validate -> (repair -> parse_doc | extract)* -> end.

Dependencies (llm, parse, is_scanned) are injected so the loop can be unit-tested without Ollama or Tesseract."""
import json
import time
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from src.eval.prompts import build_messages
from src.parsing.parse import is_scanned as _is_scanned
from src.parsing.parse import parse as _parse
from src.schema.models import GMPRecord
from src.serving.ollama_eval import chat

# Each step must CHANGE something, because decoding is greedy (same input -> same output).
LADDER = {
    "digital": [{"action": "reprompt"}],
    "scan": [
        {"action": "reprompt"},
        {"action": "reocr", "ocr": {"psm": 4}},
        {"action": "reocr", "ocr": {"psm": 3}},
    ],
}


class State(TypedDict, total=False):
    pdf_path: str
    doc_id: str
    kind: str
    ocr_kw: dict
    messages: list
    calls: list          # one dict per model call: content, n_in, n_out, truncated, secs
    errors: list
    attempt: int         # number of repairs done so far
    max_retries: int
    ladder: list
    action: str
    ok_first_try: bool
    status: str          # pending | ok | failed


def ollama_llm(model, num_predict=768, num_ctx=2048, timeout=600):
    def call(messages):
        out, secs = chat(model, messages, num_predict, num_ctx, timeout)
        n_out = out.get("eval_count", 0)
        return {
            "content": out["message"]["content"],
            "n_in": out.get("prompt_eval_count"),
            "n_out": n_out,
            "truncated": n_out >= num_predict,
            "secs": round(secs, 2),
        }
    return call


def decode(raw):
    """Return (obj, error). Accepts a bare JSON object or one preceded by prose."""
    raw = raw.strip()
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as e:
        first = e
    i = raw.find("{")
    if i >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw[i:])
            return obj, None
        except json.JSONDecodeError as e:
            first = e
    return None, f"json_decode: {first}"


def build_graph(llm, tracer, parse_fn=_parse, scanned_fn=_is_scanned):
    def classify(s):
        kind = "scan" if scanned_fn(s["pdf_path"]) else "digital"
        ladder = LADDER[kind][: s["max_retries"]]
        tracer.log("classify", doc=s["doc_id"], kind=kind, ladder=ladder)
        return {"kind": kind, "ladder": ladder, "attempt": 0, "errors": [], "calls": [], "status": "pending"}

    def parse_doc(s):
        t = time.perf_counter()
        kw = s.get("ocr_kw") or {}
        parsed = parse_fn(s["pdf_path"], **kw) if s["kind"] == "scan" else parse_fn(s["pdf_path"])
        text = parsed["text"]
        tracer.log("parse", method=parsed.get("method"), n_chars=len(text), ocr_kw=kw,
                   secs=round(time.perf_counter() - t, 2))
        # Same prompt as training and the baselines: text only, no shots.
        return {"messages": build_messages({"input_text": text}, shots=(), tables=False)}

    def extract(s):
        out = llm(s["messages"])
        tracer.log("extract", attempt=s["attempt"], n_in=out["n_in"], n_out=out["n_out"],
                   truncated=out["truncated"], secs=out["secs"], raw=out["content"][:4000])
        return {"calls": s["calls"] + [out]}

    def validate(s):
        call = s["calls"][-1]
        obj, err = decode(call["content"])
        errors = []
        if err:
            errors.append(err)
        elif not isinstance(obj, dict):
            errors.append("not_record: top-level JSON is not an object")
        else:
            try:
                GMPRecord(**obj)
            except ValidationError as e:
                errors += [f"schema: {'.'.join(str(p) for p in x['loc'])}: {x['msg']}" for x in e.errors()[:6]]
        if errors and call["truncated"]:
            errors.insert(0, "truncated: output reached the token limit")
        ok = not errors
        tracer.log("validate", attempt=s["attempt"], ok=ok, errors=errors)
        update = {"errors": errors, "status": "ok" if ok else "pending"}
        if s["attempt"] == 0:
            update["ok_first_try"] = ok
        return update

    def repair(s):
        i = s["attempt"]
        if i >= len(s["ladder"]):
            tracer.log("repair", decision="give_up", errors=s["errors"])
            return {"status": "failed", "action": "stop"}
        step = s["ladder"][i]
        if step["action"] == "reprompt":
            fix = "That JSON failed validation:\n- " + "\n- ".join(s["errors"]) + "\nReturn the corrected JSON object only."
            msgs = s["messages"] + [{"role": "assistant", "content": s["calls"][-1]["content"]},
                                    {"role": "user", "content": fix}]
            update = {"messages": msgs, "action": "reprompt"}
        else:
            update = {"ocr_kw": {**(s.get("ocr_kw") or {}), **step["ocr"]}, "action": "reocr"}
        tracer.log("repair", decision=step, errors=s["errors"])
        update["attempt"] = i + 1
        return update

    def route_repair(s):
        if s["status"] == "failed":
            return "end"
        return "parse_doc" if s["action"] == "reocr" else "extract"

    g = StateGraph(State)
    g.add_node("classify", classify)
    g.add_node("parse_doc", parse_doc)
    g.add_node("extract", extract)
    g.add_node("validate", validate)
    g.add_node("repair", repair)
    g.set_entry_point("classify")
    g.add_edge("classify", "parse_doc")
    g.add_edge("parse_doc", "extract")
    g.add_edge("extract", "validate")
    g.add_conditional_edges("validate", lambda s: "end" if s["status"] == "ok" else "repair",
                            {"end": END, "repair": "repair"})
    g.add_conditional_edges("repair", route_repair,
                            {"end": END, "parse_doc": "parse_doc", "extract": "extract"})
    return g.compile()