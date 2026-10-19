import json

from src.agent.graph import build_graph, decode
from src.agent.tracing import Tracer

VALID = {
    "document_id": "CR-1", "document_type": "cleaning_record",
    "equipment": {"id": "E1", "name": "Blender", "location": "Room 1"},
    "date": "2026-07-02", "performed_by": "A. Patel", "reviewed_by": None,
    "steps": [{"no": 1, "description": "Rinse", "result": "pass", "initials": "AP"}],
    "measurements": [{"parameter": "pH", "value": 7.0, "unit": "pH", "limit_min": 6.0, "limit_max": 8.0}],
    "deviations": [],
}


def bad():
    d = json.loads(json.dumps(VALID))
    d["steps"][0]["result"] = "PASS"
    return d


def fake_parse(path, **kw):
    return {"text": "page text", "tables": [], "method": "fake"}


def make_llm(fn, log):
    def llm(messages):
        log.append(messages)
        return {"content": fn(messages), "n_in": 1, "n_out": 1, "truncated": False, "secs": 0.0}
    return llm


def run(tmp_path, llm, kind="digital", parse_fn=fake_parse, max_retries=2):
    tr = Tracer(tmp_path / "t.jsonl")
    g = build_graph(llm, tr, parse_fn=parse_fn, scanned_fn=lambda p: kind == "scan")
    final = g.invoke({"pdf_path": "x.pdf", "doc_id": "T_1", "ocr_kw": {}, "max_retries": max_retries})
    trace = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    return final, trace


def test_success_first_try(tmp_path):
    log = []
    final, trace = run(tmp_path, make_llm(lambda m: json.dumps(VALID), log))
    assert final["status"] == "ok" and final["attempt"] == 0 and final["ok_first_try"] is True
    assert [t["node"] for t in trace] == ["classify", "parse", "extract", "validate"]
    assert len(log) == 1


def test_reprompt_recovers_and_sees_errors(tmp_path):
    log = []
    answers = iter([json.dumps(bad()), json.dumps(VALID)])
    final, _ = run(tmp_path, make_llm(lambda m: next(answers), log))
    assert final["status"] == "ok" and final["attempt"] == 1 and final["ok_first_try"] is False
    assert "steps.0.result" in log[1][-1]["content"]


def test_scan_reocr_ladder(tmp_path):
    def parse_fn(path, **kw):
        return {"text": "good" if kw.get("psm") == 4 else "bad", "tables": [], "method": "fake"}
    log = []
    llm = make_llm(lambda m: json.dumps(VALID if "good" in m[1]["content"] else bad()), log)
    final, trace = run(tmp_path, llm, kind="scan", parse_fn=parse_fn)
    assert final["status"] == "ok" and final["attempt"] == 2 and len(log) == 3
    assert [t["node"] for t in trace].count("parse") == 2


def test_gives_up_after_ladder(tmp_path):
    log = []
    final, _ = run(tmp_path, make_llm(lambda m: json.dumps(bad()), log))
    assert final["status"] == "failed" and len(log) == 2 and final["ok_first_try"] is False


def test_truncated_is_flagged(tmp_path):
    def llm(messages):
        return {"content": '{"document_id": "CR', "n_in": 1, "n_out": 768, "truncated": True, "secs": 0.0}
    final, _ = run(tmp_path, llm, max_retries=0)
    assert final["status"] == "failed" and final["errors"][0].startswith("truncated")


def test_decode_handles_prose_and_broken_json():
    obj, err = decode("Sure! " + json.dumps(VALID) + " hope that helps")
    assert err is None and obj["document_id"] == "CR-1"
    obj, err = decode('{"a":')
    assert obj is None and err.startswith("json_decode")