"""Groundedness checks on a DOCX read back from disk/bytes. They compare the document against the
extracted JSON, NOT against the original page."""
import json
from decimal import Decimal

from src.render.checks import _numbers
from src.render.docx_render import MEAS_HDR, STEPS_HDR, plain
from src.render.markdown import DASH, REQUIRED_HEADINGS


def _headings(doc):
    return [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]


def _body(doc, header):
    for t in doc.tables:
        if [c.text for c in t.rows[0].cells] == header:
            return [[c.text for c in r.cells] for r in t.rows[1:]]
    return []


def check_sections(doc):
    have = set(_headings(doc))
    return [f"missing section: {h}" for h in REQUIRED_HEADINGS if h not in have]


def check_steps(doc, rec):
    rows = _body(doc, STEPS_HDR)
    if len(rows) != len(rec.steps):
        return [f"steps: {len(rows)} rows rendered, {len(rec.steps)} in record"]
    bad = []
    for r, s in zip(rows, rec.steps):
        if len(r) != 4 or r[0] != str(s.no) or r[1] != plain(s.description) or r[2] != s.result:
            bad.append(f"steps: row for step {s.no} does not match record")
    return bad


def check_measurements(doc, rec):
    rows = _body(doc, MEAS_HDR)
    if len(rows) != len(rec.measurements):
        return [f"measurements: {len(rows)} rows rendered, {len(rec.measurements)} in record"]
    bad = []
    for r, m in zip(rows, rec.measurements):
        ok = len(r) == 6 and r[0] == plain(m.parameter) and r[2] == plain(m.unit)
        if ok:
            try:
                ok = Decimal(r[1]) == Decimal(str(m.value))
                for c, lim in ((r[3], m.limit_min), (r[4], m.limit_max)):
                    ok = ok and ((c == DASH) if lim is None else (c != DASH and Decimal(c) == Decimal(str(lim))))
            except Exception:
                ok = False
        if not ok:
            bad.append(f"measurements: row for {m.parameter} differs from record")
    return bad


def check_no_invented_numbers(doc, rec):
    texts = [p.text for p in doc.paragraphs]
    texts += [c.text for t in doc.tables for r in t.rows for c in r.cells]
    allowed = _numbers(json.dumps(rec.model_dump(mode="json")))
    return [f"number not in record: {x}" for x in sorted(_numbers("\n".join(texts)) - allowed)]


def check_docx(doc, rec):
    return (check_sections(doc) + check_steps(doc, rec) + check_measurements(doc, rec)
            + check_no_invented_numbers(doc, rec))