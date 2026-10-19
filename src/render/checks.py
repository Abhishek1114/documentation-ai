"""Groundedness checks: does the rendered draft say exactly what the validated JSON says?
These compare the document against the extracted JSON, NOT against the original page."""
import json
import re
from decimal import Decimal

from src.render.markdown import DASH, REQUIRED_HEADINGS
from src.schema.models import GMPRecord

NUM = re.compile(r"\d+(?:\.\d+)?")
SPLIT = re.compile(r"(?<!\\)\|")


def _numbers(text):
    return {Decimal(t) for t in NUM.findall(text)}


def _section(md, heading):
    m = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", md, re.S | re.M)
    return m.group(1) if m else ""


def _rows(section):
    lines = [l for l in section.splitlines() if l.startswith("|")]
    return [[c.strip() for c in SPLIT.split(l[1:-1])] for l in lines[2:]]  # drop header + separator


def check_sections(md):
    return [f"missing section: {h}" for h in REQUIRED_HEADINGS if f"\n## {h}\n" not in "\n" + md]


def check_steps(md, rec):
    rows = _rows(_section(md, "Steps"))
    if len(rows) != len(rec.steps):
        return [f"steps: {len(rows)} rows rendered, {len(rec.steps)} in record"]
    bad = []
    for r, s in zip(rows, rec.steps):
        if len(r) != 4 or r[0] != str(s.no) or r[2] != s.result:
            bad.append(f"steps: row for step {s.no} does not match record")
    return bad


def check_measurements(md, rec):
    rows = _rows(_section(md, "Measurements"))
    if len(rows) != len(rec.measurements):
        return [f"measurements: {len(rows)} rows rendered, {len(rec.measurements)} in record"]
    bad = []
    for r, m in zip(rows, rec.measurements):
        if len(r) != 6:
            bad.append(f"measurements: malformed row for {m.parameter}")
            continue
        try:
            ok = Decimal(r[1]) == Decimal(str(m.value))
            for cellv, lim in ((r[3], m.limit_min), (r[4], m.limit_max)):
                ok &= (cellv == DASH) if lim is None else (cellv != DASH and Decimal(cellv) == Decimal(str(lim)))
        except Exception:
            ok = False
        if not ok:
            bad.append(f"measurements: numbers for {m.parameter} differ from record")
    return bad


def check_no_invented_numbers(md, rec):
    allowed = _numbers(json.dumps(rec.model_dump(mode="json")))
    extra = sorted(_numbers(md) - allowed)
    return [f"number not in record: {x}" for x in extra]


def check_all(md, rec: GMPRecord):
    return (check_sections(md) + check_steps(md, rec) + check_measurements(md, rec)
            + check_no_invented_numbers(md, rec))