"""Render a validated GMPRecord as a QA-review Markdown draft. Deterministic template: no model call."""
from decimal import Decimal

from src.schema.models import GMPRecord

DASH = "—"
TITLES = {
    "cleaning_record": "Equipment Cleaning Record",
    "maintenance_log": "Equipment Maintenance Log",
    "sop_excerpt": "SOP Excerpt",
}
REQUIRED_HEADINGS = ["Document details", "Equipment", "Steps", "Measurements", "Deviations",
                     "Review flags", "Sign-off"]


def fmt_num(x) -> str:
    """Plain decimal text without exponent or trailing zeros: 7.0 -> '7', 500.70 -> '500.7'."""
    return format(Decimal(str(x)).normalize(), "f")


def fmt_opt(x) -> str:
    return DASH if x is None else fmt_num(x)


def cell(s) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ").strip()


def limit_check(m) -> str:
    if m.limit_min is None and m.limit_max is None:
        return "no limits"
    if m.limit_min is not None and m.value < m.limit_min:
        return "OUT OF LIMITS"
    if m.limit_max is not None and m.value > m.limit_max:
        return "OUT OF LIMITS"
    return "within limits"


def review_flags(rec: GMPRecord) -> list:
    flags = []
    if rec.reviewed_by is None:
        flags.append("Not yet reviewed: no reviewer recorded.")
    for s in rec.steps:
        if s.result == "fail":
            flags.append(f"Failed step {s.no}: {cell(s.description)}")
    for m in rec.measurements:
        if limit_check(m) == "OUT OF LIMITS":
            flags.append(f"Out of limits: {cell(m.parameter)} = {fmt_num(m.value)} {cell(m.unit)} "
                         f"(limits {fmt_opt(m.limit_min)} to {fmt_opt(m.limit_max)})")
    for d in rec.deviations:
        flags.append(f"Deviation recorded: {cell(d)}")
    return flags


def render_markdown(rec: GMPRecord) -> str:
    L = [f"# {TITLES[rec.document_type]}", "",
         "*Draft for QA review. Generated from extracted data; not a validated GMP document.*", ""]

    L += ["## Document details", "", "| Field | Value |", "|---|---|",
          f"| Document ID | {cell(rec.document_id)} |",
          f"| Document type | {rec.document_type} |",
          f"| Date | {rec.date.isoformat()} |",
          f"| Performed by | {cell(rec.performed_by)} |",
          f"| Reviewed by | {cell(rec.reviewed_by) if rec.reviewed_by else 'Not recorded'} |", ""]

    L += ["## Equipment", "", "| Field | Value |", "|---|---|",
          f"| ID | {cell(rec.equipment.id)} |",
          f"| Name | {cell(rec.equipment.name)} |",
          f"| Location | {cell(rec.equipment.location)} |", ""]

    L += ["## Steps", ""]
    if rec.steps:
        L += ["| No | Description | Result | Initials |", "|---|---|---|---|"]
        L += [f"| {s.no} | {cell(s.description)} | {s.result} | {cell(s.initials)} |" for s in rec.steps]
    else:
        L += ["No steps recorded."]
    L += [""]

    L += ["## Measurements", ""]
    if rec.measurements:
        L += ["| Parameter | Value | Unit | Limit min | Limit max | Limit check |", "|---|---|---|---|---|---|"]
        L += [f"| {cell(m.parameter)} | {fmt_num(m.value)} | {cell(m.unit)} | {fmt_opt(m.limit_min)} | "
              f"{fmt_opt(m.limit_max)} | {limit_check(m)} |" for m in rec.measurements]
    else:
        L += ["No measurements recorded."]
    L += [""]

    L += ["## Deviations", ""]
    L += [f"- {cell(d)}" for d in rec.deviations] if rec.deviations else ["None recorded."]
    L += [""]

    L += ["## Review flags", ""]
    flags = review_flags(rec)
    L += [f"- {f}" for f in flags] if flags else ["None."]
    L += [""]

    L += ["## Sign-off", "", "| Role | Name | Signature | Date |", "|---|---|---|---|",
          f"| Performed by | {cell(rec.performed_by)} | | |",
          f"| Reviewed by | {cell(rec.reviewed_by) if rec.reviewed_by else ''} | | |", ""]
    return "\n".join(L)