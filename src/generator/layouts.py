from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import datetime as dt
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

BOLD = {"Helvetica": "Helvetica-Bold", "Times-Roman": "Times-Bold", "Courier": "Courier-Bold"}


def _lim(v, null_marker):
    """Print a limit. A null limit is drawn with a layout-specific marker, never the string 'None'."""
    return null_marker if v is None else str(v)


def _limit_phrase(m):
    """Layout C prints limits as prose, so the model must turn words back into limit_min / limit_max."""
    lo, hi = m["limit_min"], m["limit_max"]
    if lo is None and hi is None:
        return "no limit"
    if lo is None:
        return f"limit up to {hi}"
    if hi is None:
        return f"limit at least {lo}"
    return f"limit {lo} to {hi}"


class _Page:
    """Collects the PDF story AND the ground-truth text in the same order."""
    def __init__(self, font):
        base = getSampleStyleSheet()
        self.st = {
            "normal": ParagraphStyle("n", parent=base["Normal"], fontName=font, fontSize=10, leading=14),
            "title": ParagraphStyle("t", parent=base["Title"], fontName=BOLD[font], fontSize=16),
        }
        self.story, self.lines = [], []

    def para(self, text, style="normal"):
        self.story.append(Paragraph(text, self.st[style]))
        self.lines.append(text)

    def space(self):
        self.story.append(Spacer(1, 12))

    def table(self, grid, cmds, font, col_widths=None):
        base = [("FONTNAME", (0, 0), (-1, -1), font), ("FONTSIZE", (0, 0), (-1, -1), 10)]
        self.story.append(Table(grid, colWidths=col_widths, style=TableStyle(base + cmds)))
        self.lines.extend(" ".join(r) for r in grid)

    def build(self, path):
        SimpleDocTemplate(path, pagesize=A4).build(self.story)

def _dev(rec):
    return "; ".join(rec["deviations"]) or "None"

# ---------- Layout A (train): ruled grids; null limit printed as "-" ----------
def render_a(rec: dict, out_pdf: str) -> dict:
    styles = getSampleStyleSheet()
    story, gt_lines = [], []

    def para(text, style="Normal"):
        story.append(Paragraph(text, styles[style]))
        gt_lines.append(text)

    para(f"Equipment Cleaning Record - {rec['document_id']}", "Title")
    para(f"Equipment: {rec['equipment']['name']} ({rec['equipment']['id']})")
    para(f"Location: {rec['equipment']['location']}")
    para(f"Date: {rec['date']}")
    para(f"Performed by: {rec['performed_by']}")
    story.append(Spacer(1, 12))

    step_grid = [["No", "Description", "Result", "Initials"]] + \
        [[str(s["no"]), s["description"], s["result"], s["initials"]] for s in rec["steps"]]
    meas_grid = [["Parameter", "Value", "Unit", "Min", "Max"]] + \
        [[m["parameter"], str(m["value"]), m["unit"], _lim(m["limit_min"], "-"), _lim(m["limit_max"], "-")]
         for m in rec["measurements"]]

    ts = TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                     ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)])
    story.append(Table(step_grid, style=ts)); story.append(Spacer(1, 12))
    gt_lines.extend(" ".join(row) for row in step_grid)
    story.append(Table(meas_grid, style=ts)); story.append(Spacer(1, 12))
    gt_lines.extend(" ".join(row) for row in meas_grid)
    para("Deviations: " + ("; ".join(rec["deviations"]) or "None"))
    para(f"Reviewed by: {rec['reviewed_by'] or ''}")

    SimpleDocTemplate(out_pdf, pagesize=A4).build(story)
    return {"gt_lines": gt_lines, "gt_tables": {"steps": step_grid, "measurements": meas_grid}}


# ---------- Layout B (train): no grid, columns reordered; null limit printed as "n.a." ----------
def render_b(rec, out_pdf):
    f, e = "Times-Roman", rec["equipment"]
    pg = _Page(f)
    pg.para(f"Cleaning Record No. {rec['document_id']}", "title")
    pg.para(f"Equipment: {e['name']} [{e['id']}]")
    pg.para(f"Area: {e['location']}")
    pg.para(f"Date of cleaning: {rec['date']}")
    pg.para(f"Done by: {rec['performed_by']}")
    pg.space()
    steps = [["No", "Result", "Description", "Initials"]] + \
        [[str(s["no"]), s["result"], s["description"], s["initials"]] for s in rec["steps"]]
    meas = [["Parameter", "Unit", "Value", "Min", "Max"]] + \
        [[m["parameter"], m["unit"], str(m["value"]), _lim(m["limit_min"], "n.a."), _lim(m["limit_max"], "n.a.")]
         for m in rec["measurements"]]
    cmds = [("LINEBELOW", (0, 0), (-1, 0), 1, colors.black), ("ALIGN", (0, 0), (-1, -1), "LEFT")]
    pg.table(steps, cmds, f); pg.space()
    pg.table(meas, cmds, f); pg.space()
    pg.para(f"Deviations noted: {_dev(rec)}")
    pg.para(f"Checked by: {rec['reviewed_by'] or ''}")
    pg.build(out_pdf)
    return {"gt_lines": pg.lines, "gt_tables": {"steps": steps, "measurements": meas}}


# ---------- Layout C (val): monospace, zebra rows, measurements as prose, date format changed ----------
def render_c(rec, out_pdf):
    f, e = "Courier", rec["equipment"]
    d = dt.date.fromisoformat(rec["date"]).strftime("%d-%b-%Y")       # 16-Oct-2026
    pg = _Page(f)
    pg.para(f"CLEANING LOG {rec['document_id']}", "title")
    pg.para(f"Machine: {e['name']} / {e['id']} / {e['location']}")
    pg.para(f"Cleaned on {d} by {rec['performed_by']}")
    pg.space()
    steps = [["#", "Action", "Outcome", "Sign"]] + \
        [[str(s["no"]), s["description"], s["result"], s["initials"]] for s in rec["steps"]]
    cmds = [("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey)]
    pg.table(steps, cmds, f); pg.space()
    parts = [f"{m['parameter']} was {m['value']} {m['unit']} ({_limit_phrase(m)})"
             for m in rec["measurements"]]
    pg.para("Rinse test results: " + "; ".join(parts) + ".")
    pg.para(f"Deviations: {_dev(rec)}")
    pg.para(f"Reviewer: {rec['reviewed_by'] or '-'}")                 # null is printed as "-"
    pg.build(out_pdf)
    return {"gt_lines": pg.lines, "gt_tables": {"steps": steps}}      # no measurement table drawn -> omit key


# ---------- Layout D (test): 2-column header, steps as text lines, merged-header table; null limit "N/A" ----------
def render_d(rec, out_pdf):
    f, e = "Helvetica", rec["equipment"]
    pg = _Page(f)
    pg.para("EQUIPMENT CLEANING VERIFICATION", "title")
    pg.para(f"Doc ID: {rec['document_id']}")
    header = [[f"Equipment: {e['name']} ({e['id']})", f"Date: {rec['date']}"],
              [f"Location: {e['location']}", f"Operator: {rec['performed_by']}"]]
    pg.table(header, [("ALIGN", (0, 0), (-1, -1), "LEFT")], f, col_widths=[260, 200])
    pg.space()
    for s in rec["steps"]:
        pg.para(f"Step {s['no']}: {s['description']} - {s['result'].upper()} ({s['initials']})")
    pg.space()
    meas = [["Measured parameters", "", "", "", ""], ["Parameter", "Value", "Unit", "Min", "Max"]] + \
        [[m["parameter"], str(m["value"]), m["unit"], _lim(m["limit_min"], "N/A"), _lim(m["limit_max"], "N/A")]
         for m in rec["measurements"]]
    cmds = [("SPAN", (0, 0), (-1, 0)), ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("ALIGN", (0, 0), (-1, 0), "CENTER")]
    pg.table(meas, cmds, f); pg.space()
    pg.para(f"Deviations: {_dev(rec)}")
    pg.para(f"Verified by: {rec['reviewed_by'] or 'N/A'}")             # null is printed as "N/A"
    pg.build(out_pdf)
    return {"gt_lines": pg.lines, "gt_tables": {"measurements": meas}}
