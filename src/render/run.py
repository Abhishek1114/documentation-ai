"""Render every schema-valid agent output as Markdown and DOCX and run the groundedness checks on both.
Invalid outputs are skipped and counted."""
import argparse
import io
import json
from pathlib import Path

from docx import Document

from src.render.checks import check_all
from src.render.docx_checks import check_docx
from src.render.docx_render import render_docx
from src.render.markdown import render_markdown
from src.schema.models import GMPRecord

ap = argparse.ArgumentParser()
ap.add_argument("--preds", nargs="+", required=True)
ap.add_argument("--out-dir", default="results/rendered")
ap.add_argument("--samples", type=int, default=3)
a = ap.parse_args()

out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
for path in a.preds:
    n = rendered = md_ok = docx_ok = saved = 0
    problems = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        n += 1
        try:
            rec = GMPRecord(**json.loads(row["raw"]))
        except Exception:
            continue                      # not schema-valid: nothing to render
        rendered += 1
        md = render_markdown(rec)
        e_md = check_all(md, rec)
        buf = io.BytesIO()
        render_docx(rec).save(buf)
        buf.seek(0)
        e_docx = check_docx(Document(buf), rec)   # read back from the saved bytes
        md_ok += not e_md
        docx_ok += not e_docx
        if e_md or e_docx:
            problems.append((row["id"], e_md + e_docx))
        if saved < a.samples:
            stem = f"{row['mode']}_{row['kind']}_{row['id']}"
            (out / f"{stem}.md").write_text(md, encoding="utf-8")
            (out / f"{stem}.docx").write_bytes(buf.getvalue())
            saved += 1
    print(f"{path}: {n} outputs | {rendered} rendered | markdown checks passed {md_ok} | docx checks passed {docx_ok}")
    for doc_id, errs in problems[:5]:
        print("  FAIL", doc_id, errs)