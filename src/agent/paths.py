"""Find the PDF for a doc id without hard-coding file names."""
from pathlib import Path


def find_pdf(doc_id, kind, root="data/synthetic"):
    d = Path(root) / ("pdf" if kind == "digital" else "scan")
    hits = sorted(d.glob(f"{doc_id}*"))
    hits = [h for h in hits if h.suffix.lower() == ".pdf"] or hits
    if not hits:
        raise FileNotFoundError(f"no file for {doc_id} in {d}")
    return str(hits[0])