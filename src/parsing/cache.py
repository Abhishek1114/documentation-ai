"""Parse every document ONCE and store the result, so later steps never re-run OCR.

    python -m src.parsing.cache                       # all docs, digital + scan, 1 process
    python -m src.parsing.cache --workers 4           # OCR in parallel (use your core count - 1)
    python -m src.parsing.cache --limit-per-layout 5  # quick smoke test

Output: data/parsed/<doc_id>.<kind>.json   (kind = digital | scan)
Resumable: a file that already exists AND was made for the current dataset tag is skipped,
so Ctrl+C and re-run is safe. A file made for a different dataset (stale) is recomputed.
"""
import argparse, csv, json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.parsing.parse import parse, PARSER_VERSION

SYN = Path("data/synthetic")
OUT = Path("data/parsed")
KINDS = ("digital", "scan")


def dataset_tag() -> str:
    meta = SYN / "dataset_meta.json"
    return json.loads(meta.read_text(encoding="utf-8"))["tag"] if meta.exists() else "unknown"


def out_path(doc_id: str, kind: str) -> Path:
    return OUT / f"{doc_id}.{kind}.json"


def is_fresh(doc_id: str, kind: str, tag: str) -> bool:
    p = out_path(doc_id, kind)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("dataset_tag") == tag and data.get("parser_version", "v1") == PARSER_VERSION[kind]
    except (json.JSONDecodeError, OSError):
        return False                       # half-written / corrupt file -> redo it


def parse_one(job: tuple) -> tuple:
    """Runs in a worker process. Must be a top-level function (Windows uses 'spawn')."""
    doc_id, layout, kind, pdf_path, tag = job
    t0 = time.perf_counter()
    try:
        r = parse(pdf_path)
        detected = "scan" if r["method"].startswith("tesseract") else "digital"
        rec = {"id": doc_id, "layout": layout, "kind": kind, "detected_kind": detected,
               "method": r["method"], "text": r["text"], "tables": r["tables"],
               "dataset_tag": tag, "parser_version": PARSER_VERSION[kind]}
        tmp = out_path(doc_id, kind).with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, out_path(doc_id, kind))          # atomic: a crash never leaves a half file
        return doc_id, kind, None, time.perf_counter() - t0
    except Exception as e:                               # keep going; report at the end
        return doc_id, kind, f"{type(e).__name__}: {e}", time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--limit-per-layout", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tag = dataset_tag()
    with open(SYN / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in args.splits]

    if args.limit_per_layout:
        seen, kept = {}, []
        for r in rows:
            seen[r["layout"]] = seen.get(r["layout"], 0) + 1
            if seen[r["layout"]] <= args.limit_per_layout:
                kept.append(r)
        rows = kept

    jobs = []
    for r in rows:
        for kind in args.kinds:
            if not is_fresh(r["id"], kind, tag):
                jobs.append((r["id"], r["layout"], kind,
                             r["pdf_path"] if kind == "digital" else r["scan_path"], tag))
    print(f"dataset tag: {tag} | {len(rows)} docs | {len(jobs)} files to parse | workers={args.workers}")
    if not jobs:
        print("nothing to do")
        return

    if args.workers > 1:
        os.environ["OMP_THREAD_LIMIT"] = "1"   # Tesseract's own threading would fight our processes
    errors, secs = [], {k: [] for k in KINDS}
    t_start = time.perf_counter()

    def record(res, n):
        doc_id, kind, err, dt_ = res
        secs[kind].append(dt_)
        if err:
            errors.append((doc_id, kind, err))
        if n % 25 == 0 or n == len(jobs):
            el = time.perf_counter() - t_start
            print(f"  {n}/{len(jobs)} done | {el/60:.1f} min elapsed | ~{el/n*(len(jobs)-n)/60:.1f} min left")

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(parse_one, j) for j in jobs]
            for n, fu in enumerate(as_completed(futs), 1):
                record(fu.result(), n)
    else:
        for n, j in enumerate(jobs, 1):
            record(parse_one(j), n)

    for k in KINDS:
        if secs[k]:
            print(f"{k}: {len(secs[k])} files, mean {sum(secs[k])/len(secs[k]):.2f} s/file")
    if errors:
        print(f"\n{len(errors)} FAILURES (re-run to retry):")
        for e in errors[:20]:
            print("  ", e)


if __name__ == "__main__":
    main()
