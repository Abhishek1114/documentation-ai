import argparse, csv, json, random
from pathlib import Path

from src.generator import layouts
from src.generator.values import sample_record
from src.generator.scan import pdf_to_scan
from src.schema.models import GMPRecord

ROOT = Path("data/synthetic")
SPLIT = {"A": "train", "B": "train", "C": "val", "D": "test"}   # split by LAYOUT, never random
LAYOUT_FNS = {k: getattr(layouts, f"render_{k.lower()}", None) for k in SPLIT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-layout", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layouts", default="ABCD")
    ap.add_argument("--severity", type=float, default=1.0)
    args = ap.parse_args()

    for sub in ("pdf", "scan", "gt"):
        (ROOT / sub).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)      # one seeded RNG -> fully reproducible dataset
    rows, idx = [], 0

    for key in args.layouts:
        fn = LAYOUT_FNS[key]
        if fn is None:
            print(f"[skip] render_{key.lower()} not written yet")
            continue
        for _ in range(args.per_layout):
            idx += 1
            rec = sample_record(rng, idx)
            GMPRecord(**rec)            # raises if my ground truth violates the schema

            doc_id = f"{key}_{idx:05d}"
            pdf_path = ROOT / "pdf" / f"{doc_id}.pdf"
            scan_path = ROOT / "scan" / f"{doc_id}.pdf"

            gt = fn(rec, str(pdf_path))                  # digital PDF + exact ground truth
            pdf_to_scan(str(pdf_path), str(scan_path), rng, severity=args.severity)   # image-only degraded copy

            (ROOT / "gt" / f"{doc_id}.json").write_text(
                json.dumps({"id": doc_id, "layout": key, "record": rec, **gt}, indent=2),
                encoding="utf-8")
            rows.append({"id": doc_id, "layout": key, "split": SPLIT[key],
                         "pdf_path": pdf_path.as_posix(), "scan_path": scan_path.as_posix()})
        print(f"layout {key}: done")

    with open(ROOT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "layout", "split", "pdf_path", "scan_path"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} documents -> {ROOT / 'manifest.csv'}")


if __name__ == "__main__":
    main()