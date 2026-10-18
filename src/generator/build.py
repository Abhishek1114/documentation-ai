import argparse, csv, datetime as dt, json, random, shutil
from pathlib import Path

from src.generator import layouts
from src.generator.values import sample_record
from src.generator.scan import pdf_to_scan
from src.schema.models import GMPRecord

ROOT = Path("data/synthetic")
DERIVED = [Path("data/parsed"), Path("data/pairs")]            # built FROM the synthetic set; stale after a rebuild
SPLIT = {"A": "train", "B": "train", "C": "val", "D": "test"}   # split by LAYOUT, never random
LAYOUT_FNS = {k: getattr(layouts, f"render_{k.lower()}", None) for k in SPLIT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-layout", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layouts", default="ABCD")
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--clean", action="store_true",
                    help="delete data/synthetic, data/parsed and data/pairs first (use when regenerating)")
    args = ap.parse_args()

    if args.clean:
        for d in [ROOT, *DERIVED]:
            if d.exists():
                shutil.rmtree(d)
                print(f"[clean] removed {d}")
    elif (ROOT / "gt").exists() and any((ROOT / "gt").iterdir()):
        print("[warn] data/synthetic already has files. Re-run with --clean if the dataset changed, "
              "otherwise stale parsed/pair files may not match the new data.")

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

    # Dataset fingerprint. The parse cache stores this tag and refuses to reuse entries made for a different dataset.
    tag = f"seed{args.seed}_n{args.per_layout}_sev{args.severity}_{args.layouts}"
    (ROOT / "dataset_meta.json").write_text(json.dumps({
        "tag": tag, "seed": args.seed, "per_layout": args.per_layout, "severity": args.severity,
        "layouts": args.layouts, "n_docs": len(rows),
        "created": dt.datetime.now().isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} documents -> {ROOT / 'manifest.csv'}  (dataset tag: {tag})")


if __name__ == "__main__":
    main()
