import csv, json, time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.eval.metrics import ROOT, cer, best_table_accuracy
from src.parsing.parse import parse_scanned

CONFIGS = [dict(psm=p, preprocess=m) for p in (4, 6, 11) for m in ("none", "autocontrast", "threshold")]


def main(limit=10):
    rows = list(csv.DictReader(open(ROOT / "manifest.csv", encoding="utf-8")))
    seen, sel = defaultdict(int), []
    for r in rows:
        if seen[r["layout"]] < limit:
            seen[r["layout"]] += 1; sel.append(r)

    out = []
    print(f"{'psm':<5}{'preprocess':<14}{'CER':>8}{'table_acc':>11}{'sec/doc':>9}")
    for cfg in CONFIGS:
        cers, tabs, t0 = [], [], time.time()
        for r in sel:
            gt = json.loads((ROOT / "gt" / f"{r['id']}.json").read_text(encoding="utf-8"))
            res = parse_scanned(r["scan_path"], **cfg)
            cers.append(cer("\n".join(gt["gt_lines"]), res["text"]))
            grids = [g for g in gt["gt_tables"].values() if g]
            if grids:
                tabs.append(mean(best_table_accuracy(g, res["tables"]) for g in grids))
        row = [cfg["psm"], cfg["preprocess"], round(mean(cers), 4),
               round(mean(tabs), 4) if tabs else "", round((time.time() - t0) / len(sel), 2)]
        out.append(row)
        print(f"{row[0]:<5}{row[1]:<14}{row[2]:>8.3f}{row[3] or 0:>11.3f}{row[4]:>9.2f}")
    Path("results").mkdir(exist_ok=True)
    with open("results/ocr_experiment.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["psm", "preprocess", "cer", "table_acc", "sec_per_doc"]); w.writerows(out)


if __name__ == "__main__":
    main()