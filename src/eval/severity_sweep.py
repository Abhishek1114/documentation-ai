import csv, json, random, tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean

from src.eval.metrics import ROOT, cer
from src.generator.scan import pdf_to_scan
from src.parsing.parse import parse_scanned


def main(limit=10, severities=(0.0, 0.5, 1.0, 1.5, 2.0)):
    rows = list(csv.DictReader(open(ROOT / "manifest.csv", encoding="utf-8")))
    seen, sel = defaultdict(int), []
    for r in rows:
        if seen[r["layout"]] < limit:
            seen[r["layout"]] += 1; sel.append(r)

    out = []
    print(f"{'severity':<10}{'layout':<8}{'CER':>8}")
    for sev in severities:
        rng = random.Random(123)                       # same randomness each severity -> fair comparison
        per_layout = defaultdict(list)
        with tempfile.TemporaryDirectory() as tmp:
            for r in sel:
                p = str(Path(tmp) / f"{r['id']}.pdf")
                pdf_to_scan(r["pdf_path"], p, rng, severity=sev)
                gt = json.loads((ROOT / "gt" / f"{r['id']}.json").read_text(encoding="utf-8"))
                per_layout[r["layout"]].append(cer("\n".join(gt["gt_lines"]), parse_scanned(p)["text"]))
        for k, v in sorted(per_layout.items()):
            out.append([sev, k, round(mean(v), 4)])
            print(f"{sev:<10}{k:<8}{mean(v):>8.3f}")
    with open("results/severity_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["severity", "layout", "cer"]); w.writerows(out)


if __name__ == "__main__":
    main()