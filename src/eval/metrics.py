import argparse, csv, json, re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from rapidfuzz.distance import Levenshtein
from src.parsing.parse import parse

ROOT = Path("data/synthetic")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def cer(ref: str, hyp: str) -> float:
    """Character error rate = edit distance / reference length."""
    ref, hyp = norm(ref), norm(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def _cell(x) -> str:
    return norm(str(x)) if x is not None else ""


def table_cell_accuracy(gt_grid, pred_grid) -> float:
    """Fraction of ground-truth cells whose (row, col) position holds identical text."""
    total = sum(len(r) for r in gt_grid)
    if total == 0:
        return 1.0
    correct = 0
    for i, row in enumerate(gt_grid):
        for j, cell in enumerate(row):
            if i < len(pred_grid) and j < len(pred_grid[i]) \
               and _cell(pred_grid[i][j]) == _cell(cell):
                correct += 1
    return correct / total


def best_table_accuracy(gt_grid, pred_tables) -> float:
    """Match a ground-truth table to whichever predicted table fits best (order-independent)."""
    if not pred_tables:
        return 0.0
    return max(table_cell_accuracy(gt_grid, p) for p in pred_tables)


def evaluate(split=None, limit=None):
    with open(ROOT / "manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if split:
        rows = [r for r in rows if r["split"] == split]

    seen = defaultdict(int)
    results = defaultdict(lambda: {"cer": [], "tab": []})

    for r in rows:
        if limit and seen[r["layout"]] >= limit:
            continue
        seen[r["layout"]] += 1
        gt = json.loads((ROOT / "gt" / f"{r['id']}.json").read_text(encoding="utf-8"))
        ref_text = "\n".join(gt["gt_lines"])
        grids = [g for g in gt["gt_tables"].values() if g]

        for kind, path in (("digital", r["pdf_path"]), ("scan", r["scan_path"])):
            out = parse(path)
            key = (r["layout"], kind, out["method"])
            results[key]["cer"].append(cer(ref_text, out["text"]))
            if grids:
                results[key]["tab"].append(
                    mean(best_table_accuracy(g, out["tables"]) for g in grids))
    return results


def report(results):
    header = f"{'layout':<7}{'input':<9}{'method':<18}{'n':>4}{'CER':>9}{'table_acc':>11}"
    print(header); print("-" * len(header))
    out_rows = []
    for (layout, kind, method), v in sorted(results.items()):
        n, c = len(v["cer"]), mean(v["cer"])
        t = mean(v["tab"]) if v["tab"] else float("nan")
        print(f"{layout:<7}{kind:<9}{method:<18}{n:>4}{c:>9.3f}{t:>11.3f}")
        out_rows.append([layout, kind, method, n, round(c, 4), round(t, 4)])
    Path("results").mkdir(exist_ok=True)
    with open("results/parsing_eval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layout", "input", "method", "n", "cer", "table_acc"])
        w.writerows(out_rows)
    print("\nsaved -> results/parsing_eval.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None, help="train | val | test")
    ap.add_argument("--limit", type=int, default=None, help="max docs per layout")
    a = ap.parse_args()
    report(evaluate(a.split, a.limit))