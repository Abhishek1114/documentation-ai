"""Failure analysis on layout C predictions: which items are wrong, and why.

Reuses the metric's own parse/flatten/normalise code, so error counts match the F1 table.
Error kinds per item (path):
  null_to_value  gold is null, model wrote a value
  value_to_null  gold has a value, model wrote null
  wrong_value    both non-null, normalised values differ
  missing_item   path in gold, absent from prediction (e.g. dropped list item)
  extra_item     path in prediction, absent from gold (e.g. invented list item)
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict

from src.eval.extraction_metrics import (
    TYPES, field_type, flatten, generic, normalize, salvage_parse, score, try_parse,
)

ERR_KINDS = ["null_to_value", "value_to_null", "wrong_value", "missing_item", "extra_item"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def classify(gold, pred):
    """List of (path, error_kind, gold_value, pred_value) for one document."""
    g = flatten(gold)
    p = flatten(pred) if pred else {}
    errs = []
    for path, gv in g.items():
        ft = field_type(path)
        if path not in p:
            errs.append((path, "missing_item", gv, None))
            continue
        pv = p[path]
        if normalize(pv, ft) == normalize(gv, ft):
            continue
        if gv is None:
            kind = "null_to_value"
        elif pv is None:
            kind = "value_to_null"
        else:
            kind = "wrong_value"
        errs.append((path, kind, gv, pv))
    for path, pv in p.items():
        if path not in g:
            errs.append((path, "extra_item", None, pv))
    return errs


def bucket(path, kind, gold_value):
    """Same bucket rule as extraction_metrics.score."""
    if kind == "extra_item":
        return field_type(path)
    return "null" if gold_value is None else field_type(path)


def short(v, n=40):
    s = repr(v)
    return s if len(s) <= n else s[: n - 3] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="results/failure_c_errors.csv")
    ap.add_argument("--examples", type=int, default=12)
    ap.add_argument("--allow-test", action="store_true")
    a = ap.parse_args()

    if not a.allow_test and any("_test_" in f for f in a.files):
        sys.exit("Refusing test-split files. Layout D is scored once, on purpose.")

    all_errs = []                 # dicts, one per wrong item
    doc_errs = {}                 # (kind, id) -> list of (path, ekind, gold, pred)
    doc_status = {}               # (kind, id) -> 'valid' | 'salvaged' | 'unparsable'

    for fpath in a.files:
        for r in read_jsonl(fpath):
            gold, gerr = try_parse(r["target"])
            assert gerr is None, f"bad gold for {r['id']}: {gerr}"
            pred, _ = try_parse(r["raw"])
            if pred is not None:
                status = "valid"
            else:
                pred = salvage_parse(r["raw"])
                status = "salvaged" if pred is not None else "unparsable"
            errs = classify(gold, pred)

            if status == "valid":     # consistency check against the metric itself
                c = score(pred, gold)
                deficit = sum(c[t]["n_pred"] - c[t]["tp"] + c[t]["n_gold"] - c[t]["tp"] for t in TYPES)
                n_wrong = sum(e[1] in ("null_to_value", "value_to_null", "wrong_value") for e in errs)
                expected = 2 * n_wrong + sum(e[1] in ("missing_item", "extra_item") for e in errs)
                assert deficit == expected, f"{r['id']}: metric deficit {deficit} != my errors {expected}"

            key = (r["kind"], r["id"])
            doc_errs[key] = errs
            doc_status[key] = status
            for path, ek, gv, pv in errs:
                all_errs.append({"id": r["id"], "kind": r["kind"], "status": status,
                                 "field": generic(path), "path": path, "error": ek,
                                 "bucket": bucket(path, ek, gv),
                                 "gold": gv, "pred": pv})

    kinds = sorted({k for k, _ in doc_errs})
    for kind in kinds:
        ids = [i for k, i in doc_errs if k == kind]
        st = Counter(doc_status[(kind, i)] for i in ids)
        n_bad = sum(1 for i in ids if doc_errs[(kind, i)])
        print(f"\n=== {kind}: {len(ids)} docs, status {dict(st)}, "
              f"imperfect {n_bad}, total wrong items {sum(len(doc_errs[(kind, i)]) for i in ids)}")

        # 1. by field type bucket x error kind
        tab = defaultdict(Counter)
        for e in all_errs:
            if e["kind"] == kind:
                tab[e["bucket"]][e["error"]] += 1
        print("\n-- by bucket x error kind")
        print(f"{'bucket':8}" + "".join(f"{k:>15}" for k in ERR_KINDS) + f"{'total':>8}")
        for b in TYPES:
            if b in tab:
                row = tab[b]
                print(f"{b:8}" + "".join(f"{row[k]:>15}" for k in ERR_KINDS) + f"{sum(row.values()):>8}")

        # 2. by specific field
        by_field = defaultdict(Counter)
        for e in all_errs:
            if e["kind"] == kind:
                by_field[e["field"]][e["error"]] += 1
        print("\n-- by field (top 12)")
        for f, row in sorted(by_field.items(), key=lambda kv: -sum(kv[1].values()))[:12]:
            detail = ", ".join(f"{k}={v}" for k, v in row.items())
            print(f"{f:32} {sum(row.values()):>4}  {detail}")

        # 3. by document: errors per doc and shared causes
        hist = Counter(min(len(doc_errs[(kind, i)]), 3) for i in ids)
        print(f"\n-- errors per doc: 0={hist[0]}  1={hist[1]}  2={hist[2]}  3+={hist[3]}")
        sig = Counter()
        for i in ids:
            errs = doc_errs[(kind, i)]
            if errs:
                sig[tuple(sorted({(generic(p), ek) for p, ek, _, _ in errs}))] += 1
        print("-- most common error signatures among imperfect docs")
        for s, n in sig.most_common(6):
            print(f"{n:>4} docs: " + "; ".join(f"{f} {ek}" for f, ek in s))

    # 4. OCR contribution: pair digital and scan variants by id
    if {"digital", "scan"} <= set(kinds):
        dig = {i for k, i in doc_errs if k == "digital"}
        scn = {i for k, i in doc_errs if k == "scan"}
        both = sorted(dig & scn)
        print(f"\n=== digital vs scan, paired by id ({len(both)} ids in both)")
        d_ok_s_bad = [i for i in both if not doc_errs[("digital", i)] and doc_errs[("scan", i)]]
        d_bad_s_ok = [i for i in both if doc_errs[("digital", i)] and not doc_errs[("scan", i)]]
        both_bad = [i for i in both if doc_errs[("digital", i)] and doc_errs[("scan", i)]]
        print(f"digital perfect, scan wrong (OCR-attributable): {len(d_ok_s_bad)}")
        print(f"digital wrong, scan perfect:                    {len(d_bad_s_ok)}")
        print(f"both wrong:                                     {len(both_bad)}")
        ocr = Counter()
        for i in d_ok_s_bad:
            for p, ek, _, _ in doc_errs[("scan", i)]:
                ocr[(generic(p), ek)] += 1
        print("-- wrong items on OCR-attributable docs (top 10)")
        for (f, ek), n in ocr.most_common(10):
            print(f"{n:>4}  {f} {ek}")

    # 5. concrete examples
    print(f"\n=== first {a.examples} wrong items (digital first)")
    for e in sorted(all_errs, key=lambda e: (e["kind"] != "digital", e["id"]))[: a.examples]:
        print(f"{e['kind']:7} {e['id']} {e['path']:34} {e['error']:14} "
              f"gold={short(e['gold'])} pred={short(e['pred'])}")

    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "kind", "status", "field", "path",
                                          "error", "bucket", "gold", "pred"])
        w.writeheader()
        w.writerows(all_errs)
    print(f"\nwrote {len(all_errs)} rows to {a.out}")


if __name__ == "__main__":
    main()