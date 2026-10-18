"""Extraction metrics: JSON validity + field-level exact match / F1, with a per-field-type breakdown.

POLICIES (write these in the README, they decide what the numbers mean):

1. Validity = the output contains a JSON object AND it passes GMPRecord validation.
   An output that is not valid counts as validity=0 and contributes ZERO predicted items and
   ALL gold items (recall is hurt, nothing is hidden).
2. The scored object is the VALIDATED record (model_dump), not the raw dict. So a missing
   Optional key (reviewed_by, limit_min, limit_max) becomes null, same as the schema says.
   Extra keys the model invented are dropped by pydantic and are NOT penalised (limitation).
3. Every leaf value is one "item" with a path such as 'steps[3].result'. An item is a hit only if
   the path exists in both gold and prediction AND normalised values are equal.
4. Lists are ALIGNED BY KEY, not by position: steps by `no`, measurements by `parameter`.
   So a skipped step costs that step only; it does not shift every later step into a wrong path.
   Duplicate keys in a prediction get a '#2' suffix and count as extra (false-positive) items.
   Other lists (deviations) are aligned by index.
5. Each list also emits '<path>.len' (type "count"). A model that invents or drops an item is
   penalised there as well as on the item's own fields.
6. Gold nulls are their own breakdown row ("null"): did the model output null where gold is null?
   This row is recall-like. The opposite error (model says null where gold HAS a value) is charged to
   that field's own row (e.g. "number"), not to "null". Read both rows together.
7. Micro-averaged: counts are summed over documents, then P/R/F1 are computed once.
8. SALVAGE F1 (reporting only, added AFTER the layout-D failure categories had been inspected; it
   changes no model and no prompt). When strict validation fails, the raw JSON object is still scored:
   unknown keys dropped (as pydantic does), missing Optionals -> null, NO value repair. So "PASS"
   matches "pass" (normalize lowercases enums), "cleaning_log" is one enum miss, "N/A" in a numeric
   field is a miss. Output with no decodable record-like JSON object (truncated / broken) gets
   no salvage and keeps the zero cliff; it is counted in n_unparsable. For output that passes strict
   validation, salvage F1 == strict F1.
"""
import argparse, csv, json, re, unicodedata
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError
from src.schema.models import Equipment, GMPRecord, Measurement, Step

TYPES = ["date", "enum", "number", "unit", "text", "count", "null"]

# ---------------------------------------------------------------- field types
_BRACKET = re.compile(r"\[[^\]]*\]")          # strips [3] and [pH of final rinse] from a path

FIELD_TYPES = {
    "date": "date",
    "document_type": "enum",
    "steps.result": "enum",
    "steps.no": "number",
    "measurements.value": "number",
    "measurements.limit_min": "number",
    "measurements.limit_max": "number",
    "measurements.unit": "unit",
}

# list path -> field used to align items (everything else is aligned by index)
LIST_KEYS = {"steps": "no", "measurements": "parameter"}


def generic(path: str) -> str:
    return _BRACKET.sub("", path)


def field_type(path: str) -> str:
    g = generic(path)
    if g.endswith(".len"):
        return "count"
    return FIELD_TYPES.get(g, "text")


# ---------------------------------------------------------------- parsing model output
_TOP_KEYS = set(GMPRecord.model_fields)


def _find_record_object(raw):
    """(obj, None) or (None, 'kind: message'). Shared by try_parse and salvage_parse.

    Tries each '{' as a start point (raw_decode parses ONE object and ignores trailing text, so code
    fences and prose around the JSON do not matter). Accepts a decoded object only if it has >= 3
    schema top-level keys, so a broken outer object can never be replaced by an inner one such as
    'equipment'. kind is one of: no_json | json_decode | not_record.
    """
    if not isinstance(raw, str) or "{" not in raw:
        return None, "no_json: the output contains no JSON object"
    dec, first_err = json.JSONDecoder(), None
    for m in list(re.finditer(r"\{", raw))[:20]:
        try:
            obj, _ = dec.raw_decode(raw[m.start():])
        except json.JSONDecodeError as e:
            first_err = first_err or e          # keep the FIRST error: it points at the real problem
            continue
        if isinstance(obj, dict) and len(_TOP_KEYS & obj.keys()) >= 3:
            return obj, None
    if first_err is not None:
        return None, f"json_decode: {first_err.msg} (line {first_err.lineno}, col {first_err.colno})"
    return None, "not_record: JSON found but no object looks like a record"


def try_parse(raw: str):
    """Return (record_dict, None) on success, or (None, 'kind: message') on failure.

    kind is one of: no_json | json_decode | not_record | schema.
    The message is written so it can be pasted back to the model in the agent's repair step.
    """
    obj, err = _find_record_object(raw)
    if err:
        return None, err
    try:
        rec = GMPRecord.model_validate(obj)
    except ValidationError as e:
        errs = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:8])
        return None, f"schema: {errs}"
    return rec.model_dump(mode="json"), None      # date -> 'YYYY-MM-DD', missing Optional -> None


def _pick(model, d):
    """Keep only the model's own keys. Missing Optional -> its default (None); missing required -> stays absent."""
    out = {}
    for name, f in model.model_fields.items():
        if name in d:
            out[name] = d[name]
        elif not f.is_required():
            out[name] = f.default
    return out


def _project(obj):
    """Raw dict -> dict with exactly the schema's keys, safe against wrong shapes (never raises)."""
    rec = _pick(GMPRecord, obj)
    if isinstance(rec.get("equipment"), dict):
        rec["equipment"] = _pick(Equipment, rec["equipment"])
    else:
        rec.pop("equipment", None)
    for key, model in (("steps", Step), ("measurements", Measurement)):
        v = rec.get(key)
        if isinstance(v, list):
            rec[key] = [_pick(model, i) for i in v if isinstance(i, dict)]
        else:
            rec.pop(key, None)
    if not isinstance(rec.get("deviations"), list):
        rec.pop("deviations", None)
    return rec


def salvage_parse(raw):
    """Lenient parse for REPORTING only (policy 8). Same object search as try_parse, no schema validation.
    Returns a dict that flatten() can score, or None if no record-like JSON object was decoded."""
    obj, err = _find_record_object(raw)
    return None if err else _project(obj)


# ---------------------------------------------------------------- normalisation
def _text(v) -> str:
    s = unicodedata.normalize("NFKC", str(v))
    return re.sub(r"\s+", " ", s).strip().casefold()


def normalize(value, ftype):
    """Normalisation policy (what counts as 'the same value'):
    text   NFKC, collapse whitespace, strip, casefold          ('K. Shah' == 'k. shah')
    unit   as text, plus micro sign / greek mu -> 'u'           ('µS/cm' == 'uS/cm')
    number float rounded to 6 dp                                 (5 == 5.0 == '5.0')
    date   ISO string (already ISO after validation)
    enum   strip + lowercase
    count  int
    None stays None. It never equals a non-null value.
    """
    if value is None:
        return None
    if ftype == "number":
        try:
            return round(float(value), 6)
        except (TypeError, ValueError):
            return _text(value)                # unparseable -> will simply not match a float
    if ftype == "count":
        return int(value)
    if ftype == "date":
        return str(value).strip()
    if ftype == "enum":
        return str(value).strip().lower()
    if ftype == "unit":
        return _text(value).replace("µ", "u").replace("μ", "u")
    return _text(value)


# ---------------------------------------------------------------- flatten + score
def flatten(obj, prefix=""):
    """Nested record -> {path: leaf}. See policies 4 and 5 for list handling."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        out[f"{prefix}.len"] = len(obj)
        key_field = LIST_KEYS.get(generic(prefix))
        seen = {}
        for i, item in enumerate(obj):
            if key_field and isinstance(item, dict) and item.get(key_field) is not None:
                k = _text(item[key_field])
                seen[k] = seen.get(k, 0) + 1
                label = k if seen[k] == 1 else f"{k}#{seen[k]}"     # duplicate keys -> extra items
            else:
                label = str(i)
            out.update(flatten(item, f"{prefix}[{label}]"))
    else:
        out[prefix] = obj
    return out


def score(pred, gold):
    """Per-type counts {type: {'tp','n_pred','n_gold'}} for ONE document.

    pred is None  -> invalid output: n_pred = 0, tp = 0, n_gold = every gold item.
    path in both  -> n_pred+1, n_gold+1, tp+1 if normalised values equal. Bucket = 'null' if gold is null.
    path gold-only-> n_gold+1 (missed).   path pred-only -> n_pred+1 (invented).
    """
    counts = {t: {"tp": 0, "n_pred": 0, "n_gold": 0} for t in TYPES}
    g = flatten(gold)
    p = flatten(pred) if pred is not None else {}

    for path, gv in g.items():
        bucket = "null" if gv is None else field_type(path)
        counts[bucket]["n_gold"] += 1
        if path in p:
            counts[bucket]["n_pred"] += 1
            ft = field_type(path)
            if normalize(p[path], ft) == normalize(gv, ft):
                counts[bucket]["tp"] += 1
    for path, pv in p.items():
        if path not in g:
            counts[field_type(path)]["n_pred"] += 1
    return counts


def prf(tp, n_pred, n_gold):
    """(precision, recall, f1); None where undefined (no items of that kind)."""
    p = tp / n_pred if n_pred else None
    r = tp / n_gold if n_gold else None
    if p is None and r is None:
        return None, None, None
    f1 = 0.0 if not (p and r) else 2 * p * r / (p + r)
    return p, r, f1


# ---------------------------------------------------------------- evaluate a predictions file
def evaluate(rows, group_keys=("mode", "split", "layout", "kind")):
    """rows: dicts with 'raw' (model output text) and 'target' (gold JSON string), plus the group keys.
    Returns one result dict per group."""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k, "") for k in group_keys)].append(r)

    results = []
    for key, items in sorted(groups.items()):
        tot = {t: {"tp": 0, "n_pred": 0, "n_gold": 0} for t in TYPES}
        tot_s = {t: {"tp": 0, "n_pred": 0, "n_gold": 0} for t in TYPES}
        n_valid = n_exact = n_trunc = n_salvaged = n_unparsable = 0
        fail_kinds = defaultdict(int)
        for r in items:
            gold, gerr = try_parse(r["target"])
            assert gerr is None, f"gold target for {r.get('id')} is invalid: {gerr}"
            pred, perr = try_parse(r["raw"])
            if pred is None:
                fail_kinds[perr.split(":")[0]] += 1
            else:
                n_valid += 1
            c = score(pred, gold)
            for t in TYPES:
                for f in ("tp", "n_pred", "n_gold"):
                    tot[t][f] += c[t][f]
            # salvage: strict-valid output is scored as is; invalid output gets the lenient parse
            salv = pred if pred is not None else salvage_parse(r["raw"])
            cs = score(salv, gold)
            for t in TYPES:
                for f in ("tp", "n_pred", "n_gold"):
                    tot_s[t][f] += cs[t][f]
            n_salvaged += (pred is None and salv is not None)
            n_unparsable += (salv is None)
            s_tp = sum(c[t]["tp"] for t in TYPES)
            if pred is not None and s_tp == sum(c[t]["n_pred"] for t in TYPES) == sum(c[t]["n_gold"] for t in TYPES):
                n_exact += 1
            n_trunc += bool(r.get("truncated"))

        all_tp = sum(tot[t]["tp"] for t in TYPES)
        all_np = sum(tot[t]["n_pred"] for t in TYPES)
        all_ng = sum(tot[t]["n_gold"] for t in TYPES)
        p, rc, f1 = prf(all_tp, all_np, all_ng)
        s_tp = sum(tot_s[t]["tp"] for t in TYPES)
        s_np = sum(tot_s[t]["n_pred"] for t in TYPES)
        s_ng = sum(tot_s[t]["n_gold"] for t in TYPES)
        row = dict(zip(group_keys, key))
        row.update({"n_docs": len(items), "validity": n_valid / len(items),
                    "doc_exact": n_exact / len(items), "truncated": n_trunc,
                    "precision": p, "recall": rc, "f1": f1,
                    "f1_salvage": prf(s_tp, s_np, s_ng)[2],
                    "n_salvaged": n_salvaged, "n_unparsable": n_unparsable,
                    "fail_kinds": dict(fail_kinds)})
        for t in TYPES:
            row[f"f1_{t}"] = prf(**tot[t])[2]
        results.append(row)
    return results


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _fmt(x):
    return "  -  " if x is None else f"{x:.3f}"


def format_table(results, group_keys=("mode", "split", "layout", "kind")):
    head = list(group_keys) + ["n", "valid", "exact", "F1", "F1_salv"] + [f"F1_{t}" for t in TYPES]
    lines = [" | ".join(f"{h:>8}" for h in head)]
    for r in results:
        cells = [str(r[k]) for k in group_keys] + [str(r["n_docs"]), _fmt(r["validity"]),
                 _fmt(r["doc_exact"]), _fmt(r["f1"]), _fmt(r["f1_salvage"])] + [_fmt(r[f"f1_{t}"]) for t in TYPES]
        lines.append(" | ".join(f"{c:>8}" for c in cells))
    return "\n".join(lines)


def write_csv(results, path, group_keys=("mode", "split", "layout", "kind")):
    cols = list(group_keys) + ["n_docs", "validity", "doc_exact", "truncated", "precision", "recall", "f1",
                               "f1_salvage", "n_salvaged", "n_unparsable"] \
        + [f"f1_{t}" for t in TYPES] + ["fail_kinds"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({c: (json.dumps(r[c]) if c == "fail_kinds" else r.get(c)) for c in cols})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score prediction files (jsonl with raw + target).")
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--out", default=None, help="csv path")
    a = ap.parse_args()
    rows = [r for p in a.preds for r in load_jsonl(p)]
    res = evaluate(rows)
    print(format_table(res))
    for r in res:
        if r["fail_kinds"]:
            print(r["mode"], r["split"], r["kind"], "invalid outputs by kind:", r["fail_kinds"],
                  "| salvaged:", r["n_salvaged"], "| unparsable:", r["n_unparsable"])
    if a.out:
        write_csv(res, a.out)
        print("wrote", a.out)