"""Turn parsed documents into (input, target) pairs for baselines and fine-tuning.

    python -m src.data.build_pairs

Reads : data/synthetic/manifest.csv, data/synthetic/gt/*.json, data/parsed/*.json
Writes: data/pairs/{train,val,test}.jsonl      (gitignored, big)
        data/samples/pairs_sample.jsonl        (8 pairs, commit this one)

One pair per (document, kind). kind = digital | scan. Both are built; the split comes from the
LAYOUT (manifest), never from a random shuffle.

Pair fields
  id, layout, split, kind, method   bookkeeping (layout is NOT shown to the model)
  input_text    parsed page text only
  input_tables  extracted tables serialised as ' | ' rows ('' if none). Kept SEPARATE so the baseline
                can be run with and without it (the text already contains the table rows, so this
                duplicates them; whether the duplicate helps is an ablation, not an assumption)
  target        canonical gold JSON string (validated, schema key order, compact)
"""
import csv, json, statistics
from pathlib import Path

from src.schema.models import GMPRecord

SYN = Path("data/synthetic")
PARSED = Path("data/parsed")
PAIRS = Path("data/pairs")
SAMPLES = Path("data/samples")
KINDS = ("digital", "scan")


def serialise_tables(tables) -> str:
    """[[['a','b'],['c','d']], ...] -> '[table 1]\\na | b\\nc | d'. None cells become ''. Empty rows dropped."""
    blocks = []
    for i, t in enumerate(tables or [], 1):
        rows = [" | ".join("" if c is None else str(c).replace("\n", " ").strip() for c in row) for row in t]
        rows = [r for r in rows if r.strip(" |")]
        if rows:
            blocks.append(f"[table {i}]\n" + "\n".join(rows))
    return "\n".join(blocks)


def canonical_target(record: dict) -> str:
    """Validate the gold record and dump it in schema key order, compact (fewer tokens than indent=2)."""
    clean = GMPRecord.model_validate(record).model_dump(mode="json")
    return json.dumps(clean, ensure_ascii=False, separators=(",", ":"))


def make_pair(row: dict, kind: str):
    p = PARSED / f"{row['id']}.{kind}.json"
    if not p.exists():
        return None
    parsed = json.loads(p.read_text(encoding="utf-8"))
    gt = json.loads((SYN / "gt" / f"{row['id']}.json").read_text(encoding="utf-8"))
    return {"id": row["id"], "layout": row["layout"], "split": row["split"], "kind": kind,
            "method": parsed["method"], "input_text": parsed["text"],
            "input_tables": serialise_tables(parsed["tables"]),
            "target": canonical_target(gt["record"])}


def main():
    PAIRS.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    with open(SYN / "manifest.csv", newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    by_split = {"train": [], "val": [], "test": []}
    missing = 0
    for row in manifest:
        for kind in KINDS:
            pair = make_pair(row, kind)
            if pair is None:
                missing += 1
            else:
                by_split[row["split"]].append(pair)

    for split, pairs in by_split.items():
        with open(PAIRS / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"{split}: {len(pairs)} pairs")
    if missing:
        print(f"[warn] {missing} (doc, kind) combos have no parsed file; run: python -m src.parsing.cache")

    # one small committed sample: first pair of every (layout, kind)
    sample, seen = [], set()
    for pairs in by_split.values():
        for p in pairs:
            if (p["layout"], p["kind"]) not in seen:
                seen.add((p["layout"], p["kind"])); sample.append(p)
    with open(SAMPLES / "pairs_sample.jsonl", "w", encoding="utf-8") as f:
        for p in sample:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"sample: {len(sample)} pairs -> {SAMPLES / 'pairs_sample.jsonl'}")

    # length sanity check (CHARACTERS; tokens are roughly chars/3 to chars/4, measure real tokens in the notebook)
    print("\ninput length in characters (text + tables) and empty-OCR count:")
    for split, pairs in by_split.items():
        for layout in sorted({p["layout"] for p in pairs}):
            for kind in KINDS:
                xs = [p for p in pairs if p["layout"] == layout and p["kind"] == kind]
                if not xs:
                    continue
                lens = sorted(len(p["input_text"]) + len(p["input_tables"]) for p in xs)
                empty = sum(1 for p in xs if len(p["input_text"].strip()) < 30)
                print(f"  {split:5} {layout} {kind:7} n={len(xs):4}  mean={statistics.mean(lens):6.0f}  "
                      f"p95={lens[int(0.95 * (len(lens) - 1))]:5}  max={lens[-1]:5}  empty={empty}")


if __name__ == "__main__":
    main()
