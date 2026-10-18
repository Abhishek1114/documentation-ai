"""Zip only what the Colab notebook needs, so you upload ONE small file.

    python -m src.data.make_colab_bundle           # -> colab_bundle.zip (gitignored)

Contents: schema + metrics + prompts code, and data/pairs/*.jsonl. No PDFs, no OCR code, no models.
"""
import zipfile
from pathlib import Path

CODE = ["src/schema/models.py", "src/eval/extraction_metrics.py", "src/eval/prompts.py"]
INITS = ["src/__init__.py", "src/schema/__init__.py", "src/eval/__init__.py"]   # only if they exist in your repo
DATA = ["data/pairs/train.jsonl", "data/pairs/val.jsonl", "data/pairs/test.jsonl"]


def main(out="colab_bundle.zip"):
    files = [p for p in CODE + DATA if Path(p).exists()] + [p for p in INITS if Path(p).exists()]
    missing = [p for p in CODE + DATA if not Path(p).exists()]
    if missing:
        raise SystemExit(f"missing files, run the earlier steps first: {missing}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, Path(p).as_posix())
    mb = Path(out).stat().st_size / 1e6
    print(f"wrote {out}: {len(files)} files, {mb:.1f} MB")


if __name__ == "__main__":
    main()
