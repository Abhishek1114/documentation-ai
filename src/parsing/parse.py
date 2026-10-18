import numpy as np
import pymupdf as fitz
import pdfplumber, pytesseract
from statistics import mean, median
from PIL import Image, ImageOps

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # if needed

PARSER_VERSION = {"digital": "v1", "scan": "v3-tess-nolines"}


def is_scanned(pdf_path: str, min_chars: int = 30) -> bool:
    return len(fitz.open(pdf_path)[0].get_text().strip()) < min_chars


def parse_digital(pdf_path: str) -> dict:
    with pdfplumber.open(pdf_path) as pdf:
        p = pdf.pages[0]
        text = p.extract_text() or ""
        tables, method = p.extract_tables(), "pdfplumber_lines"        # needs ruling lines
        if not tables:                                                  # fallback: infer from text alignment
            tables = p.extract_tables({"vertical_strategy": "text", "horizontal_strategy": "text"})
            method = "pdfplumber_text"
    return {"text": text, "tables": tables, "method": method}


# ---------------- scanned path ----------------
def _remove_ruling(img, k):
    """Opening with 1xk and kx1 kernels via prefix sums: any dark run >= k px is treated as a ruling
    line and whitened."""
    gray = ImageOps.autocontrast(img.convert("L"))
    arr = np.array(gray)
    dark = arr < 128
    H, W = dark.shape

    # Horizontal runs >= k
    cs_h = np.pad(np.cumsum(dark, axis=1, dtype=np.int32), ((0, 0), (1, 0)))
    eroded_h = (cs_h[:, k:] - cs_h[:, :-k]) == k
    diff_h = np.zeros((H, W + 1), dtype=np.int32)
    diff_h[:, : W - k + 1] += eroded_h
    diff_h[:, k:] -= eroded_h
    h_lines = np.cumsum(diff_h, axis=1)[:, :W] > 0

    # Vertical runs >= k
    cs_v = np.pad(np.cumsum(dark, axis=0, dtype=np.int32), ((1, 0), (0, 0)))
    eroded_v = (cs_v[k:, :] - cs_v[:-k, :]) == k
    diff_v = np.zeros((H + 1, W), dtype=np.int32)
    diff_v[: H - k + 1, :] += eroded_v
    diff_v[k:, :] -= eroded_v
    v_lines = np.cumsum(diff_v, axis=0)[:H, :] > 0

    lines = h_lines | v_lines

    # Dilate by 1 px (8-connected)
    dilated = lines.copy()
    dilated[1:, :] |= lines[:-1, :]
    dilated[:-1, :] |= lines[1:, :]
    dilated[:, 1:] |= lines[:, :-1]
    dilated[:, :-1] |= lines[:, 1:]
    dilated[1:, 1:] |= lines[:-1, :-1]
    dilated[1:, :-1] |= lines[:-1, 1:]
    dilated[:-1, 1:] |= lines[1:, :-1]
    dilated[:-1, :-1] |= lines[1:, 1:]

    cleaned = arr.copy()
    cleaned[dilated] = 255
    return Image.fromarray(cleaned)


def _preprocess(img, mode, dpi=300):
    if mode == "nolines":
        return _remove_ruling(img, k=round(0.2 * dpi))
    img = img.convert("L")
    if mode == "none":
        return img
    img = ImageOps.autocontrast(img)
    if mode == "threshold":
        img = img.point(lambda p: 255 if p > 160 else 0)
    return img


def _words(img, psm):
    d = pytesseract.image_to_data(img, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    return [{"t": t.strip(), "l": d["left"][i], "top": d["top"][i], "w": d["width"][i], "h": d["height"][i],
             "line": (d["block_num"][i], d["par_num"][i], d["line_num"][i])}
            for i, t in enumerate(d["text"]) if t.strip()]


def _group_lines(words):
    """Group words by Tesseract's own line ids; left-to-right inside a line, top-to-bottom across lines."""
    if not words:
        return [], 0
    lines = {}
    for w in words:
        lines.setdefault(w["line"], []).append(w)
    rows = [sorted(ws, key=lambda w: w["l"]) for ws in lines.values()]
    rows.sort(key=lambda r: mean(w["top"] + w["h"] / 2 for w in r))
    return rows, median(w["h"] for w in words)


def _group_rows(words):
    """Cluster words into visual lines by vertical centre."""
    if not words:
        return [], 0
    med_h = median(w["h"] for w in words)
    words = sorted(words, key=lambda w: w["top"] + w["h"] / 2)
    rows, cur = [], [words[0]]
    for w in words[1:]:
        cy = w["top"] + w["h"] / 2
        if abs(cy - mean(x["top"] + x["h"] / 2 for x in cur)) > 0.6 * med_h:
            rows.append(cur); cur = [w]
        else:
            cur.append(w)
    rows.append(cur)
    return [sorted(r, key=lambda w: w["l"]) for r in rows], med_h


def _split_cells(row, gap):
    """Split a line into cells wherever the horizontal gap between words is wide."""
    cells, cur = [], [row[0]]
    for prev, w in zip(row, row[1:]):
        if w["l"] - (prev["l"] + prev["w"]) > gap:
            cells.append(cur); cur = [w]
        else:
            cur.append(w)
    cells.append(cur)
    return [" ".join(x["t"] for x in c) for c in cells]


def _tables_from_rows(rows, gap):
    """A table = consecutive rows with >=3 cells and the same cell count (>=2 rows)."""
    cells = [_split_cells(r, gap) for r in rows]
    tops = [min(w["top"] for w in r) for r in rows]
    pitches = [b - a for a, b in zip(tops, tops[1:])]
    med_pitch = median(pitches) if pitches else 0
    tables, cur, last_top = [], [], None
    for c, top in zip(cells, tops):
        is_tab = len(c) >= 3
        broke = bool(cur) and ((last_top is not None and top - last_top > 1.5 * med_pitch)
                               or len(c) != len(cur[-1]))
        if is_tab and not broke:
            cur.append(c)
        else:
            if len(cur) >= 2:
                tables.append(cur)
            cur = [c] if is_tab else []
        last_top = top
    if len(cur) >= 2:
        tables.append(cur)
    return tables


def parse_scanned(pdf_path: str, dpi=300, psm=6, preprocess="nolines",
                  gap_factor=0.8, with_tables=True, grouping="tess") -> dict:
    page = fitz.open(pdf_path)[0]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img = _preprocess(img, preprocess, dpi=dpi)
    words = _words(img, psm)
    rows, med_h = _group_lines(words) if grouping == "tess" else _group_rows(words)
    text = "\n".join(" ".join(w["t"] for w in r) for r in rows)
    tables = _tables_from_rows(rows, gap_factor * med_h) if (rows and with_tables) else []
    return {"text": text, "tables": tables, "method": f"tesseract_psm{psm}_{preprocess}"}


def parse(pdf_path: str, **ocr_kw) -> dict:
    return parse_scanned(pdf_path, **ocr_kw) if is_scanned(pdf_path) else parse_digital(pdf_path)