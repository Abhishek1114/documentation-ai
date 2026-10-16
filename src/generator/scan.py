import io, random
import pymupdf as fitz
import numpy as np
from PIL import Image, ImageFilter


def degrade(img: Image.Image, rng: random.Random, severity: float = 1.0) -> Image.Image:
    """Simulate a bad scan. severity 0 = barely touched, 1 = default, 2 = harsh."""
    # 1) skew: pages are never fed perfectly straight
    angle = rng.uniform(-2.0, 2.0) * severity
    img = img.rotate(angle, expand=True, fillcolor=(255, 255, 255), resample=Image.BICUBIC)
    # 2) blur: out-of-focus / low-quality optics
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 1.2) * severity))
    # 3) sensor noise: add gaussian noise per pixel, clip back to valid range
    arr = np.asarray(img).astype(np.float32)
    noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, 12 * severity, arr.shape)
    img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    # 4) JPEG round-trip: compression artifacts
    quality = max(15, int(90 - 40 * severity * rng.uniform(0.3, 1.0)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def pdf_to_scan(pdf_in: str, pdf_out: str, rng: random.Random, dpi: int = 200, severity: float = 1.0):
    page = fitz.open(pdf_in)[0]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img = degrade(img, rng, severity)
    img.save(pdf_out, "PDF", resolution=dpi)   # image-only PDF: no text layer