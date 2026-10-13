"""Turns a PDF or photo (by URL or bytes) into text plus page images the AI can read.
A file name or title is never used as work: if nothing readable comes out, the caller must not grade."""
from __future__ import annotations
import base64
from dataclasses import dataclass, field
import httpx
import pymupdf
from . import config as C

MAX_BYTES = 25 * 1024 * 1024


@dataclass
class Material:
    text: str = ""
    images: list[str] = field(default_factory=list)   # base64 JPEG
    pages: int = 0

    @property
    def readable(self) -> bool:
        return bool(self.text.strip()) or bool(self.images)


def fetch_bytes(url: str, client: httpx.Client | None = None) -> bytes:
    if not url.startswith("https://"):
        raise ValueError("only https file links are fetched")
    own = client is None
    client = client or httpx.Client(timeout=60, follow_redirects=True)
    try:
        r = client.get(url)
        r.raise_for_status()
        if len(r.content) > MAX_BYTES:
            raise ValueError("file too large")
        return r.content
    finally:
        if own:
            client.close()


def _jpeg(page, max_side: int = 1400) -> str:
    zoom = min(2.0, max_side / max(page.rect.width, page.rect.height))
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB, alpha=False)
    return base64.b64encode(pix.tobytes("jpeg", jpg_quality=80)).decode()


def read_document(data: bytes, name: str = "", max_images: int = C.MAX_PAGES_REFERENCE,
                  always_images: bool = False) -> Material:
    """Text from the PDF's text layer, plus page images when it is a scan (or always_images, for
    the student's handwriting). Unsupported or corrupt files give an empty, unreadable Material."""
    try:
        kind = "pdf" if data[:4] == b"%PDF" else (name.rsplit(".", 1)[-1].lower() if "." in name else "png")
        doc = pymupdf.open(stream=data, filetype=kind)
    except Exception:
        return Material()
    try:
        texts = [p.get_text().strip() for p in doc]
        scan = sum(len(t) for t in texts) < C.MIN_TEXT_PER_PAGE * max(1, len(doc))
        m = Material(text="\n\n".join(f"[page {i + 1}]\n{t}" for i, t in enumerate(texts) if t), pages=len(doc))
        if scan or always_images:
            m.images = [_jpeg(doc[i]) for i in range(min(len(doc), max_images))]
        return m
    except Exception:
        return Material()
    finally:
        doc.close()
