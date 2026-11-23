"""Turns a PDF or photo (by URL or bytes) into text plus page images the AI can read.
A file name or title is never used as work: if nothing readable comes out, the caller must not grade."""
from __future__ import annotations
import base64
from urllib.parse import quote
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
    page_numbers: list[int] = field(default_factory=list)   # 1-based page of each image

    @property
    def readable(self) -> bool:
        return bool(self.text.strip()) or bool(self.images)


STORED = "storage://"   # private uploads (portal P0.3): "storage://uploads/<path>", read with the service key


def is_file_ref(url) -> bool:
    """A web link or a private-bucket reference: something fetch_bytes can read."""
    return isinstance(url, str) and (url.startswith("https://") or url.startswith(STORED))


def storage_request(ref: str) -> tuple[str, dict]:
    """The Storage API address and headers for "storage://<bucket>/<path>". The key is only ever sent to our own
    Supabase project; httpx drops the Authorization header if a redirect leaves that host."""
    bucket_path = ref[len(STORED):]
    if "/" not in bucket_path or ".." in bucket_path.split("/"):
        raise ValueError("bad storage reference")
    base, key = C.supabase_url(), C.supabase_key()
    if not base.startswith("https://") or not key:
        raise ValueError("private file, but SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not set")
    return (f"{base}/storage/v1/object/authenticated/{quote(bucket_path, safe='/')}",
            {"Authorization": f"Bearer {key}", "apikey": key})


def fetch_bytes(url: str, client: httpx.Client | None = None) -> bytes:
    if not is_file_ref(url):
        raise ValueError("only https file links and private uploads are fetched")
    headers = {}
    if url.startswith(STORED):
        url, headers = storage_request(url)
    own = client is None
    client = client or httpx.Client(timeout=60, follow_redirects=True)
    try:
        r = client.get(url, headers=headers)
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


def _grey(page, width: int = 160) -> tuple[int, int, bytes]:
    z = width / max(1.0, page.rect.width)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z), colorspace=pymupdf.csGRAY, alpha=False)
    return pix.width, pix.height, pix.samples


def written_pages(work: bytes, blank: bytes, margin: float = C.WRITTEN_PAGE_MARGIN) -> list[int] | None:
    """0-based pages where the student wrote, found by comparing each page of the work with the same page of the
    blank paper (the assignment's own file, e.g. a practice book the student wrote on). A page counts as written when
    it differs from the blank one clearly more than a typical page does (the median page: untouched pages still differ
    a little after the student's app re-saves the file). None when the two files do not line up (not both PDFs, a
    different page count or page size, under two pages): then the caller reads the pages in order."""
    if work[:4] != b"%PDF" or blank[:4] != b"%PDF":
        return None
    try:
        a, b = pymupdf.open(stream=work, filetype="pdf"), pymupdf.open(stream=blank, filetype="pdf")
    except Exception:
        return None
    try:
        if len(a) < 2 or len(a) != len(b):
            return None
        scores = []
        for pa, pb in zip(a, b):
            wa, ha, x = _grey(pa)
            wb, hb, y = _grey(pb)
            if (wa, ha) != (wb, hb):
                return None
            scores.append(sum(abs(p - q) for p, q in zip(x, y)) / max(1, len(x)))
        typical = sorted(scores)[len(scores) // 2]
        return [i for i, s in enumerate(scores) if s - typical > margin]
    except Exception:
        return None
    finally:
        a.close()
        b.close()


def read_document(data: bytes, name: str = "", max_images: int = C.MAX_PAGES_REFERENCE,
                  always_images: bool = False, pages: list[int] | None = None) -> Material:
    """Text from the PDF's text layer, plus page images when it is a scan (or always_images, for
    the student's handwriting): the first max_images pages, or the given 0-based `pages` (still at most max_images).
    Unsupported or corrupt files give an empty, unreadable Material."""
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
            chosen = [i for i in (pages if pages is not None else range(len(doc))) if 0 <= i < len(doc)][:max_images]
            m.images = [_jpeg(doc[i]) for i in chosen]
            m.page_numbers = [i + 1 for i in chosen]
        return m
    except Exception:
        return Material()
    finally:
        doc.close()
