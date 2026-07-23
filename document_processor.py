"""
document_processor.py — Turns the UPLOADED document into auditable segments.

WHAT CHANGED AND WHY IT MATTERS
-------------------------------
The old version of this file called Chroma.from_documents(..., persist_directory=
"./local_db") on the uploaded file. That wrote the document being AUDITED into
the same collection as the baselines. Wire that to the upload button and the
target document becomes part of its own baseline: it retrieves itself, matches
itself, and reports perfect compliance. Nothing in this file touches the vector
store any more. It is pure parsing.

The second fix is segmentation. The old app embedded the ENTIRE document as one
retrieval query. all-MiniLM-L6-v2 truncates at 256 tokens (~1000 chars), so a
13-page document became a query consisting of its cover page. Auditing happens
per-segment now, and every segment stays under the embedder's ceiling.
"""

import re
from io import BytesIO

from pypdf import PdfReader

from corpus import EMBED_CHAR_CAP

# Documents that already carry process IDs (PRC-ACC-001 etc.) segment cleanly.
PROC_ID = re.compile(r'\b(PRC-[A-Z]{2,4}-\d{2,3})\b\s*[—–-]?\s*([^\n]{0,80})')

# Fallback: numbered headings such as "4.2 Backup and Restoration".
NUM_HDR = re.compile(r'^\s*(\d{1,2}(?:\.\d{1,2})*)\s+([A-Z][^\n]{3,70})\s*$', re.M)

TARGET_CHAR_CAP = 4000   # cap on a segment handed to the LLM (num_ctx budget)
MIN_SEG = 200            # below this a "segment" is a heading, not content


def read_pdf(uploaded_file):
    """
    Streamlit UploadedFile -> plain text. Never persists anything.

    Reads from memory, not a temp file. The old version wrote a
    NamedTemporaryFile(delete=False), handed the PATH to PdfReader, then called
    os.remove() -- PdfReader still holds the handle, and Windows refuses to
    delete an open file. Silent on Linux, PermissionError on Windows. BytesIO
    sidesteps it and never touches disk.
    """
    return "\n".join(
        (pg.extract_text() or "")
        for pg in PdfReader(BytesIO(uploaded_file.getvalue())).pages
    )


def guess_org(text):
    """
    Pull the organisation name out of the TARGET document.

    The original prompt hardcoded 'FBC', which is why an audit of Meridian
    County produced findings about Fort Bend. The org name is data, not a
    constant.
    """
    head = text[:4000]
    # Title Case: "Meridian County"
    m = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s+County)\b', head)
    if m:
        return m.group(1)
    # ALL CAPS: "MERIDIAN COUNTY", "FORT BEND COUNTY EMPLOYEE INFORMATION MANUAL".
    # Policy documents put the org name in an all-caps banner, which the
    # Title-Case pattern above silently misses -- and a missed org name means the
    # draft packet is addressed to "the organization".
    m = re.search(r'\b([A-Z]{2,}(?:\s+[A-Z]{2,}){0,2}\s+COUNTY)\b', head)
    if m:
        return " ".join(w.capitalize() for w in m.group(1).split())
    return "the organization"


def _pack(label, title, body):
    body = re.sub(r'[ \t]+', ' ', body).strip()
    return {
        "id": label,
        "title": title.strip() or label,
        "text": body[:TARGET_CHAR_CAP],
        # The retrieval query is deliberately SHORT. This is the fix for the
        # silent-truncation bug: MiniLM sees all of this, not a prefix of it.
        "query": body[:EMBED_CHAR_CAP],
    }


def segment(text):
    """
    Split a target document into auditable units.

    Returns: [{id, title, text, query}]
    Strategy, in order of preference:
      1. explicit process IDs (PRC-XXX-###)
      2. numbered headings (4.2 Backup and Restoration)
      3. paragraph packing into ~1800-char blocks
    """
    text = re.sub(r'\n{3,}', '\n\n', text)

    hits = list(PROC_ID.finditer(text))
    if len(hits) >= 2:
        segs = []
        for i, m in enumerate(hits):
            end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
            body = text[m.end():end]
            if len(body) >= MIN_SEG:
                segs.append(_pack(m.group(1), m.group(2), body))
        if segs:
            return segs

    hits = list(NUM_HDR.finditer(text))
    if len(hits) >= 3:
        segs = []
        for i, m in enumerate(hits):
            end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
            body = text[m.end():end]
            if len(body) >= MIN_SEG:
                segs.append(_pack(f"SEC-{m.group(1)}", m.group(2), body))
        if segs:
            return segs

    # Fallback: pack paragraphs.
    segs, buf = [], ""
    for para in re.split(r'\n\s*\n', text):
        para = para.strip()
        if not para:
            continue
        if len(buf) + len(para) > 1800 and buf:
            segs.append(_pack(f"SEG-{len(segs) + 1:03d}", buf[:60], buf))
            buf = ""
        buf += para + "\n\n"
    if buf.strip():
        segs.append(_pack(f"SEG-{len(segs) + 1:03d}", buf[:60], buf))
    return [s for s in segs if len(s["text"]) >= MIN_SEG]


def load_target(uploaded_file=None, raw_text=None):
    """
    Returns (org_name, segments, full_text). Reads only -- writes nothing.

    full_text is the ORIGINAL extracted text, not a join of the segments. The
    revision history lives at the tail of a policy document, outside any
    numbered section, so segmentation drops it. change_control.py needs that
    tail to read the current revision number -- hand it the real thing.
    """
    text = read_pdf(uploaded_file) if uploaded_file is not None else (raw_text or "")
    if not text.strip():
        return "the organization", [], ""
    return guess_org(text), segment(text), text