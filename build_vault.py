"""builds the baseline vector store. Run: python3 build_vault.py --reset"""

import hashlib
import os
import re
import shutil
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_core.documents import Document

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

import config as C


def fold(s):
    """Filenames aren't stable: 'fbc ai policy.pdf' == 'fbc_ai_policy.pdf'."""
    return re.sub(r'[^a-z0-9]', '', (s or '').lower())


def classify(filename):
    for cfg in C.CORPUS.values():
        if fold(cfg["match"]) in fold(filename):
            return cfg
    return None


def read_pages(path, pages):
    r = PdfReader(path)
    lo, hi = (pages[0] - 1, min(pages[1], len(r.pages))) if pages else (0, len(r.pages))
    return [(i + 1, r.pages[i].extract_text() or "") for i in range(lo, hi)]


def split(text, prefix, budget=None):
    """Cut text into chunks that fit under the embedder's ceiling."""
    budget = budget or (C.EMBED_CAP - len(prefix) - 24)
    text = re.sub(r'\s+', ' ', text).strip()
    parts = [text[i:i + budget] for i in range(0, len(text), budget)] or [text]
    n = len(parts)
    return [f"{prefix}{f' ({i+1}/{n})' if n > 1 else ''}\n{p}" for i, p in enumerate(parts)]


# --- NIST ------------------------------------------------------------------
HDR = re.compile(r'^([A-Z]{2}-\d{1,2}(?:\(\d{1,2}\))?)\s+([A-Z][A-Z0-9 ,\-/&()]{3,})\s*$', re.M)
BODY = re.compile(r'Control:\s*(.*?)(?=\n\s*Discussion:|\n\s*Related Controls:|\Z)', re.S)
# Enhancements have NO "Control:" label -- miss this and you lose IA-2(1) (MFA).
ENH = re.compile(r'^\((\d{1,2})\)\s+(?=[A-Z])', re.M)
SENT = re.compile(r'\b[A-Z][a-z]{2,}')          # where the ALL-CAPS title ends
NOISE = re.compile(r'NIST SP 800-53, REV\. 5.*?ORGANIZATIONS|'
                   r'This publication is available free of charge from:\s*\S+|'
                   r'(CHAPTER (ONE|TWO|THREE)|APPENDIX [A-Z])\s+PAGE \d+', re.S)


def chunk_nist(pages, cfg):
    full = "".join(NOISE.sub(' ', t) + "\n" for _, t in pages)
    out = []

    def add(cid, title, body, page):
        body = re.sub(r'\s+', ' ', body).strip()
        if len(body) < 40 or body[:40].find('Withdrawn') != -1:
            return
        for t in split(body, f"{cid} {title}"):
            out.append({"text": t.replace("\n", "\nControl: ", 1), "meta": {
                "source_type": cfg["source_type"], "doc_name": cfg["doc_name"],
                "control_id": cid, "control_title": title[:120],
                "family": cid.split('-')[0], "page": page}})

    hits = list(HDR.finditer(full))
    for i, m in enumerate(hits):
        cid, title = m.group(1), m.group(2).strip()
        raw = full[m.end(): hits[i + 1].start() if i + 1 < len(hits) else len(full)]
        b = BODY.search(raw)
        if not b:
            continue
        add(cid, title, b.group(1), 0)
        eh = list(ENH.finditer(raw))
        for j, e in enumerate(eh):
            seg = raw[e.end(): eh[j + 1].start() if j + 1 < len(eh) else len(raw)]
            flat = re.sub(r'\s+', ' ', seg.split('Discussion:')[0]).strip()
            s = SENT.search(flat)
            if s:
                add(f"{cid}({e.group(1)})", flat[:s.start()].strip(' |-') or title,
                    flat[s.start():].split('Related Controls:')[0], 0)
    return out


# --- County ----------------------------------------------------------------
# The AI Policy's headings don't survive PDF extraction as detectable headers,
# so its section names are listed. Policy 604 has real "Section 604.xx" anchors.
AI_ANCHORS = [
    "Introduction", "Ownership", "Emerging Technologies Committee",
    "Membership and Structure", "Decision Rights and Escalation",
    "AI Approval and Exception Process", "Review Process", "Recordkeeping and Review",
    "Accuracy and Accountability", "Transparency and Explainability",
    "AI Usage Documentation", "Prohibition of Sole Reliance",
    "Prohibition of Harmful Activities", "AI Restrictions", "Responsibilities",
    "Public Complaint and Dispute Resolution", "AI Incidents",
    "Examples of AI-Related Incidents", "Detection and Reporting",
    "Classification and Severity Levels", "Containment, Eradication, and Recovery",
    "Communication and Coordination", "Post-Incident Activity and Continuous Improvement",
    "Linkage to Existing County Plans", "Intellectual Property", "Confidential Data",
    "Data Containment and Harm Mitigation", "Data Protection", "Public Information Act",
    "Enforcement Procedures", "Needs Assessment", "Approval Process",
    "System Configuration", "Monitoring", "Continuous Improvement",
    "Security and Privacy", "Secure Development Practices",
    "Third-Party Security Evaluation", "Disclosing the Use of AI", "Policy Violations",
    "Authorized Access", "AI Inventory", "Inventory Oversight and Maintenance",
    "Periodic Reviews for Alignment and Adaptability", "Training and Awareness",
    "Responding to Rapid Change", "Regular Document Updates", "Discretionary Authority",
]
SEC604 = re.compile(r'Section\s+604\.\d{2}\s*\n?\s*([^\n]{0,60})')


def chunk_county(pages, cfg):
    full = re.sub(r'FORT BEND COUNTY EMPLOYEE INFORMATION MANUAL\s*604-\s*\d+|Page \d+ of \d+',
                  ' ', "\n".join(t for _, t in pages))
    if "604" in cfg["doc_name"]:
        h = list(SEC604.finditer(full))
        blocks = [(re.sub(r'\s+', ' ', h[i].group(0)).strip(),
                   full[h[i].end(): h[i + 1].start() if i + 1 < len(h) else len(full)])
                  for i in range(len(h))]
    else:
        found = []
        cur = 0
        for a in AI_ANCHORS:
            i = full.find(a, cur)
            if i != -1:
                found.append((i, a))
                cur = i + len(a)
        blocks = [(lbl, full[pos + len(lbl): found[n + 1][0] if n + 1 < len(found) else len(full)])
                  for n, (pos, lbl) in enumerate(found)]

    out = []
    for label, body in blocks or [(cfg["doc_name"], full)]:
        if len(body.strip()) < 40:
            continue
        for t in split(body, f"{cfg['doc_name']} - {label}"):
            out.append({"text": t, "meta": {
                "source_type": cfg["source_type"], "doc_name": cfg["doc_name"],
                "control_id": label[:80], "control_title": label[:120],
                "family": "COUNTY", "page": 0}})
    return out


def main():
    if "--reset" in sys.argv and os.path.exists(C.DB_DIR):
        try:
            shutil.rmtree(C.DB_DIR)
            print(f"Removed {C.DB_DIR}")
        except PermissionError:
            sys.exit(f"Can't delete {C.DB_DIR} - stop 'streamlit run app.py' first.")

    pdfs = sorted(f for f in os.listdir(C.DOCS_DIR) if f.lower().endswith(".pdf")) \
        if os.path.isdir(C.DOCS_DIR) else []
    if not pdfs:
        sys.exit(f"No PDFs in {C.DOCS_DIR}")

    chunks, seen = [], set()
    for f in pdfs:
        cfg = classify(f)
        if not cfg:
            print(f"  SKIP  {f}")
            continue
        seen.add(cfg["match"])
        pages = read_pages(os.path.join(C.DOCS_DIR, f), cfg["pages"])
        c = chunk_nist(pages, cfg) if cfg["kind"] == "nist" else chunk_county(pages, cfg)
        print(f"  OK    {f:<40} -> {len(c):>4} chunks")
        chunks += c

    # A manifest entry that matched no file = a silently half-built vault.
    missing = [k for k, v in C.CORPUS.items() if v["match"] not in seen]
    if missing:
        sys.exit(f"\nABORT: no file matched {missing}. Found: {pdfs}")

    fed = sum(1 for c in chunks if c["meta"]["source_type"] == "federal_baseline")
    print(f"\nfederal={fed}  county={len(chunks) - fed}  ratio={fed / max(len(chunks) - fed, 1):.1f}:1")

    db = Chroma(collection_name=C.COLLECTION, persist_directory=C.DB_DIR,
                embedding_function=HuggingFaceEmbeddings(
                    model_name=C.EMBED_MODEL,
                    encode_kwargs={"normalize_embeddings": True}))

    docs = [Document(page_content=c["text"], metadata=c["meta"]) for c in chunks]
    # Deterministic IDs: same content overwrites instead of duplicating, so
    # re-running this script is safe.
    ids = [hashlib.sha256(c["text"].encode()).hexdigest() for c in chunks]

    print("Embedding...")
    for i in range(0, len(docs), 256):
        db.add_documents(documents=docs[i:i + 256], ids=ids[i:i + 256])
        print(f"  {min(i + 256, len(docs))}/{len(docs)}")
    print(f"\nDone. {db._collection.count()} vectors.")


if __name__ == "__main__":
    main()




