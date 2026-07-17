"""
corpus.py — Corpus definition and structure-aware chunkers.

Shared by build_vault.py and app.py so that chunking rules live in ONE place.

Why this file exists:
  A 492-page federal catalog and a 7-page county policy cannot be chunked the
  same way, and they cannot compete in the same similarity ranking. This file
  encodes what each source document actually IS.
"""

import hashlib
import re

# --- Embedding model hard limit -------------------------------------------
# all-MiniLM-L6-v2 has max_seq_length = 256 tokens (~1000 characters).
# Anything longer is SILENTLY TRUNCATED at embed time. Every chunk and every
# query must stay under this or you are embedding a prefix and nothing else.
EMBED_CHAR_CAP = 900


# --- Corpus manifest -------------------------------------------------------
# Add a new baseline doc by adding an entry here. Keys are matched against the
# PDF filename (case-insensitive substring).
CORPUS = {
    "nist": {
        "match": "800-53",
        "source_type": "federal_baseline",
        "doc_name": "NIST SP 800-53r5",
        # Chapter 3 (the control catalog) ONLY. Verified against the r5 PDF:
        #   pp.   1- 42  front matter, TOC, Ch.1-2 narrative   -> excluded
        #   pp.  43-420  CHAPTER THREE: the controls           -> INCLUDED
        #   pp. 421-454  Appendix A/B: glossary, acronyms      -> excluded
        #   pp. 455-492  Appendix C: control SUMMARY TABLES    -> excluded
        # Appendix C is control *names* with no requirement language. It embeds
        # beautifully (dense with control terminology) and says nothing. It is
        # what poisoned the original runs.
        "page_range": (43, 420),
        "chunker": "nist_controls",
    },
    "ai_policy": {
        "match": "ai_policy",
        "source_type": "county_policy",
        "doc_name": "FBC AI Policy v1.0",
        "page_range": None,
        "chunker": "county_sections",
    },
    "aup": {
        "match": "secuirty_policies",  # matches the repo's existing filename
        "source_type": "county_policy",
        "doc_name": "FBC Policy 604 - Acceptable Use",
        "page_range": None,
        "chunker": "county_sections",
    },
}


def classify(filename: str):
    """Map a PDF filename to its CORPUS entry. Returns None if unknown."""
    low = filename.lower()
    for cfg in CORPUS.values():
        if cfg["match"].lower() in low:
            return cfg
    return None


# --- NIST 800-53 control parser -------------------------------------------
# Matches a control header line, e.g.:
#   AC-2 ACCOUNT MANAGEMENT
#   AU-4(1) TRANSFER TO ALTERNATE STORAGE
CONTROL_HDR = re.compile(
    r'^([A-Z]{2}-\d{1,2}(?:\(\d{1,2}\))?)\s+([A-Z][A-Z0-9 ,\-/&()]{3,})\s*$',
    re.M,
)

# Everything from "Control:" up to Discussion/Related/next header is the
# NORMATIVE requirement. Discussion is explanatory prose -- it is the reason
# chunks "embed well and say nothing", so it is dropped from page_content.
CONTROL_BODY = re.compile(r'Control:\s*(.*?)(?=\n\s*Discussion:|\n\s*Related Controls:|\Z)', re.S)

# Control ENHANCEMENTS use a DIFFERENT format than base controls -- they have
# no "Control:" label at all:
#     (1) IDENTIFICATION AND AUTHENTICATION (ORGANIZATIONAL USERS) | MULTI-FACTOR
#     AUTHENTICATION TO PRIVILEGED ACCOUNTS
#     Implement multi-factor authentication for access to privileged accounts.
#     Discussion: ...
# Miss this and you lose IA-2(1) (MFA), AC-2(3), CP-9(8) -- i.e. most of the
# controls that actually matter for an audit.
ENH_START = re.compile(r'^\((\d{1,2})\)\s+(?=[A-Z])', re.M)
# The ALL-CAPS title ends where the first real sentence begins: a capitalised
# word followed by lowercase ("Implement", "Employ", "Require").
ENH_SPLIT = re.compile(r'\b[A-Z][a-z]{2,}')

# Page furniture that pollutes every chunk if left in.
NOISE = [
    re.compile(r'NIST SP 800-53, REV\. 5.*?ORGANIZATIONS', re.S),
    re.compile(r'This publication is available free of charge from:\s*\S+'),
    re.compile(r'CHAPTER (ONE|TWO|THREE)\s+PAGE \d+'),
    re.compile(r'APPENDIX [A-Z]\s+PAGE \d+'),
]


def _scrub(text: str) -> str:
    for pat in NOISE:
        text = pat.sub(' ', text)
    return re.sub(r'[ \t]+', ' ', text).strip()


def chunk_nist_controls(pages):
    """
    pages: list of (page_number:int, text:str)

    Returns list of dicts: {text, meta}
    One chunk per control (split further only if the control is enormous).
    Each chunk carries control_id/family metadata so the judge can cite it.
    """
    # Stitch the filtered pages into one stream, tracking where each page starts
    # so we can attribute a page number to each control.
    stream, offsets = [], []
    cursor = 0
    for pno, ptext in pages:
        clean = _scrub(ptext) + "\n"
        stream.append(clean)
        offsets.append((cursor, pno))
        cursor += len(clean)
    full = "".join(stream)

    def page_of(pos):
        pg = offsets[0][1]
        for start, pno in offsets:
            if start <= pos:
                pg = pno
            else:
                break
        return pg

    hits = list(CONTROL_HDR.finditer(full))
    chunks = []

    def emit(cid, title, body, page):
        body = re.sub(r'\s+', ' ', body).strip()
        if len(body) < 40 or 'Withdrawn' in body[:40]:
            return
        header = f"{cid} {title}"
        # Reserve room for header + " (part n/m)" + "\nControl: " so that the
        # FINAL string stays under the embedder's 256-token ceiling.
        budget = EMBED_CHAR_CAP - len(header) - 24
        parts = [body[j:j + budget] for j in range(0, len(body), budget)] or [body]
        for pi, part in enumerate(parts):
            suffix = f" (part {pi + 1}/{len(parts)})" if len(parts) > 1 else ""
            chunks.append({
                "text": f"{header}{suffix}\nControl: {part}",
                "meta": {
                    "source_type": "federal_baseline",
                    "doc_name": "NIST SP 800-53r5",
                    "control_id": cid,
                    "control_title": title[:120],
                    "family": cid.split('-')[0],
                    "page": page,
                },
            })

    for i, m in enumerate(hits):
        cid, title = m.group(1).strip(), m.group(2).strip()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(full)
        raw = full[m.end():end]
        page = page_of(m.start())

        body_m = CONTROL_BODY.search(raw)
        if not body_m:
            continue  # header with no "Control:" clause -> summary/TOC artifact
        emit(cid, title, body_m.group(1), page)

        # --- enhancements nested under this base control ---
        enh_hits = list(ENH_START.finditer(raw))
        for j, em in enumerate(enh_hits):
            e_end = enh_hits[j + 1].start() if j + 1 < len(enh_hits) else len(raw)
            seg = raw[em.end():e_end]
            seg = seg.split('Discussion:')[0]
            flat = re.sub(r'\s+', ' ', seg).strip()
            sm = ENH_SPLIT.search(flat)
            if not sm:
                continue
            e_title = flat[:sm.start()].strip(' |-')
            e_body = flat[sm.start():].split('Related Controls:')[0]
            emit(f"{cid}({em.group(1)})", e_title or title, e_body, page)

    return chunks


# --- County policy chunker -------------------------------------------------
# Policy 604 has machine-readable anchors ("Section 604.03"). The AI Policy
# does not -- its headings are plain lines that survive PDF extraction
# inconsistently. For a hand-curated 3-document corpus, listing the anchors is
# more reliable than guessing at them with a regex, and it means every finding
# can cite a real section instead of "FBC AI Policy, somewhere".
AI_POLICY_ANCHORS = [
    "Introduction", "Ownership", "Emerging Technologies Committee",
    "Membership and Structure", "Decision Rights and Escalation",
    "AI Approval and Exception Process", "Review Process",
    "Recordkeeping and Review", "Accuracy and Accountability",
    "Transparency and Explainability", "AI Usage Documentation",
    "Prohibition of Sole Reliance", "Prohibition of Harmful Activities",
    "AI Restrictions", "Responsibilities",
    "Public AI Used Within Fort Bend County Website and Applications",
    "Public Complaint and Dispute Resolution", "AI Incidents",
    "Examples of AI-Related Incidents", "Detection and Reporting",
    "Classification and Severity Levels",
    "Containment, Eradication, and Recovery",
    "Communication and Coordination",
    "Post-Incident Activity and Continuous Improvement",
    "Linkage to Existing County Plans", "Intellectual Property",
    "Confidential Data", "Data Containment and Harm Mitigation",
    "Data Protection", "Public Information Act", "Enforcement Procedures",
    "Needs Assessment", "Approval Process", "System Configuration",
    "Monitoring", "Continuous Improvement", "Security and Privacy",
    "Secure Development Practices", "Third-Party Security Evaluation",
    "Disclosing the Use of AI", "Policy Violations", "Authorized Access",
    "AI Inventory", "Inventory Oversight and Maintenance",
    "Periodic Reviews for Alignment and Adaptability", "Training and Awareness",
    "Responding to Rapid Change", "Regular Document Updates",
    "Discretionary Authority", "Acknowledgment of Policy Limitations",
]

SECTION_604 = re.compile(r'Section\s+604\.\d{2}\s*\n?\s*([^\n]{0,60})')


def _split_on_anchors(full, anchors):
    """Return [(label, body)] by locating anchor strings in order."""
    found = []
    cursor = 0
    for a in anchors:
        i = full.find(a, cursor)
        if i != -1:
            found.append((i, a))
            cursor = i + len(a)
    blocks = []
    for n, (pos, label) in enumerate(found):
        end = found[n + 1][0] if n + 1 < len(found) else len(full)
        blocks.append((label, full[pos + len(label):end]))
    return blocks


def chunk_county_sections(pages, doc_name):
    full = "\n".join(t for _, t in pages)
    full = re.sub(r'FORT BEND COUNTY EMPLOYEE INFORMATION MANUAL\s*604-\s*\d+', ' ', full)
    full = re.sub(r'Page \d+ of \d+', ' ', full)

    if "604" in doc_name:
        hits = list(SECTION_604.finditer(full))
        blocks = [
            (re.sub(r'\s+', ' ', hits[i].group(0)).strip(),
             full[hits[i].end(): hits[i + 1].start() if i + 1 < len(hits) else len(full)])
            for i in range(len(hits))
        ]
    else:
        blocks = _split_on_anchors(full, AI_POLICY_ANCHORS)

    if not blocks:
        blocks = [(doc_name, full)]

    chunks = []
    for label, body in blocks:
        body = re.sub(r'\s+', ' ', body).strip()
        if len(body) < 40:
            continue
        prefix = f"{doc_name} — {label}"
        budget = EMBED_CHAR_CAP - len(prefix) - 24
        parts = [body[j:j + budget] for j in range(0, len(body), budget)] or [body]
        for pi, part in enumerate(parts):
            suffix = f" (part {pi + 1}/{len(parts)})" if len(parts) > 1 else ""
            chunks.append({
                "text": f"{prefix}{suffix}\n{part}",
                "meta": {
                    "source_type": "county_policy",
                    "doc_name": doc_name,
                    "control_id": label[:80],
                    "control_title": label[:120],
                    "family": "COUNTY",
                    "page": 0,
                },
            })
    return chunks


def chunk_id(text: str, meta: dict) -> str:
    """
    Deterministic ID. Same content -> same ID -> Chroma OVERWRITES instead of
    appending. This is what makes build_vault.py safe to re-run.
    """
    key = f"{meta.get('doc_name','')}|{meta.get('control_id','')}|{text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()