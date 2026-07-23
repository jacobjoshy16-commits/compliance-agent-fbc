"""Parse the target document, then judge it. Detection only."""

import json
import re
from io import BytesIO

from pypdf import PdfReader

import config as C

WS = re.compile(r'\s+')


def norm(s):
    return WS.sub(' ', (s or "")).strip().lower()


# ===========================================================================
# 1. Parse the target document
# ===========================================================================
PROC_ID = re.compile(r'\b(PRC-[A-Z]{2,4}-\d{2,3})\b\s*[—–-]?\s*([^\n]{0,80})')
NUM_HDR = re.compile(r'^\s*(\d{1,2}(?:\.\d{1,2})*)\s+([A-Z][^\n]{3,70})\s*$', re.M)


def read_pdf(uploaded_file):
    """From memory. A temp file + PdfReader(path) + os.remove() fails on Windows."""
    return "\n".join((p.extract_text() or "")
                     for p in PdfReader(BytesIO(uploaded_file.getvalue())).pages)


def guess_org(text):
    """From the document, never hardcoded - or a Meridian audit blames Fort Bend."""
    head = text[:4000]
    m = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\s+County)\b', head)
    if m:
        return m.group(1)
    m = re.search(r'\b([A-Z]{2,}(?:\s+[A-Z]{2,}){0,2}\s+COUNTY)\b', head)   # ALL-CAPS banner
    if m:
        return " ".join(w.capitalize() for w in m.group(1).split())
    return "the organization"


def segment(text):
    """Split into auditable units. Each query stays under the embedder's cap."""
    text = re.sub(r'\n{3,}', '\n\n', text)

    def pack(pid, title, body):
        body = re.sub(r'[ \t]+', ' ', body).strip()
        return {"id": pid, "title": title.strip() or pid, "text": body[:4000],
                "query": body[:C.EMBED_CAP]}

    for pat, prefix in ((PROC_ID, ""), (NUM_HDR, "SEC-")):
        hits = list(pat.finditer(text))
        if len(hits) >= 2:
            segs = []
            for i, m in enumerate(hits):
                body = text[m.end(): hits[i + 1].start() if i + 1 < len(hits) else len(text)]
                if len(body) >= 200:
                    segs.append(pack(prefix + m.group(1), m.group(2), body))
            if segs:
                return segs

    segs, buf = [], ""
    for para in re.split(r'\n\s*\n', text):
        if len(buf) + len(para) > 1800 and buf:
            segs.append(pack(f"SEG-{len(segs)+1:03d}", buf[:60], buf))
            buf = ""
        buf += para.strip() + "\n\n"
    if buf.strip():
        segs.append(pack(f"SEG-{len(segs)+1:03d}", buf[:60], buf))
    return [s for s in segs if len(s["text"]) >= 200]


def load_target(uploaded_file=None, raw_text=None):
    """Returns (org, segments, full_text). Reads only - writes nothing."""
    text = read_pdf(uploaded_file) if uploaded_file else (raw_text or "")
    if not text.strip():
        return "the organization", [], ""
    return guess_org(text), segment(text), text


# ===========================================================================
# 2. Judge
# ===========================================================================
# A 3B model can't "find every gap and rank by risk" - that's multi-hop
# reasoning and it answers differently every run. It CAN do 3-way
# classification and copy a sentence. So: LLM classifies, Python scores.

GATE = """Requirement {cid}: {ctitle}
{ctext}

Procedure being reviewed:
{seg}

Does the requirement apply to this procedure? A control about classified
information does not apply to a county IT procedure.

Answer with ONE word: YES, NO, or UNCLEAR."""

JUDGE = """Requirement {cid}: {ctitle}
{ctext}

Procedure being reviewed:
{seg}

1. Find the sentence in the Procedure that addresses this Requirement. Copy it
   EXACTLY from the Procedure. If none, write NONE.
2. Decide: MET, PARTIAL, or NOT_MET.

Reply with ONLY this JSON:
{{"evidence": "...", "verdict": "MET"}}"""


def gate(llm, seg, ctrl):
    out = llm.invoke(GATE.format(cid=ctrl["control_id"], ctitle=ctrl["control_title"],
                                 ctext=ctrl["text"][:700], seg=seg["text"][:1500])
                     ).content.strip().upper()
    return next((w for w in ("YES", "NO", "UNCLEAR") if w in out[:12]), "UNCLEAR")


def judge(llm, seg, ctrl):
    raw = llm.invoke(JUDGE.format(cid=ctrl["control_id"], ctitle=ctrl["control_title"],
                                  ctext=ctrl["text"][:700], seg=seg["text"][:2500])
                     ).content.strip()
    m = re.search(r'\{.*\}', raw, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    v = str(d.get("verdict", "")).upper().replace(" ", "_").replace("-", "_")
    return {"evidence": str(d.get("evidence") or ""), "verdict": v} \
        if v in C.VERDICT_MULT else None


def verify(evidence, seg_text):
    """quoted | absent | hallucinated. If the model says it's there, it must be."""
    e = norm(evidence)
    if not e or e in ("none", "n/a", "null"):
        return "absent"
    if e in norm(seg_text):
        return "quoted"
    w = e.split()
    if len(w) >= 8 and " ".join(w[:8]) in norm(seg_text):
        return "quoted"      # tolerate minor drift
    return "hallucinated"


def score(family, verdict, evidence):
    """impact x verdict x evidence, then banded. Same answer every run."""
    v = (C.FAMILY_WEIGHT.get(family, C.DEFAULT_WEIGHT)
         * C.VERDICT_MULT[verdict] * C.EVIDENCE_MULT.get(evidence, 0.6))
    if v <= 0:
        return 0.0, "COMPLIANT"
    return round(v, 2), next(b for t, b in C.BANDS if v >= t)


def audit_segment(llm, seg, controls):
    """One segment vs its controls. Returns (findings, trace)."""
    findings, trace = [], []
    for c in controls:
        def log(outcome):
            trace.append({"process_id": seg["id"], "control_id": c["control_id"],
                          "outcome": outcome})

        g = gate(llm, seg, c)
        if g == "NO":
            log("not applicable")
            continue

        j = judge(llm, seg, c)
        if not j:
            log("unparseable output - dropped")
            continue

        ev = verify(j["evidence"], seg["text"])
        if ev == "hallucinated":
            log("DROPPED - quoted text not in document")
            continue

        val, band = score(c["family"], j["verdict"], ev)
        if band == "COMPLIANT":
            log("compliant")
            continue

        findings.append({
            "process_id": seg["id"], "process_title": seg["title"],
            "control_id": c["control_id"], "control_title": c["control_title"],
            "family": c["family"], "source": c["doc_name"],
            "requirement": c["text"], "verdict": j["verdict"],
            "evidence": j["evidence"] if ev == "quoted" else "",
            "confidence": "Quoted" if ev == "quoted" else "Absence",
            "applicability": g, "score": val, "risk": band,
            "authority": C.AUTHORITY.get(c["family"], C.DEFAULT_AUTHORITY),
        })
        log(f"{j['verdict']} -> {band}")
    return findings, trace