"""Every knob lives here. Change things in this file, not in the others."""

# --- models ---------------------------------------------------------------
MODEL = "qwen2.5:3b"
NUM_CTX = 8192              # Ollama defaults to 2048; the doc falls out of context
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_CAP = 900             # MiniLM silently truncates past ~256 tokens (~900 chars)

# --- storage --------------------------------------------------------------
DB_DIR = "./local_db"
DOCS_DIR = "./baseline_docs"
COLLECTION = "fbc_baseline"

# --- retrieval ------------------------------------------------------------
# Separate k per source. The index is ~10:1 NIST:county, so in one flat
# ranking the county policies never surface.
K_FEDERAL = 2
K_COUNTY = 1

# --- baseline documents ---------------------------------------------------
# match: any fragment of the filename (spaces/underscores/case ignored)
# pages: (first, last) 1-indexed, or None for all
# kind:  "nist" or "county"
CORPUS = {
    "nist": {
        "match": "800-53",
        "doc_name": "NIST SP 800-53r5",
        "source_type": "federal_baseline",
        # Ch.3 = the control catalog. Appendix C (p.455+) is control NAMES with
        # no requirement text: embeds well, says nothing.
        "pages": (43, 420),
        "kind": "nist",
    },
    "ai_policy": {
        "match": "ai policy",
        "doc_name": "FBC AI Policy v1.0",
        "source_type": "county_policy",
        "pages": None,
        "kind": "county",
    },
    "aup": {
        "match": "secuirty policies",
        "doc_name": "FBC Policy 604 - Acceptable Use",
        "source_type": "county_policy",
        "pages": None,
        "kind": "county",
    },
}

# --- risk model -----------------------------------------------------------
# score = FAMILY_WEIGHT x VERDICT_MULT x EVIDENCE_MULT, then banded.
# Transparent heuristic, not ground truth. Tune freely.
FAMILY_WEIGHT = {
    "AC": 5, "IA": 5, "CP": 5, "SC": 5, "SI": 5,
    "AU": 4, "CM": 4, "IR": 4, "RA": 4, "PT": 4, "COUNTY": 4,
    "CA": 3, "MP": 3, "PE": 3, "SA": 3, "SR": 3,
    "AT": 2, "MA": 2, "PL": 2, "PM": 2, "PS": 2,
}
DEFAULT_WEIGHT = 3
VERDICT_MULT = {"NOT_MET": 1.0, "PARTIAL": 0.6, "MET": 0.0}
EVIDENCE_MULT = {"quoted": 1.0, "absent": 0.6}   # no quote = weaker evidence
BANDS = [(3.5, "HIGH"), (1.8, "MEDIUM"), (0.0, "LOW")]

# --- citation allowlist ---------------------------------------------------
# The ONLY citations allowed to appear in a draft. Anything the model invents
# gets stripped. NIST 800-53 is a control catalog, not law -- a 3B model does
# not know what binds a Texas county, so it does not get to guess.
AUTHORITY = {
    "AC": "FBC Policy 604.02 / 604.08. CJIS Security Policy 5.5 where CJI is in scope.",
    "AT": "Tex. Gov't Code 2054.5191 (annual cybersecurity training). FBC Policy 604.10.",
    "AU": "CJIS Security Policy 5.4 where CJI is in scope.",
    "CA": "FBC AI Policy - Periodic Reviews for Alignment and Adaptability.",
    "CM": "FBC AI Policy - Secure Development Practices (change control).",
    "CP": "Continuity of county operations. Operational and contractual exposure.",
    "IA": "FBC Policy 604.08. CJIS Security Policy 5.6 where CJI is in scope.",
    "IR": "FBC Incident Response Plan (Rev 1.4). Tex. Bus. & Com. Code 521.053.",
    "MA": "Vendor support and maintenance obligations.",
    "MP": "FBC Policy 604.05 (FIPS 197 encryption for PHI/PII).",
    "PE": "County physical security standards. CJIS Security Policy 5.9 where CJI is in scope.",
    "PL": "County IT governance and planning requirements.",
    "PM": "County information security program governance.",
    "PS": "FBC Policy 604.11. Mostly HR-owned - confirm scope before flagging IT.",
    "PT": "FBC AI Policy - Confidential Data. Tex. Bus. & Com. Code Ch. 521.",
    "RA": "FBC AI Policy - risk assessment required in an ETC submission.",
    "SA": "FBC AI Policy - Secure Development / Third-Party Security Evaluation.",
    "SC": "FBC Policy 604.05. CJIS Security Policy 5.10 where CJI is in scope.",
    "SI": "FBC AI Policy - Continuous Model Testing. Vendor patching obligations.",
    "SR": "FBC AI Policy - Vendor Disclosures and Accountability.",
    "COUNTY": "Fort Bend County policy. Internal policy violation.",
}
DEFAULT_AUTHORITY = ("Internal county policy. NIST SP 800-53 is a control catalog, "
                     "not a statutory obligation for a Texas county.")

# --- change control -------------------------------------------------------
# Who a change actually has to go to. Read out of each document.
ROUTING = {
    "ai_policy": ("Emerging Technologies Committee (ETC) -> Management Approver of "
                  "record. Moderate/high-risk changes also require Change Control "
                  "Board review; variances require Commissioners Court ratification."),
    "policy_604": ("Fort Bend County Commissioners Court. Route through the IT "
                   "Director and HR before the Court agenda. NOT a management-level "
                   "change."),
    "unknown": "Change-control process not detected. Identify the approving authority.",
}