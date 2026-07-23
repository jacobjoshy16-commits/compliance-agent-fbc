"""Draft amendment language + the change-control entry. Generation only.

Detection is grounded: every finding carries a quote verified to exist in the
document. Generation has no such anchor, so it is guarded in CODE, not in the
prompt. A 3B model writing policy prose will invent statute citations, and a
fake "Tex. Gov't Code 2054.516" inside a county policy is a bad day.
"""

import re
from collections import Counter
from datetime import date

import config as C

WS = re.compile(r'\s+')

# NIST parameter slots -> County decisions. The model never fills these.
ASSIGNMENT = re.compile(r'\[(Assignment|Selection)[^\]]*:\s*([^\]]+)\]', re.I)

# Anything that smells like a legal citation.
CITATION = re.compile(
    r'(\d+\s*U\.?\s?S\.?\s?C\.?\s*(?:§+\s*)?\d+[\w.\-()]*'
    r'|\d+\s*C\.?\s?F\.?\s?R\.?\s*(?:§+\s*)?[\d.]+[\w.\-()]*'
    r"|Tex(?:as)?\.?\s+[A-Za-z'&.\s]{2,30}?Code\s*(?:§+\s*)?[\d.]+"
    r'|(?:HB|SB)\s*\d+|Public\s+Law\s+[\d\-]+|NIST\s+SP\s*[\d\-]+[a-z\d]*'
    r'|§+\s*[\d.]+)', re.I)

# Numbers the model may have invented. Not stripped - policies contain numbers -
# but surfaced, because the model had no basis for any of them.
QUANTITY = re.compile(r'\b\d+(?:\.\d+)?\s*(?:%|percent|days?|weeks?|months?|years?'
                      r'|hours?|minutes?|times?|attempts?|characters?)\b', re.I)


def key(s):
    """'§521.053' and '521.053' are the same citation."""
    return re.sub(r'[^a-z0-9]', '', (s or '').lower())


def citation_guard(text, family):
    """Strip every citation not in the curated allowlist. Returns (text, removed)."""
    allow = [key(m.group(0)) for m in
             CITATION.finditer(C.AUTHORITY.get(family, C.DEFAULT_AUTHORITY))]
    allow = [a for a in allow if len(a) >= 8]
    removed = []

    def repl(m):
        cite, k = m.group(0).strip(), key(m.group(0))
        if len(k) >= 8 and any(k in a or a in k for a in allow):
            return cite
        removed.append(cite)
        return "[CITATION REMOVED - not in curated authority map]"

    return CITATION.sub(repl, text), removed


def placeholders(control_text):
    out = []
    for m in ASSIGNMENT.finditer(control_text):
        v = WS.sub(' ', m.group(2)).strip(' .;')
        if v and v not in out:
            out.append(v)
    return out


def unverified_values(text):
    seen = []
    for m in QUANTITY.finditer(text):
        v = WS.sub(' ', m.group(0)).strip()
        if v.lower() not in [s.lower() for s in seen]:
            seen.append(v)
    return seen


PROMPT = """You are drafting replacement policy language for {org}.

Requirement not currently addressed:
{cid} {ctitle}
{ctext}

Existing procedure text:
{seg}

Write ONE short paragraph of policy language that would close this gap.

STRICT RULES:
- Do NOT cite any law, statute, code section, regulation, or standard.
- Do NOT invent values. Where one is needed write: [COUNTY MUST DEFINE]
- Match the plain style of the existing procedure.
- Output the paragraph only."""


def draft_language(llm, finding, org, seg_text):
    raw = (llm.invoke(PROMPT.format(
        org=org, cid=finding["control_id"], ctitle=finding["control_title"],
        ctext=finding["requirement"][:700], seg=seg_text[:1500])).content or "").strip()
    if len(raw) < 40:
        return None
    raw = re.sub(r'^(Policy|Draft|Paragraph)\s*:\s*', '', raw.strip('`"\' \n'), flags=re.I)

    clean, removed = citation_guard(raw, finding["family"])
    decisions = placeholders(finding["requirement"])
    if "[COUNTY MUST DEFINE]" in clean and not decisions:
        decisions = ["value(s) marked [COUNTY MUST DEFINE] in the draft"]

    return dict(finding, draft_text=clean, decisions_required=decisions,
                citations_removed=removed, unverified_values=unverified_values(clean),
                status="DRAFT - AI-generated - not reviewed")


def render_markdown(drafts, org, reviewer):
    L = [f"# Proposed Policy Amendments - {org}", "",
         "> **STATUS: DRAFT. AI-GENERATED. NOT REVIEWED.**",
         "> Per the FBC AI Policy (*Prohibition of Sole Reliance*), a qualified human",
         "> must validate this before any use. Per *Transparency and Explainability*,",
         "> its AI origin is disclosed here and must stay disclosed downstream.",
         f"> **Reviewer:** {reviewer}", "",
         "Citations below come from a human-curated map. Model-produced citations",
         "are stripped automatically.", "", "---", ""]
    for i, d in enumerate(drafts, 1):
        L += [f"## {i}. {d['process_id']} - {d['control_id']} {d['control_title']}", "",
              f"**Risk:** {d['risk']}  |  **Source:** {d['source']}", ""]
        L += ([f"**Current text (verbatim):**", "", f"> {d['evidence']}", ""]
              if d["evidence"] else ["**Current text:** none found - appears unaddressed.", ""])
        L += ["**Requirement:**", "", f"> {d['requirement']}", "",
              "**Proposed language (DRAFT):**", "", f"> {d['draft_text']}", ""]
        if d["decisions_required"]:
            L += ["**Decisions the County must make before adoption:**"] + \
                 [f"- {x}" for x in d["decisions_required"]] + [""]
        if d["unverified_values"]:
            L += ["**Values the model produced - it had no basis for these. Confirm or replace:**"] + \
                 [f"- `{x}`" for x in d["unverified_values"]] + [""]
        if d["citations_removed"]:
            L += ["**Citations stripped (not in the curated map):**"] + \
                 [f"- `{x}` - verify manually before reinstating" for x in d["citations_removed"]] + [""]
        L += [f"**Applicable authority (curated, not generated):** {d['authority']}", "", "---", ""]
    return "\n".join(L)


# ===========================================================================
# Change control
# ===========================================================================
# The two county documents define how they get amended, and define it
# DIFFERENTLY. The summary is ASSEMBLED from finding metadata - a revision log
# is a record, it doesn't get to be approximately true.

REV_TABLE = re.compile(r'Change\s+Initiator', re.I)
REV_LIST = re.compile(r'Approved\s+and\s+Adopted\s+by|^\s*Revised:\s', re.I | re.M)
REV_NUM = re.compile(r'\b(\d+)\.(\d+)\b')
REV_DATE = re.compile(r'Revised:\s*([A-Z][a-z]+\s+\d{1,2},\s*\d{4})')


def detect_document(text):
    if re.search(r'Artificial\s+Intelligence\s+Policy', text[:2000], re.I):
        return "ai_policy"
    if re.search(r'ACCEPTABLE\s+USE\s+OF\s+INFORMATION\s+SYSTEMS|604\.\d{2}', text, re.I):
        return "policy_604"
    return "unknown"


def detect_format(text):
    return "table" if REV_TABLE.search(text) else "list" if REV_LIST.search(text) else "unknown"


def current_revision(text):
    """('rev','1.0') for the table style, ('date','June 7, 2022') for the list."""
    if detect_format(text) == "table":
        nums = REV_NUM.findall(text[-1500:])
        return ("rev", f"{nums[-1][0]}.{nums[-1][1]}") if nums else ("rev", "1.0")
    d = REV_DATE.findall(text)
    return ("date", d[-1]) if d else ("date", "unknown")


def bump(rev, level="minor"):
    try:
        maj, mnr = (int(x) for x in rev.split("."))
    except (ValueError, AttributeError):
        return "1.1"
    return f"{maj+1}.0" if level == "major" else f"{maj}.{mnr+1}"


def summarize(drafts):
    if not drafts:
        return "No changes."
    procs = sorted({d["process_id"] for d in drafts})
    ctrls = sorted({d["control_id"] for d in drafts})
    risks = Counter(d["risk"] for d in drafts)
    return (f"Added policy language addressing {len(drafts)} identified gap(s) "
            f"[{', '.join(f'{v} {k.lower()}' for k, v in risks.most_common())}] across "
            f"{len(procs)} section(s): {', '.join(procs)}. Controls: {', '.join(ctrls)}. "
            f"Gaps identified by automated review; language drafted with AI assistance "
            f"and validated by the Change Initiator prior to submission.")


def revision_entry(drafts, target_text, initiator, approver, level="minor"):
    doc, fmt = detect_document(target_text), detect_format(target_text)
    kind, cur = current_revision(target_text)
    summary = summarize(drafts)
    today = date.today()

    if fmt == "table":
        new = bump(cur, level)
        paste = ("| Change Initiator | Management Approver | Summary of Changes | Date "
                 "| Revision Number |\n|---|---|---|---|---|\n"
                 f"| {initiator} | {approver} | {summary} | "
                 f"{today.strftime('%m/%d/%Y')} | {new} |")
    elif fmt == "list":
        new = None
        # "%-d" is not portable (fails on Windows). Build it by hand.
        paste = (f"Revised: {today.strftime('%B')} {today.day}, {today.year}\n\n"
                 f"(Agenda summary - this document's revision list carries dates only:)\n{summary}")
    else:
        new, paste = None, summary

    return {"document": doc, "format": fmt, "current": cur, "new_revision": new,
            "initiator": initiator, "approver": approver, "summary": summary,
            "paste_block": paste, "routing": C.ROUTING[doc],
            "status": "DRAFT - requires Change Initiator sign-off before submission"}


def render_change_control(e):
    fence = "```"
    return "\n".join([
        "## Change Control Entry", "",
        f"**Detected document:** `{e['document']}`  |  **Format:** `{e['format']}`", "",
        f"**Current:** {e['current']}" + (f"  ->  **Proposed:** {e['new_revision']}"
                                          if e["new_revision"] else ""), "",
        "**Ready to paste into the document's revision history:**", "",
        fence, e["paste_block"], fence, "",
        f"**Approval routing:** {e['routing']}", "",
        f"> {e['status']}. The Change Initiator ({e['initiator']}) is accountable "
        f"for the accuracy of this entry - not the tool.",
    ])