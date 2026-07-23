"""Offline pipeline test. No Ollama, no Chroma. Run: python3 selftest.py

The fake model is deliberately badly behaved - it fabricates a quote and
invents statute citations. The point is to prove the guards catch it.
"""

import sys

import audit
import config as C
import draft

ok = []


def check(name, cond, detail=""):
    ok.append(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


class Msg:
    def __init__(self, c):
        self.content = c


class FakeLLM:
    calls = 0

    def invoke(self, p):
        FakeLLM.calls += 1
        if "ONE word" in p:
            return Msg("YES")
        if "ONLY this JSON" in p:
            if "AU-11" in p:      # fabricated quote -> must be caught
                return Msg('{"evidence": "Logs are retained for seven years offsite.", "verdict": "NOT_MET"}')
            return Msg('{"evidence": "The Graylog retention policy is set to 14 days", "verdict": "NOT_MET"}')
        return Msg("The County shall retain audit records for 180 days in accordance "
                   "with Tex. Gov't Code 2054.516 and 45 CFR 164.312(b), as required "
                   "under NIST SP 800-53. Reviews occur every 90 days.")


TARGET = """MERIDIAN COUNTY
Information Technology Services - Operational Procedures Manual

PRC-LOG-005 - Security Log Collection and Review
Windows domain controllers forward syslog to the on-premises Graylog instance.
The Graylog retention policy is set to 14 days due to the size of the allocated
datastore. Logs are not archived to secondary storage before rotation. There is
no documented procedure assigning a named reviewer, and log review is performed
when an incident prompts it rather than on a defined schedule.

PRC-BCK-007 - Backup and Restoration
Veeam performs nightly incremental backups of all virtual machines to the
on-premises appliance. Both copies reside on the county production network and
both are reachable using the same service account credentials; there is no
immutable, offline, or logically air-gapped copy. The most recent documented
restoration test was 2022-11 and covered a single file-level restore.

Revision History
Change Initiator Management Approver Summary of Changes Date Revision Number
Emmanuel Effiong Lee Morgan Initial Version 10/31/2025 1.0
"""

CONTROLS = [
    {"control_id": "AU-11", "control_title": "AUDIT RECORD RETENTION", "family": "AU",
     "doc_name": "NIST SP 800-53r5",
     "text": "AU-11 AUDIT RECORD RETENTION\nControl: Retain audit records for "
             "[Assignment: organization-defined time period] to support investigations."},
    {"control_id": "CP-9", "control_title": "SYSTEM BACKUP", "family": "CP",
     "doc_name": "NIST SP 800-53r5",
     "text": "CP-9 SYSTEM BACKUP\nControl: Conduct backups at [Assignment: "
             "organization-defined frequency] and protect backup information."},
]

print("\n=== parse ===")
check("org from document, not hardcoded", audit.guess_org(TARGET) == "Meridian County",
      audit.guess_org(TARGET))
segs = audit.segment(TARGET)
check("segments", len(segs) == 2, [s["id"] for s in segs])
check("query under embed cap", all(len(s["query"]) <= C.EMBED_CAP for s in segs),
      max(len(s["query"]) for s in segs))

print("\n=== judge ===")
llm = FakeLLM()
finds, trace = audit.audit_segment(llm, segs[0], CONTROLS)
check("fabricated quote dropped", sum("DROPPED" in t["outcome"] for t in trace) == 1)
kept = [f for f in finds if f["control_id"] == "CP-9"]
check("real quote survives", bool(kept))
check("evidence actually in source", bool(kept) and kept[0]["evidence"] in segs[0]["text"])
check("CP NOT_MET quoted -> HIGH", bool(kept) and kept[0]["risk"] == "HIGH")
check("deterministic", audit.score("CP", "NOT_MET", "quoted") == audit.score("CP", "NOT_MET", "quoted"))
check("absence scores lower than quoted",
      audit.score("CP", "NOT_MET", "absent")[0] < audit.score("CP", "NOT_MET", "quoted")[0])
check("score and band agree", audit.score("AC", "NOT_MET", "absent") == (3.0, "MEDIUM"))
check("MET -> compliant", audit.score("AC", "MET", "quoted")[1] == "COMPLIANT")

print("\n=== draft guards ===")
d = draft.draft_language(llm, kept[0], "Meridian County", segs[0]["text"])
check("draft produced", d is not None)
check("invented citations stripped", len(d["citations_removed"]) >= 3, d["citations_removed"])
check("no raw citation survives", "2054.516" not in d["draft_text"])
check("invented values flagged", len(d["unverified_values"]) >= 2, d["unverified_values"])
check("NIST assignment -> County decision", len(d["decisions_required"]) >= 1,
      d["decisions_required"])
check("authority curated, not generated", d["authority"] == C.AUTHORITY["CP"])
c, _ = draft.citation_guard("Follow Tex. Bus. & Com. Code 521.053 timelines.", "IR")
check("curated citation survives its own family", "521.053" in c and "REMOVED" not in c)
c2, r2 = draft.citation_guard("Follow Tex. Bus. & Com. Code 521.053 timelines.", "AU")
check("same citation stripped under another family", len(r2) == 1)

print("\n=== change control ===")
check("format detected", draft.detect_format(TARGET) == "table")
check("current revision read from tail", draft.current_revision(TARGET) == ("rev", "1.0"))
check("minor bump", draft.bump("1.0") == "1.1")
check("major bump", draft.bump("1.0", "major") == "2.0")
e = draft.revision_entry([d], TARGET, "Jacob Joshy", "Lee Morgan")
check("entry built", e["new_revision"] == "1.1")
check("summary assembled, not generated", "CP-9" in e["summary"])
check("AI Policy -> ETC + CCB", "ETC" in C.ROUTING[draft.detect_document(
    "Fort Bend County Artificial Intelligence Policy")])
check("604 -> Commissioners Court", "Commissioners Court" in C.ROUTING[draft.detect_document(
    "FORT BEND COUNTY 604.01 ACCEPTABLE USE OF INFORMATION SYSTEMS")])
check("paste block renders", "Jacob Joshy" in e["paste_block"])

print("\n" + "=" * 56)
print(f"{sum(ok)}/{len(ok)} passed   |   fake LLM calls: {FakeLLM.calls}")
print("=" * 56)
sys.exit(0 if sum(ok) == len(ok) else 1)