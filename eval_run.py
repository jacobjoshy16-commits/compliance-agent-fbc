"""Eval harness. Measures the MODEL, not the guards (selftest.py does that).

Usage:
  1. Hand-label ~20 pairs in labels.json (schema below). Label BEFORE running
     the model - if you peek at its answers first, the numbers are worthless.
  2. python3 eval_run.py            -> runs whatever config.MODEL is set to
  3. Change MODEL in config.py, run again.
  4. Compare the two eval_<model>.json files it writes.

labels.json schema - one entry per (segment x control) pair:
[
  {
    "pair_id": "PRC-LOG-005 x AU-11",
    "seg_text": "<paste the segment text verbatim>",
    "control": {
      "control_id": "AU-11",
      "control_title": "AUDIT RECORD RETENTION",
      "family": "AU",
      "doc_name": "NIST SP 800-53r5",
      "text": "<paste the requirement text>"
    },
    "expected": "NOT_MET"      // MET | PARTIAL | NOT_MET | NOT_APPLICABLE
  }
]

Include a spread: some MET, some PARTIAL, some NOT_MET, and 3-4
NOT_APPLICABLE pairs (a control that clearly doesn't apply) so the gate
gets tested too. Decide your PARTIAL vs NOT_MET rule BEFORE labeling and
write it down - that boundary is where you and the model will disagree,
and if your own rule is fuzzy the accuracy number means nothing.
"""

import json
import re
import sys
import time
from collections import Counter

from langchain_ollama import ChatOllama

import audit
import config as C


def load_labels(path="labels.json"):
    try:
        with open(path, encoding="utf-8") as f:
            labels = json.load(f)
    except FileNotFoundError:
        sys.exit(f"No {path}. Create it first - schema is in this file's docstring.")
    for i, p in enumerate(labels):
        for k in ("pair_id", "seg_text", "control", "expected"):
            if k not in p:
                sys.exit(f"labels.json entry {i} missing key: {k}")
        if p["expected"] not in ("MET", "PARTIAL", "NOT_MET", "NOT_APPLICABLE"):
            sys.exit(f"entry {i}: bad expected value {p['expected']!r}")
    return labels


def run_pair(llm, pair):
    """Mirrors the real pipeline: gate -> judge -> verify. Returns one result row."""
    seg = {"id": pair["pair_id"], "title": pair["pair_id"], "text": pair["seg_text"]}
    ctrl = pair["control"]
    t0 = time.time()

    g = audit.gate(llm, seg, ctrl)
    if g == "NO":
        return {"pair_id": pair["pair_id"], "expected": pair["expected"],
                "got": "NOT_APPLICABLE", "evidence": "", "verify": "-",
                "parse": "ok", "secs": round(time.time() - t0, 1)}

    j = audit.judge(llm, seg, ctrl)
    if j is None:
        return {"pair_id": pair["pair_id"], "expected": pair["expected"],
                "got": "PARSE_FAIL", "evidence": "", "verify": "-",
                "parse": "FAIL", "secs": round(time.time() - t0, 1)}

    ev = audit.verify(j["evidence"], seg["text"])
    return {"pair_id": pair["pair_id"], "expected": pair["expected"],
            "got": j["verdict"], "evidence": j["evidence"][:80], "verify": ev,
            "parse": "ok", "secs": round(time.time() - t0, 1)}


def main():
    labels = load_labels()
    print(f"model: {C.MODEL}   pairs: {len(labels)}\n")
    llm = ChatOllama(model=C.MODEL, temperature=0.0, num_ctx=C.NUM_CTX)

    rows = []
    for p in labels:
        r = run_pair(llm, p)
        rows.append(r)
        mark = "OK " if r["got"] == r["expected"] else "MISS"
        print(f"  {mark}  {r['pair_id']:<28} expected={r['expected']:<15} "
              f"got={r['got']:<15} verify={r['verify']:<12} {r['secs']}s")

    n = len(rows)
    correct = sum(r["got"] == r["expected"] for r in rows)
    parse_fail = sum(r["parse"] == "FAIL" for r in rows)
    halluc = sum(r["verify"] == "hallucinated" for r in rows)
    confusion = Counter((r["expected"], r["got"]) for r in rows
                        if r["got"] != r["expected"])

    print("\n" + "=" * 60)
    print(f"verdict accuracy      {correct}/{n}  ({100*correct/n:.0f}%)")
    print(f"parse failures        {parse_fail}/{n}")
    print(f"hallucinated quotes   {halluc}/{n}   (caught by verify(), but each "
          f"one is a finding the model tried to fake)")
    print(f"avg secs/pair         {sum(r['secs'] for r in rows)/n:.1f}")
    if confusion:
        print("\nmisses (expected -> got):")
        for (e, g), c in confusion.most_common():
            print(f"  {e} -> {g}: {c}")
    print("=" * 60)

    out = f"eval_{re.sub(r'[^a-z0-9]', '_', C.MODEL.lower())}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": C.MODEL, "n": n, "accuracy": round(correct / n, 3),
                   "parse_failures": parse_fail, "hallucinations": halluc,
                   "rows": rows}, f, indent=2)
    print(f"\nwrote {out} - run again with the other model in config.py, then diff.")


if __name__ == "__main__":
    main()