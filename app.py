"""UI only. Everything tunable is in config.py."""

import pandas as pd
import streamlit as st
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

import audit
import config as C
import draft

st.set_page_config(page_title="Compliance Audit Agent", layout="wide")
st.title("Automated Compliance Audit Agent")
st.caption(f"Local · {C.MODEL} · NIST SP 800-53r5 + FBC AI Policy + FBC Policy 604")


@st.cache_resource
def get_db():
    return Chroma(collection_name=C.COLLECTION, persist_directory=C.DB_DIR,
                  embedding_function=HuggingFaceEmbeddings(
                      model_name=C.EMBED_MODEL,
                      encode_kwargs={"normalize_embeddings": True}))


@st.cache_resource
def get_llm():
    # Without num_ctx, Ollama uses 2048 and the document falls out of context.
    return ChatOllama(model=C.MODEL, temperature=0.0, num_ctx=C.NUM_CTX)


def retrieve(db, query):
    """Two filtered searches. One flat ranking would bury the county policies."""
    out = []
    for stype, k in (("federal_baseline", C.K_FEDERAL), ("county_policy", C.K_COUNTY)):
        try:
            hits = db.similarity_search(query, k=k, filter={"source_type": stype})
        except Exception:
            hits = []
        out += [dict(h.metadata, text=h.page_content) for h in hits]
    return out


# --- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Baseline Vault")
    try:
        db = get_db()
        n = db._collection.count()
    except Exception as e:
        st.error(f"No vault. Run `python3 build_vault.py --reset`\n\n{e}")
        st.stop()
    if not n:
        st.error("Vault empty. Run `python3 build_vault.py --reset`")
        st.stop()
    st.success(f"{n} control vectors")

    st.markdown("---")
    st.header("Audit Target")
    user_file = st.file_uploader("Upload PDF", type=["pdf"])
    user_text = st.text_area("…or paste text", height=140)
    st.caption(f"Per segment: {C.K_FEDERAL}× NIST + {C.K_COUNTY}× county")

if not (user_file or user_text.strip()):
    st.info("Upload a PDF or paste text to begin.")
    st.stop()

# Streamlit reruns the whole script on every widget click, and st.button() is
# True only on the run right after the click. Results live in session_state so
# the page survives a rerun - otherwise touching any widget wipes it.
if st.button("Run Audit", type="primary"):
    st.session_state.pop("audit", None)
    st.session_state["armed"] = True
if not st.session_state.get("armed"):
    st.stop()

if "audit" not in st.session_state:
    with st.spinner("Parsing…"):
        org, segments, raw = audit.load_target(uploaded_file=user_file, raw_text=user_text)
    if not segments:
        st.error("No text extracted. A scanned PDF needs OCR first.")
        st.stop()

    llm = get_llm()
    findings, trace, rlog = [], [], []
    bar, status = st.progress(0.0), st.empty()
    for i, seg in enumerate(segments):
        status.write(f"Auditing `{seg['id']}` — {seg['title'][:60]}")
        controls = retrieve(db, seg["query"])
        rlog += [{"segment": seg["id"], "control_id": c["control_id"],
                  "source": c["doc_name"], "preview": c["text"][:110].replace("\n", " ")}
                 for c in controls]
        f, t = audit.audit_segment(llm, seg, controls)
        findings += f
        trace += t
        bar.progress((i + 1) / len(segments))
    status.empty()
    bar.empty()

    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    findings.sort(key=lambda x: (order.get(x["risk"], 3), -x["score"]))
    st.session_state["audit"] = dict(findings=findings, trace=trace, retrieval=rlog,
                                     org=org, segments=segments, raw=raw)

A = st.session_state["audit"]
findings, trace, rlog = A["findings"], A["trace"], A["retrieval"]
org, segments, raw = A["org"], A["segments"], A["raw"]

st.write(f"**Organization:** {org}  |  **Segments audited:** {len(segments)}")

# --- metrics: count objects, not substrings --------------------------------
cols = st.columns(5)
for col, label, val in zip(
        cols, ["High", "Medium", "Low", "Pairs checked", "Hallucinations blocked"],
        [sum(f["risk"] == "HIGH" for f in findings),
         sum(f["risk"] == "MEDIUM" for f in findings),
         sum(f["risk"] == "LOW" for f in findings),
         len(trace),
         sum("DROPPED" in t["outcome"] for t in trace)]):
    col.metric(label, val)

# --- findings --------------------------------------------------------------
st.markdown("### Findings")
if not findings:
    st.success("No gaps in the applicable controls retrieved.")
    st.stop()

st.dataframe(pd.DataFrame(findings)[["risk", "score", "process_id", "control_id",
                                     "verdict", "confidence", "source"]],
             use_container_width=True, hide_index=True)

for f in findings:
    with st.expander(f"[{f['risk']}] {f['process_id']} × {f['control_id']} — {f['control_title'][:60]}"):
        st.markdown(f"**Requirement ({f['source']}):** {f['requirement']}")
        st.markdown(f"**Verdict:** {f['verdict']} | **Score:** {f['score']} | "
                    f"**Evidence:** {f['confidence']}")
        if f["evidence"]:
            st.info(f["evidence"])
        else:
            st.warning("No controlling text found - finding is based on absence and "
                       "scored lower. Confirm it isn't addressed elsewhere.")
        st.markdown(f"**Applicable authority:** {f['authority']}")

with st.expander("Retrieval trace — what came out of the vault"):
    st.dataframe(pd.DataFrame(rlog), use_container_width=True, hide_index=True)
with st.expander("Decision trace — every pair, including the clean ones"):
    st.dataframe(pd.DataFrame(trace), use_container_width=True, hide_index=True)

df = pd.DataFrame(findings)
name = user_file.name.rsplit(".", 1)[0] if user_file else "pasted"
c1, c2 = st.columns(2)
c1.download_button("Download findings (.csv)", df.to_csv(index=False),
                   f"audit_{name}.csv", "text/csv")
c2.download_button("Download findings (.json)", df.to_json(orient="records", indent=2),
                   f"audit_{name}.json", "application/json")

# --- drafting (opt-in) -----------------------------------------------------
st.markdown("---")
st.markdown("### Draft Remediation Language")
st.caption("Optional. DRAFT text for review — never final, never auto-sent.")

kind, cur = draft.current_revision(raw)
d1, d2, d3 = st.columns(3)
d1.metric("Detected document", draft.detect_document(raw))
d2.metric("Revision format", draft.detect_format(raw))
d3.metric("Current revision" if kind == "rev" else "Last revised", cur)

top_n = st.slider("Draft language for the top N findings", 1,
                  min(10, len(findings)), min(3, len(findings)))
f1, f2 = st.columns(2)
initiator = f1.text_input("Change Initiator (accountable for accuracy)")
approver = f2.text_input("Approver",
                         placeholder="AI Policy -> Management | 604 -> Commissioners Court")
level = st.radio("Version bump", ["minor", "major"], horizontal=True)

if st.button("Generate Draft Amendments"):
    if not initiator or not approver:
        st.error("Enter a Change Initiator and an Approver. An unsigned revision "
                 "log is not a record.")
        st.stop()

    by_id = {s["id"]: s for s in segments}
    llm = get_llm()
    drafts, bar = [], st.progress(0.0)
    for i, f in enumerate(findings[:top_n]):
        d = draft.draft_language(llm, f, org, by_id.get(f["process_id"], {}).get("text", ""))
        if d:
            drafts.append(d)
        bar.progress((i + 1) / top_n)
    bar.empty()

    if not drafts:
        st.warning("Model produced no usable draft text.")
        st.stop()

    m = st.columns(3)
    m[0].metric("Drafts", len(drafts))
    m[1].metric("Invented citations stripped", sum(len(d["citations_removed"]) for d in drafts))
    m[2].metric("Unverified values flagged", sum(len(d["unverified_values"]) for d in drafts))

    packet = draft.render_markdown(drafts, org, initiator) + "\n\n---\n\n" + \
        draft.render_change_control(
            draft.revision_entry(drafts, raw, initiator, approver, level))
    st.markdown(packet)
    st.download_button("Download amendment packet (.md)", packet,
                       f"amendment_packet_{org.replace(' ', '_')}.md", "text/markdown")