"""
build_vault.py — Builds the baseline vector store.

Fixes vs. the original:
  1. Chroma.from_documents() APPENDS. Re-running duplicated every chunk.
     Now uses deterministic SHA-256 IDs -> same content overwrites itself.
  2. NIST was ingested whole (492 pages), including Appendix C summary tables
     -- control NAMES with no requirement language. They embed well and say
     nothing. Now only Chapter 3 (pp. 43-420) is ingested.
  3. RecursiveCharacterTextSplitter(chunk_size=1000) exceeded the embedder's
     256-token (~1000 char) ceiling, so chunk tails were silently truncated
     at embed time. Chunks are now capped at 900 chars.
  4. No metadata -> no way to filter NIST vs county at query time. Every chunk
     now carries source_type / control_id / family / page.

Run:  python build_vault.py [--reset]
"""

import os
import shutil
import sys
from collections import Counter

# Windows consoles are not UTF-8 by default. HuggingFace also forks workers that
# can hang on Windows. Both are set before any import that reads them.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    sys.stdout.reconfigure(encoding="utf-8")   # Python 3.7+
except (AttributeError, ValueError):
    pass

from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_core.documents import Document

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # older installs
    from langchain_community.embeddings import HuggingFaceEmbeddings

import corpus

DB_DIRECTORY = "./local_db"
COLLECTION = "fbc_baseline"
DOCS_DIR = "./baseline_docs"


def load_pages(path, page_range):
    reader = PdfReader(path)
    if page_range:
        lo, hi = page_range
        hi = min(hi, len(reader.pages))
        idx = range(lo - 1, hi)
    else:
        idx = range(len(reader.pages))
    return [(i + 1, reader.pages[i].extract_text() or "") for i in idx]


def main():
    if "--reset" in sys.argv and os.path.exists(DB_DIRECTORY):
        try:
            shutil.rmtree(DB_DIRECTORY)
            print(f"Removed {DB_DIRECTORY} (full rebuild).")
        except PermissionError:
            sys.exit(
                f"Could not delete {DB_DIRECTORY} - a process is holding it open.\n"
                "On Windows this almost always means 'streamlit run app.py' is still\n"
                "running. Stop it (Ctrl+C in that terminal) and run this again."
            )

    if not os.path.isdir(DOCS_DIR):
        sys.exit(f"Error: create a '{DOCS_DIR}' folder and put the baseline PDFs inside it.")

    pdfs = [f for f in sorted(os.listdir(DOCS_DIR)) if f.lower().endswith(".pdf")]
    if not pdfs:
        sys.exit(f"Error: no PDFs found in {DOCS_DIR}.")

    all_chunks = []
    for fname in pdfs:
        cfg = corpus.classify(fname)
        if cfg is None:
            print(f"  SKIP  {fname} - not in the CORPUS manifest (corpus.py). "
                  f"Add an entry to ingest it.")
            continue

        pages = load_pages(os.path.join(DOCS_DIR, fname), cfg["page_range"])
        if cfg["chunker"] == "nist_controls":
            chunks = corpus.chunk_nist_controls(pages)
        else:
            chunks = corpus.chunk_county_sections(pages, cfg["doc_name"])

        span = f"pp.{cfg['page_range'][0]}-{cfg['page_range'][1]}" if cfg["page_range"] else "all pages"
        ids = len({c["meta"]["control_id"] for c in chunks})
        print(f"  OK    {fname:<42} {span:<14} -> {len(chunks):>4} chunks / {ids} sections")
        all_chunks.extend(chunks)

    if not all_chunks:
        sys.exit("Error: nothing ingested. Check the CORPUS manifest in corpus.py.")

    # --- Corpus balance report ------------------------------------------
    # This ratio is the whole reason retrieval must be filtered by
    # source_type. In one flat similarity ranking, the county policies
    # cannot win against a federal catalog an order of magnitude larger.
    balance = Counter(c["meta"]["source_type"] for c in all_chunks)
    fed, cty = balance.get("federal_baseline", 0), balance.get("county_policy", 0)
    print(f"\nCorpus balance: federal={fed}  county={cty}  ratio={fed / max(cty, 1):.1f}:1")
    print("  -> app.py MUST retrieve with a source_type filter, or county policy never surfaces.\n")

    docs = [Document(page_content=c["text"], metadata=c["meta"]) for c in all_chunks]
    ids = [corpus.chunk_id(c["text"], c["meta"]) for c in all_chunks]
    dupes = len(ids) - len(set(ids))
    if dupes:
        print(f"  ({dupes} identical chunks collapsed by hash - this is the de-dup working.)")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma(
        collection_name=COLLECTION,
        persist_directory=DB_DIRECTORY,
        embedding_function=embeddings,
    )

    print("Embedding & upserting...")
    B = 256
    for i in range(0, len(docs), B):
        db.add_documents(documents=docs[i:i + B], ids=ids[i:i + B])
        print(f"  {min(i + B, len(docs))}/{len(docs)}")

    print(f"\nDone. Collection '{COLLECTION}' now holds {db._collection.count()} vectors.")
    print("Re-running this script is now safe - identical chunks overwrite instead of duplicating.")


if __name__ == "__main__":
    main()