import os
import tempfile
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

# --- Setup App Configuration ---
st.set_page_config(page_title="FBC Compliance Audit", layout="wide")
st.title("FBC Automated Compliance Audit Local Agent (Qwen2.5:3b)")

DB_DIRECTORY = "./local_db"

# --- Helper: Read Uploaded PDF temporarily ---
def extract_text_from_upload(uploaded_file):
    """Reads the uploaded PDF without permanently saving it to the database."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    pages = loader.load()
    
    # Combine text for the audit
    extracted_text = "\n".join([page.page_content for page in pages])
    os.remove(temp_file_path)
    return extracted_text

def calculate_metrics(report_text):
    """Counts risk levels for the dashboard."""
    high = report_text.upper().count("HIGH")
    medium = report_text.upper().count("MEDIUM")
    low = report_text.upper().count("LOW")
    return high, medium, low

# --- Sidebar: The Scanner UI ---
with st.sidebar:
    st.header("Hardcoded Baseline Database")
    st.success("NIST SP 800-53")
    st.success("FBC IT Security Principles")
    st.success("FBC AI Policy")
    st.markdown("*(These core policies dictate all compliance audits)*")
    st.markdown("---")
    
    st.header("📂 Audit Scanner")
    st.write("Provide a draft policy to audit against the FBC Vault using either a file upload or by pasting text directly.")
    
    # Dual Input Options
    user_file = st.file_uploader("Option A: Upload Policy Draft (PDF):", type=["pdf"])
    user_text = st.text_area("Option B: Paste Raw Policy Text Here:", placeholder="Type or paste draft rules...", height=150)

# --- Main Logic Execution ---
if os.path.exists(DB_DIRECTORY):
    # Connect database & model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DB_DIRECTORY, embedding_function=embeddings)
    llm = ChatOllama(model="qwen2.5:3b", temperature=0.0)
    
    # The Prompt: Instructs the AI to compare the UPLOADED text vs the VAULT text
    template = """
    You are the Lead Cybersecurity Auditor for FBC. You are rigorous and exhaustive.
    Compare the ENTIRE "Target Document" against the "FBC Vault Standards" provided below.

    FBC Vault Standards (Context): 
    {context}
    
    Target Document to Audit: 
    {question}

    INSTRUCTIONS:
    1. You MUST identify EVERY SINGLE compliance gap present in the target document.
    2. Do NOT stop after finding just one or two issues. Keep scanning until you reach the end of the text.
    3. Format your response EXACTLY as a structured list containing ONLY these sections for EACH gap found:
    
    - **Control ID:** [Relevant NIST or FBC Control ID]
    - **Vault Requirement:** [What the baseline requires]
    - **Draft Document Gap:** [What the uploaded document is missing or violating]
    - **Risk Priority:** [State either HIGH, MEDIUM, or LOW]
    - **Operational & Legal Impact:** [Explain the exact vulnerability and potential consequences.]
    """
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    # Create the retriever setup
    retriever = db.as_retriever(search_kwargs={"k": 15})
    compliance_engine = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt}
    )
    
    # Check if we have either a file OR pasted text
    has_input = user_file is not None or user_text.strip() != ""
    
    if has_input:
        if user_file is not None:
            st.write(f"### Target Acquired via File: `{user_file.name}`")
        else:
            st.write("### Target Acquired via Direct Text Stream")
            
        if st.button("Generate Comprehensive Audit", type="primary"):
            with st.spinner("Processing local text embeddings..."):
                
                # 1. Determine target source text
                if user_file is not None:
                    target_text = extract_text_from_upload(user_file)
                else:
                    target_text = user_text
                
                # Keep target summary to stable processing length
                target_summary = target_text
                
                # 2. STEP INSIDE THE VAULT: Manually fetch the matching chunks to display them
                retrieved_chunks = retriever.invoke(target_summary)
                
                # 3. EXPOSE THE DATA PROCESS (Visual Local Database Trace)
                st.markdown("### 🔍 Local Database Retrieval Trace (Internal Processing)")
                st.write("Below is a real-time view showing how text data stored in your local hard drive is being extracted and translated without internet access:")
                
                with st.expander("📂 View Local Vector Database Matches Found", expanded=True):
                    for idx, chunk in enumerate(retrieved_chunks):
                        st.markdown(f"**Matched Chunk #{idx + 1}**")
                        st.caption(f"Source Document Metadata: `{chunk.metadata.get('source', 'Baseline Vault File')}`")
                        st.info(chunk.page_content)
                
                # 4. Run the final local inference
                with st.spinner("Analyzing rules and generating official text document..."):
                    response = compliance_engine.invoke({"query": target_summary})
                    report_text = response["result"]
                
                # 5. Display Dashboard Metrics
                high_count, med_count, low_count = calculate_metrics(report_text)
                
                st.markdown("---")
                st.markdown("### 📊 Executive Audit Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("High Risk Gaps", high_count)
                col2.metric("Medium Risk Gaps", med_count)
                col3.metric("Low Risk Gaps", low_count)

                st.markdown("---")
                st.markdown("### Policy Analysis")
                st.success(report_text)
                
                # 6. Output Document Generator
                st.markdown("### Export Documentation")
                doc_name = user_file.name if user_file is not None else "Text_Stream"
                st.download_button(
                    label=" Download Comprehensive Review (.txt)",
                    data=report_text,
                    file_name=f"Audit_Report_{doc_name}.txt",
                    mime="text/plain"
                )
    else:
        st.info("👈 Please upload a PDF or paste text into the sidebar scanner to begin the audit.")
else:
    st.error("⚠️ FBC Vault Database not found. Please ensure your `local_db` is initialized using `build_vault.py`.")