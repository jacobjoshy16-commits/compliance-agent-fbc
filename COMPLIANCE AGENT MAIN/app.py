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
st.title("🛡️ FBC Automated Compliance Audit Engine")

DB_DIRECTORY = "./local_db"

# --- Helper: Read Uploaded PDF temporarily ---
def extract_text_from_upload(uploaded_file):
    """Reads the uploaded PDF without permanently saving it to the database."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    pages = loader.load()
    
    # Combine the first few pages of text for the audit
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
    st.header("🏛️ Hardcoded Baseline Vault")
    st.success("🔒 NIST SP 800-53")
    st.success("🔒 FBC IT Security Principles")
    st.success("🔒 FBC AI Policy")
    st.markdown("*(These core policies dictate all compliance audits)*")
    st.markdown("---")
    
    st.header("📂 Audit Scanner")
    st.write("Upload a draft document to audit it against the FBC Vault.")
    user_file = st.file_uploader("Upload Policy Draft (PDF):", type=["pdf"])

# --- Main Logic Execution ---
if os.path.exists(DB_DIRECTORY):
    # Connect database & model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DB_DIRECTORY, embedding_function=embeddings)
    llm = ChatOllama(model="qwen2.5:3b", temperature=0.0)
    
    # The Prompt: Instructs the AI to compare the UPLOADED text vs the VAULT text
    template = """
    You are the Lead Cybersecurity Auditor for FBC. 
    Compare the "Target Document" against the "FBC Vault Standards" provided below.

    FBC Vault Standards (Context): 
    {context}
    
    Target Document to Audit: 
    {question}

    Format your response EXACTLY as a structured list containing ONLY these sections.
    
    - **Control ID:** [Relevant NIST or FBC Control ID]
    - **Vault Requirement:** [What the FBC/NIST baseline requires]
    - **Draft Document Gap:** [What the uploaded document is missing or violating]
    - **Risk Priority:** [State either HIGH, MEDIUM, or LOW]
    - **Operational & Legal Impact:** [Explain the exact vulnerability, potential legal consequences, fines, or operational downtime FBC might face.]
    """
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    compliance_engine = RetrievalQA.from_chain_type(
        llm=llm, retriever=db.as_retriever(search_kwargs={"k": 5}), chain_type_kwargs={"prompt": prompt}
    )
    
    if user_file is not None:
        st.write(f"### Target Acquired: `{user_file.name}`")
        if st.button("Generate Comprehensive Audit", type="primary"):
            with st.spinner("Scanning document against FBC Vault..."):
                
                # 1. Extract text from the uploaded PDF
                target_text = extract_text_from_upload(user_file)
                
                # 2. To keep the model fast on 32GB RAM, we limit the scan size
                target_summary = target_text[:4000] # Audit the first ~4000 characters
                
                # 3. Run the LLM
                response = compliance_engine.invoke({"query": target_summary})
                report_text = response["result"]
                
                # 4. Display Dashboard
                high_count, med_count, low_count = calculate_metrics(report_text)
                
                st.markdown("---")
                st.markdown("### 📊 Executive Audit Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("🔴 High Risk Gaps", high_count)
                col2.metric("🟡 Medium Risk Gaps", med_count)
                col3.metric("🟢 Low Risk Gaps", low_count)
                
                st.markdown("---")
                st.markdown("### 📋 Official Gap Analysis")
                st.info(report_text)
                
                # 5. Output Document Generator
                st.download_button(
                    label="📥 Download Comprehensive Review (.txt)",
                    data=report_text,
                    file_name=f"Audit_Report_{user_file.name}.txt",
                    mime="text/plain"
                )
    else:
        st.info("👈 Please upload a target document in the sidebar to begin the audit.")
else:
    st.error("⚠️ FBC Vault Database not found. Please ensure your `local_db` is initialized.")