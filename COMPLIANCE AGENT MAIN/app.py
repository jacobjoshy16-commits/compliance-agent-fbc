import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Import the processing engine we just created
from document_processor import process_and_store_document

# --- UI Configuration ---
st.set_page_config(page_title="County Compliance Agent", layout="wide")
st.title("🛡️ IT Security Compliance Agent")
st.markdown("Automated gap analysis between NIST standards and County IT Policy.")

# --- Sidebar: The Upload Zone ---
with st.sidebar:
    st.header("📂 Document Upload Zone")
    st.write("Upload NIST standards or County Policies here to update the system knowledge.")
    
    # The Drag-and-Drop Uploader for non-technical users
    uploaded_file = st.file_uploader("Drag and drop a PDF", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Ingesting document into secure local database..."):
                success, message = process_and_store_document(uploaded_file)
                if success:
                    st.success(message)
                else:
                    st.error(message)
                    
    st.markdown("---")
    st.header("System Status")
    st.success("🟢 Engine: Qwen 2.5 (3B) - Local")
    db_exists = os.path.exists("./local_db")
    if db_exists:
        st.success("✅ Secure Database Active")
    else:
        st.warning("⚠️ Database empty. Please upload a policy above.")

# --- Database & Model Connection ---
@st.cache_resource
def load_agent():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./local_db", embedding_function=embeddings)
    
    # Using Qwen 2.5 on your i7 processor
    llm = ChatOllama(model="qwen2.5:3b", temperature=0.0)
    
    # specific output format
    template = """
    You are a cybersecurity compliance agent for a county government. 
    Analyze the NIST framework against the county policy provided in the context below.

    Context: {context}
    Question: {question}

    Format your response EXACTLY as a list containing ONLY these sections. Do not add introductory or concluding text:
    
    - **Control ID:** [The specific NIST Control ID]
    - **What it requires:** [Brief summary of the NIST requirement]
    - **What our policy doesn't cover:** [The exact gap or missing information in the county policy]
    - **Risk Priority:** [Rate as High, Medium, or Low]
    - **So What?:** [Explain the real-world operational danger of this gap]
    """
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# --- Main Interface: Gap Analysis ---
if db_exists:
    agent_chain = load_agent()
    
    st.write("### Run Gap Analysis")
    query = st.text_input(
        "Enter a control topic to analyze:", 
        placeholder="e.g., Does our policy meet NIST standards for data backups?"
    )
    
    if st.button("Generate Compliance Report", type="primary"):
        if query:
            with st.spinner("Analyzing county policy against NIST framework..."):
                response = agent_chain.invoke({"query": query})
                
                st.markdown("### Analysis Results")
                st.info(response["result"])
        else:
            st.warning("Please enter a topic to search.")
else:
    st.info("Welcome! To get started, use the sidebar on the left to upload your NIST standards and County Policy PDFs.")