import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_DIRECTORY = "./local_db"

def process_and_store_document(uploaded_file):
    """
    Takes a file uploaded via Streamlit, saves it temporarily, 
    processes the text, and appends it to the Chroma vector database.
    """
    try:
        # 1. Save the uploaded file temporarily so LangChain can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        # 2. Load the PDF
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # 3. Chunk the document (keeps NIST rules intact)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # 4. Initialize Local Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 5. Store in the local Chroma database
        vector_db = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=DB_DIRECTORY
        )

        # 6. Clean up the temporary file
        os.remove(temp_file_path)

        return True, f"Successfully processed {len(chunks)} sections from {uploaded_file.name}."

    except Exception as e:
        return False, f"An error occurred while processing the document: {e}"