import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

print("🔒 Starting the FBC Vault Builder...")

if not os.path.exists("./baseline_docs"):
    print("❌ Error: Please create a 'baseline_docs' folder and put your PDFs inside it.")
    exit()

loader = PyPDFDirectoryLoader("./baseline_docs")
documents = loader.load()
print(f"📄 Loaded {len(documents)} pages of baseline policies.")

if len(documents) == 0:
    print("❌ Error: No PDFs found in the 'baseline_docs' folder!")
    exit()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"✂️ Split into {len(chunks)} searchable compliance rules.")

print("⚙️ Building the database... (This might take a minute)")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./local_db"
)

print("✅ SUCCESS! The FBC Vault (local_db) is now permanently hardcoded on your machine.")
