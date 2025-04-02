import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

# Setting paths
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="mental_docs")

# Load PDFs
print("Loading PDFs from:", DATA_PATH)
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
chunks = text_splitter.split_documents(raw_documents)

# Prepare for ChromaDB
documents, metadata, ids = [], [], []
for i, chunk in enumerate(chunks):
    documents.append(chunk.page_content)
    metadata.append(chunk.metadata)
    ids.append(f"ID{i}")

# Insert into ChromaDB
collection.upsert(documents=documents, metadatas=metadata, ids=ids)

print(f"✅ Successfully stored {len(documents)} document chunks in ChromaDB!")
