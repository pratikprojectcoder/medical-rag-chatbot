from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import os

PDF_FOLDER = "data/medical_pdfs"

all_documents = []

# ============================================
# LOAD PDFs
# ============================================

for file in os.listdir(PDF_FOLDER):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(PDF_FOLDER, file)

        print(f"\nLoading PDF: {file}")

        loader = PyMuPDFLoader(pdf_path)

        documents = loader.load()

        print(f"Pages loaded: {len(documents)}")

        # FILTER SMALL/NOISY PAGES
        for doc in documents:

            text = doc.page_content.strip()

            if len(text) < 100:
                continue

            lower_text = text.lower()

            if (
                "project editor" in lower_text or
                "permissions manager" in lower_text or
                "manufacturing manager" in lower_text or
                "image catalogers" in lower_text
            ):
                continue

            all_documents.append(doc)

print("\nTotal pages loaded:", len(all_documents))

# ============================================
# TEXT SPLITTING
# ============================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(all_documents)

print("\nTotal chunks created:", len(chunks))

# ============================================
# EMBEDDINGS
# ============================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("\nEmbedding model loaded successfully!")

# ============================================
# CREATE FAISS DB
# ============================================

print("\nCreating FAISS vector database...")

vectorstore = FAISS.from_documents(
    chunks,
    embedding_model
)

# SAVE DATABASE
vectorstore.save_local("faiss_index")

print("\nFAISS vector database created successfully!")