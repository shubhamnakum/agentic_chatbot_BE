import pdfplumber
from pathlib import Path
from tqdm import tqdm
import torch
from config import CHROMA_PERSIST_DIR, Embedding_Model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document
import shutil
import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Define embedding model
Embedding_Model = Embedding_Model
embedding_fn = HuggingFaceEmbeddings(
    model_name=Embedding_Model,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

# Define base directory and categories
BASE_DIR = "reports_data"
CATEGORY_FOLDERS = [
    "aviation", "energy", "finance", "healthcare", "retail", "tech"
]

# Initialize LangChain Chroma vectorstore
vectordb = Chroma(
    collection_name="AllSectors1",
    embedding_function=embedding_fn,
    persist_directory=CHROMA_PERSIST_DIR,
    collection_metadata={"hnsw:space": "cosine"},
)


# Extract text and tables from file
def extract_data(path):
    content_blocks = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                content_blocks.append(text)
            table = page.extract_tables()
            if table:
                table_text = "\n".join([" | ".join(map(str, row))
                                       for row in table if any(row)])
                content_blocks.append(table_text)
    return "\n\n".join(content_blocks)

# Split into chunks


def split_text_to_docs(text, source_path, category):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
    )
    docs = splitter.split_text(text)
    file_name = Path(source_path).name
    return [
        {
            "content": chunk,
            "metadata": {
                "source": file_name,
                "domain": category
            }
        }
        for chunk in docs
    ]

# Track existing sources


def get_existing_sources():
    try:
        existing = vectordb.get(include=["metadatas"], limit=None)
        metadatas = existing.get("metadatas", [])
        return set(md["source"] for md in metadatas if "source" in md)
    except:
        return set()

# Main Ingestion Pipeline with batching


def ingest_pipeline():
    print("Starting ingest pipeline...")
    existing_sources = get_existing_sources()

    for folder in CATEGORY_FOLDERS:
        category_path = Path(BASE_DIR) / folder
        category_name = folder

        for pdf_path in tqdm(category_path.rglob("*.pdf"), desc=f"Ingesting {category_name}"):
            file_name = pdf_path.name
            if file_name in existing_sources:
                print(f"{file_name} - already embedded.")
                continue

            print(f"Processing: {file_name} ({category_name})")

            # 1. Extract content
            full_text = extract_data(str(pdf_path))

            # 2. Split
            docs = split_text_to_docs(
                full_text, file_name, category=category_name)

            # 3. Wrap into LangChain Document
            langchain_docs = [
                Document(page_content=doc["content"], metadata=doc["metadata"])
                for doc in docs
            ]

            print(f"Embedding {len(langchain_docs)} chunks from {file_name}")

            # 4. Batch & store
            batch_size = 32
            for i in range(0, len(langchain_docs), batch_size):
                batch = langchain_docs[i:i + batch_size]
                vectordb.add_documents(batch)

    print("Ingestion pipeline completed.")

# Optional: Semantic search function


def query_Chroma(query, top_k=4):
    embedded_query = embedding_fn.embed_query(query)
    results = vectordb.similarity_search_by_vector(embedded_query, k=top_k)

    print(f"\nTop {top_k} results for query: \"{query}\"\n")
    for i, doc in enumerate(results):
        print(f"--- Result {i + 1} ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Domain: {doc.metadata.get('domain', 'Unknown')}")
        print(doc.page_content[:500])
        print("\n")


if __name__ == '__main__':
    ingest_pipeline()

    query_Chroma("what is the test code of Hemoglobin A1c?")
    # shutil.rmtree(CHROMA_PERSIST_DIR)
