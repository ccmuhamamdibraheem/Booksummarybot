import os
import glob
from pathlib import Path
from utils.chunking import chunk_text
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
from tracing import tracer
from langchain.docstore.document import Document



# ------------------- CONFIG -------------------
MAX_CHARS = 1000   # smaller chunks = more precise retrieval
OVERLAP = 100
BOOKS_FOLDER = "data/books/*.txt"

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("⚠️ GOOGLE_API_KEY not found in .env file")

# ------------------- HELPERS -------------------
def clean_gutenberg_text(text: str):
    """Remove Project Gutenberg headers/footers if present."""
    start_marker = "*** START"
    end_marker = "*** END"

    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)

    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]

    return text.strip()

def extract_metadata(text: str, file_path: str):
    """Extract title and author from the first 2000 chars if possible."""
    snippet = text[:2000]

    title = "Unknown Title"
    author = "Unknown Author"

    for line in snippet.splitlines():
        if line.lower().startswith("title:"):
            title = line.replace("Title:", "").strip()
        if line.lower().startswith("author:"):
            author = line.replace("Author:", "").strip()

    # fallback from filename
    if title == "Unknown Title":
        title = Path(file_path).stem.replace("_", " ").title()

    return title, author

# ------------------- MAIN -------------------
# 1. Load embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 2. Read all book files
with tracer.start_as_current_span("load_books"):
    book_files = glob.glob(BOOKS_FOLDER)

docs = []

with tracer.start_as_current_span("process_books") as proc_span:
    for file_path in book_files:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        text = clean_gutenberg_text(raw_text)
        title, author = extract_metadata(raw_text, file_path)

        chunks = chunk_text(text, chunk_size=MAX_CHARS, chunk_overlap=OVERLAP)
        proc_span.set_attribute(f"{title}_num_chunks", len(chunks))

        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "title": title,
                        "author": author,
                        "page_number": i + 1  # NEW
                    }
                )
            )

with tracer.start_as_current_span("build_faiss_index") as idx_span:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Debug: Check if docs is empty
    if not docs:
        print("[ERROR] No documents to index. Check if book files were loaded and chunked correctly.")
        raise ValueError("No documents to index. Aborting FAISS index build.")

    # Debug: Try to get embeddings for the first doc to check if embedding model works
    try:
        test_emb = embeddings.embed_query(docs[0].page_content)
        if not test_emb or not isinstance(test_emb, list):
            print("[ERROR] Embedding model did not return a valid embedding for the first document.")
            raise ValueError("Embedding model failed to generate embedding.")
    except Exception as e:
        print(f"[ERROR] Exception when generating embedding for first document: {e}")
        raise

    # Build FAISS index
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("storage/books_index")
        idx_span.set_attribute("num_documents", len(docs))
        print(f"[INFO] FAISS index built and saved. Number of documents indexed: {len(docs)}")
    except Exception as e:
        print(f"[ERROR] Exception during FAISS index build: {e}")
        raise



with tracer.start_as_current_span("debug_chunks"):
    books_seen = set()
    for doc in docs:
        title = doc.metadata["title"]
        if title not in books_seen:
            books_seen.add(title)
            chunks_for_book = [d.page_content for d in docs if d.metadata["title"] == title]
            for i, chunk in enumerate(chunks_for_book[:3]):
                print(f"\n--- Chunk {i+1} from {title} ---\n{chunk[:300]}\n")


for file_path in book_files:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text, chunk_size=MAX_CHARS, chunk_overlap=OVERLAP)
    
    # Print first 3 chunks for debugging
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} from {file_path} ---\n{chunk[:300]}\n")
    