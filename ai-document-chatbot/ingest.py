"""
ingest.py — Document Ingestion Pipeline

This script implements the full ingestion pipeline that prepares a PDF
document for semantic search:

    PDF File
       |
       v
    [1. Load PDF] — Extract raw text from each page using PyPDFLoader
       |
       v
    [2. Split Text] — Break text into overlapping chunks using
                       RecursiveCharacterTextSplitter
       |
       v
    [3. Generate Embeddings] — Convert each chunk into a vector using
                                SentenceTransformer (all-MiniLM-L6-v2)
       |
       v
    [4. Store in Endee] — Insert vectors + text metadata into the
                           Endee vector database

Why chunking?
    LLMs and embedding models have token limits. A full PDF may contain
    thousands of words. By splitting into smaller chunks (500 chars with
    50-char overlap), we ensure:
    - Each chunk fits within the embedding model's context window
    - Search results are granular (specific paragraphs, not entire pages)
    - Overlapping text prevents information loss at chunk boundaries

Why overlapping chunks?
    If a sentence spans two chunks, the overlap ensures the full sentence
    appears in at least one chunk, preserving semantic meaning.
"""

import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from vector_store import EndeeVectorStore

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
PDF_PATH = os.path.join("documents", "sample.pdf")
INDEX_NAME = "documents"
ENDEE_URL = os.environ.get("ENDEE_URL", "http://localhost:8080")

# Embedding model: all-MiniLM-L6-v2
# - Produces 384-dimensional vectors
# - Fast and lightweight (~80MB)
# - Good balance of speed and accuracy for semantic search
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Text splitting parameters
CHUNK_SIZE = 500        # Maximum characters per chunk
CHUNK_OVERLAP = 50      # Characters of overlap between adjacent chunks


def ingest_document(pdf_path: str) -> None:
    """
    Run the full ingestion pipeline: load PDF -> chunk -> embed -> store.

    Args:
        pdf_path: Path to the PDF file to ingest.
    """

    # ------------------------------------------------------------------
    # Step 1: Load the PDF document
    # ------------------------------------------------------------------
    # PyPDFLoader reads each page of the PDF and returns a list of
    # Document objects, each containing the page text and metadata
    # (page number, source file path).
    print(f"[1/4] Loading PDF: {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found: {pdf_path}")
        print("Please place a PDF file at 'documents/sample.pdf' and try again.")
        sys.exit(1)

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"       Loaded {len(pages)} pages.")

    # ------------------------------------------------------------------
    # Step 2: Split text into chunks
    # ------------------------------------------------------------------
    # RecursiveCharacterTextSplitter tries to split on natural boundaries
    # (paragraphs, sentences, words) before falling back to character-level
    # splits. This preserves semantic coherence within each chunk.
    print(f"[2/4] Splitting text into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(pages)
    print(f"       Created {len(chunks)} chunks.")

    if len(chunks) == 0:
        print("ERROR: No text chunks extracted. The PDF may be empty or image-based.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Generate embeddings for each chunk
    # ------------------------------------------------------------------
    # SentenceTransformer converts text into dense vectors (arrays of floats).
    # The model maps semantically similar texts to nearby points in
    # 384-dimensional vector space.
    print(f"[3/4] Generating embeddings using model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    print(f"       Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}.")

    # ------------------------------------------------------------------
    # Step 4: Store embeddings in Endee vector database
    # ------------------------------------------------------------------
    # Each vector is stored with:
    #   - id: A unique identifier (chunk_0, chunk_1, ...)
    #   - vector: The 384-dimensional embedding
    #   - meta: The original text chunk (retrieved during search for context)
    print(f"[4/4] Storing embeddings in Endee ({ENDEE_URL})")

    store = EndeeVectorStore(base_url=ENDEE_URL)

    # Check that Endee server is running
    if not store.health_check():
        print(f"ERROR: Cannot connect to Endee at {ENDEE_URL}")
        print("Make sure the Endee server is running. See README.md for instructions.")
        sys.exit(1)

    # Delete old index if it exists, then create a fresh one
    if store.index_exists(INDEX_NAME):
        print(f"       Deleting existing index '{INDEX_NAME}'...")
        store.delete_index(INDEX_NAME)

    result = store.create_index(INDEX_NAME, dim=EMBEDDING_DIM, space_type="cosine")
    print(f"       Created index '{INDEX_NAME}': {result['message']}")

    # Prepare vectors for batch insertion
    vectors = []
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        vectors.append({
            "id": f"chunk_{i}",
            "vector": embedding.tolist(),  # Convert numpy array to Python list
            "meta": text,                  # Store original text for retrieval
        })

    # Insert in batches of 100 to avoid oversized requests
    batch_size = 100
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start : start + batch_size]
        result = store.insert_vectors(INDEX_NAME, batch)
        print(f"       Inserted batch {start // batch_size + 1}: {result['message']}")

    print()
    print("Ingestion complete!")
    print(f"  - Index: {INDEX_NAME}")
    print(f"  - Chunks stored: {len(vectors)}")
    print(f"  - Vector dimension: {EMBEDDING_DIM}")
    print(f"  - Distance metric: cosine")


if __name__ == "__main__":
    # Allow passing a custom PDF path as a command-line argument
    path = sys.argv[1] if len(sys.argv) > 1 else PDF_PATH
    ingest_document(path)
