# AI Document Chatbot using Endee Vector Database

An intelligent document Q&A system that lets users upload PDF documents and ask natural language questions about their content. Built on **Endee**, a high-performance open-source vector database, the system uses semantic search and RAG (Retrieval Augmented Generation) to deliver accurate, context-grounded answers.

## Problem Statement

Extracting specific information from large PDF documents is time-consuming and unreliable with traditional keyword search. When a user asks "When was the organization established?" but the document says "The company was founded in 2010", keyword matching fails entirely because no words overlap.

This project solves that problem using **semantic search** powered by vector embeddings. Instead of matching keywords, it understands the *meaning* of text. The user's question is converted into a numerical vector, and the system finds document passages with the most similar meaning — regardless of the exact words used.

The system goes beyond simple retrieval by incorporating an LLM-powered **generation step**: retrieved passages are fed as context to a language model that produces a clear, natural-language answer instead of dumping raw text chunks on the user.

## System Design

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│                    Streamlit (port 8501)                          │
│                                                                  │
│   ┌───────────────────┐        ┌────────────────────────────┐   │
│   │  Upload PDF        │        │  Chat Interface             │   │
│   │  (drag & drop)     │        │  Ask questions → Get answers│   │
│   └────────┬──────────┘        └──────────┬─────────────────┘   │
└────────────┼───────────────────────────────┼─────────────────────┘
             │ POST /ingest                  │ POST /ask
             ▼                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                       FASTAPI BACKEND (port 8000)                │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │               INGESTION PIPELINE                         │   │
│   │                                                          │   │
│   │   PDF ──► Extract Text ──► Split into ──► Generate ──►  │   │
│   │           (PyPDF /         Chunks        Embeddings     │   │
│   │            PyMuPDF)        (500 chars)   (384-dim)      │   │
│   └──────────────────────────────────────────────┬──────────┘   │
│                                                   │ Store        │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                RAG QUERY PIPELINE                        │   │
│   │                                                          │   │
│   │   Question ──► Embed ──► Vector ──► Build ──► LLM ──►  │   │
│   │                Query     Search    Context   Generate   │   │
│   │                (384-d)   (Top K)   String    Answer     │   │
│   └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                    ENDEE VECTOR DATABASE (port 8080)              │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  HNSW Index: "documents"                                  │  │
│   │  ├── chunk_0: [0.12, -0.34, 0.56, ...] + "text content" │  │
│   │  ├── chunk_1: [0.78, 0.23, -0.11, ...] + "text content" │  │
│   │  ├── chunk_2: [-0.45, 0.67, 0.89, ...] + "text content" │  │
│   │  └── ... (384-dimensional vectors with metadata)          │  │
│   │  Distance metric: cosine similarity                       │  │
│   └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Technical Approach

The system operates in two phases:

**Phase 1 — Document Ingestion:**
1. User uploads a PDF through the Streamlit UI
2. Text is extracted using PyPDFLoader (with PyMuPDF fallback for complex PDFs)
3. Text is split into 500-character chunks with 50-character overlap to preserve context at boundaries
4. Each chunk is converted into a 384-dimensional vector using the `all-MiniLM-L6-v2` embedding model
5. Vectors and their original text (as metadata) are stored in Endee's HNSW index

**Phase 2 — Question Answering (RAG Pipeline):**
1. **Embed** — The user's question is converted into a 384-dimensional vector using the same embedding model
2. **Retrieve** — Endee performs approximate nearest neighbor search using cosine similarity, returning the top 5 most relevant chunks
3. **Augment** — Retrieved chunks are combined into a single context string
4. **Generate** — The context and question are sent to an LLM which produces a natural-language answer grounded in the document content
5. **Respond** — The answer and source chunks are returned to the UI

### LLM Strategy (Answer Generation)

The system tries three LLMs in order, falling back automatically:

| Priority | Model | Type | Quality | Requirement |
|----------|-------|------|---------|-------------|
| 1 | Google Gemini (`gemini-2.0-flash-lite`) | Cloud API | Best | `GEMINI_API_KEY` |
| 2 | Groq Llama 3.1 8B (`llama-3.1-8b-instant`) | Free Cloud API | Great | `GROQ_API_KEY` |
| 3 | Google flan-t5-base | Local (no internet) | Good | None |

## How Endee Is Used

**Endee** is the core component of this system — it serves as the vector database that powers semantic search. Here is exactly how the application interacts with Endee:

### Endee's Role

Endee stores document chunks as high-dimensional vectors and performs fast similarity search using the **HNSW (Hierarchical Navigable Small World)** algorithm. When a user asks a question, Endee finds the most semantically similar document passages in O(log n) time, far faster than brute-force comparison.

### REST API Interactions

The application communicates with Endee through its REST API (`http://localhost:8080`). The `vector_store.py` module wraps these calls:

| Operation | Endee API Endpoint | When Used |
|---|---|---|
| Health check | `GET /api/v1/health` | Verify Endee is running |
| Create index | `POST /api/v1/index/create` | When a new PDF is uploaded |
| Delete index | `DELETE /api/v1/index/{name}/delete` | Before re-ingesting a new document |
| Check index | `GET /api/v1/index/{name}/info` | Check if an index already exists |
| Insert vectors | `POST /api/v1/index/{name}/vector/insert` | Store document chunk embeddings |
| Search | `POST /api/v1/index/{name}/search` | Find similar chunks for a question |

### Index Configuration

- **Index name:** `documents`
- **Dimensions:** 384 (matches `all-MiniLM-L6-v2` output)
- **Distance metric:** Cosine similarity
- **Search results:** Top 5 most similar chunks

### Response Format

Endee returns search results as **MessagePack** (binary format for efficiency). Each result is a list: `[similarity_score, vector_id, metadata_bytes, filter_string, norm, vector]`. The client decodes the metadata bytes to recover the original chunk text.

## Project Structure

```
ai-document-chatbot/
├── app.py              # FastAPI backend — RAG pipeline, LLM integration, /ask and /ingest endpoints
├── frontend.py         # Streamlit UI — PDF upload, chat interface
├── vector_store.py     # Endee REST API client (create, insert, search, delete)
├── ingest.py           # CLI ingestion script (alternative to UI upload)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Multi-stage Docker image (bakes in ML models)
├── docker-compose.yml  # 3-service deployment (Endee + Backend + Frontend)
├── .env.example        # Template for API keys
├── .dockerignore       # Exclude secrets and cache from Docker builds
├── documents/
│   └── sample.pdf      # Sample document for testing
└── README.md
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Vector Database | **Endee** | HNSW-based vector storage and similarity search |
| Backend API | FastAPI + Uvicorn | REST API for ingestion and question answering |
| Frontend | Streamlit | Chat UI with PDF upload |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) | Convert text to 384-dim vectors |
| LLM (Primary) | Groq API (Llama 3.1 8B) | Natural language answer generation |
| LLM (Fallback) | flan-t5-base (local) | Offline answer generation |
| PDF Extraction | PyPDF + PyMuPDF | Text extraction from PDFs |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` | Chunk documents with overlap |
| Serialization | msgpack | Decode Endee's binary search responses |
| Containerization | Docker Compose | Production deployment |

## Setup and Execution

### Option 1: Docker Compose (Recommended)

Deploy all three services with a single command.

**Prerequisites:** Docker Desktop installed and running.

**Step 1 — Build the Endee Docker image** (from the repository root):
```bash
cd /path/to/endee
docker build -f infra/Dockerfile --build-arg BUILD_ARCH=neon -t endee-oss:latest .
# Use BUILD_ARCH=avx2 on x86 Linux, or BUILD_ARCH=neon on Apple Silicon
```

**Step 2 — Configure API keys** (optional but recommended):
```bash
cd ai-document-chatbot
cp .env.example .env
```
Edit `.env` and add your Groq API key (free at https://console.groq.com/keys):
```
GROQ_API_KEY=your_key_here
```

**Step 3 — Launch everything:**
```bash
docker compose up --build -d
```

**Step 4 — Verify:**
```bash
docker compose ps   # All 3 services should show "healthy"
```

**Step 5 — Open the chatbot:**
```
http://localhost:8501
```

Upload a PDF and start asking questions.

**Stop the services:**
```bash
docker compose down
```

### Option 2: Local Development

Run each component directly on your machine.

**Prerequisites:**
- Python 3.10+
- Endee server (built from source or running via Docker)

**Step 1 — Start the Endee server:**
```bash
# Build from source (one-time):
cd /path/to/endee
./install.sh --release --neon   # Use --avx2 on x86
./run.sh

# Verify:
curl http://localhost:8080/api/v1/health
```

**Step 2 — Install Python dependencies:**
```bash
cd ai-document-chatbot
pip install -r requirements.txt
```

**Step 3 — Set API keys** (optional):
```bash
export GROQ_API_KEY="your_key_here"
```

**Step 4 — Start the backend:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Step 5 — Start the frontend** (in a new terminal):
```bash
streamlit run frontend.py --server.port 8501 --server.headless true
```

**Step 6 — Open the chatbot:**
```
http://localhost:8501
```

### Option 3: CLI-only (No Frontend)

Use the API directly without the Streamlit UI.

```bash
# Ingest a PDF
python3 ingest.py documents/sample.pdf

# Or ingest via the API
curl -X POST http://localhost:8000/ingest -F "file=@documents/sample.pdf"

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

## API Reference

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "running",
  "message": "AI Document Chatbot API is running. Use POST /ask to ask questions."
}
```

### `POST /ingest`
Upload and ingest a PDF document into Endee.

**Request:** `multipart/form-data` with a `file` field containing the PDF.

**Response:**
```json
{
  "status": "success",
  "filename": "report.pdf",
  "pages": 12,
  "chunks": 47,
  "message": "Ingested 47 chunks from 'report.pdf' into Endee."
}
```

### `POST /ask`
Ask a question about the ingested document. This triggers the full RAG pipeline.

**Request:**
```json
{
  "question": "What algorithm does Endee use?"
}
```

**Response:**
```json
{
  "question": "What algorithm does Endee use?",
  "answer": "Endee uses the HNSW (Hierarchical Navigable Small World) algorithm for approximate nearest neighbor search on dense vectors.",
  "context": "[Chunk 1 | Similarity: 0.4551]\n...",
  "sources": [
    {
      "chunk_id": "chunk_1",
      "similarity": 0.4551,
      "text": "Endee is a high-performance open-source vector database..."
    }
  ]
}
```

### Interactive Docs
FastAPI auto-generates Swagger UI at:
```
http://localhost:8000/docs
```

## Key Concepts

### Embeddings
Embeddings are numerical representations of text in a high-dimensional vector space. The `all-MiniLM-L6-v2` model converts any text into a 384-dimensional array of floating-point numbers. The key property: **semantically similar texts produce similar vectors**, enabling meaning-based search instead of keyword matching.

### Vector Search (Cosine Similarity)
Cosine similarity measures the angle between two vectors, producing a score from 0 to 1 (1 = identical meaning, 0 = unrelated). Endee uses HNSW graphs for efficient approximate nearest neighbor search in O(log n) time, rather than brute-force O(n) comparison.

### RAG (Retrieval Augmented Generation)
RAG grounds LLM responses in real document data by combining retrieval with generation:
1. **Retrieve** relevant passages via vector search
2. **Augment** the LLM prompt with retrieved context
3. **Generate** a natural-language answer based on the context

This ensures answers are factual, specific, and traceable to source documents.
