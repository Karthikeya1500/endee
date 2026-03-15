# AI Document Chatbot using Endee Vector Database

An AI-powered document Q&A system that uses semantic search and RAG (Retrieval Augmented Generation) to answer questions about PDF documents.

## Problem Statement

Users often need to quickly find specific information within large PDF documents. Traditional keyword search fails when the user's question uses different words than the document. This project solves that problem using **semantic search** — understanding the *meaning* of text, not just matching keywords.

For example, if a document says "The company was founded in 2010" and the user asks "When did the organization start?", keyword search would fail (no matching words), but semantic search finds the answer because the *meaning* is similar.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                        │
│                     (ingest.py)                              │
│                                                              │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐  │
│   │ Load PDF │-->│ Split    │-->│ Generate │-->│ Store  │  │
│   │ (PyPDF)  │   │ Chunks   │   │ Embed.   │   │ in     │  │
│   │          │   │ (500ch)  │   │ (384-dim)│   │ Endee  │  │
│   └──────────┘   └──────────┘   └──────────┘   └────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE (RAG)                       │
│                      (app.py)                                │
│                                                              │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐  │
│   │ User     │-->│ Embed    │-->│ Search   │-->│ Return │  │
│   │ Question │   │ Question │   │ Endee DB │   │ Answer │  │
│   │          │   │ (384-dim)│   │ (Top K)  │   │ + Ctx  │  │
│   └──────────┘   └──────────┘   └──────────┘   └────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    ENDEE VECTOR DATABASE                      │
│                  (localhost:8080)                             │
│                                                              │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  HNSW Index: "documents"                             │  │
│   │  ├── chunk_0: [0.12, -0.34, 0.56, ...] + text       │  │
│   │  ├── chunk_1: [0.78, 0.23, -0.11, ...] + text       │  │
│   │  ├── chunk_2: [-0.45, 0.67, 0.89, ...] + text       │  │
│   │  └── ...                                             │  │
│   │  Distance metric: cosine similarity                  │  │
│   │  Dimensions: 384                                     │  │
│   └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Embeddings

Embeddings are numerical representations of text in a high-dimensional vector space. When text is passed through an embedding model (like `all-MiniLM-L6-v2`), it produces a fixed-length array of floating-point numbers (384 dimensions in our case).

The key property: **semantically similar texts produce similar vectors**. This means:
- "The cat sat on the mat" and "A feline rested on the rug" will have vectors pointing in nearly the same direction
- "The cat sat on the mat" and "Stock prices rose today" will have vectors pointing in very different directions

### Vector Search (Cosine Similarity)

Cosine similarity measures the angle between two vectors, producing a score from 0 to 1:
- **1.0** = Vectors point in the same direction (maximum similarity)
- **0.0** = Vectors are perpendicular (no similarity)

Endee uses **HNSW (Hierarchical Navigable Small World)** graphs for efficient approximate nearest neighbor search. Instead of comparing a query against every stored vector (O(n)), HNSW navigates a multi-layer graph structure to find similar vectors in O(log n) time.

### RAG (Retrieval Augmented Generation)

RAG enhances AI responses by grounding them in real document data:
1. **Retrieve**: Find relevant passages using vector search
2. **Augment**: Add retrieved passages as context to the prompt
3. **Generate**: The AI generates an answer based on the context

This ensures answers are factual, specific, and traceable to source documents.

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backend API | FastAPI | REST API endpoints |
| Server | Uvicorn | ASGI server |
| Embeddings | Sentence Transformers | Text-to-vector conversion |
| Document Loading | LangChain + PyPDF | PDF text extraction |
| Text Splitting | LangChain | Chunking with overlap |
| Vector Database | Endee | Storage and similarity search |
| HTTP Client | Requests + msgpack | Communication with Endee API |

## Setup Instructions

### Prerequisites

- Python 3.12+
- Endee vector database server running on port 8080

### 1. Start the Endee Server

**Option A — Docker (recommended):**
```bash
docker run -p 8080:8080 -v ./endee-data:/data --name endee-server endeeio/endee-server:latest
```

**Option B — Build from source:**
```bash
cd /path/to/endee
chmod +x ./install.sh ./run.sh
./install.sh --release --neon   # Use --avx2 or --avx512 on x86
./run.sh
```

Verify it's running:
```bash
curl http://localhost:8080/api/v1/health
```

### 2. Install Python Dependencies

```bash
cd ai-document-chatbot
pip install -r requirements.txt
```

### 3. Add Your PDF Document

Place your PDF file in the `documents/` folder:
```bash
cp /path/to/your/document.pdf documents/sample.pdf
```

### 4. Run the Ingestion Pipeline

```bash
python3 ingest.py
```

This will:
- Load the PDF and extract text
- Split text into 500-character chunks with 50-character overlap
- Generate 384-dimensional embeddings for each chunk
- Create an index in Endee and store all vectors

You can also ingest a different PDF:
```bash
python3 ingest.py /path/to/another/document.pdf
```

### 5. Start the API Server

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`.

## Usage

### Health Check

```bash
curl http://localhost:8000/
```

Response:
```json
{
  "status": "running",
  "message": "AI Document Chatbot API is running. Use POST /ask to ask questions."
}
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

Response:
```json
{
  "question": "What is the main topic of the document?",
  "context": "[Chunk 1 | Similarity: 0.8234]\n...",
  "answer": "Based on the document, here is the relevant information...",
  "sources": [
    {
      "chunk_id": "chunk_5",
      "similarity": 0.8234,
      "text": "..."
    }
  ]
}
```

### Interactive API Docs

FastAPI provides auto-generated Swagger UI:
```
http://localhost:8000/docs
```

## Project Structure

```
ai-document-chatbot/
├── app.py              # FastAPI server with /ask RAG endpoint
├── ingest.py           # PDF ingestion pipeline (load → chunk → embed → store)
├── vector_store.py     # Endee vector database Python client
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── documents/
    └── sample.pdf      # Your PDF document
```

## Quick Start Summary

```bash
# 1. Start Endee server (in a separate terminal)
docker run -p 8080:8080 -v ./endee-data:/data endeeio/endee-server:latest

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your PDF
cp your-document.pdf documents/sample.pdf

# 4. Ingest the document
python3 ingest.py

# 5. Start the API
uvicorn app:app --reload

# 6. Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```
