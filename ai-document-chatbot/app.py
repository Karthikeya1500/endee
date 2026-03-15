"""
app.py — FastAPI Application for AI Document Chatbot

This module implements a full RAG (Retrieval Augmented Generation) pipeline:

    User Question
         |
         v
    [1. Embed Question]  — Convert question text into a 384-dim vector
         |
         v
    [2. Vector Search]   — Search Endee for the most similar document chunks
         |
         v
    [3. Build Context]   — Combine retrieved chunks into a single context string
         |
         v
    [4. Send to LLM]     — Pass context + question to the LLM for answer generation
         |
         v
    [5. Return Answer]   — Send natural language answer + source chunks to the UI

This converts the system from simple retrieval into a full RAG pipeline
by adding the GENERATION step (Step 4). Instead of displaying raw chunks,
the LLM reads the retrieved context and produces a clear, human-readable
answer grounded in the document content.

LLM Strategy (tried in order):
  1. Google Gemini   (cloud API, best quality — needs GEMINI_API_KEY)
  2. Groq            (free cloud API, Llama 3 — needs GROQ_API_KEY)
  3. flan-t5-base    (local model, no API key needed)
"""

import os
import tempfile

import requests as http_requests
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import T5ForConditionalGeneration, T5Tokenizer

from vector_store import EndeeVectorStore

# ------------------------------------------------------------------
# Initialize the application
# ------------------------------------------------------------------
app = FastAPI(
    title="AI Document Chatbot",
    description="Ask questions about PDF documents using semantic search and RAG",
    version="1.0.0",
)

# ------------------------------------------------------------------
# Load embedding model (used for both ingestion and query)
# ------------------------------------------------------------------
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")

# ------------------------------------------------------------------
# Initialize LLMs for answer generation
# ------------------------------------------------------------------
# Primary: Google Gemini (cloud API — best quality answers)
# Set GEMINI_API_KEY environment variable to enable.
gemini_client = None
GEMINI_MODEL = ""
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = "gemini-2.0-flash-lite"
        print(f"Gemini LLM initialized (model: {GEMINI_MODEL}).")
    except Exception as e:
        print(f"Gemini init failed: {e}. Will use local LLM.")

# Secondary: Groq (free cloud API — Llama 3, very fast)
# Set GROQ_API_KEY environment variable to enable.
# Get a free key at https://console.groq.com/keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"
if GROQ_API_KEY:
    print(f"Groq LLM enabled (model: {GROQ_MODEL}).")
else:
    print("No GROQ_API_KEY set. Groq LLM disabled.")

# Fallback: flan-t5-base (runs locally, no API key needed)
LLM_MODEL_NAME = "google/flan-t5-base"
print(f"Loading local LLM: {LLM_MODEL_NAME}...")
llm_tokenizer = T5Tokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = T5ForConditionalGeneration.from_pretrained(LLM_MODEL_NAME)
print("Local LLM loaded.")

# ------------------------------------------------------------------
# Endee vector store client
# ------------------------------------------------------------------
ENDEE_BASE_URL = os.environ.get("ENDEE_URL", "http://localhost:8080")
vector_store = EndeeVectorStore(base_url=ENDEE_BASE_URL)
INDEX_NAME = "documents"
TOP_K = 5

# Ingestion settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_DIM = 384


# ------------------------------------------------------------------
# Request/Response models
# ------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question: str
    context: str
    answer: str
    sources: list[dict]


# ------------------------------------------------------------------
# Helper: Generate answer using Gemini (primary LLM)
# ------------------------------------------------------------------
RAG_SYSTEM_PROMPT = (
    "You are a helpful document assistant. Answer the user's question "
    "using ONLY the information provided in the context below. "
    "Write a clear, complete, natural-language answer in 2-4 sentences. "
    "If the context does not contain enough information, say so honestly. "
    "Do NOT mention 'chunks', 'passages', or 'context' in your answer — "
    "just answer as if you know the information."
)

def generate_answer_gemini(context: str, question: str) -> str | None:
    """
    Send context + question to Google Gemini and return a natural answer.
    Returns None if Gemini is unavailable or fails.
    """
    if not gemini_client:
        return None

    prompt = (
        f"{RAG_SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return response.text
    except Exception:
        return None


# ------------------------------------------------------------------
# Helper: Generate answer using Groq (secondary — free cloud LLM)
# ------------------------------------------------------------------
def generate_answer_groq(context: str, question: str) -> str | None:
    """
    Send context + question to Groq's free API (Llama 3) and return a natural answer.
    Returns None if Groq is unavailable or fails.
    """
    if not GROQ_API_KEY:
        return None

    try:
        resp = http_requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
                "temperature": 0.3,
                "max_tokens": 512,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        else:
            print(f"Groq API error: {resp.status_code} — {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"Groq request failed: {e}")
        return None


# ------------------------------------------------------------------
# Helper: Generate answer using flan-t5 (fallback local LLM)
# ------------------------------------------------------------------
def _clean_answer(text: str) -> str:
    """Clean up extracted text: remove chapter headings, fix punctuation."""
    import re
    # Remove "Chapter N: Title" lines (only full heading patterns)
    text = re.sub(r"Chapter\s+\d+:\s*[A-Za-z ]+(?=\s[A-Z])", "", text).strip()
    # Fix double periods and stray periods
    text = text.replace("..", ".").replace(". .", ".")
    # Remove leading periods/whitespace
    text = text.lstrip(". ")
    # Remove leading/trailing whitespace, collapse multiple spaces
    text = " ".join(text.split())
    # Remove trailing " ."
    if text.endswith(" ."):
        text = text[:-2] + "."
    return text


def generate_answer_local(chunks: list[str], question: str) -> str:
    """
    Generate an answer using the local flan-t5 model.

    Since flan-t5-base is a small model (250M params), we use two strategies:
      1. Try Q&A extraction from the best chunk
      2. If the extracted answer is too short, build a formatted answer
         by combining the extraction with relevant chunk text
    """
    best_chunk = chunks[0] if chunks else ""

    # Strategy 1: Direct Q&A extraction from the top chunk
    prompt = (
        f"Read the passage and answer the question.\n\n"
        f"Passage: {best_chunk}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    extracted = ""
    try:
        input_ids = llm_tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        ).input_ids

        outputs = llm_model.generate(
            input_ids,
            max_new_tokens=200,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.5,
        )
        extracted = llm_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception:
        pass

    # Strategy 2: If the extraction is too short or generic,
    # build a better answer by combining the extraction with chunk context
    if not extracted or len(extracted) < 20:
        # Use extractive summarization — take the most relevant sentences
        sentences = []
        for chunk in chunks[:3]:
            for sentence in chunk.replace("\n", " ").split(". "):
                s = sentence.strip()
                # Skip chapter headings and very short fragments
                if s.lower().startswith("chapter ") or len(s) < 20:
                    continue
                if any(w.lower() in s.lower() for w in question.split() if len(w) > 3):
                    sentences.append(s)

        if sentences:
            answer = ". ".join(sentences[:3])
            if not answer.endswith("."):
                answer += "."
            return _clean_answer(answer)
        # Absolute fallback — return first chunk as readable text
        return _clean_answer(best_chunk[:400]) if best_chunk else ""

    # The extracted answer is decent — enhance it if possible
    if len(extracted) < 50:
        # Short answer — try to expand with relevant sentence from chunk
        for sentence in best_chunk.replace("\n", " ").split(". "):
            s = sentence.strip()
            if len(s) > 30 and extracted.lower()[:20] in s.lower():
                return _clean_answer(s + "." if not s.endswith(".") else s)

    return _clean_answer(extracted)


# ------------------------------------------------------------------
# Endpoint: GET /
# ------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "AI Document Chatbot API is running. Use POST /ask to ask questions.",
    }


# ------------------------------------------------------------------
# Endpoint: POST /ask  —  Full RAG Pipeline
# ------------------------------------------------------------------
@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Full RAG (Retrieval Augmented Generation) Pipeline:

    1. EMBED    — Convert user question into a vector embedding
    2. RETRIEVE — Search Endee vector DB for the most similar chunks
    3. AUGMENT  — Combine retrieved chunks into a single context string
    4. GENERATE — Send context + question to LLM for natural answer
    5. RESPOND  — Return the generated answer + source chunks
    """
    question = request.question

    # ------------------------------------------------------------------
    # Step 1: Convert question to vector embedding
    # ------------------------------------------------------------------
    query_vector = embedding_model.encode(question).tolist()

    # ------------------------------------------------------------------
    # Step 2: Search Endee for the most relevant document chunks
    # ------------------------------------------------------------------
    results = vector_store.search(INDEX_NAME, query_vector, k=TOP_K)

    if not results:
        return AskResponse(
            question=question,
            context="No relevant documents found.",
            answer="I could not find any relevant information in the document to answer your question.",
            sources=[],
        )

    # ------------------------------------------------------------------
    # Step 3: Combine retrieved chunks into a single context string
    # ------------------------------------------------------------------
    # This is the AUGMENTATION step — we build a context block from
    # the top retrieved passages that will be sent to the LLM.
    chunk_texts = [result["meta"] for result in results]
    context = "\n".join(chunk_texts)

    # Also build the labeled context and sources for the UI
    context_labeled = "\n\n---\n\n".join(
        f"[Chunk {i+1} | Similarity: {r['similarity']:.4f}]\n{r['meta']}"
        for i, r in enumerate(results)
    )
    sources = [
        {
            "chunk_id": result["id"],
            "similarity": round(result["similarity"], 4),
            "text": result["meta"],
        }
        for result in results
    ]

    # ------------------------------------------------------------------
    # Step 4: GENERATE — Send context + question to the LLM
    # ------------------------------------------------------------------
    # This step converts the system from simple retrieval into a full
    # RAG pipeline. Instead of returning raw chunks, the LLM reads the
    # retrieved context and produces a clear, human-readable answer
    # grounded in the document content.
    #
    # Pipeline: context + question → LLM → natural language answer
    #
    # We try Gemini first (best quality), then fall back to the local
    # flan-t5 model if Gemini is unavailable.

    # Try Gemini (primary — cloud API, best quality)
    answer = generate_answer_gemini(context, question)

    # Try Groq (secondary — free cloud API, Llama 3)
    if not answer:
        answer = generate_answer_groq(context, question)

    # Fall back to local flan-t5 if cloud LLMs are unavailable
    if not answer:
        answer = generate_answer_local(chunk_texts, question)

    # Final fallback — return raw context if both LLMs fail
    if not answer:
        answer = (
            "Based on the document, here is the relevant information:\n\n"
            + context_labeled
        )

    # ------------------------------------------------------------------
    # Step 5: Return the answer along with source chunks for transparency
    # ------------------------------------------------------------------
    return AskResponse(
        question=question,
        context=context_labeled,
        answer=answer,
        sources=sources,
    )


# ------------------------------------------------------------------
# Endpoint: POST /ingest
# ------------------------------------------------------------------
@app.post("/ingest")
def ingest_pdf(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF document into Endee.

    Steps:
        1. Save uploaded PDF to a temp file
        2. Load and extract text using PyPDFLoader
        3. Split into chunks using RecursiveCharacterTextSplitter
        4. Generate embeddings using SentenceTransformer
        5. Delete old index and create fresh one in Endee
        6. Insert all vectors into Endee
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        # Try PyPDFLoader first
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # Filter out pages with no meaningful text
        page_texts = [p.page_content.strip() for p in pages if p.page_content.strip()]

        # If PyPDFLoader found no text, try PyMuPDF (handles more PDF formats)
        if not page_texts:
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(tmp_path)
                page_texts = [page.get_text().strip() for page in doc if page.get_text().strip()]
                doc.close()
            except Exception:
                pass

        if not page_texts:
            return {"status": "error", "message": "PDF has no readable text. It may be a scanned image — OCR is not yet supported."}

        # Join all page texts and split into chunks
        full_text = "\n\n".join(page_texts)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = text_splitter.split_text(full_text)

        if not chunks:
            return {"status": "error", "message": "No text chunks extracted from PDF."}

        texts = chunks  # split_text returns list of strings directly
        embeddings = embedding_model.encode(texts)

        if vector_store.index_exists(INDEX_NAME):
            vector_store.delete_index(INDEX_NAME)
        vector_store.create_index(INDEX_NAME, dim=EMBEDDING_DIM, space_type="cosine")

        vectors = []
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            vectors.append({
                "id": f"chunk_{i}",
                "vector": emb.tolist(),
                "meta": text,
            })

        batch_size = 100
        for start in range(0, len(vectors), batch_size):
            vector_store.insert_vectors(INDEX_NAME, vectors[start:start + batch_size])

        return {
            "status": "success",
            "filename": file.filename,
            "pages": len(page_texts),
            "chunks": len(chunks),
            "message": f"Ingested {len(chunks)} chunks from '{file.filename}' into Endee.",
        }

    finally:
        os.unlink(tmp_path)
