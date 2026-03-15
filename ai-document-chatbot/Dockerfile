# ============================================================
# Stage 1: Install dependencies and download ML models
# ============================================================
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ML models so they are baked into the image
# This prevents multi-minute downloads on every container start
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from transformers import T5ForConditionalGeneration, T5Tokenizer; T5Tokenizer.from_pretrained('google/flan-t5-base'); T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')"

# ============================================================
# Stage 2: Runtime image
# ============================================================
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r chatbot && useradd -r -g chatbot -d /home/chatbot -m chatbot

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy cached models from builder (HuggingFace cache)
COPY --from=builder /root/.cache /home/chatbot/.cache

# Copy application code
COPY app.py frontend.py vector_store.py ingest.py ./
COPY documents/ ./documents/

# Fix ownership
RUN chown -R chatbot:chatbot /app /home/chatbot

USER chatbot

# Default env vars (overridden by docker-compose)
ENV ENDEE_URL="http://endee:8080" \
    API_URL="http://backend:8000" \
    HF_HOME="/home/chatbot/.cache/huggingface" \
    TRANSFORMERS_CACHE="/home/chatbot/.cache/huggingface/hub"

EXPOSE 8000
EXPOSE 8501

# Default command is backend; overridden per service in compose
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
