"""
frontend.py — Streamlit Frontend for AI Document Chatbot

A chat-style web UI that communicates with the FastAPI backend.
Users can upload a PDF document and then ask questions about it.
"""

import streamlit as st
import requests
import os

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="AI Document Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Document Chatbot")
st.caption("Upload a PDF document and ask questions about it — powered by Endee Vector Database")

# ------------------------------------------------------------------
# Sidebar — system info, settings, and document upload
# ------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        "This chatbot uses **semantic search** and **RAG** "
        "(Retrieval Augmented Generation) to answer questions "
        "about your PDF documents."
    )
    st.divider()

    # Backend health check
    st.subheader("System Status")
    try:
        resp = requests.get(f"{API_URL}/", timeout=3)
        if resp.status_code == 200:
            st.success("FastAPI backend: Connected")
        else:
            st.error("FastAPI backend: Error")
    except requests.ConnectionError:
        st.error("FastAPI backend: Not reachable")

    st.divider()

    st.subheader("How it works")
    st.markdown(
        "1. Upload a PDF document below\n"
        "2. The system splits it into chunks and creates embeddings\n"
        "3. Embeddings are stored in **Endee** vector database\n"
        "4. Ask questions and get answers from the document"
    )

# ------------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
if "document_name" not in st.session_state:
    st.session_state.document_name = ""

# ------------------------------------------------------------------
# Document Upload Section
# ------------------------------------------------------------------
if not st.session_state.document_uploaded:
    st.markdown("---")
    st.subheader("Step 1: Upload your PDF document")
    st.markdown("Upload a PDF file to get started. The document will be processed and stored in the Endee vector database for semantic search.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF document you want to ask questions about.",
    )

    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner(f"Processing '{uploaded_file.name}'... This may take a moment."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    resp = requests.post(f"{API_URL}/ingest", files=files, timeout=120)

                    if resp.status_code == 200:
                        data = resp.json()
                        if data["status"] == "success":
                            st.session_state.document_uploaded = True
                            st.session_state.document_name = uploaded_file.name
                            st.session_state.messages = []
                            st.success(
                                f"Document processed successfully!\n\n"
                                f"- **File:** {data['filename']}\n"
                                f"- **Pages:** {data['pages']}\n"
                                f"- **Chunks:** {data['chunks']}"
                            )
                            st.rerun()
                        else:
                            st.error(f"Processing failed: {data['message']}")
                    else:
                        st.error(f"Server error: {resp.status_code} — {resp.text}")

                except requests.ConnectionError:
                    st.error("Cannot connect to the backend. Make sure the FastAPI server is running.")

else:
    # ------------------------------------------------------------------
    # Document loaded — show info bar and allow new upload
    # ------------------------------------------------------------------
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"Current document: **{st.session_state.document_name}** — Ask your questions below.")
    with col2:
        if st.button("Upload New Document"):
            st.session_state.document_uploaded = False
            st.session_state.document_name = ""
            st.session_state.messages = []
            st.rerun()

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("View source chunks"):
                    for src in message["sources"]:
                        st.markdown(
                            f"**{src['chunk_id']}** (similarity: {src['similarity']:.4f})"
                        )
                        st.text(src["text"])
                        st.divider()

    # ------------------------------------------------------------------
    # Chat input
    # ------------------------------------------------------------------
    if question := st.chat_input("Ask a question about your document..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Call the FastAPI backend
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/ask",
                        json={"question": question},
                        timeout=30,
                    )

                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])

                        st.markdown(answer)

                        if sources:
                            with st.expander("View source chunks"):
                                for src in sources:
                                    st.markdown(
                                        f"**{src['chunk_id']}** (similarity: {src['similarity']:.4f})"
                                    )
                                    st.text(src["text"])
                                    st.divider()

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        })
                    else:
                        error_msg = f"API error: {resp.status_code} — {resp.text}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                        })

                except requests.ConnectionError:
                    error_msg = (
                        "Cannot connect to the backend. "
                        "Make sure the FastAPI server is running: "
                        "`uvicorn app:app --reload`"
                    )
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })
