"""
PDF Q&A AI Agent — Streamlit Web Application.

Connects:
- src.pdf_processor: Page-level PDF text extraction and chunking
- src.gemini_client: Embedding generation and LLM answer synthesis
- src.vector_store: ChromaDB vector indexing and similarity retrieval
- src.rag_engine: Grounded RAG orchestration with source citation
"""

import streamlit as st

from src.config import (
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_MODEL,
    validate_config,
)
from src.gemini_client import (
    GeminiClientError,
    GeminiConfigurationError,
    embed_texts,
)
from src.pdf_processor import (
    PDFProcessingError,
    process_pdf_bytes,
)
from src.rag_engine import (
    RAGError,
    RAGEngine,
    RAGResponse,
)
from src.vector_store import (
    ChromaVectorStore,
    VectorStoreError,
)

# --- Page Configuration ---
st.set_page_config(
    page_title="PDF Q&A AI Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Initialize Persistent Singletons in Session State ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = ChromaVectorStore()

if "messages" not in st.session_state:
    st.session_state.messages = []

vector_store: ChromaVectorStore = st.session_state.vector_store
rag_engine: RAGEngine = RAGEngine(vector_store=vector_store)

# --- Configuration Validation ---
config_status = validate_config()
is_api_key_configured = config_status["valid"]


# --- Helper to format file size ---
def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


# --- Sidebar: Document Ingestion & Status ---
with st.sidebar:
    st.title("📁 Document Manager")

    # API Key Alert if not configured
    if not is_api_key_configured:
        st.warning(
            "⚠️ **API Key Missing**\n\n"
            "Please set `GEMINI_API_KEY` in your `.env` file to process documents and generate answers.",
            icon="⚠️",
        )

    # 1. Multi-file PDF Uploader
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF documents to index for question answering.",
    )

    if uploaded_files:
        st.caption(f"**{len(uploaded_files)}** file(s) selected:")
        for uf in uploaded_files:
            st.markdown(f"- 📄 `{uf.name}` ({format_file_size(uf.size)})")

    # 2. Process Documents Button
    process_btn = st.button("⚡ Process Documents", type="primary", use_container_width=True)

    if process_btn:
        if not is_api_key_configured:
            st.error("Cannot process documents: `GEMINI_API_KEY` is not set in `.env`.")
        elif not uploaded_files:
            st.warning("Please select at least one PDF file first.")
        else:
            indexed_sources_before = set(vector_store.list_sources())
            total_processed_chunks = 0

            with st.status("Indexing documents into ChromaDB...", expanded=True) as status:
                for file in uploaded_files:
                    filename = file.name

                    # Performance check: skip already indexed files
                    if filename in indexed_sources_before:
                        st.write(f"ℹ️ `{filename}` is already indexed. Skipping.")
                        continue

                    try:
                        st.write(f"🔍 Reading and chunking `{filename}`...")
                        file_bytes = file.getvalue()
                        chunks = process_pdf_bytes(file_bytes, source_file=filename)

                        if not chunks:
                            st.warning(f"⚠️ `{filename}` contained no extractable text.")
                            continue

                        pages_count = len({c.page_number for c in chunks})
                        st.write(
                            f"📄 `{filename}`: extracted {pages_count} page(s), created {len(chunks)} chunks."
                        )

                        st.write(f"🧬 Generating embeddings via `{GEMINI_EMBEDDING_MODEL}`...")
                        chunk_texts = [c.text for c in chunks]
                        embeddings = embed_texts(
                            texts=chunk_texts,
                            task_type="RETRIEVAL_DOCUMENT",
                        )

                        st.write(f"💾 Upserting chunks into ChromaDB...")
                        added_count = vector_store.add_chunks(chunks, embeddings)
                        total_processed_chunks += added_count
                        st.write(f"✅ `{filename}` successfully indexed ({added_count} chunks).")

                    except (PDFProcessingError, GeminiClientError, VectorStoreError) as err:
                        st.error(f"Error processing `{filename}`: {err}")
                    except Exception as err:
                        st.error(f"Unexpected error processing `{filename}`: {err}")

                status.update(
                    label="Document processing completed!",
                    state="complete",
                    expanded=False,
                )

            st.rerun()

    st.divider()

    # 3. Document Knowledge Base Status
    st.subheader("📊 Knowledge Base Status")
    try:
        current_chunk_count = vector_store.count()
        current_sources = vector_store.list_sources()
    except Exception:
        current_chunk_count = 0
        current_sources = []

    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Documents", len(current_sources))
    with col_stat2:
        st.metric("Chunks", current_chunk_count)

    if current_sources:
        st.markdown("**Currently Indexed Files:**")
        for src in current_sources:
            st.markdown(f"- 📄 `{src}`")
    else:
        st.info("No documents indexed yet.")

    # 4. Clear Knowledge Base Button with confirmation
    st.divider()
    with st.popover("🗑️ Clear Knowledge Base", use_container_width=True):
        st.markdown("⚠️ **Confirm Reset**")
        st.write("This will permanently delete all indexed chunks and clear current chat history.")
        if st.button("Yes, Clear Everything", type="primary", use_container_width=True):
            try:
                vector_store.reset_collection()
                st.session_state.messages = []
                st.success("Knowledge base and chat history cleared!")
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to reset collection: {exc}")

    # Model settings info
    st.divider()
    st.caption(f"**Generation Model:** `{GEMINI_MODEL}`")
    st.caption(f"**Embedding Model:** `{GEMINI_EMBEDDING_MODEL}`")


# --- Main Header ---
st.title("📚 PDF Q&A AI Agent")
st.subheader("Ask questions and get answers grounded in your documents.")

# Show warning in main area if API key is absent
if not is_api_key_configured:
    st.warning(
        "⚠️ **Gemini API Key Required:** Please configure `GEMINI_API_KEY` in your `.env` file "
        "to allow document processing and response generation.",
        icon="🔑",
    )


# --- Render Conversation History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display sources if present in assistant response
        sources = msg.get("sources", [])
        if sources:
            with st.expander("📖 View Verified Sources & Citations", expanded=False):
                for s in sources:
                    st.markdown(
                        f"- 📄 **`{s['source_file']}`** — 📖 **Page {s['page_number']}** "
                        f"*(Relevance: `{s['relevance_score']:.2f}`)*"
                    )

        # Display confidence indicator if available
        confidence = msg.get("confidence_score")
        if confidence is not None and confidence > 0 and msg.get("has_sufficient_context"):
            st.caption(f"🎯 Relevance Score: `{confidence:.2f}`")


# --- Chat Input & Response Handling ---
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    # 1. Render and record user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Check document index status
    try:
        active_docs = vector_store.count()
    except Exception:
        active_docs = 0

    with st.chat_message("assistant"):
        if active_docs == 0:
            warning_text = "Please upload and process at least one PDF before asking questions."
            st.markdown(warning_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": warning_text,
                "sources": [],
                "confidence_score": 0.0,
                "has_sufficient_context": False,
            })
        elif not is_api_key_configured:
            key_err_text = (
                "Cannot generate answer: `GEMINI_API_KEY` is not configured. "
                "Please add it to your `.env` file."
            )
            st.error(key_err_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": key_err_text,
                "sources": [],
                "confidence_score": 0.0,
                "has_sufficient_context": False,
            })
        else:
            with st.spinner("Searching document context and generating answer..."):
                try:
                    # Pass the recent history (excluding the current user question which was just added)
                    recent_history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]

                    rag_response: RAGResponse = rag_engine.answer_question(
                        question=user_query,
                        conversation_history=recent_history,
                    )

                    # Display Answer
                    st.markdown(rag_response.answer)

                    # Display Sources
                    sources_data = [s.to_dict() for s in rag_response.sources]
                    if sources_data:
                        with st.expander("📖 View Verified Sources & Citations", expanded=False):
                            for s in sources_data:
                                st.markdown(
                                    f"- 📄 **`{s['source_file']}`** — 📖 **Page {s['page_number']}** "
                                    f"*(Relevance: `{s['relevance_score']:.2f}`)*"
                                )

                    # Display Relevance Indicator
                    if rag_response.has_sufficient_context and rag_response.confidence_score > 0:
                        st.caption(f"🎯 Relevance Score: `{rag_response.confidence_score:.2f}`")

                    # Append to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": rag_response.answer,
                        "sources": sources_data,
                        "confidence_score": rag_response.confidence_score,
                        "has_sufficient_context": rag_response.has_sufficient_context,
                    })

                except (RAGError, GeminiClientError, VectorStoreError) as exc:
                    error_msg = f"An error occurred while answering your question: {exc}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                        "confidence_score": 0.0,
                        "has_sufficient_context": False,
                    })
                except Exception as exc:
                    error_msg = f"Unexpected error: {exc}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                        "confidence_score": 0.0,
                        "has_sufficient_context": False,
                    })
