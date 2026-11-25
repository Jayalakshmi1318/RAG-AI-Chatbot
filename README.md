# PDF Q&A AI Agent

A lightweight, local Retrieval-Augmented Generation (RAG) system for question-answering over uploaded PDF documents using Google Gemini and ChromaDB.

## Features

- **Document Ingestion:** Fast, reliable text extraction and page tracking using PyMuPDF.
- **Semantic Search:** Vector retrieval powered by ChromaDB and Gemini embeddings (`gemini-embedding-2`).
- **Grounded Responses:** Accurate answers generated with Google Gemini Flash models with direct page-level citations.
- **Multiple Document Support:** Upload and query across multiple PDFs in a single session.
- **Conversation History:** Retains recent context for natural multi-turn conversations.
- **Lightweight Architecture:** Custom Python pipeline with no heavy framework overhead, designed to run smoothly on standard laptops.

## Project Structure

```text
pdf-qa-agent/
├── .env.example             # Template for API keys and configuration
├── .gitignore               # Ignores venv, local db, and sensitive files
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
├── app.py                   # Streamlit UI entrypoint
└── src/
    ├── __init__.py
    ├── config.py            # Central settings & environment loading
    ├── pdf_processor.py     # PDF text extraction & chunking
    ├── gemini_client.py     # Google GenAI SDK wrapper (Embeddings & LLM)
    ├── vector_store.py      # ChromaDB collection management & similarity search
    └── rag_engine.py        # Core RAG orchestration and prompt grounding
```

## Setup Instructions

1. **Clone or navigate to the project directory:**
   ```bash
   cd "pdf QA bot"
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Windows (cmd):
   .\venv\Scripts\activate.bat
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   - Copy `.env.example` to `.env`:
     ```bash
     copy .env.example .env
     ```
   - Open `.env` and insert your Google Gemini API key:
     ```ini
     GEMINI_API_KEY="your_api_key_here"
     ```

5. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
