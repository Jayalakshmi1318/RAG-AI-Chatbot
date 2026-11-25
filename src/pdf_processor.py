"""
PDF Processor Module.

Responsible for:
- Reading PDF byte streams (e.g., from Streamlit uploaded files) using PyMuPDF (fitz)
- Extracting text page-by-page to preserve accurate page numbers
- Cleaning extracted text while preserving semantic structure
- Slicing text into configurable overlapping chunks without crossing page boundaries
- Returning structured chunks with metadata (source_file, page_number, chunk_id)
"""

import re
from dataclasses import asdict, dataclass
from typing import Any

import fitz  # PyMuPDF

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


class PDFProcessingError(Exception):
    """Custom exception raised when an error occurs during PDF parsing or extraction."""
    pass


@dataclass
class PDFChunk:
    """Represents a text chunk extracted from a specific page of a PDF document."""

    text: str
    source_file: str
    page_number: int
    chunk_id: str

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk into a standard dictionary format suitable for ChromaDB and RAG."""
        return {
            "text": self.text,
            "metadata": {
                "source_file": self.source_file,
                "page_number": self.page_number,
                "chunk_id": self.chunk_id,
            },
        }


def clean_text(raw_text: str) -> str:
    """
    Cleans extracted raw text while preserving meaningful paragraphs.

    Operations:
    1. Normalizes carriage return line endings (\\r\\n and \\r -> \\n).
    2. Reconnects hyphenated words broken across line wraps (e.g., 'trans-\\nformer' -> 'transformer').
    3. Normalizes horizontal spaces/tabs to a single space.
    4. Limits consecutive newlines to at most two (paragraph separation).
    5. Cleans trailing spaces on each line and strips boundary whitespace.
    """
    if not raw_text:
        return ""

    # Normalize line endings
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix hyphenated words broken across line wraps
    text = re.sub(r"(\b\w+)-\n(\w+\b)", r"\1\2", text)

    # Normalize multiple horizontal whitespaces (spaces/tabs) to a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive newlines to at most 2 (preserving paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Clean whitespace around newlines
    text = re.sub(r" ?\n ?", "\n", text)

    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Splits text into chunks of maximum `chunk_size` characters with `chunk_overlap`.
    Attempts to break on natural sentence or word boundaries whenever possible.

    Args:
        text: The cleaned text string to split.
        chunk_size: Maximum characters per chunk. Must be > 0.
        chunk_overlap: Number of characters to overlap between adjacent chunks. Must be < chunk_size.

    Returns:
        A list of non-empty text chunks.
    """
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap must be >= 0 and < chunk_size ({chunk_size}), got {chunk_overlap}"
        )

    # If text is already smaller than or equal to chunk size, return it as single chunk
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Look back from `end` to find a natural boundary (paragraph, newline, period, space)
        # Search within a lookback window to prevent breaking words midway
        lookback_window = min(chunk_size // 4, max(chunk_overlap, 30))
        boundary_found = -1

        for delimiter in ("\n\n", "\n", ". ", " "):
            split_pos = text.rfind(delimiter, end - lookback_window, end)
            if split_pos != -1:
                boundary_found = split_pos + len(delimiter)
                break

        split_index = boundary_found if boundary_found != -1 else end

        chunk = text[start:split_index].strip()
        if chunk:
            chunks.append(chunk)

        # Advance start position ensuring strictly positive forward progress
        next_start = split_index - chunk_overlap
        if next_start <= start:
            next_start = split_index

        start = next_start

    return chunks


def extract_pages_from_pdf(
    pdf_bytes: bytes,
    source_file: str = "document.pdf",
) -> list[dict[str, Any]]:
    """
    Extracts text page-by-page from raw PDF bytes. Completely empty pages are ignored.

    Args:
        pdf_bytes: Byte content of the PDF file.
        source_file: Name or path identifier of the PDF file.

    Returns:
        List of dicts: [{'page_number': int, 'text': str}, ...]

    Raises:
        PDFProcessingError: If the PDF is corrupt, password protected, or empty.
    """
    if not pdf_bytes:
        raise PDFProcessingError(f"Provided PDF byte stream for '{source_file}' is empty.")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFProcessingError(f"Failed to open/parse PDF '{source_file}': {exc}") from exc

    try:
        if doc.is_encrypted:
            raise PDFProcessingError(
                f"PDF '{source_file}' is password-protected and cannot be processed."
            )

        if doc.page_count == 0:
            raise PDFProcessingError(f"PDF '{source_file}' contains no pages.")

        pages_data: list[dict[str, Any]] = []

        for page_idx in range(doc.page_count):
            page_number = page_idx + 1  # 1-indexed page number
            page = doc.load_page(page_idx)
            raw_text = page.get_text()
            cleaned = clean_text(raw_text)

            # Ignore completely empty pages
            if not cleaned:
                continue

            pages_data.append({
                "page_number": page_number,
                "text": cleaned,
            })

        return pages_data

    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(f"Error extracting text from '{source_file}': {exc}") from exc
    finally:
        doc.close()


def process_pdf_bytes(
    pdf_bytes: bytes,
    source_file: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[PDFChunk]:
    """
    Processes a PDF byte stream: extracts pages, cleans text, and splits into chunks.
    Preserves exact page numbers for every chunk without cross-page splitting.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF file.
        source_file: Original filename of the document.
        chunk_size: Maximum character length per chunk (defaults to config.CHUNK_SIZE).
        chunk_overlap: Overlapping characters between adjacent chunks (defaults to config.CHUNK_OVERLAP).

    Returns:
        A list of PDFChunk dataclass instances containing text and metadata.

    Raises:
        PDFProcessingError: If the PDF cannot be opened or parsed.
    """
    pages = extract_pages_from_pdf(pdf_bytes=pdf_bytes, source_file=source_file)
    all_chunks: list[PDFChunk] = []

    # Clean filename for IDs
    safe_filename = source_file.replace("/", "_").replace("\\", "_")

    for page in pages:
        page_num = page["page_number"]
        page_text = page["text"]

        text_chunks = chunk_text(
            text=page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_idx, chunk_content in enumerate(text_chunks):
            chunk_id = f"{safe_filename}_p{page_num}_c{chunk_idx}"
            all_chunks.append(
                PDFChunk(
                    text=chunk_content,
                    source_file=source_file,
                    page_number=page_num,
                    chunk_id=chunk_id,
                )
            )

    return all_chunks


def validate_pdf_processor() -> bool:
    """
    Internal validation function that generates an in-memory sample PDF with 3 pages
    (page 1: text, page 2: empty, page 3: long text), processes it, and verifies:
    1. Empty pages are skipped.
    2. Page numbers are strictly preserved.
    3. Chunks contain valid metadata and text.
    """
    doc = fitz.open()

    # Page 1: normal text
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Hello world from Page 1.\nThis is a sample document for RAG testing.")

    # Page 2: completely blank/empty
    _ = doc.new_page()

    # Page 3: multi-sentence text to test chunking
    p3 = doc.new_page()
    sample_text = (
        "Artificial intelligence and retrieval-augmented generation have transformed "
        "how organizations query unstructured document repositories. "
        "By combining semantic search with large language models, applications can "
        "provide accurate, cited answers while minimizing factual hallucination."
    ) * 10
    p3.insert_text((50, 50), sample_text)

    sample_bytes = doc.tobytes()
    doc.close()

    chunks = process_pdf_bytes(
        pdf_bytes=sample_bytes,
        source_file="test_document.pdf",
        chunk_size=300,
        chunk_overlap=50,
    )

    # Verification assertions
    assert len(chunks) > 0, "No chunks produced"
    page_numbers = {chunk.page_number for chunk in chunks}
    assert 2 not in page_numbers, "Page 2 was blank and should have been ignored"
    assert 1 in page_numbers, "Page 1 missing"
    assert 3 in page_numbers, "Page 3 missing"

    for chunk in chunks:
        assert chunk.source_file == "test_document.pdf"
        assert chunk.chunk_id.startswith("test_document.pdf_p")
        dict_repr = chunk.to_dict()
        assert "text" in dict_repr and "metadata" in dict_repr
        assert dict_repr["metadata"]["page_number"] in (1, 3)

    return True


if __name__ == "__main__":
    is_valid = validate_pdf_processor()
    print(f"PDF Processor Validation Passed: {is_valid}")
