"""
RAG Engine Module.

Responsible for:
- Orchestrating the end-to-end RAG pipeline:
  User Question -> Query Contextualization -> Embedding -> Vector Retrieval -> Relevance Thresholding -> Grounded Generation
- Enforcing strict grounding constraints to prevent hallucinations
- Preserving source metadata and deduplicating citations by document page
- Resolving pronouns and follow-up references using recent conversation history
- Returning structured response objects containing answer, sources, and similarity metrics
"""

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from src.config import DISTANCE_THRESHOLD, TOP_K_RESULTS
from src.gemini_client import (
    _sanitize_error,
    embed_query,
    generate_response,
)
from src.vector_store import ChromaVectorStore, SearchResult

NO_CONTEXT_FALLBACK = "I couldn't find sufficient information about this in the uploaded documents."

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are an academic and technical document Q&A assistant.\n"
    "Follow these guidelines:\n"
    "1. Answer the user question thoroughly and accurately based on the document titles, module numbers, and context excerpts (e.g. BIS701 Big Data Analytics / Information Systems Module 1).\n"
    "2. If the user asks about foundational topics or syllabus concepts related to the retrieved documents (e.g., types of data such as structured, semi-structured, and unstructured data, Big Data 5Vs, Hadoop architecture, etc.), explain the concepts clearly and accurately within that academic subject context.\n"
    "3. Only if the question is completely unrelated to the topics or domain of the uploaded documents, state exactly: "
    f"'{NO_CONTEXT_FALLBACK}'\n"
    "4. Be direct, clear, and technically precise."
)


class RAGError(Exception):
    """Exception raised for RAG pipeline errors."""
    pass


@dataclass
class SourceReference:
    """Represents a unique source citation grouped by document and page."""

    source_file: str
    page_number: int
    chunk_ids: list[str]
    relevance_score: float
    min_distance: float

    def to_dict(self) -> dict[str, Any]:
        """Serializes source citation to dictionary."""
        return {
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_ids": self.chunk_ids,
            "relevance_score": self.relevance_score,
            "min_distance": self.min_distance,
        }


@dataclass
class RAGResponse:
    """Represents the complete structured response from the RAG engine."""

    answer: str
    sources: list[SourceReference]
    retrieved_chunks: list[SearchResult]
    has_sufficient_context: bool
    confidence_score: float
    query_used: str

    def to_dict(self) -> dict[str, Any]:
        """Serializes RAG response to dictionary."""
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "retrieved_chunks": [c.to_dict() for c in self.retrieved_chunks],
            "has_sufficient_context": self.has_sufficient_context,
            "confidence_score": self.confidence_score,
            "query_used": self.query_used,
        }


def deduplicate_sources(chunks: Sequence[SearchResult]) -> list[SourceReference]:
    """
    Groups retrieved chunks by (source_file, page_number) to eliminate duplicate page citations.
    Calculates the highest relevance score and minimum distance for each page.

    Args:
        chunks: Sequence of retrieved SearchResult chunks.

    Returns:
        Sorted list of unique SourceReference items ordered by descending relevance.
    """
    groups: dict[tuple[str, int], list[SearchResult]] = {}
    for chunk in chunks:
        key = (chunk.source_file, chunk.page_number)
        if key not in groups:
            groups[key] = []
        groups[key].append(chunk)

    sources: list[SourceReference] = []
    for (source_file, page_num), group_chunks in groups.items():
        chunk_ids = [c.chunk_id for c in group_chunks]
        best_sim = max(c.similarity_score for c in group_chunks)
        min_dist = min(c.distance for c in group_chunks)
        sources.append(
            SourceReference(
                source_file=source_file,
                page_number=page_num,
                chunk_ids=chunk_ids,
                relevance_score=round(best_sim, 4),
                min_distance=round(min_dist, 4),
            )
        )

    sources.sort(key=lambda s: s.relevance_score, reverse=True)
    return sources


def format_context_blocks(chunks: Sequence[SearchResult]) -> str:
    """
    Formats retrieved chunks into labeled context sections preserving document names and page numbers.

    Args:
        chunks: Sequence of SearchResult items.

    Returns:
        Formatted multi-block string.
    """
    blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        block = (
            f"[EXCERPT {idx}] (Document: '{chunk.source_file}', Page {chunk.page_number})\n"
            f"{chunk.text}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def contextualize_query(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
    generate_fn: Callable[..., str] | None = None,
) -> str:
    """
    Uses recent conversation history to reformulate an ambiguous follow-up question
    into a standalone search query before querying the vector store.

    Args:
        question: User's latest question.
        conversation_history: Optional list of previous chat messages ({'role': ..., 'content': ...}).
        generate_fn: Callable for LLM text generation (defaults to generate_response).

    Returns:
        Contextualized standalone search query.
    """
    if not conversation_history:
        return question.strip()

    recent_history = [
        msg for msg in conversation_history[-4:]
        if isinstance(msg, dict) and msg.get("content")
    ]

    if not recent_history:
        return question.strip()

    history_text = "\n".join(
        f"{msg.get('role', 'User').capitalize()}: {msg.get('content', '')}"
        for msg in recent_history
    )

    prompt = (
        "Given the recent conversation history and a follow-up question from the user, "
        "rewrite the follow-up question into a clear, standalone search query suitable for document retrieval.\n"
        "- Do NOT answer the question.\n"
        "- Resolve all ambiguous references, pronouns, and implied subjects (e.g., 'it', 'its', 'they', 'these', 'those').\n"
        "- Keep the query concise and preserve key technical terms.\n"
        "- If the question is already clear and standalone, return it unchanged.\n\n"
        f"Conversation History:\n{history_text}\n\n"
        f"Follow-up Question: {question.strip()}\n\n"
        "Standalone Search Query:"
    )

    try:
        fn = generate_fn or generate_response
        rewritten = fn(
            prompt=prompt,
            temperature=0.0,
        ).strip()

        if rewritten and len(rewritten) > 3:
            return rewritten.strip("\"'")
    except Exception:
        # Fall back to original question on any reformulation failure
        pass

    return question.strip()


def build_grounded_prompt(
    question: str,
    context_blocks: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """
    Assembles the final generation prompt combining document context, recent history, and the user question.

    Args:
        question: The user's query.
        context_blocks: Formatted document excerpts with metadata.
        conversation_history: Optional recent dialogue turns.

    Returns:
        Complete augmented prompt string.
    """
    sections = [f"DOCUMENT CONTEXT:\n{context_blocks}"]

    if conversation_history:
        recent_history = [
            msg for msg in conversation_history[-4:]
            if isinstance(msg, dict) and msg.get("content")
        ]
        if recent_history:
            history_str = "\n".join(
                f"{msg.get('role', 'User').capitalize()}: {msg.get('content', '')}"
                for msg in recent_history
            )
            sections.append(f"RECENT CONVERSATION CONTEXT:\n{history_str}")

    sections.append(f"USER QUESTION: {question}\n\nANSWER:")
    return "\n\n".join(sections)


class RAGEngine:
    """
    End-to-end Retrieval-Augmented Generation engine for PDF question answering.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore | None = None,
        distance_threshold: float = DISTANCE_THRESHOLD,
        top_k: int = TOP_K_RESULTS,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        embed_fn: Callable[[str], list[float]] | None = None,
        generate_fn: Callable[..., str] | None = None,
    ):
        """
        Initializes the RAG Engine.

        Args:
            vector_store: Vector store instance (defaults to persistent ChromaVectorStore).
            distance_threshold: Maximum cosine distance cutoff for retrieved chunks.
            top_k: Number of chunks to retrieve.
            system_instruction: Strict grounding instructions for Gemini generation.
            embed_fn: Query embedding function (defaults to embed_query).
            generate_fn: Answer generation function (defaults to generate_response).
        """
        self.vector_store = vector_store or ChromaVectorStore()
        self.distance_threshold = distance_threshold
        self.top_k = top_k
        self.system_instruction = system_instruction
        self._embed_fn = embed_fn or embed_query
        self._generate_fn = generate_fn or generate_response

    def answer_question(
        self,
        question: str,
        conversation_history: list[dict[str, str]] | None = None,
        source_file_filter: str | None = None,
    ) -> RAGResponse:
        """
        Executes the RAG pipeline to answer a user question.

        Args:
            question: The user's question.
            conversation_history: Optional list of past chat messages.
            source_file_filter: Optional specific document filename to scope retrieval.

        Returns:
            Structured RAGResponse containing answer, citations, chunks, and metadata.

        Raises:
            RAGError: If a fatal retrieval or generation error occurs.
        """
        clean_question = question.strip() if question else ""
        if not clean_question:
            return RAGResponse(
                answer="Please provide a non-empty question.",
                sources=[],
                retrieved_chunks=[],
                has_sufficient_context=False,
                confidence_score=0.0,
                query_used="",
            )

        # 1. Verify that documents exist in vector store
        try:
            doc_count = self.vector_store.count()
        except Exception as exc:
            clean_msg = _sanitize_error(str(exc))
            raise RAGError(f"Failed to access vector store: {clean_msg}") from exc

        if doc_count == 0:
            return RAGResponse(
                answer="No documents are currently indexed. Please upload at least one PDF first.",
                sources=[],
                retrieved_chunks=[],
                has_sufficient_context=False,
                confidence_score=0.0,
                query_used=clean_question,
            )

        # 2. Contextualize query if conversation history is present
        retrieval_query = contextualize_query(
            question=clean_question,
            conversation_history=conversation_history,
            generate_fn=self._generate_fn,
        )

        # 3. Generate embedding for the retrieval query
        try:
            query_embedding = self._embed_fn(retrieval_query)
        except Exception as exc:
            clean_msg = _sanitize_error(str(exc))
            raise RAGError(f"Failed to generate query embedding: {clean_msg}") from exc

        # 4. Search vector store with distance thresholding
        try:
            chunks = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=self.top_k,
                distance_threshold=self.distance_threshold,
                source_file=source_file_filter,
            )
        except Exception as exc:
            clean_msg = _sanitize_error(str(exc))
            raise RAGError(f"Vector search failed: {clean_msg}") from exc

        # 5. Check if relevant context was found
        # Requirement 6: If no sufficiently relevant chunks found, DO NOT call Gemini
        if not chunks:
            return RAGResponse(
                answer=NO_CONTEXT_FALLBACK,
                sources=[],
                retrieved_chunks=[],
                has_sufficient_context=False,
                confidence_score=0.0,
                query_used=retrieval_query,
            )

        # 6. Deduplicate sources by document and page
        sources = deduplicate_sources(chunks)
        confidence = max(c.similarity_score for c in chunks)

        # 7. Format context and prompt
        context_blocks = format_context_blocks(chunks)
        prompt = build_grounded_prompt(
            question=clean_question,
            context_blocks=context_blocks,
            conversation_history=conversation_history,
        )

        # 8. Generate grounded answer
        try:
            answer = self._generate_fn(
                prompt=prompt,
                system_instruction=self.system_instruction,
                temperature=0.0,
            )
        except Exception as exc:
            clean_msg = _sanitize_error(str(exc))
            raise RAGError(f"Failed to generate answer from Gemini: {clean_msg}") from exc

        is_fallback = not answer or NO_CONTEXT_FALLBACK in answer
        return RAGResponse(
            answer=answer.strip() if answer and not is_fallback else NO_CONTEXT_FALLBACK,
            sources=sources,
            retrieved_chunks=chunks,
            has_sufficient_context=not is_fallback,
            confidence_score=round(confidence, 4),
            query_used=retrieval_query,
        )
