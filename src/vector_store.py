"""
ChromaDB Vector Store Module.

Responsible for:
- Managing a local persistent ChromaDB client using CHROMA_PERSIST_DIR from src.config
- Initializing or retrieving collections with cosine distance metric
- Upserting PDF chunks and precomputed embeddings while preventing duplicate entries
- Preserving chunk text and page-level metadata (source_file, page_number, chunk_id)
- Querying top-k most relevant chunks using query embeddings
- Providing distance/relevance thresholding and collection reset functionality
"""

from dataclasses import dataclass
from typing import Any, Sequence

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from src.config import CHROMA_PERSIST_DIR, TOP_K_RESULTS
from src.pdf_processor import PDFChunk


class VectorStoreError(Exception):
    """Base exception for all ChromaDB vector store errors."""
    pass


@dataclass
class SearchResult:
    """Represents a retrieved document chunk with similarity metrics and metadata."""

    chunk_id: str
    text: str
    source_file: str
    page_number: int
    distance: float
    similarity_score: float

    def to_dict(self) -> dict[str, Any]:
        """Serializes search result into a dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "distance": self.distance,
            "similarity_score": self.similarity_score,
        }


class ChromaVectorStore:
    """
    Manages local persistent vector storage and similarity retrieval using ChromaDB.
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = "pdf_knowledge_base",
    ):
        """
        Initializes the ChromaVectorStore.

        Args:
            persist_dir: Local filesystem directory for ChromaDB persistence.
            collection_name: Name of the collection to store PDF chunks in.
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client: ClientAPI | None = None
        self._collection: Collection | None = None

    @property
    def client(self) -> ClientAPI:
        """Lazy-loaded persistent ChromaDB client."""
        if self._client is None:
            try:
                self._client = chromadb.PersistentClient(path=self.persist_dir)
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to initialize ChromaDB PersistentClient at '{self.persist_dir}': {exc}"
                ) from exc
        return self._client

    @property
    def collection(self) -> Collection:
        """Lazy-loaded ChromaDB collection with cosine distance metric."""
        if self._collection is None:
            self._collection = self.get_or_create_collection(self.collection_name)
        return self._collection

    def get_or_create_collection(self, collection_name: str | None = None) -> Collection:
        """
        Retrieves or creates a ChromaDB collection configured for cosine distance.

        Args:
            collection_name: Optional custom collection name; defaults to self.collection_name.

        Returns:
            The ChromaDB Collection object.

        Raises:
            VectorStoreError: If collection retrieval or creation fails.
        """
        name = collection_name or self.collection_name
        try:
            return self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to get or create collection '{name}': {exc}") from exc

    def add_chunks(
        self,
        chunks: Sequence[PDFChunk | dict[str, Any]],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """
        Upserts PDF chunks and their precomputed embeddings into ChromaDB.
        Uses stable chunk_ids to ensure idempotent operations without duplicate records.

        Args:
            chunks: Sequence of PDFChunk dataclasses or dicts containing text and metadata.
            embeddings: Sequence of vector embeddings corresponding 1-to-1 with chunks.

        Returns:
            Number of chunks successfully upserted.

        Raises:
            VectorStoreError: If inputs are invalid or ChromaDB upsert fails.
        """
        if not chunks or not embeddings:
            return 0

        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)})."
            )

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        valid_embeddings: list[list[float]] = []

        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            if not emb:
                raise VectorStoreError(f"Embedding vector at index {idx} is empty.")

            if isinstance(chunk, PDFChunk):
                chunk_id = chunk.chunk_id
                text = chunk.text
                source_file = chunk.source_file
                page_number = chunk.page_number
            elif isinstance(chunk, dict):
                meta = chunk.get("metadata", {})
                chunk_id = meta.get("chunk_id") or chunk.get("chunk_id")
                text = chunk.get("text", "")
                source_file = meta.get("source_file") or chunk.get("source_file", "unknown")
                page_number = int(meta.get("page_number") or chunk.get("page_number", 1))
            else:
                raise VectorStoreError(
                    f"Unsupported chunk type at index {idx}: expected PDFChunk or dict, got {type(chunk)}"
                )

            if not chunk_id:
                raise VectorStoreError(f"Missing required 'chunk_id' for chunk at index {idx}.")

            ids.append(str(chunk_id))
            documents.append(text)
            metadatas.append({
                "source_file": str(source_file),
                "page_number": int(page_number),
                "chunk_id": str(chunk_id),
            })
            valid_embeddings.append(list(emb))

        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=valid_embeddings,
                metadatas=metadatas,
            )
            return len(ids)
        except Exception as exc:
            raise VectorStoreError(f"Failed to upsert chunks into ChromaDB: {exc}") from exc

    def query(
        self,
        query_embedding: Sequence[float],
        top_k: int = TOP_K_RESULTS,
        distance_threshold: float | None = None,
        source_file: str | None = None,
    ) -> list[SearchResult]:
        """
        Executes a vector similarity query against stored chunks.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of nearest neighbors to retrieve (defaults to config.TOP_K_RESULTS).
            distance_threshold: Optional maximum cosine distance cutoff (0.0=identical, 1.0=orthogonal).
                                Chunks with distance > distance_threshold are excluded.
            source_file: Optional document filter to retrieve chunks only from a specific file.

        Returns:
            List of SearchResult items ordered by increasing distance (highest similarity first).

        Raises:
            VectorStoreError: If query embedding is empty or ChromaDB query fails.
        """
        if not query_embedding:
            raise VectorStoreError("Query embedding cannot be empty.")

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        total_count = self.count()
        if total_count == 0:
            return []

        effective_k = min(top_k, total_count)
        where_filter = {"source_file": source_file} if source_file else None

        try:
            results = self.collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=effective_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to query ChromaDB collection: {exc}") from exc

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        search_results: list[SearchResult] = []
        ids_list = results["ids"][0]
        docs_list = results["documents"][0] if results.get("documents") else []
        metas_list = results["metadatas"][0] if results.get("metadatas") else []
        distances_list = results["distances"][0] if results.get("distances") else []

        for cid, doc, meta, dist in zip(ids_list, docs_list, metas_list, distances_list):
            distance_val = float(dist) if dist is not None else 0.0

            # Distance thresholding: skip chunks further than threshold
            if distance_threshold is not None and distance_val > distance_threshold:
                continue

            # Cosine similarity calculation (1.0 = identical, 0.0 = orthogonal)
            sim_score = max(0.0, 1.0 - distance_val)

            search_results.append(
                SearchResult(
                    chunk_id=cid,
                    text=doc or "",
                    source_file=meta.get("source_file", "unknown") if meta else "unknown",
                    page_number=int(meta.get("page_number", 1)) if meta else 1,
                    distance=distance_val,
                    similarity_score=sim_score,
                )
            )

        return search_results

    def reset_collection(self) -> None:
        """
        Clears and resets the active collection, deleting all indexed chunks.
        """
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self._collection = None
        self._collection = self.get_or_create_collection(self.collection_name)

    def count(self) -> int:
        """
        Returns the number of documents currently indexed in the active collection.
        """
        try:
            return self.collection.count()
        except Exception as exc:
            raise VectorStoreError(f"Failed to count documents in ChromaDB collection: {exc}") from exc

    def list_sources(self) -> list[str]:
        """
        Returns a sorted list of unique document filenames indexed in the collection.
        """
        try:
            if self.count() == 0:
                return []
            data = self.collection.get(include=["metadatas"])
            if not data or not data.get("metadatas"):
                return []
            sources = {
                meta.get("source_file")
                for meta in data["metadatas"]
                if meta and "source_file" in meta
            }
            return sorted([s for s in sources if s])
        except Exception as exc:
            raise VectorStoreError(f"Failed to list sources from ChromaDB collection: {exc}") from exc
