"""
Gemini Client Module.

Responsible for:
- Initializing the official Google GenAI SDK client securely
- Generating embeddings for document chunks and user queries using gemini-embedding-2
- Handling batch chunk embeddings with asymmetric retrieval task types (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY)
- Generating grounded text completions using the configured Gemini Flash model
- Providing robust error handling and API key sanitization
"""

import os
import time
from typing import Any

from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL, GEMINI_MODEL


class GeminiClientError(Exception):
    """Base exception for all Gemini client errors."""
    pass


class GeminiConfigurationError(GeminiClientError):
    """Raised when configuration or API key is missing or invalid."""
    pass


class GeminiAPIError(GeminiClientError):
    """Raised when an error occurs during an API call to Gemini."""
    pass


def _sanitize_error(message: str, api_key: str | None = None) -> str:
    """
    Strips raw API keys from exception messages to prevent accidental leakage in logs or UI.
    """
    sanitized = message
    if api_key and api_key in sanitized:
        sanitized = sanitized.replace(api_key, "[REDACTED_API_KEY]")
    if GEMINI_API_KEY and GEMINI_API_KEY in sanitized:
        sanitized = sanitized.replace(GEMINI_API_KEY, "[REDACTED_API_KEY]")
    return sanitized


def get_gemini_client(api_key: str | None = None) -> genai.Client:
    """
    Initializes and returns a Google GenAI Client instance.

    Args:
        api_key: Optional explicit API key. If omitted, uses GEMINI_API_KEY from config.

    Returns:
        An initialized genai.Client instance.

    Raises:
        GeminiConfigurationError: If no API key is configured.
    """
    key = (api_key or GEMINI_API_KEY).strip()
    if not key:
        raise GeminiConfigurationError(
            "GEMINI_API_KEY is not configured. Please add your Gemini API key to the .env file "
            "or set the GEMINI_API_KEY environment variable."
        )

    try:
        return genai.Client(api_key=key)
    except Exception as exc:
        clean_msg = _sanitize_error(str(exc), key)
        raise GeminiConfigurationError(f"Failed to initialize Gemini Client: {clean_msg}") from exc


def embed_texts(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    model: str = GEMINI_EMBEDDING_MODEL,
    batch_size: int = 50,
    client: genai.Client | None = None,
) -> list[list[float]]:
    """
    Generates vector embeddings for a list of text strings in configurable batches.

    Args:
        texts: List of text strings to embed.
        task_type: Retrieval task type ('RETRIEVAL_DOCUMENT', 'RETRIEVAL_QUERY', etc.).
        model: Gemini embedding model name (defaults to GEMINI_EMBEDDING_MODEL).
        batch_size: Maximum number of texts per API request (defaults to 50).
        client: Optional pre-initialized genai.Client instance.

    Returns:
        List of embedding vectors (each vector is a list of floats) matching input order.

    Raises:
        ValueError: If batch_size <= 0.
        GeminiAPIError: If the API request fails or returns unexpected data.
    """
    if not texts:
        return []

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    cli = client or get_gemini_client()
    config = types.EmbedContentConfig(task_type=task_type)
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        contents_payload = [types.Content(parts=[types.Part.from_text(text=t)]) for t in batch]
        try:
            response = cli.models.embed_content(
                model=model,
                contents=contents_payload,
                config=config,
            )

            if not response.embeddings:
                raise GeminiAPIError(
                    f"API returned no embeddings for batch starting at index {i}."
                )

            for emb in response.embeddings:
                if emb.values is None:
                    raise GeminiAPIError(
                        f"Received null embedding values from API for batch at index {i}."
                    )
                all_embeddings.append(list(emb.values))

        except GeminiClientError:
            raise
        except Exception as exc:
            clean_msg = _sanitize_error(str(exc))
            raise GeminiAPIError(
                f"Error embedding texts batch ({i} to {i + len(batch)}) using model '{model}': {clean_msg}"
            ) from exc

    return all_embeddings


def embed_query(
    query: str,
    model: str = GEMINI_EMBEDDING_MODEL,
    client: genai.Client | None = None,
) -> list[float]:
    """
    Generates a vector embedding for a single user query using task_type='RETRIEVAL_QUERY'.

    Args:
        query: The user question or query string.
        model: Gemini embedding model name (defaults to GEMINI_EMBEDDING_MODEL).
        client: Optional pre-initialized genai.Client instance.

    Returns:
        A list of floats representing the query embedding vector.

    Raises:
        ValueError: If query is empty.
        GeminiAPIError: If the embedding generation fails.
    """
    if not query or not query.strip():
        raise ValueError("Cannot generate embedding for an empty query.")

    results = embed_texts(
        texts=[query.strip()],
        task_type="RETRIEVAL_QUERY",
        model=model,
        batch_size=1,
        client=client,
    )

    if not results:
        raise GeminiAPIError("Failed to generate embedding for query.")

    return results[0]


def _is_transient_error(exc: Exception) -> bool:
    """
    Determines if an API error is transient and safe to retry (e.g., 503 UNAVAILABLE, 429 RATE_LIMIT).
    Permanent errors (400 Invalid Argument, 401/403 Authentication, 404 Not Found) are not retried.
    """
    code = getattr(exc, "code", None)
    if code in (503, 502, 504, 429):
        return True

    err_str = str(exc).upper()

    # Fast check for permanent errors that should never be retried
    permanent_signals = (
        "400",
        "401",
        "403",
        "404",
        "INVALID_ARGUMENT",
        "PERMISSION_DENIED",
        "UNAUTHENTICATED",
        "NOT_FOUND",
    )
    if any(sig in err_str for sig in permanent_signals):
        return False

    transient_signals = (
        "503",
        "UNAVAILABLE",
        "HIGH DEMAND",
        "RESOURCE_EXHAUSTED",
        "429",
        "RATE LIMIT",
        "QUOTA EXCEEDED",
        "TEMPORARILY UNAVAILABLE",
        "DEADLINE_EXCEEDED",
    )
    return any(sig in err_str for sig in transient_signals)


def generate_response(
    prompt: str,
    system_instruction: str | None = None,
    temperature: float = 0.0,
    model: str = GEMINI_MODEL,
    client: genai.Client | None = None,
    max_retries: int = 3,
    initial_delay: float = 1.0,
) -> str:
    """
    Generates a text completion using the configured Gemini generation model,
    with automatic exponential backoff retry for transient 503/UNAVAILABLE errors.

    Args:
        prompt: The augmented user prompt containing context and question.
        system_instruction: Optional system instruction for grounding and behavior.
        temperature: Sampling temperature (0.0 for deterministic, factual outputs).
        model: Gemini model identifier (defaults to GEMINI_MODEL).
        client: Optional pre-initialized genai.Client instance.
        max_retries: Maximum number of retry attempts for transient errors (defaults to 3).
        initial_delay: Initial delay in seconds before the first retry (defaults to 1.0s).

    Returns:
        The generated response string.

    Raises:
        ValueError: If prompt is empty.
        GeminiAPIError: If the generation request fails permanently or retries are exhausted.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    cli = client or get_gemini_client()

    config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction if system_instruction else None,
    )

    delay = initial_delay
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = cli.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return response.text or ""
        except Exception as exc:
            last_exc = exc
            # Only retry if transient and retries remain
            if attempt < max_retries and _is_transient_error(exc):
                # If quota limit is hit (e.g. 20 reqs/day on free tier), sleeping seconds won't help; try available flash model
                if "RESOURCE_EXHAUSTED" in str(exc) or getattr(exc, "code", None) == 429:
                    for fb_model in ("gemini-3.5-flash-lite", "gemini-flash-latest"):
                        if fb_model != model:
                            try:
                                fb_resp = cli.models.generate_content(
                                    model=fb_model,
                                    contents=prompt,
                                    config=config,
                                )
                                return fb_resp.text or ""
                            except Exception:
                                pass
                time.sleep(delay)
                delay *= 2  # Exponential backoff (1s -> 2s -> 4s)
                continue

            clean_msg = _sanitize_error(str(exc))
            raise GeminiAPIError(
                f"Error generating response from Gemini model '{model}': {clean_msg}"
            ) from exc

    if last_exc:
        clean_msg = _sanitize_error(str(last_exc))
        raise GeminiAPIError(
            f"Error generating response from Gemini model '{model}' after {max_retries} retries: {clean_msg}"
        ) from last_exc

    return ""
