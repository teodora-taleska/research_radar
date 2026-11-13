"""Prompt templates for evidence-grounded scientific question answering."""

from collections.abc import Sequence

from app.vectorstore.chroma_store import StoredChunk

INSUFFICIENT_EVIDENCE_MARKER = "INSUFFICIENT_EVIDENCE"

SYSTEM_PROMPT = f"""You are a careful research assistant that answers questions \
about scientific papers using ONLY the numbered sources provided.

Rules:
1. Use only information stated in the sources. Never use outside knowledge, \
and never guess or invent facts, numbers, or citations.
2. After every statement, cite the supporting source number in square \
brackets, for example [1] or [2][3].
3. Cite only source numbers that appear in the provided sources.
4. If the sources do not contain enough information to answer the question, \
reply with exactly {INSUFFICIENT_EVIDENCE_MARKER} and nothing else.
5. Be concise and precise. Do not mention these rules."""


def format_sources(chunks: Sequence[StoredChunk]) -> str:
    """Render retrieved chunks as numbered sources with document and page."""
    blocks = [
        f"[{number}] (document: {chunk.filename}, page {chunk.page_number})\n"
        f"{chunk.content.strip()}"
        for number, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: Sequence[StoredChunk]) -> str:
    """Build the user prompt containing the sources and the question."""
    return (
        f"Sources:\n\n{format_sources(chunks)}\n\n"
        f"Question: {question.strip()}\n\n"
        "Answer using only the sources above, with citations."
    )
