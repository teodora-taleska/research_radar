"""Chat message rendering for the Research Assistant."""

import streamlit as st
from components.citations import render_citations, render_retrieved_passages


def render_answer(response: dict) -> None:
    """Render a generated answer, its citations and retrieved evidence."""
    if not response["sufficient_evidence"]:
        st.warning(response["answer"])
    else:
        st.markdown(response["answer"])
        st.caption(
            "Generated from the passages below. Check the cited text "
            "before relying on a claim."
        )
        render_citations(response["citations"])

    if response["retrieved_passages"]:
        render_retrieved_passages(response["retrieved_passages"])


def render_history(messages: list[dict]) -> None:
    """Render all stored chat messages."""
    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            elif "error" in message:
                st.error(message["error"])
            else:
                render_answer(message["response"])
