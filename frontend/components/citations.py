"""Display of cited sources and retrieved evidence."""

import streamlit as st


def citation_label(citation: dict) -> str:
    """Return the heading for a cited source, e.g. '[1] paper.pdf - page 3'."""
    return (
        f"[{citation['reference']}] {citation['filename']} "
        f"- page {citation['page_number']}"
    )


def passage_label(index: int, passage: dict) -> str:
    """Return the heading for a retrieved passage."""
    return (
        f"{index}. {passage['filename']} - page {passage['page_number']} "
        f"(distance {passage['distance']:.3f})"
    )


def render_citations(citations: list[dict]) -> None:
    """Show each cited source with its original text from the paper."""
    for citation in citations:
        with st.expander(citation_label(citation)):
            st.text(citation["excerpt"])
            st.caption(
                f"Distance {citation['distance']:.3f} (lower means closer "
                "to the question)"
            )


def render_retrieved_passages(passages: list[dict]) -> None:
    """Show every passage the search returned, cited or not."""
    with st.expander(f"All retrieved passages ({len(passages)})"):
        for index, passage in enumerate(passages, start=1):
            st.markdown(f"**{passage_label(index, passage)}**")
            st.text(passage["content"])
