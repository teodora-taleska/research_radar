"""Research Radar home page."""

import streamlit as st

st.set_page_config(page_title="Research Radar", layout="wide")

st.title("Research Radar")
st.subheader("Scan research. Discover evidence. Save time.")

st.write(
    "Research Radar helps you work with scientific papers. Upload PDFs, "
    "ask questions in plain language, and get answers grounded in your "
    "papers, with the document and page for every claim. When your papers "
    "do not contain the answer, it says so instead of guessing."
)

st.markdown(
    """
**How to use it**

1. Open **Research Assistant** in the sidebar.
2. Upload one or more PDFs and click **Index papers**.
3. Ask a question and check the cited passages.

Everything runs locally: your papers never leave your computer.
"""
)

st.info(
    "Planned for version 2: structured paper summaries, side-by-side "
    "paper comparison, and saved conversations."
)
