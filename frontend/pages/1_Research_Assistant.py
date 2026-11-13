"""Research Assistant: ask questions about uploaded papers."""

import streamlit as st
from api_client import ApiError, ResearchRadarClient
from components.chat import render_answer, render_history

st.set_page_config(page_title="Research Assistant", layout="wide")

st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("flash", [])

client = ResearchRadarClient()


def index_files(files) -> None:
    """Upload each selected PDF and remember the outcome to show it."""
    for file in files:
        with st.spinner(f"Indexing {file.name}... this can take a minute."):
            try:
                result = client.upload_document(file.name, file.getvalue())
                st.session_state.flash.append(
                    (
                        "success",
                        f"{file.name}: {result['page_count']} pages, "
                        f"{result['chunk_count']} passages indexed.",
                    )
                )
            except ApiError as exc:
                st.session_state.flash.append(
                    ("error", f"{file.name}: {exc}")
                )
    st.session_state.uploader_key += 1
    st.rerun()


st.title("Research Assistant")
st.caption("Ask questions about your papers. Answers cite document and page.")

if not client.is_healthy():
    st.error(
        f"Cannot reach the Research Radar API at {client.base_url}. "
        "Start the backend with: uvicorn app.api.main:app"
    )
    st.stop()

try:
    documents = client.list_documents()
except ApiError as exc:
    st.error(str(exc))
    st.stop()

selected_ids: list[str] = []

with st.sidebar:
    st.header("Papers")

    for kind, text in st.session_state.flash:
        (st.success if kind == "success" else st.error)(text)
    st.session_state.flash = []

    files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if st.button("Index papers", disabled=not files, type="primary"):
        index_files(files)

    st.subheader(f"Uploaded ({len(documents)})")
    if not documents:
        st.caption("No papers yet.")

    for document in documents:
        name_column, delete_column = st.columns([5, 1])
        with name_column:
            if st.checkbox(
                document["filename"],
                value=True,
                key=f"select_{document['document_id']}",
                help=(
                    f"{document['page_count']} pages, "
                    f"{document['chunk_count']} passages. "
                    "Untick to leave this paper out of the search."
                ),
            ):
                selected_ids.append(document["document_id"])
        with delete_column:
            if st.button(
                "",
                icon=":material/delete:",
                key=f"delete_{document['document_id']}",
                help="Delete this paper",
            ):
                try:
                    client.delete_document(document["document_id"])
                except ApiError as exc:
                    st.session_state.flash.append(("error", str(exc)))
                st.rerun()

    st.divider()
    top_k = st.slider(
        "Passages to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="How many passages are sent to the model as evidence.",
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if not documents:
    st.info("Upload one or more PDFs in the sidebar to get started.")
elif not selected_ids:
    st.warning("Tick at least one paper in the sidebar to search it.")

render_history(st.session_state.messages)

question = st.chat_input(
    "Ask a question about your papers",
    disabled=not selected_ids,
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your papers and writing an answer..."):
            try:
                response = client.ask(
                    question,
                    top_k=top_k,
                    document_ids=(
                        None
                        if len(selected_ids) == len(documents)
                        else selected_ids
                    ),
                )
                message = {"role": "assistant", "response": response}
            except ApiError as exc:
                response = None
                message = {"role": "assistant", "error": str(exc)}

        if response is None:
            st.error(message["error"])
        else:
            render_answer(response)

    st.session_state.messages.append(message)
