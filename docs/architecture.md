
## PDF Ingestion Architecture

The document ingestion pipeline is responsible for transforming uploaded
scientific PDF files into structured text chunks.

### Processing stages

1. PDF extraction:
   PyMuPDF extracts text from each PDF page independently.

2. Text cleaning:
   Extracted text is normalized to remove unnecessary whitespace,
   line-break artifacts, and common hyphenation problems.

3. Text chunking:
   LangChain RecursiveCharacterTextSplitter divides each page into
   overlapping text chunks.

4. Metadata preservation:
   Every chunk retains its page number, global chunk index, and
   character position within its source page.

### Initial chunking configuration

| Parameter | Value |
|---|---:|
| Chunk size | 1200 characters |
| Chunk overlap | 200 characters |
| Splitting strategy | Recursive |
| Page numbering | One-based |
| Chunk numbering | Zero-based |

These values represent an initial baseline and will be evaluated
against alternative configurations in the evaluation phase.

### API integration

The FastAPI endpoint POST /documents/upload accepts a PDF file
and returns document metadata and chunk previews.

### Current limitations

- Scanned PDFs are not processed because OCR is not implemented.
- Documents and chunks are not persisted yet.
- Uploaded PDFs are processed in memory.
- No embedding generation or semantic retrieval is performed yet.
