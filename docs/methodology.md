
## Document preprocessing methodology

Research Radar uses a page-aware document preprocessing pipeline.

The original page boundaries are preserved during extraction and chunking
to support evidence traceability and page-level citations.

Text cleaning normalizes line endings, whitespace, non-breaking spaces,
and common line-break hyphenation artifacts.

The initial chunking baseline uses a recursive character splitter
with a chunk size of 1200 characters and overlap of 200 characters.

Chunk size and overlap are treated as experimental parameters rather
than fixed optimal values.

Future experiments will compare alternative configurations using
retrieval metrics such as Precision@K, Recall@K, MRR, and nDCG@K.


## 3.1 Local Embedding Provider

Research Radar uses Ollama as its local model runtime and
EmbeddingGemma as its initial text embedding model.

The embedding provider is implemented in
`app/embeddings/provider.py`.

### Embedding workflow

1. Receive text or a batch of text chunks.
2. Send the input to the locally running Ollama service.
3. Generate numerical vector representations using EmbeddingGemma.
4. Validate the returned embeddings.
5. Return vectors to the application.

### Configuration

- Runtime: Ollama
- Initial model: EmbeddingGemma
- Model dimensions: 768
- Default Ollama host: http://localhost:11434

### Design decisions

The application uses an `EmbeddingProvider` interface to
separate embedding functionality from the underlying runtime.

The initial implementation uses Ollama, but this interface
allows alternative embedding providers to be introduced
without changing the rest of the application.

Batch embedding is supported to reduce the number of
requests when processing document chunks.

### Limitations

- Ollama must be running locally.
- The embedding model must be downloaded before use.
- Document and query embeddings must use the same model
  and compatible embedding configuration.
