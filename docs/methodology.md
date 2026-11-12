
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
