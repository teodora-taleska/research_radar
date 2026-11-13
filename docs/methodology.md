
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


## 3.2 Vector storage

Chunks are stored in one persistent ChromaDB collection configured for
cosine distance. Each entry keeps the chunk text, its embedding and the
metadata needed for citations: document id, filename, page number and
chunk index. Search returns the stored text and metadata together with the
cosine distance (0 is identical; lower means closer).

## 4. Evidence-grounded answer generation

The retrieved chunks are given to the language model as numbered sources
with their document name and page. The system prompt requires the model to
use only those sources, to cite every statement as `[n]`, and to reply with
a fixed marker when the sources are not sufficient.

The model output is not trusted for citations. After generation:

- citation numbers that do not match a supplied source are discarded;
- an answer with no valid citation is rejected;
- a refusal marker, or an empty retrieval, produces an explicit
  insufficient-evidence result.

So every citation shown to the user refers to a passage that was actually
retrieved. This verifies that a citation exists, not that the claim is
faithful to it. Faithfulness is assessed separately (manual scoring in the
planned LLM comparison experiment).

Reasoning models such as Qwen3 may emit chain-of-thought text. The LLM
provider strips it, so only the final answer reaches the user.

Generation uses a low temperature (0.1) to favour consistent, factual
answers.

## 5. Retrieval evaluation

Retrieval is evaluated on a labelled benchmark (see
`evaluation/datasets/README.md`). Relevance is labelled per paper page, and
a retrieved passage is relevant if its (paper, page) is labelled relevant.

| Metric | Definition |
|---|---|
| Precision@K | Relevant passages in the top K divided by K |
| Recall@K | Labelled relevant pages found in the top K divided by the number of labelled pages |
| Latency | Time to embed the question and search: mean, median and 95th percentile |

Unanswerable questions are excluded from these metrics and reserved for
refusal testing.

Limitations of the method:

- Page-level relevance is lenient: a passage from a relevant page counts
  even if it is not the sentence containing the answer.
- With one relevant page per question, Precision@K cannot exceed 1/K.
- Results depend on the size and difficulty of the benchmark.
- MRR and nDCG are not reported yet.

## 6. Language model comparison

Language models are compared on the same benchmark with identical retrieved
passages, so only the generator differs. For each model the evaluation
records latency, whether it answered or declined, and whether its citations
point to labelled relevant pages (see `docs/experiments.md` for the metric
definitions).

Faithfulness cannot be computed reliably by simple text matching, so it is
scored by a person on a 1 to 5 scale (5: every claim supported by the cited
text; 1: wrong or contradicted), comparing each answer with the cited passage
and the recorded expected answer. This is a subjective judgement by one
scorer on a small sample.

Limitations:

- The automatic citation checks only verify that a cited page is a labelled
  one, not that the claim is correct.
- Correct refusal is measured on only 4 unanswerable questions, so one
  different answer changes the rate by 25 percentage points.
- Latency depends on the hardware and on whether the model was already
  loaded. Models are loaded before timing starts.
- Results apply to the prompt, the top-5 retrieval and the sampling
  temperature (0.1) used here.
