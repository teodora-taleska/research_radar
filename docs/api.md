
## Document endpoints

### POST /documents/upload

Uploads a PDF, processes it, then embeds and indexes its chunks.

**Request:** multipart/form-data

| Field | Type | Required |
|---|---|---|
| file | PDF file | Yes |

**Successful response:** HTTP 201

The response includes:
- document_id
- filename
- page_count
- chunk_count
- status (`indexed`)
- chunk previews

**Possible errors:**

| Status | Meaning |
|---|---|
| 400 | Empty upload |
| 413 | File exceeds the upload limit (20 MB) |
| 415 | Unsupported file type |
| 422 | Invalid PDF or no extractable text |
| 503 | The document could not be indexed (for example Ollama is not running) |

### GET /documents

Returns all uploaded documents, newest first. Each item contains
`document_id`, `filename`, `page_count`, `chunk_count`, `status` and
`created_at`.

### GET /documents/{document_id}

Returns one document. Status 404 if it does not exist.

### DELETE /documents/{document_id}

Deletes the document's vectors and its record. Status 204 on success,
404 if it does not exist, 503 if the vectors could not be removed (the
record is kept so the deletion can be retried).

## Research endpoint

### POST /research/ask

Answers a question using only the indexed documents.

**Request:** JSON

| Field | Type | Required | Notes |
|---|---|---|---|
| question | string | Yes | 1 to 1000 characters, whitespace trimmed |
| top_k | integer | No | 1 to 20, default 5 |
| document_ids | list of strings | No | Restrict the search to these documents |

**Successful response:** HTTP 200

| Field | Meaning |
|---|---|
| question | The trimmed question |
| answer | Generated answer, or the insufficient-evidence message |
| sufficient_evidence | `false` when the documents do not support an answer |
| citations | Verified sources: `reference`, `document_id`, `filename`, `page_number`, `chunk_index`, `excerpt`, `distance` |
| retrieved_passages | Every retrieved chunk with its `distance` (lower is closer) |

**Possible errors:**

| Status | Meaning |
|---|---|
| 422 | Invalid request (empty question, `top_k` out of range) |
| 503 | Ollama or the vector store is unavailable |
