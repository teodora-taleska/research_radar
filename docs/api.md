
## Document endpoints

### POST /documents/upload

Uploads and processes a PDF document.

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
- status
- chunk previews

**Possible errors:**

| Status | Meaning |
|---|---|
| 400 | Empty upload |
| 413 | File exceeds the upload limit |
| 415 | Unsupported file type |
| 422 | Invalid PDF or no extractable text |
