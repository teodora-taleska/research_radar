"""HTTP client used by the Streamlit pages to talk to the FastAPI backend."""

import os

import requests

DEFAULT_API_URL = "http://127.0.0.1:8000"
UPLOAD_TIMEOUT_SECONDS = 600
ASK_TIMEOUT_SECONDS = 300
QUICK_TIMEOUT_SECONDS = 15


class ApiError(RuntimeError):
    """Raised when the backend cannot be reached or returns an error."""


def _error_detail(response: requests.Response) -> str:
    """Extract a readable message from an error response."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None

    if isinstance(detail, list):
        detail = "; ".join(str(item.get("msg", item)) for item in detail)

    return str(detail or f"The API returned status {response.status_code}.")


class ResearchRadarClient:
    """Thin wrapper around the Research Radar REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("RESEARCH_RADAR_API_URL", DEFAULT_API_URL)
        ).rstrip("/")
        self._session = session or requests.Session()

    def _request(
        self,
        method: str,
        path: str,
        timeout: float = QUICK_TIMEOUT_SECONDS,
        **kwargs,
    ) -> requests.Response:
        try:
            response = self._session.request(
                method,
                f"{self.base_url}{path}",
                timeout=timeout,
                **kwargs,
            )
        except requests.exceptions.ConnectionError as exc:
            raise ApiError(
                f"Cannot reach the Research Radar API at {self.base_url}. "
                "Is the backend running?"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise ApiError("The API took too long to respond.") from exc
        except requests.exceptions.RequestException as exc:
            raise ApiError(f"Request failed: {exc}") from exc

        if not response.ok:
            raise ApiError(_error_detail(response))

        return response

    def is_healthy(self) -> bool:
        """Return True if the backend answers its health check."""
        try:
            self._request("GET", "/health", timeout=5)
        except ApiError:
            return False
        return True

    def list_documents(self) -> list[dict]:
        """Return all uploaded documents, newest first."""
        return self._request("GET", "/documents").json()

    def upload_document(self, filename: str, content: bytes) -> dict:
        """Upload and index a PDF."""
        return self._request(
            "POST",
            "/documents/upload",
            timeout=UPLOAD_TIMEOUT_SECONDS,
            files={"file": (filename, content, "application/pdf")},
        ).json()

    def delete_document(self, document_id: str) -> None:
        """Delete a document and its indexed vectors."""
        self._request("DELETE", f"/documents/{document_id}")

    def ask(
        self,
        question: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> dict:
        """Ask a question and return the answer with its evidence."""
        payload: dict = {"question": question, "top_k": top_k}
        if document_ids:
            payload["document_ids"] = document_ids

        return self._request(
            "POST",
            "/research/ask",
            timeout=ASK_TIMEOUT_SECONDS,
            json=payload,
        ).json()
