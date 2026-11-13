"""Application-wide exceptions."""


class DocumentNotFoundError(LookupError):
    """Raised when a document does not exist."""
