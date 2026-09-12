"""Custom exception hierarchy for the Byte-Level BPE Tokenizer project."""

class TokenizerBaseException(Exception):
    """Base exception for all BPE tokenizer errors."""
    def __init__(self, message: str = "", details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigValidationError(TokenizerBaseException):
    """Raised when YAML configuration validation fails."""
    pass


class CorpusDownloadError(TokenizerBaseException):
    """Raised when streaming or downloading dataset chunks fails."""
    pass


class CorpusBuildError(TokenizerBaseException):
    """Raised when assembling or cleaning the corpus fails."""
    pass


class BpeTrainingError(TokenizerBaseException):
    """Raised during vocabulary and merge rule training."""
    pass


class TokenizerInferenceError(TokenizerBaseException):
    """Raised during encoding or decoding."""
    pass


class SerializationError(TokenizerBaseException):
    """Raised when exporting or importing tokenizer models fails."""
    pass


class BenchmarkEvaluationError(TokenizerBaseException):
    """Raised during multi-domain benchmark evaluation or LM evaluation."""
    pass


class HubPublishError(TokenizerBaseException):
    """Raised when pushing tokenizer models or cards to Hugging Face Hub fails."""
    pass
