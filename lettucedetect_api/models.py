from pydantic import BaseModel


class DetectionRequest(BaseModel):
    """Request model for hallucination detection.

    A request contains a list of contexts, a question and an answer. This
    mirrors the interface of the `HallucinationDetector`.
    """

    contexts: list[str]
    question: str
    answer: str


class TokenDetectionItem(BaseModel):
    """Response model list item of token-level detection."""

    token: str
    hallucination_score: float


class SpanDetectionItem(BaseModel):
    """Response model list item of span-level detection."""

    start: int
    end: int
    text: str
    hallucination_score: float


class TokenDetectionResponse(BaseModel):
    """Response model for token-level detection."""

    predictions: list[TokenDetectionItem]


class SpanDetectionResponse(BaseModel):
    """Response model for span-level detection."""

    predictions: list[SpanDetectionItem]
