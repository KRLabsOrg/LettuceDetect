"""Request and result types shared by the detectors.

All models are frozen and hide the input in validation errors, so a validation error never
contains customer text (sources, question, answer or prompt).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator

Triplet = list[str]
"""A (subject, predicate, object) triplet produced by the RAG fact checker."""


class ResultWarning(BaseModel):
    """A note that a result is correct but degraded.

    :param code: stable snake_case identifier, for example ``input_truncated``.
    :param message: optional human-readable text without customer text.
    """

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    code: str = Field(min_length=1)
    message: str | None = None


class Chunk(BaseModel):
    """A piece of source text with an id.

    :param id: non-empty identifier of the source.
    :param text: the source text.
    :param metadata: free-form data about the source.
    :param source_start: position of the chunk in its source document, if known.
    :param source_end: end position of the chunk in its source document, if known.
    """

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    id: str = Field(min_length=1)
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_start: int | None = Field(default=None, ge=0)
    source_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_source_offsets_in_order(self) -> Chunk:
        if (
            self.source_start is not None
            and self.source_end is not None
            and self.source_end < self.source_start
        ):
            raise ValueError("source_end must not precede source_start")
        return self


class Span(BaseModel):
    """A part of a text, given by offsets in Unicode code points.

    :param start: first code point, inclusive.
    :param end: last code point, exclusive.
    :param text: the text between ``start`` and ``end``.
    :param score: score in [0, 1].
    """

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    start: int = Field(ge=0)
    end: int
    text: str
    score: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_text_fits_offsets(self) -> Span:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if len(self.text) != self.end - self.start:
            raise ValueError("text must be exactly as long as the span from start to end")
        return self


class DetectedSpan(Span):
    """A hallucinated span of the answer.

    :param category: taxonomy category, ``None`` when not produced.
    :param subcategory: taxonomy subcategory, ``None`` when not produced.
    :param reasoning: the model's explanation, ``None`` when not produced.
    """

    category: str | None = None
    subcategory: str | None = None
    reasoning: str | None = None


class TokenScore(BaseModel):
    """The score of one answer token.

    :param token: the token text.
    :param start: first code point of the token in the answer, inclusive.
    :param end: last code point of the token in the answer, exclusive.
    :param score: probability that the token is hallucinated.
    :param flagged: whether the detector marks the token as hallucinated.
    """

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    token: str
    start: int = Field(ge=0)
    end: int
    score: float = Field(ge=0.0, le=1.0)
    flagged: bool


class Usage(BaseModel):
    """What one model or service did during a call.

    :param model_id: id of the model, ``None`` for work without a model.
    :param model_version: version of the model, ``None`` when unknown.
    :param units: counts of work done, for example ``{"input_tokens": 3410}``.
    """

    # Older pydantic 2 releases reserve the "model_" prefix for their own methods.
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True, protected_namespaces=())

    model_id: str | None
    model_version: str | None
    units: dict[str, NonNegativeInt]


class Triplets(BaseModel):
    """The triplets the RAG fact checker worked with.

    :param answer: triplets of the answer.
    :param context: triplets of the sources.
    :param hallucinated: answer triplets that the sources do not support.
    """

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    answer: list[Triplet] = Field(default_factory=list)
    context: list[Triplet] = Field(default_factory=list)
    hallucinated: list[Triplet] = Field(default_factory=list)


class DetailedResult(BaseModel):
    """The evidence the RAG fact checker produced.

    :param triplets: the answer, context and hallucinated triplets.
    :param fact_check_results: for each answer triplet index, whether the sources support it.
    :param span_triplets: the source triplet of each span, parallel to ``Detection.spans``.
    """

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    kind: Literal["detailed"] = "detailed"
    triplets: Triplets
    fact_check_results: dict[int, bool] = Field(default_factory=dict)
    span_triplets: list[Triplet | None] = Field(default_factory=list)


class SourcesInput(BaseModel):
    """A request made of sources and an answer.

    :param sources: the sources the answer is checked against.
    :param answer: the answer to check.
    :param question: the question the answer responds to, ``None`` for summarisation.
    """

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    kind: Literal["sources"] = "sources"
    sources: list[Chunk]
    answer: str
    question: str | None = None


class PromptInput(BaseModel):
    """A request made of a full prompt and an answer.

    :param prompt: the full prompt the answer responds to.
    :param answer: the answer to check.
    """

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    kind: Literal["prompt"] = "prompt"
    prompt: str
    answer: str


DetectRequest = Annotated[SourcesInput | PromptInput, Field(discriminator="kind")]
"""A request to a detector, discriminated on ``kind``."""


class Detection(BaseModel):
    """The result of one detector call.

    :param spans: the hallucinated spans of the answer.
    :param tokens: token scores, ``None`` unless requested.
    :param detail: fact checker evidence, ``None`` unless requested.
    :param usage: what each model that took part did.
    :param warnings: notes that the result is degraded.
    """

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    spans: list[DetectedSpan] = Field(default_factory=list)
    tokens: list[TokenScore] | None = None
    detail: DetailedResult | None = None
    usage: list[Usage] = Field(default_factory=list)
    warnings: list[ResultWarning] = Field(default_factory=list)
