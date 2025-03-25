import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic_settings import BaseSettings

from lettucedetect.models.inference import HallucinationDetector
from lettucedetect_api.models import DetectionRequest, SpanDetectionResponse, TokenDetectionResponse


class Settings(BaseSettings):
    """Lettuce Web API configuration.

    Uses the built-in settings functionallity of FastAPI. The values of an
    instance of this class will automatically be populated with environment
    variables of the same name.
    """

    lettucedetect_model: str = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
    lettucedetect_method: str = "transformer"


settings = Settings()
detector: HallucinationDetector | None = None


@asynccontextmanager
async def init_detector(app: FastAPI) -> AsyncIterator[None]:
    """Load model and initialize hallucination detector object.

    This fastapi livespan event is run once during fastapi startup. It is used
    to load and initialize the hallucination detector. All subsequent requests
    can then use the hallucination detector without repeating the initialization
    steps over and over again.

    :param app: The FastAPI object for this livespan event.
    """
    global detector
    detector = HallucinationDetector(
        method=settings.lettucedetect_method,
        model_path=settings.lettucedetect_model,
    )
    yield


app = FastAPI(lifespan=init_detector)
detector_lock = asyncio.Lock()


async def run_detector_safe(request: DetectionRequest, output_format: str) -> dict:
    """Run detector safely in a async environment without blocking."""
    async with detector_lock:
        preds = await run_in_threadpool(
            detector.predict,
            context=request.contexts,
            question=request.question,
            answer=request.answer,
            output_format=output_format,
        )
    return preds


@app.post(
    "/v1/lettucedetect/token",
    response_model=TokenDetectionResponse,
    summary="Run token-level hallucination detection.",
)
async def run_token_detection(request: DetectionRequest) -> dict:
    """Run token-level hallucination detection.

    Predicts hallucination scores for each token in `answer`. A higher score
    correlates to a higher probability that this token is hallucinated.
    """
    preds = await run_detector_safe(request, output_format="tokens")
    preds_converted = [{"token": p["token"], "hallucination_score": p["prob"]} for p in preds]
    return {"predictions": preds_converted}


@app.post(
    "/v1/lettucedetect/spans",
    response_model=SpanDetectionResponse,
    summary="Run span-level hallucination detection.",
)
async def run_span_detection(request: DetectionRequest) -> dict:
    """Run span-level hallucination detection.

    Predicts hallucination scores for spans of text in `answer`. A higher score
    correlates to a higher probability that this token is hallucinated.  The
    hallucination score of a span corresponds to the highest hallucination score
    of the tokens part of the span.
    """
    preds = await run_detector_safe(request, output_format="spans")
    preds_converted = [
        {
            "start": p["start"],
            "end": p["end"],
            "text": p["text"],
            "hallucination_score": p["confidence"],
        }
        for p in preds
    ]
    return {"predictions": preds_converted}
