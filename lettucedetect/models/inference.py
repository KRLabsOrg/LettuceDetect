# lettucedetect/inference.py
"""Public façade for LettuceDetect.

Down-stream code should keep importing **HallucinationDetector** from here;
the concrete detector classes now live in :pymod:`lettucedetect.detectors.*`.
Nothing in the public API has changed.
"""

from lettucedetect.detectors.factory import make_detector


class HallucinationDetector:
    """Facade class that delegates to a concrete detector chosen by *method*.

    :param method: ``"transformer"`` (token-classifier) or ``"llm"`` (OpenAI function-calling).
    :param kwargs: Passed straight through to the chosen detector's constructor.
    """

    def __init__(self, method: str = "transformer", **kwargs) -> None:
        """Initialize the detector.

        :param method: Detection method to use.
        :param kwargs: Passed to the detector constructor.
        """
        self.detector = make_detector(method, **kwargs)

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
        min_confidence: float = 0.0,
    ) -> list:
        """Predict hallucination tokens or spans given passages and an answer.

        This is the call most RAG pipelines use.

        :param min_confidence: Drop ``"spans"`` whose ``confidence`` is below this threshold
            (in ``[0, 1]``; ``0.0`` keeps every span). Ignored for ``"tokens"`` output.

        See the concrete detector docs for the structure of the returned list.
        """
        return self.detector.predict(
            context, answer, question, output_format, min_confidence=min_confidence
        )

    def predict_prompt(
        self, prompt: str, answer: str, output_format: str = "tokens", min_confidence: float = 0.0
    ) -> list:
        """Predict hallucinations when you already have a *single* full prompt string.

        :param prompt: The prompt string.
        :param answer: The answer string.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        :param min_confidence: Drop ``"spans"`` below this confidence threshold (``[0, 1]``).
        """
        return self.detector.predict_prompt(
            prompt, answer, output_format, min_confidence=min_confidence
        )

    def predict_prompt_batch(
        self,
        prompts: list[str],
        answers: list[str],
        output_format: str = "tokens",
        min_confidence: float = 0.0,
    ) -> list:
        """Batch version of :py:meth:`predict_prompt`.

        Length of *prompts* and *answers* must match.

        :param prompts: List of prompt strings.
        :param answers: List of answer strings.
        :param output_format: "tokens" to return token-level predictions, or "spans" to return grouped spans.
        :param min_confidence: Drop ``"spans"`` below this confidence threshold (``[0, 1]``).
        """
        return self.detector.predict_prompt_batch(
            prompts, answers, output_format, min_confidence=min_confidence
        )
