# Integrations

LettuceDetect can be used with popular LLM frameworks. Integration examples are available in the repository's `integrations/` directory (see the [GitHub repo](https://github.com/KRLabsOrg/LettuceDetect)).

## LangChain

Use LettuceDetect as a callback, chain component, or tool within LangChain pipelines.

```python
from lettucedetect.models.inference import HallucinationDetector

# Create detector
detector = HallucinationDetector(
    method="transformer",
    model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"
)

# Use in your LangChain pipeline
def check_hallucination(context, question, answer):
    spans = detector.predict(
        context=context, question=question,
        answer=answer, output_format="spans"
    )
    return len(spans) == 0  # True if no hallucinations detected
```

## Pydantic AI

Use LettuceDetect within Pydantic AI agents for structured hallucination checking.

## Haystack

Add LettuceDetect as a pipeline component in Haystack for post-generation verification.

## General Pattern

Any framework that gives you access to the retrieved context and generated answer can use LettuceDetect:

```python
from lettucedetect.models.inference import HallucinationDetector

detector = HallucinationDetector(
    method="transformer",
    model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"
)

# After your RAG pipeline generates an answer:
spans = detector.predict(
    context=retrieved_documents,
    question=user_query,
    answer=generated_answer,
    output_format="spans"
)

if spans:
    print(f"Found {len(spans)} hallucinated spans")
    for span in spans:
        print(f"  '{span['text']}' (confidence: {span['confidence']:.2f})")
```
