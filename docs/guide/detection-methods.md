# Detection Methods

LettuceDetect supports three ways to detect hallucinations, each with different trade-offs.

## Transformer (Recommended)

Fine-tuned encoder models that classify each token in the answer as supported or hallucinated. Best balance of speed and accuracy.

```python
detector = HallucinationDetector(
    method="transformer",
    model_path="KRLabsOrg/lettucedetect-large-modernbert-en-v1"
)
```

**How it works:** The model reads the context, question, and answer together. It labels each answer token, then merges consecutive hallucinated tokens into character spans. A single forward pass — fast enough for production use (30-60 samples/sec on GPU).

**When to use:** Production systems, latency-sensitive applications, or when you need precise span locations.

## LLM-based

Uses OpenAI-compatible APIs (GPT-4, Claude, etc.) for hallucination detection. No fine-tuning needed.

```python
detector = HallucinationDetector(method="llm", model_path="gpt-4o-mini")
```

**How it works:** Sends context + question + answer to the LLM with a prompt requesting hallucination spans in a structured format.

**When to use:** Quick prototyping, or when you want the LLM to explain *why* something is hallucinated.

### Backends

Beyond OpenAI-compatible APIs, the detector now ships an `LLMClient` abstraction with an AWS Bedrock backend. Select it with `provider`, or pass your own `client` for a custom backend:

```python
detector = HallucinationDetector(
    method="llm",
    provider="bedrock",
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
    region_name="us-east-1",
)
```

The OpenAI backend uses strict JSON-schema structured output; Bedrock obtains the same structured output through a forced tool call.

### Reasoning and confidence

Set `include_reasoning=True` to make the model reason before it judges. The response opens with a claim-by-claim `analysis` scratchpad, and each span carries a short `reasoning`, a `confidence` score (0–1), and an `is_hallucination` verdict. Spans the model talks itself out of (verdict `false`) are discarded automatically, and the `confidence`/`reasoning` keys are added to the returned spans.

```python
detector = HallucinationDetector(
    method="llm",
    model="gpt-4.1-mini",
    include_reasoning=True,
    min_confidence=0.7,   # drop low-confidence spans
)
```

`min_confidence` filters out spans scored below the threshold (only meaningful together with `include_reasoning`).

### Taxonomy classification

Set `include_taxonomy=True` to classify each hallucination into one of the unified taxonomy categories, or pass a list of category names to use a custom set. The chosen label is returned under a `category` key on each span.

```python
detector = HallucinationDetector(method="llm", model="gpt-4.1-mini", include_taxonomy=True)
```

With `include_taxonomy=True` the unified taxonomy is used, with these categories:

| Category | Definition |
|---|---|
| `contradiction` | The span directly conflicts with the context: the context states one thing and the answer asserts a different, incompatible thing. |
| `unsupported_addition` | The span asserts something the context neither states nor implies — plausible, but not derivable from the context. |
| `fabricated_reference` | The span references a named element (identifier, section, file, entity, attribute) that does not appear anywhere in the context. |

Pass a `{category: description}` dict instead of `True` to classify against your own taxonomy. The descriptions are injected into the prompt so the model knows what each label means:

```python
detector = HallucinationDetector(
    method="llm",
    model="gpt-4.1-mini",
    include_taxonomy={
        "unsupported_claim": "a statement not backed by anything in the source",
        "contradiction": "directly contradicts the source",
        "wrong_entity": "right relation but the wrong name, place, or number",
    },
)
```

You can also pass a plain list of names; definitions are reused from the built-in taxonomy where the name matches and omitted otherwise.

### Verification pass

Set `verify=True` to run a second, deliberately strict pass that re-judges each flagged span against the source and drops the ones it cannot confirm. Spans the verifier does not return are kept, so a malformed verdict never silently discards confirmed detections.

```python
detector = HallucinationDetector(method="llm", model="gpt-4.1-mini", verify=True)
```

These options compose freely and are all cached, so repeated runs over the same inputs avoid redundant API calls.

## RAG Fact Checker

Triplet-based fact checking that breaks the answer into structured claims and verifies each one.

```python
detector = HallucinationDetector(method="rag_fact_checker", model_path="gpt-4o-mini")
```

**How it works:** Extracts (subject, predicate, object) claims from the answer, then checks each claim against the context.

**When to use:** When you want claim-level granularity and structured verification results.
