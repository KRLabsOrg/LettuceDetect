# Detection Methods

LettuceDetect supports four ways to detect hallucinations, each with different trade-offs.

## Transformer (Recommended)

Fine-tuned encoder models that classify each token in the answer as supported or hallucinated. Best balance of speed and accuracy.

```python
detector = HallucinationDetector(
    method="transformer",
    model_path="KRLabsOrg/lettucedect-large-modernbert-en-v1"
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

## Vela 2.0

[Vela 2.0](https://huggingface.co/collections/vllm-sr/vela-20) models (a 0.3B encoder; 0.8B, 4B and 9B decoders) answer typed SystemOne questions about a request, a context and an answer. `VelaDetector` asks the hallucination span question and returns the unsupported spans of the answer with their probability as score.

Run the model in-process. Its code ships in the model repository and runs with `trust_remote_code=True`, so pin `revision` to a commit you have reviewed:

```python
from lettucedetect.detectors import Chunk, SourcesInput, VelaDetector

detector = VelaDetector.from_pretrained("vllm-sr/Vela-2.0-0.3B", revision="<commit>")
result = detector.detect(
    SourcesInput(
        question="When does it ship?",
        sources=[Chunk(id="doc-1", text="Ships Friday.")],
        answer="Ships Monday.",
    ),
    min_score=0.5,
)
```

Or call a SystemOne server (`POST /v1/systemone`), such as the `vela2_serve.py` script in the model repository. The URL and the key come from the caller; nothing is read from the environment:

```python
detector = VelaDetector.from_server(
    "http://127.0.0.1:8001", model="vllm-sr/Vela-2.0-0.3B", api_key="<key>"
)
```

A rejected key raises `ConfigurationError`. A timeout, a connection error, a rate limit, a server error or an unreadable reply raises `UpstreamError`, with `retryable` set when a retry can help. Install the in-process runtime with `pip install 'lettucedetect[vela]'`; the server mode needs no extra.

### When to choose Vela

Hallucination example-F1 on the 10,698-example test set and on RAGTruth, as reported by Vela ([EVALUATION.md](https://huggingface.co/vllm-sr/Vela-2.0-4B/blob/main/EVALUATION.md)); the 0.3B number is reproduced in [Vela 2.0 0.3B results](../code-hallucination/results_vela.md):

| Model | 10,698 examples | RAGTruth |
|---|--:|--:|
| Vela 2.0 0.3B (encoder, CPU) | 0.848 | 0.706 |
| Vela 2.0 4B | 0.874 | 0.770 |
| Vela 2.0 9B | 0.885 | 0.774 |
| lettucedect-v2-mmbert-base | 0.869 | 0.743 |
| lettucedect-v2-qwen-2b | 0.921 | 0.817 |

**Choose Vela** when one model should answer more than hallucination: the same Vela call can also return PII spans and prompt-attack and safety scores (not yet exposed by `VelaDetector`), or when you need a CPU model with an 8,192-token window between the encoder and generative lines.

**Choose the LettuceDetect models** for the best hallucination F1, and when you need typed categories: they label each span with the hallucination taxonomy, while Vela returns one `unsupported` label.
