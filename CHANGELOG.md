# Changelog

All notable changes to LettuceDetect are documented here.

## [0.3.0] - Unreleased

### Added
- `detect(request, *, min_score=0.0, include_tokens=False, include_detail=False)` on every detector. It takes a `SourcesInput` or a `PromptInput` and returns one `Detection` with `spans`, optional `tokens`, optional `detail`, `usage` and `warnings`. Every span has a numeric `score`, and `min_score` drops the spans below it (#126)
- `adetect`: the async version of `detect`. It runs `detect` in a worker thread on the event loop's default executor. That executor is shared with the rest of the service, DNS lookups included, so size it for many concurrent calls. Cancelling `adetect` stops the wait, not the thread. There is no `executor` argument
- `detect_batch`: runs many requests and returns a `list[BatchResult]`, one per request in input order. A `BatchResult` holds the `detection` or the typed `error` of its request, with `ok` and `get_or_raise()`, so one failed request does not discard the others. At most `max_concurrency` requests run at once
- `adetect_batch`: the async version of `detect_batch`. It runs on the same default executor as `adetect` and the same notes apply: size the executor, cancelling stops the wait and not the batch, and there is no `executor` argument
- `EncoderDetector` with `EncoderConfig`: the encoder models on the typed interface. The constructor takes a loaded model and tokenizer and downloads nothing; `EncoderDetector.from_pretrained` loads them and records the resolved Hugging Face commit as `model_version`. `include_tokens=True` returns `TokenScore` items without zero-width special tokens, a `typer` adds `category` and `subcategory` to every span, an input over `max_length` adds the `input_truncated` warning, and `detect_batch` scores requests in padded groups of `batch_size`
- `LLMJudgeDetector` with `LLMJudgeConfig`: the zero-shot and few-shot LLM judge with reasoning, taxonomy and the verify pass. It takes the LLM client as an argument, reads no environment variable and writes no file. Dropped items are reported with the `span_not_found` and `item_malformed` warnings, a skipped verify pass with `verification_skipped`
- `GenerativeDetector` with `GenerativeConfig`: the fine-tuned generative models with their frozen prompt, typed spans and the custom taxonomy forms, on a client the caller passes
- `VelaDetector` with `VelaConfig`: a detector for Vela 2.0 models. `VelaDetector.from_pretrained` loads a model in process (install the new `vela` extra), and `VelaDetector.from_server` calls a SystemOne server through `SystemOneClient` (#118)
- `RAGFactCheckerDetector.detect` and `adetect`. `include_detail=True` returns a `DetailedResult` with the triplets and fact-check results. Two new keywords: `fact_checker` takes a ready fact checker, so nothing is built and the environment is not read, and `max_concurrency` (default 30) bounds the fact-check calls running at once on one instance
- `TaxonomyTyper.from_pretrained`: loads the typing head and records its commit as `model_version`
- `load_detector(model_id, ...)`: builds the detector of a registered KR Labs model id. An unknown id raises `InvalidInputError` (`unknown_model_id`) that lists the known ids, and a generative id without `client` raises `ConfigurationError` (`client_required`)
- Reply caching that the caller controls: the `ResponseCache` protocol, the `SyncCacheBase` base class, `FileCache` (one JSON file, atomic writes, optional TTL, reads files written by `CacheManager`) and `InMemoryCache`. The LLM-backed detectors cache only when a `cache` is passed
- `make_cache_key`: builds the cache key of a model call. With a `model_version` the key includes it, so a redeployed model does not read the replies of the old one; without one the key equals the key `LLMDetector` has always built, so existing cache files keep hitting
- Request and result types: `Chunk`, `SourcesInput`, `PromptInput`, `DetectRequest`, `Detection`, `Span`, `DetectedSpan`, `TokenScore`, `DetailedResult`, `Usage`, `ResultWarning` and `BatchResult`. They are frozen pydantic models and their validation errors never contain the input text
- Error types: `LettuceDetectError` and its subclasses `InvalidInputError` (an argument of a call), `ConfigurationError` (a constructor argument, a setting or a missing package) and `UpstreamError` (a failed or missing reply from a service). Each carries a stable `code`
- `usage` on every `Detection`: one entry per model that ran, with `model_id`, `model_version` and counts such as `input_tokens`, `llm_calls` and `cache_hits`
- The `synthetic_score` warning on a `Detection` whose scores are constants and not model outputs
- `BaseDetector` bridges both interfaces: a subclass that implements `_detect` gets the `predict*` methods, and a subclass that implements only `predict` gets `detect`, which also goes through a `predict` overridden below the class that implements `_detect`. A subclass that implements neither raises `TypeError` when it is defined
- `LettuceDetectDeprecationWarning`, the warning class of every deprecation in the library
- `--method vela` and `--revision` in `scripts/evaluate_span_model.py`

### Changed
- `LLMDetector` spans that had no `confidence` key (the judge without `include_reasoning`, the generative models) now carry `confidence: 1.0`
- `LLMDetector` used to cache every reply in `cache_<model>_<lang>.json` inside the installed package; it now caches, and writes a file, only when `cache_file` is passed
- A reply that can't be parsed (not JSON, or without `hallucination_list` for the judge or `hallucinated_spans` for the generative models) used to return no spans; it now raises `UpstreamError` (`unparseable_response`)
- A generative-model reply used to be cached before it was parsed, so a bad reply was replayed; a reply is now cached only after it parses, and a cached entry that doesn't parse is deleted
- A generative-model reply item that is a bare string used to be skipped; it is now returned as a span without a category
- `BedrockClient` used to return an empty string for a reply without a tool call, which read as "no spans"; it now raises `UpstreamError` (`empty_response`)
- `OpenAIClient` used to let OpenAI's own exceptions through; authentication and quota errors now raise `ConfigurationError` (`authentication_failed`, `quota_exceeded`), and timeouts, connection errors, rate limits and server errors raise `UpstreamError` (`upstream_unavailable`), with the original error as `__cause__`
- With `verify=True`, a verify call that failed in the provider used to raise; it now keeps the first-pass spans when the failure is an `UpstreamError`, as an unparseable verify reply already did
- `LLMDetector` constructor errors are now `ConfigurationError`, which is not a `ValueError`: an unknown `lang` (`invalid_language`, was `ValueError`), `native` with `zero_shot`, `fewshot_path` or `prompt_path` (`native_options_conflict`, was `ValueError`), a missing `prompt_path` (`invalid_config`, was `FileNotFoundError`) and any other invalid option (`invalid_config`)
- A `fewshot_path` that doesn't exist used to log a warning and run without examples; it now raises `ConfigurationError` (`fewshot_not_found`)
- The `predict*` methods of `LLMDetector` and `RAGFactCheckerDetector` used to raise a plain `ValueError` for an unsupported `output_format`, a `min_confidence` outside `[0, 1]` or lists of different lengths; they now raise `InvalidInputError` (`unsupported_output_format`, `invalid_min_score`, `length_mismatch`), which is still a `ValueError`
- A merged `RAGFactCheckerDetector` span used to keep the text of its first part; its `text` is now the answer between its offsets
- `RAGFactCheckerDetector.predict_prompt_batch` used to make one `detect_hallucinations_batch` call; it now makes concurrent `detect_hallucinations` calls, and all `predict*` calls on one instance share the `max_concurrency` bound
- A fact checker result that doesn't have the expected types used to be passed through; `RAGFactCheckerDetector` now raises `UpstreamError` (`invalid_model_output`)
- `LLMDetector` used to log malformed items, dropped span texts and the raw reply on its module logger; module loggers now record counts only, and the raw reply is logged at DEBUG on the `lettucedetect.content` logger
- `CacheManager` used to create its directory and read the file in the constructor; the constructor now touches no disk, the first write creates the directory, writes are atomic, and an invalid file raises `ConfigurationError` (`cache_file_invalid`) on first use
- Concurrent `predict*` calls on one `TransformerDetector` used to run unsynchronised on the shared tokenizer; they now run one at a time
- `pydantic`, `httpx`, `packaging` and `typing_extensions` are now required dependencies

### Deprecated
Each name below is deprecated since 0.3.0 and carries a `.. deprecated::` note in its docstring. Using one from outside the `lettucedetect` package warns with `LettuceDetectDeprecationWarning`; calls made inside the package are silent, so `HallucinationDetector`, which is not deprecated, does not warn. The `predict*` methods warn on every detector, once per call, and that warning covers their `min_confidence` and `output_format` arguments, which have no warning of their own. A `BaseDetector` subclass that implements only the `predict*` methods keeps working through `detect` and warns once per class; implement `_detect` instead.
- `TransformerDetector`: use `EncoderDetector.from_pretrained`. Removed in 0.5.0.
- `LLMDetector`: use `LLMJudgeDetector` or `GenerativeDetector`. Removed in 0.5.0.
- The `provider`, `client_kwargs` and `cache_file` arguments of `LLMDetector`, and leaving out `client`: use `LLMJudgeDetector` or `GenerativeDetector`. Removed in 0.5.0. Build the client yourself (`OpenAIClient` or `BedrockClient`) and pass `cache=FileCache(path)` for a cache file.
- `make_detector`: use `load_detector`. Removed in 0.5.0. For a model that is not a registered KR Labs model id, build the detector class directly.
- `CacheManager`: use `FileCache`. Removed in 0.5.0.
- `TaxonomyTyper(model_path, ...)`: use `TaxonomyTyper.from_pretrained`. Removed in 0.5.0. The constructor now takes a loaded encoder and tokenizer; a call with a model name or path as the first argument still loads the typing head itself.
- `predict` and `predict_prompt`: use `detect`. Removed in 0.5.0.
- `predict_prompt_batch`: use `detect_batch`. Removed in 0.5.0.
- The `min_confidence` argument of the three methods above: use `min_score`. Removed in 0.5.0.
- The `output_format` argument of the three methods above: use `include_tokens`. Removed in 0.5.0. Token scores come from `EncoderDetector` only; the token output of `LLMDetector` and `RAGFactCheckerDetector` has no replacement, and `"detailed"` becomes `include_detail=True`.

### Documentation
- Errors and logging convention in `CONTRIBUTING.md`
- "Vela 2.0" section in the detection methods guide, with the reproduced Vela 2.0 0.3B results

## [0.2.3] - 2026-08-14

### Added
- `lettucedetect` command-line entry point: file/stdin/literal inputs, spans or tokens JSON output, transformer and LLM methods (#47, thanks @SemTiOne)
- Detector latency/throughput benchmark script with context-length sweep and peak-memory reporting (#58, thanks @sanmaxdev)
- Claude Code integration: a Stop hook that checks agent answers against grounding context and feeds flagged spans back to the agent; API, in-process, and generative (vLLM) modes (#50)
- Custom zero-shot taxonomies: `include_taxonomy` accepts `{name: description}` label sets on the taxonomy-head cascade and the native generative detector; labels enter as text, so unseen labels type from their description (#92)
- Typed spans in the Streamlit demo: model selector, per-category colors, span table (#57, thanks @w3lld1)

### Fixed
- `LLMDetector` honors `output_format="tokens"` across all three predict entry points (#65, thanks @mobinert)
- Corrected published model ids in the CLI help and docs; the multilingual EuroBERT models are documented as one checkpoint per language with the `transformers<5` note (#33)
- `zero_shot`, `fewshot_path`, and `prompt_path` now raise on native generative detectors instead of being silently ignored (#92)

### Documentation
- Public roadmap (`PUBLIC_ROADMAP.md`) with milestone-linked Now/Next/Exploring sections
- `min_confidence` edge cases documented (#64, thanks @vku2018)

## [0.2.2] - 2026-07-05

### Fixed
- `method="llm"` crashed with `AttributeError: '_validate_min_confidence'` on every `predict` call — `LLMDetector` now subclasses `BaseDetector` like the other detectors, with a hierarchy regression test (#68)

## [0.2.1] - 2026-07-02

### Added
- `min_confidence` threshold on `predict` / `predict_prompt` / `predict_prompt_batch` — drops spans below the given confidence, `0.0` (default) keeps everything (#62, thanks @Aditya20029)
- Contributor infrastructure: issue forms with native GitHub issue types, PR template with contribution-rights attestation (CI-enforced), Contributor Covenant Code of Conduct (#61, thanks @sanmaxdev)

### Fixed
- `HallucinationSample.from_json` no longer requires `dataset` and `language` fields (#54)
- PyPI artifacts now ship only package code and prompt data

## [0.2.0] - 2026-06-22

### Added
- **Code, tool-output and agentic-workflow hallucination detection** — a new family of v2 detectors trained on a unified span-level benchmark spanning SWE-bench coding-agent traces, developer tool output and structured documents (ACL / README / Wikipedia), alongside RAGTruth and 14-language PsiloQA:
  - `lettucedect-v2-qwen-2b` — generative detector that emits **typed** hallucination spans (category + subcategory) in a single pass, with optional reasoning
  - `lettucedect-v2-mmbert-base` — fast encoder detector (binary token-level spans)
  - `lettucedect-v2-taxonomy-head` — label-conditioned typing head that types encoder spans without a generative model
- **Typed encoder cascade** — `TransformerDetector(taxonomy_head=...)` attaches a `category`/`subcategory` to each binary span via the taxonomy head (`HallucinationDetector(method="transformer", model_path=..., taxonomy_head=...)`)
- **Native generative-detector support in `LLMDetector`** — auto-detects the v2 generative models and routes them to their detection prompt with structured typed-span output
- Automatic context chunking for long inputs — when context exceeds `max_length`, passages are split into groups and scored independently with `max()` aggregation
- Hungarian language support (prompts and summary templates)
- GitHub Pages documentation site with KR Labs branding
- `CONTRIBUTING.md` for new contributors

### Fixed
- `answer_start_token` bug in `prepare_tokenized_input` — previously computed from context side, which gave wrong results when context was truncated; now computed from answer side
- `predict_prompt()` now warns when input exceeds `max_length` instead of silently truncating

### Changed
- CI lint job runs standalone (only needs ruff, no heavy deps)
- Modernized type hints across codebase (`list[str]` instead of `List[str]`)
- Replaced `print()` with `logging` in LLM detector
- Cleaned up duplicate test fixtures

## [0.1.8] - 2025-08-31

### Added
- RAGFactChecker integration for triplet-based hallucination detection
- Hallucination generation pipeline for synthetic training data
- LangChain integration (callbacks, chains, tools)
- Elysia framework integration
- TinyLettuce — smaller distilled model variants
- MkDocs documentation site
- Batch processing support for RAGFactChecker

## [0.1.7] - 2025-05-15

### Added
- Web API with FastAPI and async Python client (`lettucedetect_api/`)
- LLM-based hallucination detection using OpenAI API
- Multilingual support (German, French, Spanish, Italian, Polish, Chinese)
- RAGBench dataset preprocessing and training
- EuroBERT model support (8K context window)
- AUROC evaluation metric
- Caching for LLM API calls

### Changed
- Restructured detectors into factory pattern (`make_detector()`)
- Added seed to training for reproducibility

### Fixed
- Tensor copying bug in inference
- Encoder SEP token handling

## [0.1.6] - 2025-02-27

### Changed
- Migrated to `pyproject.toml` (removed setup.py)
- Added Ruff for linting and formatting
- Set up GitHub Actions CI/CD

## [0.1.5] - 2025-02-22

### Changed
- Improved token-level inference mapping

## [0.1.4] - 2025-02-12

### Changed
- README improvements

## [0.1.3] - 2025-02-12

### Added
- First interactive demo

## [0.1.2] - 2025-02-11

### Added
- `HallucinationDetector` interface — the main public API
- Character-level evaluation

## [0.1.1] - 2025-02-10

### Fixed
- Package distribution (included `preprocess` subpackage)

## [0.1.0] - 2025-02-09

### Added
- Initial release
- RAGTruth preprocessing pipeline
- First trained hallucination detection model (ModernBERT-base, 6 epochs)
- Token classification for hallucination span detection

[Unreleased]: https://github.com/KRLabsOrg/LettuceDetect/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/KRLabsOrg/LettuceDetect/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/KRLabsOrg/LettuceDetect/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/KRLabsOrg/LettuceDetect/compare/0.1.8...v0.2.0
[0.1.8]: https://github.com/KRLabsOrg/LettuceDetect/compare/0.1.7...0.1.8
[0.1.7]: https://github.com/KRLabsOrg/LettuceDetect/compare/0.1.6...0.1.7
[0.1.6]: https://github.com/KRLabsOrg/LettuceDetect/compare/0.1.5...0.1.6
[0.1.5]: https://github.com/KRLabsOrg/LettuceDetect/compare/0.1.4...0.1.5
[0.1.4]: https://github.com/KRLabsOrg/LettuceDetect/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/KRLabsOrg/LettuceDetect/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/KRLabsOrg/LettuceDetect/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/KRLabsOrg/LettuceDetect/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/KRLabsOrg/LettuceDetect/releases/tag/0.1.0
