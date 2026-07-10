# LettuceDetect roadmap

LettuceDetect is maintained and evidence-driven. This roadmap separates near-term release work from research directions, and it intentionally has no delivery dates. Every item links to a GitHub issue where the discussion happens; milestones group them.

## Now: [v0.3 — Real batching, lighter installs](https://github.com/KRLabsOrg/LettuceDetect/milestone/1)

The next PyPI release is about making the existing detectors cheaper to install and faster to run, not about new detection methods.

- [Real batched inference](https://github.com/KRLabsOrg/LettuceDetect/issues/23): `predict_prompt_batch` currently loops one sample at a time; the plan is padded-batch tokenization with a single forward pass.
- [Lighter installs](https://github.com/KRLabsOrg/LettuceDetect/issues/69): lazy top-level imports so the LLM detector never touches torch. How to slim the default install without breaking existing users is an open design question; input welcome on the issue.
- [Fix tokens output from the LLM detector](https://github.com/KRLabsOrg/LettuceDetect/issues/65): `output_format="tokens"` should return token-level output, matching the transformer detector.
- [A `lettucedetect` CLI](https://github.com/KRLabsOrg/LettuceDetect/issues/47) and [a latency/throughput benchmark](https://github.com/KRLabsOrg/LettuceDetect/issues/58): both have contributor PRs in review.
- [Document `min_confidence` edge cases](https://github.com/KRLabsOrg/LettuceDetect/issues/64): contributor PR in review.

## Next: [Typed spans in the wild](https://github.com/KRLabsOrg/LettuceDetect/milestone/3)

The detectors already emit typed spans (category and subcategory per detected span); this milestone makes that visible and usable in real workflows.

- [Typed spans in the Streamlit demo](https://github.com/KRLabsOrg/LettuceDetect/issues/57)
- [Dataset-level hallucination-rate evaluation and reporting](https://github.com/KRLabsOrg/LettuceDetect/issues/56)
- [A Claude Code hook that flags hallucinations in agent answers](https://github.com/KRLabsOrg/LettuceDetect/issues/50)

## Exploring: [safety spans and long context](https://github.com/KRLabsOrg/LettuceDetect/milestone/2)

A collaborator playground. These issues are experiments and design discussions, not committed release features or stable APIs. Reproducible comparisons, failure analyses, and short design notes are valuable contributions even when the result is negative.

- [Span-based safety classification](https://github.com/KRLabsOrg/LettuceDetect/issues/43) and [span-level supervision from Aegis 2.0](https://github.com/KRLabsOrg/LettuceDetect/issues/44): extending span detection from hallucination to safety dimensions.
- [Span-level prompt-injection and jailbreak detection](https://github.com/KRLabsOrg/LettuceDetect/issues/55)
- [Long-context bidirectional encoders](https://github.com/KRLabsOrg/LettuceDetect/issues/42)
- [Zero-shot token classification via label conditioning](https://github.com/KRLabsOrg/LettuceDetect/issues/70): GLiNER-style label conditioning, but token-level, so unseen span taxonomies work without retraining.
- [A rule-based detection tier](https://github.com/KRLabsOrg/LettuceDetect/issues/15): interpretable lexical checks (numbers, dates, entities unsupported by the context) as a zero-cost, torch-free first tier, potentially synthesized from RAGTruth annotations with [RuleChef](https://github.com/KRLabsOrg/rulechef).

## How this roadmap changes

- Items move from Exploring to Now only with evidence: a working prototype, a benchmark result, or a design note that survived discussion.
- Simpler baselines and negative results can narrow or stop a direction.
- Anything user-facing lands with documentation and an entry in the changelog.
- Suggestions belong in issues; the roadmap is updated when milestones change, not on a schedule.
