# LettuceDetect roadmap

LettuceDetect is maintained and evidence-driven. This roadmap separates near-term release work from research directions, and it intentionally has no delivery dates. Every item links to a GitHub issue where the discussion happens; milestones group them.

## Now: [v0.3 — Faster, lighter, consistent inference](https://github.com/KRLabsOrg/LettuceDetect/milestone/1)

The next PyPI release is about making the existing detectors more reliable, cheaper to install, and faster to run—not about adding new detection methods.

Open:

- [Real batched inference](https://github.com/KRLabsOrg/LettuceDetect/issues/23): `predict_prompt_batch` currently loops one sample at a time; the plan is padded-batch tokenization with a single forward pass.
- [Lighter installs](https://github.com/KRLabsOrg/LettuceDetect/issues/69): lazy top-level imports so the LLM detector never touches torch. How to slim the default install without breaking existing users is an open design question; input welcome on the issue.
- [Define evidence aggregation for chunked inference](https://github.com/KRLabsOrg/LettuceDetect/issues/74): benchmark whether evidence in any chunk or every chunk is required before changing the current conservative behavior.
- [Map repeated LLM span strings to distinct answer occurrences](https://github.com/KRLabsOrg/LettuceDetect/issues/85)

Shipped in this milestone so far: the network-free unit suite as a merge gate ([#73](https://github.com/KRLabsOrg/LettuceDetect/issues/73)), the `lettucedetect` CLI ([#47](https://github.com/KRLabsOrg/LettuceDetect/issues/47)), the detector latency/throughput benchmark ([#58](https://github.com/KRLabsOrg/LettuceDetect/issues/58)), token output from the LLM detector ([#65](https://github.com/KRLabsOrg/LettuceDetect/issues/65)), and the `min_confidence` documentation ([#64](https://github.com/KRLabsOrg/LettuceDetect/issues/64)).

## Next: [Validation — Typed-span workflows](https://github.com/KRLabsOrg/LettuceDetect/milestone/3)

The detectors already emit typed spans (category and subcategory per detected span). This milestone tests whether that localization improves real debugging, evaluation, monitoring, and agent workflows.

- [Dataset-level hallucination-rate evaluation and reporting](https://github.com/KRLabsOrg/LettuceDetect/issues/56)
- [A Claude Code hook that flags hallucinations in agent answers](https://github.com/KRLabsOrg/LettuceDetect/issues/50)
- [A stable HTTP contract for typed spans and detector selection](https://github.com/KRLabsOrg/LettuceDetect/issues/75)

Shipped: typed spans in the Streamlit demo ([#57](https://github.com/KRLabsOrg/LettuceDetect/issues/57)).

## Exploring: [transferable span detection](https://github.com/KRLabsOrg/LettuceDetect/milestone/2)

A collaborator playground. These issues are experiments and design discussions, not committed release features or stable APIs. Reproducible comparisons, failure analyses, and short design notes are valuable contributions even when the result is negative.

- [Span-based safety classification](https://github.com/KRLabsOrg/LettuceDetect/issues/43) and [span-level supervision from Aegis 2.0](https://github.com/KRLabsOrg/LettuceDetect/issues/44): extending span detection from hallucination to safety dimensions.
- [Span-level prompt-injection and jailbreak detection](https://github.com/KRLabsOrg/LettuceDetect/issues/55)
- [Zero-shot token classification via label conditioning](https://github.com/KRLabsOrg/LettuceDetect/issues/70): GLiNER-style label conditioning, but token-level, so unseen span taxonomies work without retraining.

## Exploring: [efficient span detection](https://github.com/KRLabsOrg/LettuceDetect/milestone/4)

Experiments on lower-cost and long-context paths. As above, these are not release commitments.

- [Long-context bidirectional encoders](https://github.com/KRLabsOrg/LettuceDetect/issues/42)
- [A rule-based detection tier](https://github.com/KRLabsOrg/LettuceDetect/issues/15): low-compute lexical checks for unsupported numbers, dates, durations, and entities, with inspectable rule outputs and optional candidate synthesis through [RuleChef](https://github.com/KRLabsOrg/rulechef). Synthesized rules remain hypotheses until they are frozen and evaluated against held-out gold labels.

## How this roadmap changes

- Items move from Exploring to Now only with evidence: a working prototype, a benchmark result, or a design note that survived discussion.
- Simpler baselines and negative results can narrow or stop a direction.
- Anything user-facing lands with documentation and an entry in the changelog.
- Accepted, scoped work belongs in issues. Open product and research questions belong in [Discussions](https://github.com/KRLabsOrg/LettuceDetect/discussions).
- The roadmap is updated when evidence or milestones change, not on a schedule.
