"""Dataset-generation pipeline for LettuceDetect.

Composable primitives for building hallucination-detection datasets from any
grounded source:

- :mod:`lettucedetect.generation.questions` — derive a question from raw data.
- :mod:`lettucedetect.generation.answers` — generate a correct, grounded answer.
- :mod:`lettucedetect.generation.injection` — corrupt a correct answer into a
  hallucinated one with exact character-level spans, driven by the unified
  taxonomy and aware of the context modality.
- :mod:`lettucedetect.generation.classify` — label an existing, untyped span
  (e.g. a natural PsiloQA hallucination) with a unified ``(category, subcategory)``
  using an LLM, for sources that ship spans without a native type.

This is distinct from :class:`lettucedetect.models.generation.HallucinationGenerator`,
which *synthesizes* hallucinated content via RAGFactChecker (the TinyLettuce
recipe). The injector here corrupts a known-correct answer to obtain exact spans.
"""
