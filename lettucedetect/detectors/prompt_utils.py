"""Utilities for loading and formatting prompts."""

from __future__ import annotations

from pathlib import Path
from string import Template

# Type for supported languages
Lang = str  # "en", "de", "fr", "es", "it", "pl", "cn"

LANG_TO_PASSAGE = {
    "en": "passage",
    "de": "Passage",
    "fr": "passage",
    "es": "pasaje",
    "it": "brano",
    "pl": "fragment",
    "cn": "段落",
}

# Full language names for each language code
LANG_TO_FULL_NAME = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pl": "Polish",
    "cn": "Chinese",
}

PROMPT_DIR = Path(__file__).parent.parent / "prompts"


class PromptUtils:
    """Utility class for loading and formatting prompts."""

    @staticmethod
    def load_prompt(filename: str) -> Template:
        """Load a prompt template from the prompts directory.

        :param filename: Name of the prompt file
        :return: Template object for the prompt
        :raises FileNotFoundError: If the prompt file doesn't exist
        """
        path = PROMPT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return Template(path.read_text(encoding="utf-8"))

    @staticmethod
    def format_context(context: list[str], question: str | None, lang: Lang) -> str:
        """Format context and question into a prompt.

        :param context: List of passages
        :param question: Question (None for summarization tasks)
        :param lang: Language code
        :return: Formatted prompt
        """
        p_word = LANG_TO_PASSAGE[lang]
        ctx_block = "\n".join(f"{p_word} {i + 1}: {p}" for i, p in enumerate(context))

        if question is None:
            tmpl = PromptUtils.load_prompt(f"summary_prompt_{lang.lower()}.txt")
            return tmpl.substitute(text=ctx_block)

        tmpl = PromptUtils.load_prompt(f"qa_prompt_{lang.lower()}.txt")
        return tmpl.substitute(question=question, num_passages=len(context), context=ctx_block)

    @staticmethod
    def get_full_language_name(lang: Lang) -> str:
        """Get the full language name for a language code.

        :param lang: Language code
        :return: Full language name
        """
        return LANG_TO_FULL_NAME.get(lang, "Unknown")
