"""Shared fixtures for pytest tests."""

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


@pytest.fixture(scope="session")
def local_wordpiece_tokenizer() -> PreTrainedTokenizerFast:
    """Build a tiny BERT-style tokenizer without reading from the network."""
    tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "the",
        "capital",
        "of",
        "france",
        "is",
        "paris",
        ".",
        "short",
        "answer",
        "word",
    ]
    tokenizer = Tokenizer(WordPiece(vocab={token: index for index, token in enumerate(tokens)}))
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",  # noqa: S106 - tokenizer special token, not a credential
        unk_token="[UNK]",  # noqa: S106 - tokenizer special token, not a credential
        cls_token="[CLS]",  # noqa: S106 - tokenizer special token, not a credential
        sep_token="[SEP]",  # noqa: S106 - tokenizer special token, not a credential
        mask_token="[MASK]",  # noqa: S106 - tokenizer special token, not a credential
    )
