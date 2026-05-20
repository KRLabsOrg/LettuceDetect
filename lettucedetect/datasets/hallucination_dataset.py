from dataclasses import dataclass, field
from typing import Literal

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


@dataclass
class HallucinationSample:
    """A single hallucination detection sample.

    Attributes:
        prompt: Context text (source documents, code files, documentation, user query).
        answer: The LLM-generated answer to check for hallucinations.
        labels: List of span annotations. Each dict has ``start``, ``end`` (character offsets
            within ``answer``), ``label`` (native source label), and optionally ``category``
            and ``subcategory`` (v2 unified taxonomy fields). Empty list for clean samples.
        split: Dataset split (``train``, ``dev``, or ``test``).
        task_type: Task type (e.g. ``summarization``, ``qa``, ``code_generation``).
        dataset: Source dataset name.
        language: Language code.
        context_modality: Modality of the retrieved context (``prose``, ``code``, ``markdown``).
        category: Top-level taxonomy category for the sample (v2). ``None`` for clean samples.
        subcategory: Optional fine-grained sub-type within the category (v2).
        metadata: Arbitrary source-specific provenance fields (e.g. ``instance_id``, ``repo``).

    """

    prompt: str
    answer: str
    labels: list[dict]
    split: Literal["train", "dev", "test"]
    task_type: str
    dataset: str
    language: Literal["en", "de", "fr", "es", "it", "pl", "cn", "hu"]
    # v2 fields (optional for backwards compatibility)
    context_modality: Literal["prose", "code", "markdown"] = "prose"
    category: str | None = None
    subcategory: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "prompt": self.prompt,
            "answer": self.answer,
            "labels": self.labels,
            "split": self.split,
            "task_type": self.task_type,
            "dataset": self.dataset,
            "language": self.language,
            "context_modality": self.context_modality,
            "category": self.category,
            "subcategory": self.subcategory,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, json_dict: dict) -> "HallucinationSample":
        """Deserialize from a JSON dict."""
        return cls(
            prompt=json_dict["prompt"],
            answer=json_dict["answer"],
            labels=json_dict["labels"],
            split=json_dict["split"],
            task_type=json_dict["task_type"],
            dataset=json_dict["dataset"],
            language=json_dict["language"],
            context_modality=json_dict.get("context_modality", "prose"),
            category=json_dict.get("category"),
            subcategory=json_dict.get("subcategory"),
            metadata=json_dict.get("metadata", {}),
        )


@dataclass
class HallucinationData:
    """A collection of hallucination detection samples.

    Attributes:
        samples: List of :class:`HallucinationSample` instances.

    """

    samples: list[HallucinationSample]

    def to_json(self) -> list[dict]:
        """Serialize all samples to a JSON-compatible list."""
        return [sample.to_json() for sample in self.samples]

    @classmethod
    def from_json(cls, json_dict: list[dict]) -> "HallucinationData":
        """Deserialize from a list of JSON dicts."""
        return cls(
            samples=[HallucinationSample.from_json(sample) for sample in json_dict],
        )


class HallucinationDataset(Dataset):
    """Dataset for Hallucination data."""

    def __init__(
        self,
        samples: list[HallucinationSample],
        tokenizer: AutoTokenizer,
        max_length: int = 4096,
    ):
        """Initialize the dataset.

        :param samples: List of HallucinationSample objects.
        :param tokenizer: Tokenizer to use for encoding the data.
        :param max_length: Maximum length of the input sequence.
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    @classmethod
    def prepare_tokenized_input(
        cls,
        tokenizer: AutoTokenizer,
        context: str,
        answer: str,
        max_length: int = 4096,
    ) -> tuple[dict[str, torch.Tensor], list[int], torch.Tensor, int]:
        """Tokenize context and answer, compute answer start index, and initialize labels.

        Computes the answer start token index and initializes a labels list
        (using -100 for context tokens and 0 for answer tokens).

        :param tokenizer: The tokenizer to use.
        :param context: The context string.
        :param answer: The answer string.
        :param max_length: Maximum input sequence length.
        :return: A tuple containing:
                 - encoding: A dict of tokenized inputs without offset mapping.
                 - labels: A list of initial token labels.
                 - offsets: Offset mappings for each token (as a tensor of shape [seq_length, 2]).
                 - answer_start_token: The index where answer tokens begin.
        """
        encoding = tokenizer(
            context,
            answer,
            truncation="only_first",
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        offsets = encoding.pop("offset_mapping")[0]  # shape: (seq_length, 2)

        # Compute answer_start_token from the answer side.  This is correct
        # even when the context has been truncated by truncation="only_first",
        # because the answer is never truncated in that mode.
        # Layout: [CLS] context_tokens [SEP] answer_tokens [SEP]
        answer_only = tokenizer(answer, add_special_tokens=False, return_tensors="pt")
        answer_token_count = answer_only["input_ids"].shape[1]
        total_seq_len = encoding["input_ids"].shape[1]
        answer_start_token = total_seq_len - answer_token_count - 1  # -1 for trailing [SEP]

        # Initialize labels: -100 for tokens before the asnwer, 0 for tokens in the answer.
        labels = [-100] * encoding["input_ids"].shape[1]

        return encoding, labels, offsets, answer_start_token

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get an item from the dataset.

        :param idx: Index of the item to get.
        :return: Dictionary with input IDs, attention mask, and labels.
        """
        sample = self.samples[idx]

        # Use the shared class method to perform tokenization and initial label setup.
        encoding, labels, offsets, answer_start = HallucinationDataset.prepare_tokenized_input(
            self.tokenizer, sample.prompt, sample.answer, self.max_length
        )
        # Adjust the token labels based on the annotated hallucination spans.
        # Compute the character offset of the first answer token.

        answer_char_offset = offsets[answer_start][0] if answer_start < len(offsets) else None

        for i in range(answer_start, encoding["input_ids"].shape[1]):
            token_start, token_end = offsets[i]
            # Adjust token offsets relative to answer text.
            token_abs_start = (
                token_start - answer_char_offset if answer_char_offset is not None else token_start
            )
            token_abs_end = (
                token_end - answer_char_offset if answer_char_offset is not None else token_end
            )

            # Default label is 0 (supported content).
            token_label = 0
            # If token overlaps any annotated hallucination span, mark it as hallucinated (1).
            for ann in sample.labels:
                if token_abs_end > ann["start"] and token_abs_start < ann["end"]:
                    token_label = 1
                    break

            labels[i] = token_label

        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }
