from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


@dataclass
class HallucinationSample:
    prompt: str
    answer: str
    labels: list[dict]
    split: Literal["train", "dev", "test"]
    task_type: str
    dataset: Literal["ragtruth", "ragbench"]
    language: Literal["en", "de"]

    def to_json(self) -> dict:
        return {
            "prompt": self.prompt,
            "answer": self.answer,
            "labels": self.labels,
            "split": self.split,
            "task_type": self.task_type,
            "dataset": self.dataset,
            "language": self.language,
        }

    @classmethod
    def from_json(cls, json_dict: dict) -> "HallucinationSample":
        return cls(
            prompt=json_dict["prompt"],
            answer=json_dict["answer"],
            labels=json_dict["labels"],
            split=json_dict["split"],
            task_type=json_dict["task_type"],
            dataset=json_dict["dataset"],
            language=json_dict["language"],
        )


@dataclass
class HallucinationData:
    samples: list[HallucinationSample]

    def to_json(self) -> list[dict]:
        return [sample.to_json() for sample in self.samples]

    @classmethod
    def from_json(cls, json_dict: list[dict]) -> "HallucinationData":
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
        return len(self.samples)

    @classmethod
    def prepare_tokenized_input(
        cls,
        tokenizer: AutoTokenizer,
        context: str,
        answer: str,
        max_length: int = 4096,
    ) -> tuple[dict[str, torch.Tensor], list[int], torch.Tensor, int]:
        """Tokenizes the context and answer together, computes the answer start token index,
        and initializes a labels list (using -100 for context tokens and 0 for answer tokens).

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

        # Simple approach: encode just the context with special tokens
        # For most tokenizers, the answer starts right after this
        context_only = tokenizer(context, add_special_tokens=True, return_tensors="pt")
        # The answer starts after the context sequence (with its special tokens)
        answer_start_token = context_only["input_ids"].shape[1]

        # Handle any edge cases where this might land on a special token
        if (
            answer_start_token < offsets.size(0)
            and offsets[answer_start_token][0] == offsets[answer_start_token][1]
        ):
            # If we landed on a special token, move forward
            answer_start_token += 1

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
