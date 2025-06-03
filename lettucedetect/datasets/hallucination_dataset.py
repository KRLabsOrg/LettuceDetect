from dataclasses import dataclass
from typing import Literal

import nltk
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

nltk.download("punkt_tab")


@dataclass
class HallucinationSample:
    prompt: str
    answer: str
    labels: list[dict]
    split: Literal["train", "dev", "test"]
    task_type: str
    dataset: Literal["ragtruth", "ragbench"]
    language: Literal["en", "de"]
    answer_sentences: list = None

    def to_json(self) -> dict:
        json_dict = {
            "prompt": self.prompt,
            "answer": self.answer,
            "labels": self.labels,
            "split": self.split,
            "task_type": self.task_type,
            "dataset": self.dataset,
            "language": self.language,
        }

        if self.answer_sentences is not None:
            json_dict["answer_sentences"] = self.answer_sentences

        return json_dict

    @classmethod
    def from_json(cls, json_dict: dict) -> "HallucinationSample":
        return cls(
            prompt=json_dict["prompt"],
            answer=json_dict["answer"],
            answer_sentences=json_dict.get("answer_sentences"),
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


def find_hallucinated_sent(sample):
    hallu_sent = []
    for label in sample.labels:
        hallu_sent.append(sample.answer[label["start"] : label["end"]])
    return hallu_sent


def define_sentence_label(sentences, hallucinated_sentences):
    labels = [
        int(any(hallu_sent in sentence for hallu_sent in hallucinated_sentences))
        for sentence in sentences
    ]
    return labels


class HallucinationDataset(Dataset):
    """Dataset for Hallucination data."""

    def __init__(
        self,
        samples: list[HallucinationSample],
        tokenizer: AutoTokenizer,
        method: Literal["transformer", "sentencetransformer"] = "transformer",
        max_length: int = 4096,
    ):
        """Initialize the dataset.x

        :param samples: List of HallucinationSample objects.
        :param tokenizer: Tokenizer to use for encoding the data.
        :param max_length: Maximum length of the input sequence.
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.method = method
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

    @classmethod
    def encode_context_and_sentences_with_offset(
        cls,
        tokenizer: AutoTokenizer,
        context: str,
        sentences: list,
        max_length: int = 4096,
    ) -> dict:
        max_length = max_length - 2

        # -------------------------------------------------------------------------
        # 1) Encode the context with special tokens
        # -------------------------------------------------------------------------
        encoded_context = tokenizer.encode_plus(
            context,
            add_special_tokens=True,
            return_offsets_mapping=True,
            max_length=max_length,
            truncation=True,
        )
        context_ids = encoded_context["input_ids"]
        context_attn_mask = encoded_context["attention_mask"]
        context_offsets = encoded_context["offset_mapping"]

        if len(context_ids) > 1 and context_ids[-1] == tokenizer.sep_token_id:
            context_ids.pop()
            context_attn_mask.pop()
            context_offsets.pop()

        input_ids = context_ids[:]
        attention_mask = context_attn_mask[:]
        offset_mapping = context_offsets[:]

        sentence_boundaries = []
        sentence_offset_mappings = []

        # -------------------------------------------------------------------------
        # 2) Encode each sentence and check if it fits within max_length
        # -------------------------------------------------------------------------
        for sent in sentences:
            # First check if adding this sentence would exceed max_length
            # Encode the sentence to check its length
            encoded_sent = tokenizer.encode_plus(
                sent,
                add_special_tokens=False,
                return_offsets_mapping=True,
                max_length=max_length,
                truncation=True,
            )

            sent_ids = encoded_sent["input_ids"]
            sent_offsets = encoded_sent["offset_mapping"]

            # +1 for [SEP] token
            if len(input_ids) + len(sent_ids) + 1 > max_length:
                # If this sentence won't fit, stop processing more sentences
                break

            # If we get here, we can add the sentence
            # Insert [SEP] for boundary

            input_ids.append(tokenizer.sep_token_id)
            attention_mask.append(1)
            offset_mapping.append((0, 0))

            sent_start_idx = len(input_ids)

            # Add the sentence tokens
            input_ids.extend(sent_ids)
            attention_mask.extend([1] * len(sent_ids))
            offset_mapping.extend(sent_offsets)

            sent_end_idx = len(input_ids) - 1  # inclusive end

            # Mark this sentence boundary and store its offsets and label
            sentence_boundaries.append((sent_start_idx, sent_end_idx))
            sentence_offset_mappings.append(sent_offsets)

        # Add final [SEP] if there's room
        if len(input_ids) < max_length:
            input_ids.append(tokenizer.sep_token_id)
            attention_mask.append(1)
            offset_mapping.append((0, 0))

        # -------------------------------------------------------------------------
        # 3) Handle truncation by only including complete sentences
        # -------------------------------------------------------------------------
        if len(input_ids) > max_length:
            # Find the last complete sentence that fits
            last_valid_idx = 0
            for i, (start, end) in enumerate(sentence_boundaries):
                if end < max_length:
                    last_valid_idx = i
                else:
                    break

            if last_valid_idx >= 0:
                last_token = sentence_boundaries[last_valid_idx][1]
                input_ids = input_ids[: last_token + 1]  # +1 to include the last [SEP]
                attention_mask = attention_mask[: last_token + 1]
                offset_mapping = offset_mapping[: last_token + 1]
                sentence_boundaries = sentence_boundaries[: last_valid_idx + 1]
                sentence_offset_mappings = sentence_offset_mappings[: last_valid_idx + 1]

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return (
            input_ids,
            attention_mask,
            offset_mapping,
            sentence_boundaries,
            sentence_offset_mappings,
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get an item from the dataset.

        :param idx: Index of the item to get.
        :return: Dictionary with input IDs, attention mask, and labels.
        """
        sample = self.samples[idx]

        # -------------------------------------------------------------------------
        # 1) Token-level Model
        # -------------------------------------------------------------------------

        if self.method == "transformer":
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
                    token_start - answer_char_offset
                    if answer_char_offset is not None
                    else token_start
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

        # -------------------------------------------------------------------------
        # 2) Sentence-Level Model
        # -------------------------------------------------------------------------
        else:
            # If the sample is coming from ragbench we will use the response sentences already defined in the dataset; otherwise the sample.answer will be split using nltk library
            sentences = sample.answer_sentences
            if sentences is None:
                sentences = nltk.sent_tokenize(sample.answer)

            (
                input_ids,
                attention_mask,
                offset_mapping,
                sentence_boundaries,
                sentence_offset_mappings,
            ) = HallucinationDataset.encode_context_and_sentences_with_offset(
                self.tokenizer, sample.prompt, sentences, max_length=4096
            )

            # Add labels for included sentences
            hallucinated_sentences = find_hallucinated_sent(sample=sample)
            sentence_labels = define_sentence_label(
                sentences=sentences[: len(sentence_boundaries)],
                hallucinated_sentences=hallucinated_sentences,
            )
            sentence_labels = torch.tensor(sentence_labels, dtype=torch.long)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "offset_mapping": offset_mapping,
                "sentence_boundaries": sentence_boundaries,
                "sentence_offset_mappings": sentence_offset_mappings,
                "labels": sentence_labels,
            }
