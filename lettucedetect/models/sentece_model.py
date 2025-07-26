import logging
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedModel

logger = logging.getLogger(__name__)


class SentenceModel(PreTrainedModel):
    """
    Sentence Level Model for sentence classification.
    """

    config_class = AutoConfig
    base_model_prefix = "bert"

    def __init__(
        self,
        config=None,
        model_name="answerdotai/ModernBERT-base",
        hidden_dim: int = 768,
        num_labels=2,
    ):
        # Create config if not provided
        if config is None:
            config = AutoConfig.from_pretrained(model_name)
            # Add our custom config values
            config.model_name = model_name
            config.hidden_dim = hidden_dim
            config.num_labels = num_labels

        super().__init__(config)

        # Set properties from config
        self.model_name = getattr(config, "model_name", model_name)
        self.hidden_dim = getattr(config, "hidden_dim", hidden_dim)
        self.num_labels = getattr(config, "num_labels", num_labels)

        # Initialize base model
        self.bert = AutoModel.from_pretrained(self.model_name)

        # Two classification heads
        self.sentence_classifier = nn.Linear(self.hidden_dim, self.num_labels)
        # Initialize weights
        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_boundaries: list[list[tuple[int, int]]],
    ) -> list[torch.Tensor]:
        """
        Forward pass of the model.

        :param input_ids: Token IDs
        :param attention_mask: Attention mask
        :param sentence_boundaries: List of lists of tuples (start, end) for sentence boundaries

        :return: List of tensors with sentence classification logits
        """
        # Get contextualized representations from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)

        # Extract sentence representations and classify each sentence
        batch_size = sequence_output.size(0)
        sentence_preds = []

        for batch_idx in range(batch_size):
            # Get the sentence boundaries for this batch item
            batch_sentence_boundaries = sentence_boundaries[batch_idx]

            # Collect sentence representations
            sentence_reprs = []
            for start, end in batch_sentence_boundaries:
                # If the sentence extends beyond the sequence, adjust the end
                if end >= sequence_output.size(1):
                    end = sequence_output.size(1) - 1

                # Skip empty or invalid sentences
                if end < start or start < 0:
                    continue

                # Get the token embeddings for this sentence
                sentence_tokens = sequence_output[batch_idx, start : end + 1]

                # Average the token embeddings to get a sentence embedding
                sentence_repr = torch.mean(sentence_tokens, dim=0)
                sentence_reprs.append(sentence_repr)

            # If no valid sentences, skip this batch item
            if not sentence_reprs:
                sentence_preds.append(None)
                continue

            # Stack and classify all sentence representations
            if sentence_reprs:
                stacked_reprs = torch.stack(sentence_reprs)
                predictions = self.sentence_classifier(stacked_reprs)
                sentence_preds.append(predictions)
            else:
                sentence_preds.append(None)

        return sentence_preds

    def get_config(self) -> dict[str, Any]:
        """Get model configuration as a dictionary.

        :return: Model configuration
        """
        return self.config.to_dict()

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, *model_args, **kwargs
    ) -> "SentenceModel":
        """Load a model from a pretrained model or path.

        This overrides the from_pretrained method from PreTrainedModel to handle
        our specific model architecture.

        :param pretrained_model_name_or_path: Pretrained model name or path
        :param model_args: Additional model arguments
        :param kwargs: Additional keyword arguments
        :return: SentencekModel instance
        """
        # Let HuggingFace handle the downloading, caching, etc.
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def save_pretrained(self, save_directory, **kwargs) -> None:
        """Save the model to a directory.
        This overrides the save_pretrained method from PreTrainedModel to handle
        our specific model architecture.

        :param save_directory: Directory to save the model
        :param kwargs: Additional keyword arguments
        :return: None
        """
        # Let HuggingFace's built-in method handle the saving
        super().save_pretrained(save_directory, **kwargs)
