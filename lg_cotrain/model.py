"""Transformer classifier wrapper for sequence classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForSequenceClassification


class BertClassifier(nn.Module):
    """Wrapper around AutoModelForSequenceClassification.

    Class name is kept as ``BertClassifier`` for backward compatibility with
    existing imports, but this wrapper now supports any HuggingFace sequence
    classification model whose architecture can be auto-detected from its
    config (e.g., bert-base-uncased, vinai/bertweet-base, roberta-base).
    """

    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        # Suppress expected UNEXPECTED/MISSING key warnings when loading a
        # base checkpoint into a sequence classification head.
        orig_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        transformers.logging.set_verbosity(orig_verbosity)

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass returning logits."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # We compute loss ourselves for weighted CE
        )
        return outputs.logits

    @torch.no_grad()
    def predict_proba(self, input_ids, attention_mask):
        """Return softmax probabilities (no gradient)."""
        logits = self.forward(input_ids, attention_mask)
        return F.softmax(logits, dim=-1)


def create_fresh_model(config) -> BertClassifier:
    """Factory to create a fresh BertClassifier from config."""
    return BertClassifier(
        model_name=config.model_name,
        num_labels=config.num_labels,
    )
