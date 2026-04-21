"""Data loading, label encoding, dataset splitting, and PyTorch Dataset."""

import logging
import random
from typing import Dict, List, Tuple

logger = logging.getLogger("lg_cotrain")

# Full 10-class superset (alphabetical) for the humanitarian task.
# Not all events contain every class; use detect_event_classes() to find
# the subset present in a specific event's data.
CLASS_LABELS = sorted([
    "caution_and_advice",
    "displaced_people_and_evacuations",
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "missing_or_found_people",
    "not_humanitarian",
    "other_relevant_information",
    "requests_or_urgent_needs",
    "rescue_volunteering_or_donation_effort",
    "sympathy_and_support",
])


def detect_event_classes(*dataframes) -> List[str]:
    """Detect the sorted list of unique class labels across DataFrames or list-of-dicts.

    Accepts pandas DataFrames (with a 'class_label' column) or lists of dicts
    (with a 'class_label' key). Returns the alphabetically sorted union.
    """
    classes = set()
    for df in dataframes:
        if isinstance(df, list):
            classes.update(rec["class_label"] for rec in df)
        else:
            classes.update(df["class_label"].unique())
    return sorted(classes)


def build_label_encoder(labels=None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build label-to-id and id-to-label mappings.

    Args:
        labels: Optional list of class names. If None, uses the full CLASS_LABELS.
    """
    labels = labels if labels is not None else CLASS_LABELS
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    return label2id, id2label


# --- Functions requiring pandas ---

def load_tsv(path: str):
    """Load a tab-separated file with columns: tweet_id, tweet_text, class_label."""
    import pandas as pd
    df = pd.read_csv(path, sep="\t", dtype={"tweet_id": str})
    expected = {"tweet_id", "tweet_text", "class_label"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"TSV at {path} missing columns. Found: {list(df.columns)}")
    return df


def load_pseudo_labels(path: str):
    """Load GPT-4o pseudo-label CSV with predicted_label, confidence, etc."""
    import pandas as pd
    df = pd.read_csv(path, dtype={"tweet_id": str})
    required = {"tweet_id", "tweet_text", "predicted_label", "confidence"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"Pseudo-label CSV at {path} missing columns. Found: {list(df.columns)}"
        )
    return df


def split_labeled_set(df, seed: int):
    """Split labeled set into D_l1 and D_l2 with stratified per-class split.

    For each class, shuffle indices and assign first half to D_l1, second to D_l2.
    Works with pandas DataFrame or list-of-dicts.
    """
    try:
        import numpy as np
        return _split_labeled_set_pandas(df, seed)
    except ImportError:
        return _split_labeled_set_pure(df, seed)


def _split_labeled_set_pandas(df, seed: int):
    """Pandas-based split implementation."""
    import numpy as np
    rng = np.random.RandomState(seed)
    idx1, idx2 = [], []

    for label in sorted(df["class_label"].unique()):
        class_indices = df.index[df["class_label"] == label].tolist()
        rng.shuffle(class_indices)
        mid = len(class_indices) // 2
        split = mid if len(class_indices) % 2 == 0 else mid + 1
        idx1.extend(class_indices[:split])
        idx2.extend(class_indices[split:])

    return df.loc[idx1].reset_index(drop=True), df.loc[idx2].reset_index(drop=True)


def _split_labeled_set_pure(records: list, seed: int):
    """Pure-Python split on list-of-dicts. Returns (list, list)."""
    rng = random.Random(seed)
    by_class = {}
    for i, rec in enumerate(records):
        label = rec["class_label"]
        by_class.setdefault(label, []).append(i)

    idx1, idx2 = [], []
    for label in sorted(by_class):
        indices = by_class[label][:]
        rng.shuffle(indices)
        mid = len(indices) // 2
        split = mid if len(indices) % 2 == 0 else mid + 1
        idx1.extend(indices[:split])
        idx2.extend(indices[split:])

    return [records[i] for i in idx1], [records[i] for i in idx2]


def build_d_lg(df_unlabeled, df_pseudo):
    """Join unlabeled tweets with pseudo-labels on tweet_id.

    Result has predicted_label (for training) and class_label (for evaluation).
    Warns if any tweet_text values disagree between the two sources after joining.
    """
    pseudo_cols = df_pseudo[["tweet_id", "tweet_text", "predicted_label", "confidence"]].copy()
    pseudo_cols = pseudo_cols.rename(columns={"tweet_text": "tweet_text_pseudo"})
    merged = df_unlabeled.merge(pseudo_cols, on="tweet_id", how="inner")

    n_unmatched = len(df_unlabeled) - len(merged)
    if n_unmatched > 0:
        logger.warning(
            f"build_d_lg: {n_unmatched} unlabeled tweets had no matching pseudo-label"
        )

    text_mismatch = merged["tweet_text"].str.strip() != merged["tweet_text_pseudo"].str.strip()
    if text_mismatch.any():
        mismatched_ids = merged.loc[text_mismatch, "tweet_id"].tolist()
        logger.warning(
            f"build_d_lg: {text_mismatch.sum()} entries have mismatched tweet_text "
            f"between unlabeled TSV and pseudo-label CSV (tweet_ids: {mismatched_ids})"
        )

    return merged.drop(columns=["tweet_text_pseudo"]).reset_index(drop=True)


# --- TweetDataset requires torch + transformers ---

def _get_tweet_dataset_class():
    """Lazy import to avoid requiring torch/transformers at module load."""
    import torch
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer

    class TweetDataset(Dataset):
        """PyTorch Dataset for tokenized tweets with labels."""

        def __init__(
            self,
            texts: List[str],
            labels: List[int],
            tokenizer: PreTrainedTokenizer,
            max_length: int = 128,
        ):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            encoding = self.tokenizer(
                self.texts[idx],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
                "sample_idx": torch.tensor(idx, dtype=torch.long),
            }

    return TweetDataset


# Expose TweetDataset as a module-level name that's lazy
class TweetDataset:
    """Proxy that lazily imports the real TweetDataset when instantiated."""
    _real_class = None

    def __new__(cls, *args, **kwargs):
        if cls._real_class is None:
            cls._real_class = _get_tweet_dataset_class()
        return cls._real_class(*args, **kwargs)
