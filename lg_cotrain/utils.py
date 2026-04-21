"""Utility functions: seed setting, logging, early stopping, device selection."""

import copy
import logging
import os
import random
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def set_seed(seed: int):
    """Set random seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    if HAS_NUMPY:
        np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_override=None):
    """Return the specified device, or CUDA if available, else CPU."""
    if not HAS_TORCH:
        raise RuntimeError("torch is required for get_device()")
    if device_override is not None:
        return torch.device(device_override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging to both file and console."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("lg_cotrain")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(output_dir, "experiment.log"))
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


class EarlyStopping:
    """Early stopping based on a monitored metric (higher is better).

    Works with any model that supports state_dict()/load_state_dict(),
    or with a simple dict for testing without torch.
    """

    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_score = -float("inf")
        self.counter = 0
        self.best_state_dict = None

    def step(self, score: float, model) -> bool:
        """Update with new score. Returns True if training should stop."""
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best(self, model):
        """Restore the model to the best checkpoint."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)


class PerClassEarlyStopping:
    """Early stopping that tracks patience independently per class.

    Stops only when ALL classes have individually plateaued for `patience` epochs.
    Checkpoints the model on improvement of the mean per-class F1.
    """

    def __init__(self, patience: int = 5, num_classes: int = 10):
        self.patience = patience
        self.num_classes = num_classes
        self.best_class_scores = [-float("inf")] * num_classes
        self.counters = [0] * num_classes
        self.stopped = [False] * num_classes
        self.best_aggregate = -float("inf")
        self.best_state_dict = None

    @property
    def counter(self):
        """Max patience counter across all classes (for logging compatibility)."""
        return max(self.counters)

    def step(self, per_class_f1: list, model) -> bool:
        """Update per-class scores. Returns True when ALL classes have plateaued."""
        for i, f1 in enumerate(per_class_f1):
            if f1 > self.best_class_scores[i]:
                self.best_class_scores[i] = f1
                self.counters[i] = 0
            else:
                self.counters[i] += 1
                if self.counters[i] >= self.patience:
                    self.stopped[i] = True
        aggregate = sum(per_class_f1) / len(per_class_f1)
        if aggregate > self.best_aggregate:
            self.best_aggregate = aggregate
            self.best_state_dict = copy.deepcopy(model.state_dict())
        return all(self.stopped)

    def restore_best(self, model):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)


class EarlyStoppingWithDelta:
    """Early stopping requiring a minimum improvement delta before resetting patience.

    The delta scales with the training-set imbalance ratio so that highly
    imbalanced events require more meaningful improvements to reset patience.
    """

    def __init__(self, patience: int = 5, base_delta: float = 0.001,
                 imbalance_ratio: float = 1.0, max_imbalance_cap: float = 20.0):
        self.patience = patience
        self.min_delta = base_delta * min(imbalance_ratio, max_imbalance_cap)
        self.best_score = -float("inf")
        self.counter = 0
        self.best_state_dict = None

    def step(self, score: float, model) -> bool:
        """Update with new score. Returns True if training should stop."""
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)


def compute_class_weights(class_labels, label2id: dict) -> list:
    """Compute inverse-frequency weights for each class, normalised so mean == 1.0.

    Args:
        class_labels: Iterable of string class labels (e.g. a pandas Series).
        label2id: Mapping from class-name string to integer class id.

    Returns:
        List of floats of length ``len(label2id)``, indexed by class id.
    """
    from collections import Counter
    counts = Counter(class_labels)
    total = sum(counts.values())
    n = len(label2id)
    id2label = {v: k for k, v in label2id.items()}
    raw = [total / (n * max(counts.get(id2label[i], 0), 1)) for i in range(n)]
    mean_w = sum(raw) / len(raw)
    return [w / mean_w for w in raw]


def compute_imbalance_ratio(class_labels) -> float:
    """Return the ratio of the most-frequent to least-frequent class count.

    Returns 1.0 for uniform distributions, single-class inputs, or empty inputs.
    """
    from collections import Counter
    counts = Counter(class_labels)
    if len(counts) < 2:
        return 1.0
    return max(counts.values()) / max(min(counts.values()), 1)
