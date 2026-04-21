"""Confidence/variability tracking and lambda weight computation."""

import math
from typing import List

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class WeightTracker:
    """Track per-sample probability history and compute lambda weights.

    Records p(pseudo_label | x; theta) for each sample across epochs,
    then computes confidence (mean) and variability (std).

    Works with both numpy arrays and plain Python lists.
    """

    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        self.prob_history: list = []

    def record_epoch(self, probs):
        """Record one epoch's per-sample probabilities.

        Args:
            probs: Array/list of length num_samples with p(pseudo_label | x; theta).
        """
        if HAS_NUMPY and isinstance(probs, np.ndarray):
            assert probs.shape == (self.num_samples,), (
                f"Expected shape ({self.num_samples},), got {probs.shape}"
            )
            self.prob_history.append(probs.copy())
        else:
            assert len(probs) == self.num_samples, (
                f"Expected length {self.num_samples}, got {len(probs)}"
            )
            self.prob_history.append(list(probs))

    def compute_confidence(self):
        """Mean probability across recorded epochs. Returns ndarray or list."""
        n = len(self.prob_history)
        if HAS_NUMPY and isinstance(self.prob_history[0], np.ndarray):
            stacked = np.stack(self.prob_history, axis=0)
            return stacked.mean(axis=0)
        result = []
        for j in range(self.num_samples):
            result.append(sum(self.prob_history[i][j] for i in range(n)) / n)
        return result

    def compute_variability(self):
        """Std (population) of probability across recorded epochs. Returns ndarray or list."""
        n = len(self.prob_history)
        if HAS_NUMPY and isinstance(self.prob_history[0], np.ndarray):
            stacked = np.stack(self.prob_history, axis=0)
            return stacked.std(axis=0)
        conf = self.compute_confidence()
        result = []
        for j in range(self.num_samples):
            variance = sum(
                (self.prob_history[i][j] - conf[j]) ** 2 for i in range(n)
            ) / n
            result.append(math.sqrt(variance))
        return result

    def compute_lambda_optimistic(self):
        """Lambda1 = confidence + variability."""
        c = self.compute_confidence()
        v = self.compute_variability()
        if HAS_NUMPY and isinstance(c, np.ndarray):
            return c + v
        return [ci + vi for ci, vi in zip(c, v)]

    def compute_lambda_conservative(self):
        """Lambda2 = max(confidence - variability, 0)."""
        c = self.compute_confidence()
        v = self.compute_variability()
        if HAS_NUMPY and isinstance(c, np.ndarray):
            return np.clip(c - v, a_min=0, a_max=None)
        return [max(ci - vi, 0.0) for ci, vi in zip(c, v)]

    @classmethod
    def seed_from_tracker(cls, source: "WeightTracker") -> "WeightTracker":
        """Create a new tracker pre-seeded with another tracker's full history.

        Copies the full epoch history from source into a new tracker.
        Note: this carries ALL Phase 1 epochs into Phase 2, which differs
        from Algorithm 1 (which seeds with only the final epoch).
        See seed_from_last_epoch() for the paper-aligned variant.
        """
        new_tracker = cls(source.num_samples)
        for probs in source.prob_history:
            if HAS_NUMPY and isinstance(probs, np.ndarray):
                new_tracker.prob_history.append(probs.copy())
            else:
                new_tracker.prob_history.append(list(probs))
        return new_tracker

    @classmethod
    def seed_from_epoch(cls, source: "WeightTracker", epoch_idx: int) -> "WeightTracker":
        """Create a new tracker seeded with a specific epoch from the source.

        Args:
            source: The source WeightTracker containing recorded epochs.
            epoch_idx: Zero-based index of the epoch to use as seed.

        Returns:
            A new WeightTracker with exactly one recorded epoch (the selected one).

        Raises:
            IndexError: If epoch_idx is out of range.
        """
        if epoch_idx < 0 or epoch_idx >= len(source.prob_history):
            raise IndexError(
                f"epoch_idx {epoch_idx} out of range for tracker with "
                f"{len(source.prob_history)} recorded epochs"
            )
        new_tracker = cls(source.num_samples)
        selected_probs = source.prob_history[epoch_idx]
        if HAS_NUMPY and isinstance(selected_probs, np.ndarray):
            new_tracker.prob_history.append(selected_probs.copy())
        else:
            new_tracker.prob_history.append(list(selected_probs))
        return new_tracker

    @classmethod
    def seed_from_last_epoch(cls, source: "WeightTracker") -> "WeightTracker":
        """Create a new tracker seeded with only the source's final epoch.

        Per Algorithm 1 in the paper: the final Phase 1 epoch's probabilities
        serve as the single initial data point for Phase 2 confidence/variability
        tracking. Variability starts at 0 and becomes meaningful only after Phase 2
        records additional epochs.
        """
        new_tracker = cls(source.num_samples)
        last_probs = source.prob_history[-1]
        if HAS_NUMPY and isinstance(last_probs, np.ndarray):
            new_tracker.prob_history.append(last_probs.copy())
        else:
            new_tracker.prob_history.append(list(last_probs))
        return new_tracker

    @property
    def num_epochs_recorded(self) -> int:
        return len(self.prob_history)
