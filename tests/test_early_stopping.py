"""Tests for alternative early stopping classes and helper functions in utils.py."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Allow importing without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from lg_cotrain.utils import (
    EarlyStoppingWithDelta,
    PerClassEarlyStopping,
    compute_class_weights,
    compute_imbalance_ratio,
)


class MockModel:
    """Minimal model stub that supports state_dict / load_state_dict."""

    def __init__(self, value=0):
        self._state = {"value": value}

    def state_dict(self):
        return dict(self._state)

    def load_state_dict(self, state):
        self._state = dict(state)


# ---------------------------------------------------------------------------
# PerClassEarlyStopping
# ---------------------------------------------------------------------------


class TestPerClassEarlyStopping(unittest.TestCase):

    def _make(self, patience=3, num_classes=3):
        return PerClassEarlyStopping(patience=patience, num_classes=num_classes)

    def test_stops_when_all_classes_plateau(self):
        """Returns True after `patience` stale epochs for every class."""
        es = self._make(patience=2, num_classes=2)
        model = MockModel()
        # Step 1: improves from -inf (no stop)
        # Steps 2-3: same score, counters reach patience=2 (stop on step 3)
        results = [es.step([0.5, 0.5], model) for _ in range(3)]
        self.assertFalse(results[1])
        self.assertTrue(results[2])

    def test_does_not_stop_if_any_class_improving(self):
        """Returns False while any one class still improves."""
        es = self._make(patience=2, num_classes=2)
        model = MockModel()
        # Class 0 improves each time; class 1 stagnates
        scores = [[0.4, 0.5], [0.5, 0.5], [0.6, 0.5]]
        results = [es.step(s, model) for s in scores]
        # Should not have stopped after 3 rounds because class 0 kept improving
        self.assertFalse(results[-1])

    def test_independent_counters_per_class(self):
        """Counter for class 1 can be 0 while counter for class 0 is at patience."""
        es = self._make(patience=3, num_classes=2)
        model = MockModel()
        # Class 0 stagnates; class 1 improves each time
        scores = [[0.5, 0.1], [0.5, 0.2], [0.5, 0.3]]
        for s in scores:
            es.step(s, model)
        # Step 1: class 0 improves from -inf → 0.5 (counter=0)
        # Steps 2-3: class 0 stagnates at 0.5 (counter=1, then 2)
        self.assertEqual(es.counters[0], 2)   # class 0 plateaued on steps 2 and 3
        self.assertEqual(es.counters[1], 0)   # class 1 always improved

    def test_stops_when_last_class_plateaus(self):
        """Triggers on the epoch the slowest class exhausts patience."""
        es = self._make(patience=2, num_classes=2)
        model = MockModel()
        # Class 0 plateaus first; class 1 improves one extra time
        results = []
        results.append(es.step([0.5, 0.5], model))   # both stagnate round 1
        results.append(es.step([0.5, 0.6], model))   # class 1 improves (resets)
        results.append(es.step([0.5, 0.6], model))   # both stagnate round 2
        results.append(es.step([0.5, 0.6], model))   # both stagnate round 3 → stop
        self.assertFalse(results[2])
        self.assertTrue(results[3])

    def test_checkpoint_on_aggregate_improvement(self):
        """restore_best returns state from the best mean-F1 epoch."""
        es = self._make(patience=5, num_classes=2)
        model = MockModel(value=0)
        es.step([0.4, 0.4], model)       # aggregate 0.4  ← best so far; checkpoint state=0
        model._state["value"] = 1
        es.step([0.6, 0.6], model)       # aggregate 0.6  ← new best; checkpoint state=1
        model._state["value"] = 2
        es.step([0.5, 0.5], model)       # aggregate 0.5  (no improvement)

        restore_target = MockModel(value=99)
        es.restore_best(restore_target)
        self.assertEqual(restore_target._state["value"], 1)

    def test_restore_noop_before_any_step(self):
        """No crash if restore_best called before any step."""
        es = self._make()
        model = MockModel(value=7)
        es.restore_best(model)           # should not raise
        self.assertEqual(model._state["value"], 7)

    def test_counter_property_is_max(self):
        """.counter equals max(self.counters)."""
        es = self._make(patience=10, num_classes=3)
        model = MockModel()
        # class 0: stagnate x3, class 1: improve each time, class 2: stagnate x1
        es.step([0.5, 0.1, 0.5], model)
        es.step([0.5, 0.2, 0.5], model)
        es.step([0.5, 0.3, 0.4], model)
        self.assertEqual(es.counter, max(es.counters))


# ---------------------------------------------------------------------------
# EarlyStoppingWithDelta
# ---------------------------------------------------------------------------


class TestEarlyStoppingWithDelta(unittest.TestCase):

    def _make(self, patience=3, base_delta=0.01, imbalance_ratio=1.0,
              max_imbalance_cap=20.0):
        return EarlyStoppingWithDelta(
            patience=patience,
            base_delta=base_delta,
            imbalance_ratio=imbalance_ratio,
            max_imbalance_cap=max_imbalance_cap,
        )

    def test_no_trigger_without_sufficient_delta(self):
        """Improvements below min_delta do not reset counter."""
        es = self._make(patience=3, base_delta=0.05, imbalance_ratio=1.0)
        model = MockModel()
        # min_delta = 0.05; improvements of 0.01 are not enough
        es.step(0.50, model)   # first step always improves (best=-inf)
        es.step(0.51, model)   # +0.01 < 0.05 → counter += 1
        es.step(0.52, model)   # +0.01 < 0.05 → counter += 1
        result = es.step(0.53, model)  # counter == 3 → stop
        self.assertTrue(result)

    def test_resets_counter_on_sufficient_improvement(self):
        """Improvement strictly greater than min_delta resets counter to 0."""
        es = self._make(patience=3, base_delta=0.01, imbalance_ratio=1.0)
        model = MockModel()
        es.step(0.50, model)   # first step (improves from -inf)
        # Need score > best + min_delta = 0.50 + 0.01 = 0.51, so use 0.52
        es.step(0.52, model)   # +0.02 > 0.01 → reset counter
        self.assertEqual(es.counter, 0)

    def test_imbalance_ratio_1_gives_base_delta(self):
        """`ratio=1.0` → min_delta == base_delta."""
        es = self._make(base_delta=0.005, imbalance_ratio=1.0)
        self.assertAlmostEqual(es.min_delta, 0.005)

    def test_high_ratio_multiplies_delta(self):
        """`ratio=10.0`, `base=0.001` → min_delta == 0.01."""
        es = self._make(base_delta=0.001, imbalance_ratio=10.0)
        self.assertAlmostEqual(es.min_delta, 0.01)

    def test_cap_limits_delta(self):
        """`ratio=100, cap=20` → min_delta == base * 20."""
        es = self._make(base_delta=0.001, imbalance_ratio=100.0, max_imbalance_cap=20.0)
        self.assertAlmostEqual(es.min_delta, 0.02)

    def test_checkpoint_saved_correctly(self):
        """restore_best returns state from the first exceeding-delta epoch."""
        es = self._make(patience=5, base_delta=0.05, imbalance_ratio=1.0)
        model = MockModel(value=0)
        es.step(0.50, model)             # best, saved state=0
        model._state["value"] = 1
        es.step(0.60, model)             # +0.10 >= 0.05 → new best, saved state=1
        model._state["value"] = 2
        es.step(0.61, model)             # +0.01 < 0.05 → no checkpoint

        restore_target = MockModel(value=99)
        es.restore_best(restore_target)
        self.assertEqual(restore_target._state["value"], 1)

    def test_best_score_starts_at_neg_inf(self):
        """Any finite score on first call is treated as improvement."""
        es = self._make(base_delta=0.001)
        model = MockModel()
        result = es.step(-999.0, model)  # -inf + delta still < -999, so should improve
        self.assertFalse(result)
        self.assertEqual(es.counter, 0)


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------


class TestComputeClassWeights(unittest.TestCase):

    def _label2id(self, classes):
        return {c: i for i, c in enumerate(sorted(classes))}

    def test_uniform_gives_unit_weights(self):
        """All weights == 1.0 for perfectly balanced input."""
        labels = ["a", "a", "b", "b", "c", "c"]
        l2id = self._label2id(["a", "b", "c"])
        weights = compute_class_weights(labels, l2id)
        for w in weights:
            self.assertAlmostEqual(w, 1.0)

    def test_rare_class_gets_higher_weight(self):
        """Class with 10 samples has higher weight than class with 100."""
        labels = ["common"] * 100 + ["rare"] * 10
        l2id = {"common": 0, "rare": 1}
        weights = compute_class_weights(labels, l2id)
        self.assertGreater(weights[1], weights[0])

    def test_mean_is_one(self):
        """`mean(weights) == 1.0` always."""
        labels = ["x"] * 50 + ["y"] * 30 + ["z"] * 20
        l2id = {"x": 0, "y": 1, "z": 2}
        weights = compute_class_weights(labels, l2id)
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0)

    def test_absent_class_gets_max_weight(self):
        """Class in label2id but absent from labels → highest weight."""
        labels = ["a"] * 50 + ["b"] * 50
        l2id = {"a": 0, "b": 1, "c": 2}  # "c" never appears
        weights = compute_class_weights(labels, l2id)
        self.assertEqual(weights.index(max(weights)), 2)

    def test_output_length_equals_num_classes(self):
        """`len(weights) == len(label2id)`."""
        labels = ["a", "b"]
        l2id = {"a": 0, "b": 1, "c": 2, "d": 3}
        weights = compute_class_weights(labels, l2id)
        self.assertEqual(len(weights), 4)


# ---------------------------------------------------------------------------
# compute_imbalance_ratio
# ---------------------------------------------------------------------------


class TestComputeImbalanceRatio(unittest.TestCase):

    def test_uniform_returns_one(self):
        """Equal counts → 1.0."""
        from lg_cotrain.utils import compute_imbalance_ratio
        ratio = compute_imbalance_ratio(["a"] * 10 + ["b"] * 10 + ["c"] * 10)
        self.assertAlmostEqual(ratio, 1.0)

    def test_ratio_correct(self):
        """max=50, min=5 → ratio=10.0."""
        from lg_cotrain.utils import compute_imbalance_ratio
        labels = ["majority"] * 50 + ["minority"] * 5
        ratio = compute_imbalance_ratio(labels)
        self.assertAlmostEqual(ratio, 10.0)

    def test_single_class_returns_one(self):
        """One class → 1.0."""
        from lg_cotrain.utils import compute_imbalance_ratio
        ratio = compute_imbalance_ratio(["only"] * 20)
        self.assertAlmostEqual(ratio, 1.0)

    def test_empty_returns_one(self):
        """Empty input → 1.0."""
        from lg_cotrain.utils import compute_imbalance_ratio
        ratio = compute_imbalance_ratio([])
        self.assertAlmostEqual(ratio, 1.0)


# ---------------------------------------------------------------------------
# TestRunAllWithStoppingStrategy — integration of config + CLI
# ---------------------------------------------------------------------------


class TestRunAllWithStoppingStrategy(unittest.TestCase):

    def test_strategy_forwarded_to_config(self):
        """`stopping_strategy='per_class_patience'` reaches LGCoTrainConfig."""
        from lg_cotrain.config import LGCoTrainConfig
        cfg = LGCoTrainConfig(stopping_strategy="per_class_patience")
        self.assertEqual(cfg.stopping_strategy, "per_class_patience")

    def test_default_strategy_is_baseline(self):
        """Omitting stopping_strategy → config.stopping_strategy == 'baseline'."""
        from lg_cotrain.config import LGCoTrainConfig
        cfg = LGCoTrainConfig()
        self.assertEqual(cfg.stopping_strategy, "baseline")

    def test_invalid_strategy_cli_rejected(self):
        """argparse choices= raises SystemExit for unknown value."""
        import argparse
        parser = argparse.ArgumentParser()
        valid = [
            "baseline", "no_early_stopping", "per_class_patience",
            "weighted_macro_f1", "balanced_dev", "scaled_threshold",
        ]
        parser.add_argument("--stopping-strategy", choices=valid, default="baseline")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--stopping-strategy", "totally_invalid"])


if __name__ == "__main__":
    unittest.main()
