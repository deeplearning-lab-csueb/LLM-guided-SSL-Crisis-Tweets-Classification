"""Tests for utils.py — works without torch using a mock model."""

import logging
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, "/workspace")

from lg_cotrain.utils import EarlyStopping, set_seed, setup_logging


class MockModel:
    """Simple model mock with state_dict/load_state_dict for testing EarlyStopping."""

    def __init__(self, value=0.0):
        self._value = value

    def state_dict(self):
        return {"value": self._value}

    def load_state_dict(self, state_dict):
        self._value = state_dict["value"]

    @property
    def value(self):
        return self._value


class TestSetSeed(unittest.TestCase):
    """set_seed makes stdlib random reproducible."""

    def test_random_reproducible(self):
        set_seed(42)
        a = [random.random() for _ in range(10)]
        set_seed(42)
        b = [random.random() for _ in range(10)]
        self.assertEqual(a, b)

    def test_different_seeds_differ(self):
        set_seed(42)
        a = [random.random() for _ in range(10)]
        set_seed(99)
        b = [random.random() for _ in range(10)]
        self.assertNotEqual(a, b)


class TestEarlyStoppingPatience(unittest.TestCase):
    """EarlyStopping triggers correctly based on patience."""

    def test_no_trigger_before_patience(self):
        es = EarlyStopping(patience=3)
        model = MockModel(1.0)
        self.assertFalse(es.step(0.5, model))  # best
        self.assertFalse(es.step(0.4, model))  # worse, counter=1
        self.assertFalse(es.step(0.3, model))  # worse, counter=2

    def test_triggers_at_patience(self):
        es = EarlyStopping(patience=3)
        model = MockModel(1.0)
        es.step(0.5, model)   # best
        es.step(0.4, model)   # counter=1
        es.step(0.3, model)   # counter=2
        self.assertTrue(es.step(0.2, model))  # counter=3 → triggers

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=3)
        model = MockModel(1.0)
        es.step(0.5, model)   # best
        es.step(0.4, model)   # counter=1
        es.step(0.6, model)   # improvement → counter=0
        self.assertFalse(es.step(0.5, model))  # counter=1
        self.assertFalse(es.step(0.4, model))  # counter=2

    def test_patience_1_triggers_immediately(self):
        es = EarlyStopping(patience=1)
        model = MockModel()
        es.step(0.5, model)   # best
        self.assertTrue(es.step(0.4, model))  # counter=1 → triggers

    def test_counter_tracks_correctly(self):
        es = EarlyStopping(patience=5)
        model = MockModel()
        es.step(0.5, model)
        self.assertEqual(es.counter, 0)
        es.step(0.4, model)
        self.assertEqual(es.counter, 1)
        es.step(0.3, model)
        self.assertEqual(es.counter, 2)
        es.step(0.6, model)  # new best
        self.assertEqual(es.counter, 0)
        self.assertAlmostEqual(es.best_score, 0.6)


class TestEarlyStoppingRestore(unittest.TestCase):
    """EarlyStopping correctly saves and restores best model state."""

    def test_restores_best_state(self):
        model = MockModel(value=1.0)
        es = EarlyStopping(patience=2)

        # Record best at value=1.0
        es.step(0.9, model)

        # Change model state
        model._value = 0.0
        es.step(0.5, model)  # worse

        # Restore
        es.restore_best(model)
        self.assertEqual(model.value, 1.0)

    def test_restores_the_actual_best(self):
        """After multiple improvements, restores the most recent best."""
        model = MockModel(value=10.0)
        es = EarlyStopping(patience=3)

        es.step(0.5, model)  # best, value=10
        model._value = 20.0
        es.step(0.7, model)  # new best, value=20
        model._value = 30.0
        es.step(0.6, model)  # worse, counter=1

        es.restore_best(model)
        self.assertEqual(model.value, 20.0)  # best was at score=0.7

    def test_restore_noop_when_no_step(self):
        """restore_best is a no-op if step() was never called."""
        model = MockModel(value=42.0)
        es = EarlyStopping(patience=2)
        es.restore_best(model)
        self.assertEqual(model.value, 42.0)


class TestSetupLogging(unittest.TestCase):
    """setup_logging creates directory, log file, and returns logger."""

    def test_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "subdir", "logs")
            # Clear existing handlers to avoid interference
            logger = logging.getLogger("lg_cotrain")
            logger.handlers.clear()
            setup_logging(out)
            self.assertTrue(os.path.isdir(out))

    def test_creates_log_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "log_test")
            logger = logging.getLogger("lg_cotrain")
            logger.handlers.clear()
            setup_logging(out)
            logger.info("test message")
            log_path = os.path.join(out, "experiment.log")
            self.assertTrue(os.path.exists(log_path))

    def test_returns_logger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "logger_test")
            logger = logging.getLogger("lg_cotrain")
            logger.handlers.clear()
            result = setup_logging(out)
            self.assertIsInstance(result, logging.Logger)
            self.assertEqual(result.name, "lg_cotrain")


if __name__ == "__main__":
    unittest.main(verbosity=2)
