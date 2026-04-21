"""Tests for supervised_baseline.optuna_tuner.

Pure-Python tests that mock optuna and the trainer. No GPU required.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_result(event="test_event", budget=5, seed_set=1, **overrides):
    """Create a fake metrics result dict as SupervisedTrainer.run() returns."""
    result = {
        "event": event,
        "budget": budget,
        "seed_set": seed_set,
        "method": "supervised",
        "model_name": "vinai/bertweet-base",
        "test_error_rate": 25.0,
        "test_macro_f1": 0.75,
        "test_ece": 0.05,
        "test_per_class_f1": [0.7, 0.8],
        "dev_error_rate": 20.0,
        "dev_macro_f1": 0.80,
        "dev_ece": 0.04,
        "dev_per_class_f1": [0.75, 0.85],
        "epochs_trained": 30,
        "best_epoch": 20,
        "best_dev_macro_f1": 0.80,
        "training_time_seconds": 100.0,
        "lr": 0.0001,
        "batch_size": 32,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_seq_length": 128,
        "max_epochs": 50,
        "patience": 5,
        "num_labels": 10,
        "class_labels": ["a", "b"],
    }
    result.update(overrides)
    return result


def _make_trial_info(number, dev_f1=0.80, **param_overrides):
    """Create a fake trial info dict as stored in best_params.json."""
    params = {
        "lr": 0.0001,
        "batch_size": 32,
        "max_epochs": 50,
    }
    params.update(param_overrides)
    return {
        "number": number,
        "state": "COMPLETE",
        "params": params,
        "dev_macro_f1": dev_f1,
    }


class MockTrial:
    """Mock Optuna trial that records suggest calls."""

    def __init__(self):
        self.suggestions = {}
        self.user_attrs_store = {}
        self.number = 0

    def suggest_float(self, name, low, high, log=False):
        val = (low + high) / 2 if not log else (low * high) ** 0.5
        self.suggestions[name] = val
        return val

    def suggest_categorical(self, name, choices):
        val = choices[0]
        self.suggestions[name] = val
        return val

    def suggest_int(self, name, low, high):
        val = (low + high) // 2
        self.suggestions[name] = val
        return val

    def set_user_attr(self, key, value):
        self.user_attrs_store[key] = value


class TestSearchSpace(unittest.TestCase):
    """Verify objective samples 3 hyperparameters from a trial."""

    def test_objective_samples_3_params(self):
        from supervised_baseline.optuna_tuner import create_supervised_objective

        mock_trainer = MagicMock()
        mock_trainer.return_value.run.return_value = _make_result()

        objective = create_supervised_objective(
            event="test_event", budget=5, seed_set=1,
            _trainer_cls=mock_trainer,
        )

        trial = MockTrial()
        objective(trial)

        expected_params = {"lr", "batch_size", "max_epochs"}
        self.assertEqual(set(trial.suggestions.keys()), expected_params)

    def test_lr_is_log_scale(self):
        from supervised_baseline.optuna_tuner import create_supervised_objective

        mock_trainer = MagicMock()
        mock_trainer.return_value.run.return_value = _make_result()

        objective = create_supervised_objective(
            event="test_event", budget=5, seed_set=1,
            _trainer_cls=mock_trainer,
        )

        log_params = []
        trial = MockTrial()
        original_suggest_float = trial.suggest_float

        def tracking_suggest_float(name, low, high, log=False):
            if log:
                log_params.append(name)
            return original_suggest_float(name, low, high, log=log)

        trial.suggest_float = tracking_suggest_float
        objective(trial)

        self.assertIn("lr", log_params)


class TestSearchSpaceConstant(unittest.TestCase):
    """Verify the SEARCH_SPACE dict matches what the objective uses."""

    def test_search_space_has_3_params(self):
        from supervised_baseline.optuna_tuner import SEARCH_SPACE

        self.assertEqual(len(SEARCH_SPACE), 3)
        expected = {"lr", "batch_size", "max_epochs"}
        self.assertEqual(set(SEARCH_SPACE.keys()), expected)

    def test_lr_is_log(self):
        from supervised_baseline.optuna_tuner import SEARCH_SPACE
        self.assertTrue(SEARCH_SPACE["lr"].get("log", False))

    def test_batch_size_is_categorical(self):
        from supervised_baseline.optuna_tuner import SEARCH_SPACE
        self.assertEqual(SEARCH_SPACE["batch_size"]["type"], "categorical")
        self.assertEqual(SEARCH_SPACE["batch_size"]["choices"], [8, 16, 32, 64])

    def test_max_epochs_is_int(self):
        from supervised_baseline.optuna_tuner import SEARCH_SPACE
        self.assertEqual(SEARCH_SPACE["max_epochs"]["type"], "int")
        self.assertEqual(SEARCH_SPACE["max_epochs"]["low"], 20)
        self.assertEqual(SEARCH_SPACE["max_epochs"]["high"], 100)


class TestObjectiveReturnValue(unittest.TestCase):
    """Verify objective returns dev_macro_f1."""

    def test_returns_dev_macro_f1(self):
        from supervised_baseline.optuna_tuner import create_supervised_objective

        mock_trainer = MagicMock()
        mock_trainer.return_value.run.return_value = _make_result(
            dev_macro_f1=0.8765,
        )

        objective = create_supervised_objective(
            event="test_event", budget=5, seed_set=1,
            _trainer_cls=mock_trainer,
        )

        trial = MockTrial()
        result = objective(trial)

        self.assertEqual(result, 0.8765)


class TestObjectiveUsesDevNotTest(unittest.TestCase):
    """Verify the objective uses dev_macro_f1, not test_macro_f1."""

    def test_uses_dev_not_test(self):
        from supervised_baseline.optuna_tuner import create_supervised_objective

        mock_trainer = MagicMock()
        mock_trainer.return_value.run.return_value = _make_result(
            dev_macro_f1=0.80,
            test_macro_f1=0.90,
        )

        objective = create_supervised_objective(
            event="test_event", budget=5, seed_set=1,
            _trainer_cls=mock_trainer,
        )

        trial = MockTrial()
        result = objective(trial)

        self.assertEqual(result, 0.80)
        self.assertNotEqual(result, 0.90)


class TestObjectiveAttachesFullMetrics(unittest.TestCase):
    """Verify the objective saves full metrics to trial.user_attrs."""

    def test_full_metrics_attached(self):
        from supervised_baseline.optuna_tuner import create_supervised_objective

        mock_trainer = MagicMock()
        full_result = _make_result(dev_macro_f1=0.80, test_macro_f1=0.90)
        mock_trainer.return_value.run.return_value = full_result

        objective = create_supervised_objective(
            event="test_event", budget=5, seed_set=1,
            _trainer_cls=mock_trainer,
        )

        trial = MockTrial()
        objective(trial)

        self.assertIn("full_metrics", trial.user_attrs_store)
        self.assertEqual(trial.user_attrs_store["full_metrics"]["test_macro_f1"], 0.90)


class TestBestParamsSaved(unittest.TestCase):
    """Verify best_params.json is written after study completes."""

    def test_saves_best_params_json(self):
        try:
            import optuna  # noqa: F401
        except ImportError:
            self.skipTest("optuna not installed")

        from supervised_baseline.optuna_tuner import run_single_study

        mock_trainer = MagicMock()
        call_count = [0]

        def fake_run():
            call_count[0] += 1
            return _make_result(dev_macro_f1=0.70 + call_count[0] * 0.01)

        mock_trainer.return_value.run = fake_run

        with tempfile.TemporaryDirectory() as tmpdir:
            run_single_study(
                event="test_event", budget=5, seed_set=1,
                n_trials=3, storage_dir=tmpdir,
                _trainer_cls=mock_trainer,
            )

            best_path = Path(tmpdir) / "test_event" / "5_set1" / "trials_3" / "best_params.json"
            self.assertTrue(best_path.exists())

            with open(best_path) as f:
                saved = json.load(f)

            self.assertEqual(saved["event"], "test_event")
            self.assertEqual(saved["budget"], 5)
            self.assertEqual(saved["seed_set"], 1)
            self.assertEqual(saved["status"], "done")
            self.assertIsNotNone(saved["best_params"])
            self.assertIsNotNone(saved["best_value"])
            self.assertEqual(saved["n_trials"], 3)
            self.assertEqual(len(saved["trials"]), 3)
            # best_full_metrics should be attached
            self.assertIsNotNone(saved["best_full_metrics"])

    def test_skips_existing_study(self):
        from supervised_baseline.optuna_tuner import run_single_study

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "test_event" / "5_set1" / "trials_10"
            out_dir.mkdir(parents=True)
            existing = {
                "event": "test_event", "budget": 5, "seed_set": 1,
                "status": "done", "best_params": {"lr": 0.001},
                "best_value": 0.85, "n_trials": 10, "trials": [],
            }
            (out_dir / "best_params.json").write_text(json.dumps(existing))

            result = run_single_study(
                event="test_event", budget=5, seed_set=1,
                n_trials=10, storage_dir=tmpdir,
            )

            self.assertEqual(result["best_value"], 0.85)


class TestIncrementalTrials(unittest.TestCase):
    """Verify incremental trial support works with supervised tuner."""

    def test_continues_from_previous(self):
        try:
            import optuna  # noqa: F401
        except ImportError:
            self.skipTest("optuna not installed")

        from supervised_baseline.optuna_tuner import run_single_study

        mock_trainer = MagicMock()
        new_call_count = [0]

        def fake_run():
            new_call_count[0] += 1
            return _make_result(dev_macro_f1=0.80 + new_call_count[0] * 0.005)

        mock_trainer.return_value.run = fake_run

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_event" / "5_set1"
            d = experiment_dir / "trials_5"
            d.mkdir(parents=True)
            prev_trials = [_make_trial_info(i, dev_f1=0.70 + i * 0.01) for i in range(5)]
            prev_data = {
                "event": "test_event", "budget": 5, "seed_set": 1,
                "status": "done", "best_params": {"lr": 0.001},
                "best_value": 0.74, "n_trials": 5,
                "trials": prev_trials,
            }
            (d / "best_params.json").write_text(json.dumps(prev_data))

            result = run_single_study(
                event="test_event", budget=5, seed_set=1,
                n_trials=10, storage_dir=tmpdir,
                _trainer_cls=mock_trainer,
            )

            # Only 5 new trials should have run
            self.assertEqual(new_call_count[0], 5)
            self.assertEqual(result["n_trials"], 10)
            self.assertEqual(len(result["trials"]), 10)
            self.assertEqual(result["continued_from"], 5)

            trials_10_path = experiment_dir / "trials_10" / "best_params.json"
            self.assertTrue(trials_10_path.exists())

    def test_no_previous_runs_all(self):
        try:
            import optuna  # noqa: F401
        except ImportError:
            self.skipTest("optuna not installed")

        from supervised_baseline.optuna_tuner import run_single_study

        mock_trainer = MagicMock()
        call_count = [0]

        def fake_run():
            call_count[0] += 1
            return _make_result(dev_macro_f1=0.70 + call_count[0] * 0.01)

        mock_trainer.return_value.run = fake_run

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_single_study(
                event="test_event", budget=5, seed_set=1,
                n_trials=3, storage_dir=tmpdir,
                _trainer_cls=mock_trainer,
            )

            self.assertEqual(call_count[0], 3)
            self.assertEqual(result["n_trials"], 3)
            self.assertIsNone(result["continued_from"])


class TestBuildDistributions(unittest.TestCase):
    """Verify _build_distributions produces the 3 expected Optuna distributions."""

    def test_three_distributions(self):
        try:
            import optuna  # noqa: F401
        except ImportError:
            self.skipTest("optuna not installed")

        from supervised_baseline.optuna_tuner import _build_distributions

        dists = _build_distributions()
        self.assertEqual(set(dists.keys()), {"lr", "batch_size", "max_epochs"})


if __name__ == "__main__":
    unittest.main()
