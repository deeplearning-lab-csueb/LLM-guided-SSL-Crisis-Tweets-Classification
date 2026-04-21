"""Tests for per-experiment Optuna hyperparameter tuning.

Pure-Python tests that mock optuna and the trainer. No GPU required.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_result(event="test_event", budget=5, seed_set=1, **overrides):
    """Create a fake metrics result dict."""
    result = {
        "event": event,
        "budget": budget,
        "seed_set": seed_set,
        "test_error_rate": 25.0,
        "test_macro_f1": 0.75,
        "test_ece": 0.05,
        "test_per_class_f1": [0.7, 0.8],
        "dev_error_rate": 20.0,
        "dev_macro_f1": 0.80,
        "dev_ece": 0.04,
        "stopping_strategy": "baseline",
        "lambda1_mean": 0.5,
        "lambda1_std": 0.1,
        "lambda2_mean": 0.4,
        "lambda2_std": 0.1,
    }
    result.update(overrides)
    return result


def _make_trial_info(number, dev_f1=0.80, **param_overrides):
    """Create a fake trial info dict as stored in best_params.json."""
    params = {
        "lr": 0.0001,
        "batch_size": 32,
        "cotrain_epochs": 10,
        "finetune_patience": 5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
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


class TestSearchSpace(unittest.TestCase):
    """Verify objective samples 6 hyperparameters from a trial."""

    def test_objective_samples_6_params(self):
        from lg_cotrain.optuna_per_experiment import create_per_experiment_objective

        mock_trainer = MagicMock()
        mock_trainer.return_value.run.return_value = _make_result()

        objective = create_per_experiment_objective(
            event="test_event", budget=5, seed_set=1,
            _trainer_cls=mock_trainer,
        )

        trial = MockTrial()
        objective(trial)

        expected_params = {"lr", "batch_size", "cotrain_epochs",
                           "finetune_patience", "weight_decay", "warmup_ratio"}
        self.assertEqual(set(trial.suggestions.keys()), expected_params)

    def test_lr_is_log_scale(self):
        """LR should be sampled with log=True."""
        from lg_cotrain.optuna_per_experiment import create_per_experiment_objective

        mock_trainer = MagicMock()
        mock_trainer.return_value.run.return_value = _make_result()

        objective = create_per_experiment_objective(
            event="test_event", budget=5, seed_set=1,
            _trainer_cls=mock_trainer,
        )

        # Use a real-ish mock trial that tracks log param
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

    def test_search_space_has_6_params(self):
        from lg_cotrain.optuna_per_experiment import SEARCH_SPACE

        self.assertEqual(len(SEARCH_SPACE), 6)
        expected = {"lr", "batch_size", "cotrain_epochs",
                    "finetune_patience", "weight_decay", "warmup_ratio"}
        self.assertEqual(set(SEARCH_SPACE.keys()), expected)

    def test_lr_is_log(self):
        from lg_cotrain.optuna_per_experiment import SEARCH_SPACE
        self.assertTrue(SEARCH_SPACE["lr"].get("log", False))

    def test_batch_size_is_categorical(self):
        from lg_cotrain.optuna_per_experiment import SEARCH_SPACE
        self.assertEqual(SEARCH_SPACE["batch_size"]["type"], "categorical")
        self.assertEqual(SEARCH_SPACE["batch_size"]["choices"], [8, 16, 32, 64])


class TestObjectiveReturnValue(unittest.TestCase):
    """Verify objective returns dev_macro_f1."""

    def test_returns_dev_macro_f1(self):
        from lg_cotrain.optuna_per_experiment import create_per_experiment_objective

        mock_trainer = MagicMock()
        mock_trainer.return_value.run.return_value = _make_result(
            dev_macro_f1=0.8765,
        )

        objective = create_per_experiment_objective(
            event="test_event", budget=5, seed_set=1,
            _trainer_cls=mock_trainer,
        )

        trial = MockTrial()
        result = objective(trial)

        self.assertEqual(result, 0.8765)


class TestObjectiveUsesDevNotTest(unittest.TestCase):
    """Verify the objective uses dev_macro_f1, not test_macro_f1."""

    def test_uses_dev_not_test(self):
        from lg_cotrain.optuna_per_experiment import create_per_experiment_objective

        mock_trainer = MagicMock()
        mock_trainer.return_value.run.return_value = _make_result(
            dev_macro_f1=0.80,
            test_macro_f1=0.90,  # different from dev
        )

        objective = create_per_experiment_objective(
            event="test_event", budget=5, seed_set=1,
            _trainer_cls=mock_trainer,
        )

        trial = MockTrial()
        result = objective(trial)

        # Should return dev, not test
        self.assertEqual(result, 0.80)
        self.assertNotEqual(result, 0.90)


class TestBestParamsSaved(unittest.TestCase):
    """Verify best_params.json is written after study completes."""

    def test_saves_best_params_json(self):
        try:
            import optuna
        except ImportError:
            self.skipTest("optuna not installed")

        from lg_cotrain.optuna_per_experiment import run_single_study

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

            # Check file was created under trials_3/
            best_path = Path(tmpdir) / "test_event" / "5_set1" / "trials_3" / "best_params.json"
            self.assertTrue(best_path.exists())

            # Check contents
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

    def test_skips_existing_study(self):
        """If trials_{n}/best_params.json exists, study is skipped."""
        from lg_cotrain.optuna_per_experiment import run_single_study

        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create trials_15/best_params.json
            out_dir = Path(tmpdir) / "test_event" / "5_set1" / "trials_15"
            out_dir.mkdir(parents=True)
            existing = {
                "event": "test_event", "budget": 5, "seed_set": 1,
                "status": "done", "best_params": {"lr": 0.001},
                "best_value": 0.85, "n_trials": 15, "trials": [],
            }
            (out_dir / "best_params.json").write_text(json.dumps(existing))

            result = run_single_study(
                event="test_event", budget=5, seed_set=1,
                n_trials=15, storage_dir=tmpdir,
            )

            # Should return existing data
            self.assertEqual(result["best_value"], 0.85)


class TestFindLatestTrials(unittest.TestCase):
    """Verify _find_latest_trials finds the highest trial-count folder."""

    def test_finds_highest(self):
        from lg_cotrain.optuna_per_experiment import _find_latest_trials

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_event" / "5_set1"

            for n in [5, 10, 20]:
                d = experiment_dir / f"trials_{n}"
                d.mkdir(parents=True)
                data = {
                    "event": "test_event", "budget": 5, "seed_set": 1,
                    "status": "done", "best_params": {"lr": 0.001},
                    "best_value": 0.80 + n * 0.001, "n_trials": n,
                    "trials": [_make_trial_info(i) for i in range(n)],
                }
                (d / "best_params.json").write_text(json.dumps(data))

            result = _find_latest_trials(experiment_dir)
            self.assertIsNotNone(result)
            n, data = result
            self.assertEqual(n, 20)
            self.assertEqual(data["n_trials"], 20)

    def test_empty_dir(self):
        from lg_cotrain.optuna_per_experiment import _find_latest_trials

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_event" / "5_set1"
            experiment_dir.mkdir(parents=True)

            result = _find_latest_trials(experiment_dir)
            self.assertIsNone(result)

    def test_nonexistent_dir(self):
        from lg_cotrain.optuna_per_experiment import _find_latest_trials

        result = _find_latest_trials(Path("/nonexistent/path"))
        self.assertIsNone(result)

    def test_ignores_non_trial_dirs(self):
        """Folders not matching trials_* pattern are ignored."""
        from lg_cotrain.optuna_per_experiment import _find_latest_trials

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_event" / "5_set1"

            # Create a non-matching directory
            other = experiment_dir / "some_other_folder"
            other.mkdir(parents=True)
            (other / "best_params.json").write_text("{}")

            # Create one matching
            d = experiment_dir / "trials_5"
            d.mkdir(parents=True)
            data = {
                "event": "test_event", "budget": 5, "seed_set": 1,
                "status": "done", "best_params": {"lr": 0.001},
                "best_value": 0.80, "n_trials": 5,
                "trials": [_make_trial_info(i) for i in range(5)],
            }
            (d / "best_params.json").write_text(json.dumps(data))

            result = _find_latest_trials(experiment_dir)
            self.assertIsNotNone(result)
            n, _ = result
            self.assertEqual(n, 5)


class TestIncrementalTrials(unittest.TestCase):
    """Verify incremental trial support."""

    def test_continues_from_previous(self):
        """Run 20 trials when 10 already exist → only 10 new trials run."""
        try:
            import optuna
        except ImportError:
            self.skipTest("optuna not installed")

        from lg_cotrain.optuna_per_experiment import run_single_study

        mock_trainer = MagicMock()
        new_call_count = [0]

        def fake_run():
            new_call_count[0] += 1
            return _make_result(dev_macro_f1=0.80 + new_call_count[0] * 0.005)

        mock_trainer.return_value.run = fake_run

        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create trials_10 with 10 trial records
            experiment_dir = Path(tmpdir) / "test_event" / "5_set1"
            d = experiment_dir / "trials_10"
            d.mkdir(parents=True)
            prev_trials = [_make_trial_info(i, dev_f1=0.70 + i * 0.01) for i in range(10)]
            prev_data = {
                "event": "test_event", "budget": 5, "seed_set": 1,
                "status": "done", "best_params": {"lr": 0.001},
                "best_value": 0.79, "n_trials": 10,
                "trials": prev_trials,
            }
            (d / "best_params.json").write_text(json.dumps(prev_data))

            # Now run with n_trials=20
            result = run_single_study(
                event="test_event", budget=5, seed_set=1,
                n_trials=20, storage_dir=tmpdir,
                _trainer_cls=mock_trainer,
            )

            # Only 10 new trials should have run
            self.assertEqual(new_call_count[0], 10)

            # Result should have all 20 trials
            self.assertEqual(result["n_trials"], 20)
            self.assertEqual(len(result["trials"]), 20)
            self.assertEqual(result["status"], "done")
            self.assertEqual(result["continued_from"], 10)

            # File should be saved under trials_20/
            trials_20_path = experiment_dir / "trials_20" / "best_params.json"
            self.assertTrue(trials_20_path.exists())

    def test_skips_exact_match(self):
        """If trials_{n} already exists, skip (no optuna needed)."""
        from lg_cotrain.optuna_per_experiment import run_single_study

        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create trials_15
            experiment_dir = Path(tmpdir) / "test_event" / "5_set1"
            d = experiment_dir / "trials_15"
            d.mkdir(parents=True)
            existing = {
                "event": "test_event", "budget": 5, "seed_set": 1,
                "status": "done", "best_params": {"lr": 0.001},
                "best_value": 0.85, "n_trials": 15, "trials": [],
            }
            (d / "best_params.json").write_text(json.dumps(existing))

            result = run_single_study(
                event="test_event", budget=5, seed_set=1,
                n_trials=15, storage_dir=tmpdir,
            )

            self.assertEqual(result["best_value"], 0.85)

    def test_no_previous_runs_all(self):
        """With no previous trials, all n_trials run from scratch."""
        try:
            import optuna
        except ImportError:
            self.skipTest("optuna not installed")

        from lg_cotrain.optuna_per_experiment import run_single_study

        mock_trainer = MagicMock()
        call_count = [0]

        def fake_run():
            call_count[0] += 1
            return _make_result(dev_macro_f1=0.70 + call_count[0] * 0.01)

        mock_trainer.return_value.run = fake_run

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_single_study(
                event="test_event", budget=5, seed_set=1,
                n_trials=5, storage_dir=tmpdir,
                _trainer_cls=mock_trainer,
            )

            self.assertEqual(call_count[0], 5)
            self.assertEqual(result["n_trials"], 5)
            self.assertIsNone(result["continued_from"])

    def test_returns_previous_when_enough_trials(self):
        """If previous trials >= n_trials, return previous results."""
        try:
            import optuna
        except ImportError:
            self.skipTest("optuna not installed")

        from lg_cotrain.optuna_per_experiment import run_single_study

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create trials_20 (more than we'll request)
            experiment_dir = Path(tmpdir) / "test_event" / "5_set1"
            d = experiment_dir / "trials_20"
            d.mkdir(parents=True)
            prev_data = {
                "event": "test_event", "budget": 5, "seed_set": 1,
                "status": "done", "best_params": {"lr": 0.0005},
                "best_value": 0.90, "n_trials": 20,
                "trials": [_make_trial_info(i) for i in range(20)],
            }
            (d / "best_params.json").write_text(json.dumps(prev_data))

            # Request only 15 — should return the 20-trial results
            result = run_single_study(
                event="test_event", budget=5, seed_set=1,
                n_trials=15, storage_dir=tmpdir,
            )

            self.assertEqual(result["best_value"], 0.90)
            self.assertEqual(result["n_trials"], 20)


class TestReplayTrials(unittest.TestCase):
    """Verify _replay_trials_into_study adds trials correctly."""

    def test_replays_correct_count(self):
        try:
            import optuna
        except ImportError:
            self.skipTest("optuna not installed")

        from lg_cotrain.optuna_per_experiment import _replay_trials_into_study

        study = optuna.create_study(direction="maximize")
        trials_info = [_make_trial_info(i, dev_f1=0.70 + i * 0.01) for i in range(5)]

        _replay_trials_into_study(study, trials_info)

        self.assertEqual(len(study.trials), 5)
        # Best value should be the highest
        self.assertAlmostEqual(study.best_value, 0.74, places=2)

    def test_skips_non_complete_trials(self):
        try:
            import optuna
        except ImportError:
            self.skipTest("optuna not installed")

        from lg_cotrain.optuna_per_experiment import _replay_trials_into_study

        study = optuna.create_study(direction="maximize")
        trials_info = [
            _make_trial_info(0, dev_f1=0.80),
            {"number": 1, "state": "FAIL", "params": {}, "dev_macro_f1": 0.0},
            _make_trial_info(2, dev_f1=0.85),
        ]

        _replay_trials_into_study(study, trials_info)

        # Only 2 COMPLETE trials should be added
        self.assertEqual(len(study.trials), 2)


class TestStudyWorkerPicklable(unittest.TestCase):
    """Verify _run_study_worker receives a plain dict."""

    def test_worker_receives_dict(self):
        """Worker function signature accepts a dict."""
        from lg_cotrain.optuna_per_experiment import _run_study_worker
        import inspect

        sig = inspect.signature(_run_study_worker)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["kwargs"])

        # Check the annotation is dict
        param = sig.parameters["kwargs"]
        self.assertEqual(param.annotation, dict)


class TestGPUAssignmentForStudies(unittest.TestCase):
    """Verify dynamic GPU assignment across studies."""

    def test_2_gpus_initial_seeding(self):
        """First num_gpus studies each get a unique GPU."""
        configs = [{"event": "e", "budget": b, "seed_set": s}
                   for b in [5, 10] for s in [1, 2, 3]]
        num_gpus = 2
        for gpu_id in range(min(num_gpus, len(configs))):
            configs[gpu_id]["device"] = f"cuda:{gpu_id}"
        self.assertEqual(configs[0]["device"], "cuda:0")
        self.assertEqual(configs[1]["device"], "cuda:1")

    def test_3_gpus_3_studies(self):
        """With 3 GPUs and 3 studies, each gets a unique GPU."""
        configs = [{"event": "e", "budget": 5, "seed_set": s}
                   for s in [1, 2, 3]]
        num_gpus = 3
        for gpu_id in range(min(num_gpus, len(configs))):
            configs[gpu_id]["device"] = f"cuda:{gpu_id}"
        devices = [c["device"] for c in configs]
        self.assertEqual(devices, ["cuda:0", "cuda:1", "cuda:2"])

    def test_freed_gpu_reused(self):
        """When a GPU finishes, the next study gets that same GPU."""
        config_queue = list(range(4))
        num_gpus = 2
        active = {}
        assigned = []

        for gpu_id in range(min(num_gpus, len(config_queue))):
            idx = config_queue.pop(0)
            active[idx] = gpu_id
            assigned.append((idx, gpu_id))

        # GPU 0 finishes -> task 2 gets GPU 0
        freed_gpu = active.pop(0)
        next_idx = config_queue.pop(0)
        active[next_idx] = freed_gpu
        assigned.append((next_idx, freed_gpu))

        self.assertEqual(assigned[2], (2, 0))  # task 2 on GPU 0


class TestAllStudiesSkipsCompleted(unittest.TestCase):
    """Verify run_all_studies skips experiments with existing results."""

    def test_all_skipped_no_worker_call(self):
        from lg_cotrain.optuna_per_experiment import run_all_studies

        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create all results for a small grid under trials_10/
            for seed_set in [1, 2]:
                out_dir = Path(tmpdir) / "test_event" / f"5_set{seed_set}" / "trials_10"
                out_dir.mkdir(parents=True)
                result = {
                    "event": "test_event", "budget": 5, "seed_set": seed_set,
                    "status": "done", "best_params": {"lr": 0.001},
                    "best_value": 0.85, "n_trials": 10, "trials": [],
                }
                (out_dir / "best_params.json").write_text(json.dumps(result))

            with patch("lg_cotrain.optuna_per_experiment._run_study_worker") as mock_worker:
                results = run_all_studies(
                    events=["test_event"],
                    budgets=[5],
                    seed_sets=[1, 2],
                    n_trials=10,
                    storage_dir=tmpdir,
                )

            mock_worker.assert_not_called()
            self.assertEqual(len(results), 2)
            self.assertTrue(all(r["status"] == "done" for r in results))

    def test_partial_skip(self):
        """Only pending studies should be run."""
        from lg_cotrain.optuna_per_experiment import run_all_studies

        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create seed_set=1 only under trials_10/
            out_dir = Path(tmpdir) / "test_event" / "5_set1" / "trials_10"
            out_dir.mkdir(parents=True)
            existing = {
                "event": "test_event", "budget": 5, "seed_set": 1,
                "status": "done", "best_params": {"lr": 0.001},
                "best_value": 0.85, "n_trials": 10, "trials": [],
            }
            (out_dir / "best_params.json").write_text(json.dumps(existing))

            new_result = {
                "event": "test_event", "budget": 5, "seed_set": 2,
                "status": "done", "best_params": {"lr": 0.0005},
                "best_value": 0.82, "n_trials": 10, "trials": [],
            }

            with patch("lg_cotrain.optuna_per_experiment._run_study_worker") as mock_worker:
                mock_worker.return_value = new_result
                results = run_all_studies(
                    events=["test_event"],
                    budgets=[5],
                    seed_sets=[1, 2],
                    n_trials=10,
                    storage_dir=tmpdir,
                )

            # Only 1 study should have been dispatched
            self.assertEqual(mock_worker.call_count, 1)
            # Both results should be present
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["best_value"], 0.85)  # skipped
            self.assertEqual(results[1]["best_value"], 0.82)  # ran

    def test_summary_json_written(self):
        """summary_{n}.json should be written after all studies complete."""
        from lg_cotrain.optuna_per_experiment import run_all_studies

        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create all results
            for seed_set in [1, 2]:
                out_dir = Path(tmpdir) / "test_event" / f"5_set{seed_set}" / "trials_10"
                out_dir.mkdir(parents=True)
                result = {
                    "event": "test_event", "budget": 5, "seed_set": seed_set,
                    "status": "done", "best_params": {"lr": 0.001},
                    "best_value": 0.85, "n_trials": 10, "trials": [],
                }
                (out_dir / "best_params.json").write_text(json.dumps(result))

            run_all_studies(
                events=["test_event"],
                budgets=[5],
                seed_sets=[1, 2],
                n_trials=10,
                storage_dir=tmpdir,
            )

            summary_path = Path(tmpdir) / "summary_10.json"
            self.assertTrue(summary_path.exists())

            with open(summary_path) as f:
                summary = json.load(f)

            self.assertEqual(summary["total_studies"], 2)
            self.assertEqual(len(summary["studies"]), 2)
            self.assertEqual(summary["n_trials_per_study"], 10)


class TestLoadBestParams(unittest.TestCase):
    """Verify load_best_params reads correct files."""

    def test_loads_existing(self):
        from lg_cotrain.optuna_per_experiment import load_best_params

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 2 best_params.json files under trials_10/
            for seed_set in [1, 2]:
                out_dir = Path(tmpdir) / "test_event" / f"5_set{seed_set}" / "trials_10"
                out_dir.mkdir(parents=True)
                data = {
                    "event": "test_event", "budget": 5, "seed_set": seed_set,
                    "status": "done", "best_params": {"lr": 0.001 * seed_set},
                    "best_value": 0.80 + seed_set * 0.01,
                    "n_trials": 10, "trials": [],
                }
                (out_dir / "best_params.json").write_text(json.dumps(data))

            results = load_best_params(
                storage_dir=tmpdir,
                events=["test_event"],
                budgets=[5],
                seed_sets=[1, 2],
                n_trials=10,
            )

            self.assertEqual(len(results), 2)
            self.assertIn(("test_event", 5, 1), results)
            self.assertIn(("test_event", 5, 2), results)
            self.assertEqual(results[("test_event", 5, 1)]["best_params"]["lr"], 0.001)
            self.assertEqual(results[("test_event", 5, 2)]["best_params"]["lr"], 0.002)

    def test_missing_files_skipped(self):
        """Missing best_params.json files are silently skipped."""
        from lg_cotrain.optuna_per_experiment import load_best_params

        with tempfile.TemporaryDirectory() as tmpdir:
            # Only create seed_set=1
            out_dir = Path(tmpdir) / "test_event" / "5_set1" / "trials_10"
            out_dir.mkdir(parents=True)
            data = {
                "event": "test_event", "budget": 5, "seed_set": 1,
                "status": "done", "best_params": {"lr": 0.001},
                "best_value": 0.85, "n_trials": 10, "trials": [],
            }
            (out_dir / "best_params.json").write_text(json.dumps(data))

            results = load_best_params(
                storage_dir=tmpdir,
                events=["test_event"],
                budgets=[5],
                seed_sets=[1, 2, 3],
                n_trials=10,
            )

            self.assertEqual(len(results), 1)
            self.assertIn(("test_event", 5, 1), results)
            self.assertNotIn(("test_event", 5, 2), results)


class TestLoadBestParamsWithTrials(unittest.TestCase):
    """Verify load_best_params with n_trials parameter."""

    def test_loads_specific_n_trials(self):
        """load_best_params(n_trials=10) loads from trials_10/."""
        from lg_cotrain.optuna_per_experiment import load_best_params

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_event" / "5_set1"

            # Create trials_10 and trials_20
            for n in [10, 20]:
                d = experiment_dir / f"trials_{n}"
                d.mkdir(parents=True)
                data = {
                    "event": "test_event", "budget": 5, "seed_set": 1,
                    "status": "done", "best_params": {"lr": 0.001 if n == 10 else 0.0005},
                    "best_value": 0.80 if n == 10 else 0.85,
                    "n_trials": n, "trials": [],
                }
                (d / "best_params.json").write_text(json.dumps(data))

            results = load_best_params(
                storage_dir=tmpdir,
                events=["test_event"],
                budgets=[5],
                seed_sets=[1],
                n_trials=10,
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[("test_event", 5, 1)]["best_value"], 0.80)
            self.assertEqual(results[("test_event", 5, 1)]["n_trials"], 10)

    def test_loads_latest_by_default(self):
        """load_best_params() without n_trials loads the latest."""
        from lg_cotrain.optuna_per_experiment import load_best_params

        with tempfile.TemporaryDirectory() as tmpdir:
            experiment_dir = Path(tmpdir) / "test_event" / "5_set1"

            for n in [10, 20]:
                d = experiment_dir / f"trials_{n}"
                d.mkdir(parents=True)
                data = {
                    "event": "test_event", "budget": 5, "seed_set": 1,
                    "status": "done", "best_params": {"lr": 0.001 if n == 10 else 0.0005},
                    "best_value": 0.80 if n == 10 else 0.85,
                    "n_trials": n, "trials": [],
                }
                (d / "best_params.json").write_text(json.dumps(data))

            results = load_best_params(
                storage_dir=tmpdir,
                events=["test_event"],
                budgets=[5],
                seed_sets=[1],
            )

            self.assertEqual(len(results), 1)
            # Should load from trials_20 (latest)
            self.assertEqual(results[("test_event", 5, 1)]["best_value"], 0.85)
            self.assertEqual(results[("test_event", 5, 1)]["n_trials"], 20)


class TestCLIFlags(unittest.TestCase):
    """CLI flags are accepted and forwarded."""

    def test_help_flag(self):
        """--help should not crash."""
        from lg_cotrain.optuna_per_experiment import main
        with self.assertRaises(SystemExit) as ctx:
            with patch("sys.argv", ["prog", "--help"]):
                main()
        self.assertEqual(ctx.exception.code, 0)

    def test_n_trials_forwarded(self):
        from lg_cotrain.optuna_per_experiment import main

        with patch("lg_cotrain.optuna_per_experiment.run_all_studies") as mock_run:
            mock_run.return_value = []
            with patch("sys.argv", [
                "prog", "--n-trials", "25", "--num-gpus", "2",
                "--events", "test_event",
            ]):
                main()

            call_kwargs = mock_run.call_args
            self.assertEqual(call_kwargs.kwargs.get("n_trials"), 25)
            self.assertEqual(call_kwargs.kwargs.get("num_gpus"), 2)
            self.assertEqual(call_kwargs.kwargs.get("events"), ["test_event"])

    def test_model_name_forwarded(self):
        """--model-name is forwarded to run_all_studies."""
        from lg_cotrain.optuna_per_experiment import main

        with patch("lg_cotrain.optuna_per_experiment.run_all_studies") as mock_run:
            mock_run.return_value = []
            with patch("sys.argv", [
                "prog", "--n-trials", "5",
                "--events", "test_event",
                "--model-name", "vinai/bertweet-base",
            ]):
                main()

            call_kwargs = mock_run.call_args
            self.assertEqual(
                call_kwargs.kwargs.get("model_name"), "vinai/bertweet-base"
            )

    def test_model_name_default_none(self):
        """If --model-name is omitted, model_name kwarg is None (use config default)."""
        from lg_cotrain.optuna_per_experiment import main

        with patch("lg_cotrain.optuna_per_experiment.run_all_studies") as mock_run:
            mock_run.return_value = []
            with patch("sys.argv", [
                "prog", "--n-trials", "5", "--events", "test_event",
            ]):
                main()

            call_kwargs = mock_run.call_args
            self.assertIsNone(call_kwargs.kwargs.get("model_name"))


# ------------------------------------------------------------------
# Tests for the test-metrics-on-trial shortcut (added 2026-04-06)
# ------------------------------------------------------------------


class MockTrialWithUserAttrs(MockTrial):
    """MockTrial that also supports set_user_attr / user_attrs."""

    def __init__(self):
        super().__init__()
        self.user_attrs = {}

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


class TestModelNamePlumbing(unittest.TestCase):
    """Verify model_name flows from create_per_experiment_objective into the
    LGCoTrainConfig that's passed to the trainer."""

    def test_model_name_passed_to_trainer_config(self):
        from lg_cotrain.optuna_per_experiment import create_per_experiment_objective

        captured_configs = []

        class _CapturingTrainer:
            def __init__(self, config):
                captured_configs.append(config)

            def run(self):
                return _make_result()

        objective = create_per_experiment_objective(
            event="test_event", budget=5, seed_set=1,
            model_name="vinai/bertweet-base",
            _trainer_cls=_CapturingTrainer,
        )

        objective(MockTrial())

        self.assertEqual(len(captured_configs), 1)
        self.assertEqual(captured_configs[0].model_name, "vinai/bertweet-base")

    def test_model_name_none_uses_config_default(self):
        """When model_name=None, the LGCoTrainConfig keeps its default."""
        from lg_cotrain.optuna_per_experiment import create_per_experiment_objective

        captured_configs = []

        class _CapturingTrainer:
            def __init__(self, config):
                captured_configs.append(config)

            def run(self):
                return _make_result()

        objective = create_per_experiment_objective(
            event="test_event", budget=5, seed_set=1,
            model_name=None,
            _trainer_cls=_CapturingTrainer,
        )

        objective(MockTrial())

        self.assertEqual(len(captured_configs), 1)
        # The config should have whatever the LGCoTrainConfig default is.
        # We don't hard-code the default here — we just verify the field is
        # populated to a non-empty string (i.e., the dataclass default kicked in).
        self.assertTrue(captured_configs[0].model_name)
        self.assertIsInstance(captured_configs[0].model_name, str)


class TestFullMetricsUserAttr(unittest.TestCase):
    """Verify the objective attaches the full result dict via set_user_attr."""

    def test_objective_sets_full_metrics_user_attr(self):
        from lg_cotrain.optuna_per_experiment import create_per_experiment_objective

        full_result = _make_result(
            test_macro_f1=0.8765,
            test_error_rate=15.5,
            test_ece=0.042,
            test_per_class_f1=[0.8, 0.9, 0.85],
            dev_macro_f1=0.81,
        )

        mock_trainer = MagicMock()
        mock_trainer.return_value.run.return_value = full_result

        objective = create_per_experiment_objective(
            event="test_event", budget=5, seed_set=1,
            _trainer_cls=mock_trainer,
        )

        trial = MockTrialWithUserAttrs()
        ret = objective(trial)

        # Objective return value is dev_macro_f1 ONLY (no test leakage)
        self.assertEqual(ret, 0.81)

        # The full result dict should be attached to the trial
        self.assertIn("full_metrics", trial.user_attrs)
        attached = trial.user_attrs["full_metrics"]
        self.assertEqual(attached["test_macro_f1"], 0.8765)
        self.assertEqual(attached["test_error_rate"], 15.5)
        self.assertEqual(attached["test_ece"], 0.042)
        self.assertEqual(attached["test_per_class_f1"], [0.8, 0.9, 0.85])
        self.assertEqual(attached["dev_macro_f1"], 0.81)

    def test_objective_handles_trial_without_set_user_attr(self):
        """Trials without set_user_attr (e.g. some mocks) do not crash."""
        from lg_cotrain.optuna_per_experiment import create_per_experiment_objective

        mock_trainer = MagicMock()
        mock_trainer.return_value.run.return_value = _make_result(dev_macro_f1=0.7)

        objective = create_per_experiment_objective(
            event="test_event", budget=5, seed_set=1,
            _trainer_cls=mock_trainer,
        )

        # Plain MockTrial has no set_user_attr — should not crash
        trial = MockTrial()
        ret = objective(trial)
        self.assertEqual(ret, 0.7)


class TestNoTestLeakageInOptunaDecision(unittest.TestCase):
    """Verify that Optuna's best_trial is chosen by dev_macro_f1 even when
    test_macro_f1 would suggest a different winner."""

    def test_best_trial_chosen_by_dev_not_test(self):
        try:
            import optuna
        except ImportError:
            self.skipTest("optuna not installed")

        from lg_cotrain.optuna_per_experiment import run_single_study

        # Construct fake trial outputs where dev and test disagree on which
        # trial is best.  dev_macro_f1 is HIGHEST on call_count==1, but
        # test_macro_f1 is HIGHEST on call_count==3.  Optuna should pick
        # call_count==1 because it only sees dev.
        call_count = [0]

        def fake_run():
            call_count[0] += 1
            n = call_count[0]
            return _make_result(
                dev_macro_f1=0.90 if n == 1 else (0.80 if n == 2 else 0.70),
                test_macro_f1=0.50 if n == 1 else (0.60 if n == 2 else 0.95),
                test_error_rate=50.0 if n == 1 else (40.0 if n == 2 else 5.0),
                test_ece=0.10,
            )

        mock_trainer = MagicMock()
        mock_trainer.return_value.run = fake_run

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_single_study(
                event="test_event", budget=5, seed_set=1,
                n_trials=3, storage_dir=tmpdir,
                _trainer_cls=mock_trainer,
            )

            # Optuna should pick the trial with the highest dev_macro_f1
            # (call_count==1, dev=0.90), NOT the one with the highest
            # test_macro_f1 (call_count==3, test=0.95).
            self.assertEqual(result["best_value"], 0.9)

            # And the saved best_full_metrics should belong to that trial:
            # the test metrics for the trial that won on dev (test=0.50,
            # NOT 0.95).
            best_full = result["best_full_metrics"]
            self.assertIsNotNone(best_full)
            self.assertEqual(best_full["dev_macro_f1"], 0.90)
            self.assertEqual(best_full["test_macro_f1"], 0.50)
            self.assertEqual(best_full["test_error_rate"], 50.0)


class TestBestFullMetricsSaved(unittest.TestCase):
    """Verify best_full_metrics is persisted in best_params.json."""

    def test_best_full_metrics_in_saved_json(self):
        try:
            import optuna
        except ImportError:
            self.skipTest("optuna not installed")

        from lg_cotrain.optuna_per_experiment import run_single_study

        call_count = [0]

        def fake_run():
            call_count[0] += 1
            return _make_result(
                dev_macro_f1=0.70 + call_count[0] * 0.01,
                test_macro_f1=0.60 + call_count[0] * 0.01,
                test_error_rate=20.0 - call_count[0],
                test_ece=0.05,
                test_per_class_f1=[0.5, 0.6, 0.7, 0.8],
                lambda1_mean=0.5,
                lambda1_std=0.1,
                lambda2_mean=0.4,
                lambda2_std=0.1,
            )

        mock_trainer = MagicMock()
        mock_trainer.return_value.run = fake_run

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_single_study(
                event="test_event", budget=5, seed_set=1,
                n_trials=3, storage_dir=tmpdir,
                _trainer_cls=mock_trainer,
            )

            # In-memory result has best_full_metrics
            self.assertIn("best_full_metrics", result)
            self.assertIsNotNone(result["best_full_metrics"])

            # Best trial is trial 3 (highest dev_f1 = 0.73)
            self.assertEqual(result["best_full_metrics"]["dev_macro_f1"], 0.73)
            self.assertEqual(result["best_full_metrics"]["test_macro_f1"], 0.63)
            self.assertEqual(result["best_full_metrics"]["test_per_class_f1"], [0.5, 0.6, 0.7, 0.8])
            self.assertEqual(result["best_full_metrics"]["lambda1_mean"], 0.5)

            # Saved JSON has the same field
            best_path = Path(tmpdir) / "test_event" / "5_set1" / "trials_3" / "best_params.json"
            saved = json.loads(best_path.read_text())
            self.assertIn("best_full_metrics", saved)
            self.assertEqual(saved["best_full_metrics"]["test_macro_f1"], 0.63)

            # And per-trial full_metrics is also saved
            self.assertEqual(len(saved["trials"]), 3)
            for t in saved["trials"]:
                self.assertIn("full_metrics", t)


class TestExportMetricsFromStudies(unittest.TestCase):
    """Verify the export_metrics_from_studies helper materializes metrics.json
    files from saved best_params.json files."""

    def _write_study(self, storage_dir, event, budget, seed_set, n_trials,
                     best_full_metrics):
        """Create a fake best_params.json under storage_dir."""
        d = Path(storage_dir) / event / f"{budget}_set{seed_set}" / f"trials_{n_trials}"
        d.mkdir(parents=True)
        data = {
            "event": event, "budget": budget, "seed_set": seed_set,
            "status": "done",
            "best_params": {"lr": 0.001, "batch_size": 32},
            "best_value": best_full_metrics.get("dev_macro_f1", 0.0),
            "best_full_metrics": best_full_metrics,
            "n_trials": n_trials,
            "trials": [],
        }
        (d / "best_params.json").write_text(json.dumps(data))

    def test_writes_metrics_json(self):
        from lg_cotrain.optuna_per_experiment import export_metrics_from_studies

        full_metrics = _make_result(
            test_macro_f1=0.78,
            test_error_rate=12.5,
            test_ece=0.06,
            test_per_class_f1=[0.7, 0.8, 0.9],
        )

        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "studies"
            results_root = Path(tmp) / "results"

            self._write_study(storage, "test_event", 5, 1, 10, full_metrics)

            counts = export_metrics_from_studies(
                storage_dir=str(storage),
                results_root=str(results_root),
                n_trials=10,
                events=["test_event"],
                budgets=[5],
                seed_sets=[1],
            )

            self.assertEqual(counts["written"], 1)
            self.assertEqual(counts["skipped"], 0)

            metrics_path = results_root / "test_event" / "5_set1" / "metrics.json"
            self.assertTrue(metrics_path.exists())
            saved = json.loads(metrics_path.read_text())
            self.assertEqual(saved["test_macro_f1"], 0.78)
            self.assertEqual(saved["test_error_rate"], 12.5)
            self.assertEqual(saved["test_ece"], 0.06)
            self.assertEqual(saved["test_per_class_f1"], [0.7, 0.8, 0.9])

    def test_skips_existing_metrics_json(self):
        from lg_cotrain.optuna_per_experiment import export_metrics_from_studies

        full_metrics = _make_result(test_macro_f1=0.78)

        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "studies"
            results_root = Path(tmp) / "results"

            self._write_study(storage, "test_event", 5, 1, 10, full_metrics)

            # Pre-create the metrics.json with different content
            existing_path = results_root / "test_event" / "5_set1" / "metrics.json"
            existing_path.parent.mkdir(parents=True)
            existing_path.write_text(json.dumps({"test_macro_f1": 0.99}))

            counts = export_metrics_from_studies(
                storage_dir=str(storage),
                results_root=str(results_root),
                n_trials=10,
                events=["test_event"],
                budgets=[5],
                seed_sets=[1],
            )

            self.assertEqual(counts["written"], 0)
            self.assertEqual(counts["skipped"], 1)

            # Existing file should be unchanged
            saved = json.loads(existing_path.read_text())
            self.assertEqual(saved["test_macro_f1"], 0.99)

    def test_overwrite_true_replaces_existing(self):
        from lg_cotrain.optuna_per_experiment import export_metrics_from_studies

        full_metrics = _make_result(test_macro_f1=0.78)

        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "studies"
            results_root = Path(tmp) / "results"

            self._write_study(storage, "test_event", 5, 1, 10, full_metrics)

            # Pre-create with stale content
            existing_path = results_root / "test_event" / "5_set1" / "metrics.json"
            existing_path.parent.mkdir(parents=True)
            existing_path.write_text(json.dumps({"test_macro_f1": 0.99}))

            counts = export_metrics_from_studies(
                storage_dir=str(storage),
                results_root=str(results_root),
                n_trials=10,
                events=["test_event"],
                budgets=[5],
                seed_sets=[1],
                overwrite=True,
            )

            self.assertEqual(counts["written"], 1)
            self.assertEqual(counts["skipped"], 0)

            saved = json.loads(existing_path.read_text())
            self.assertEqual(saved["test_macro_f1"], 0.78)

    def test_missing_study_counted(self):
        from lg_cotrain.optuna_per_experiment import export_metrics_from_studies

        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "studies"
            results_root = Path(tmp) / "results"

            counts = export_metrics_from_studies(
                storage_dir=str(storage),
                results_root=str(results_root),
                n_trials=10,
                events=["test_event"],
                budgets=[5],
                seed_sets=[1],
            )

            self.assertEqual(counts["written"], 0)
            self.assertEqual(counts["missing_study"], 1)

    def test_missing_metrics_counted(self):
        """If best_full_metrics is None (e.g. old study), it's counted."""
        from lg_cotrain.optuna_per_experiment import export_metrics_from_studies

        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "studies"
            results_root = Path(tmp) / "results"

            # Write a study with best_full_metrics=None (old schema)
            d = storage / "test_event" / "5_set1" / "trials_10"
            d.mkdir(parents=True)
            (d / "best_params.json").write_text(json.dumps({
                "event": "test_event", "budget": 5, "seed_set": 1,
                "status": "done", "best_params": {"lr": 0.001},
                "best_value": 0.85,
                "best_full_metrics": None,  # missing
                "n_trials": 10, "trials": [],
            }))

            counts = export_metrics_from_studies(
                storage_dir=str(storage),
                results_root=str(results_root),
                n_trials=10,
                events=["test_event"],
                budgets=[5],
                seed_sets=[1],
            )

            self.assertEqual(counts["written"], 0)
            self.assertEqual(counts["missing_metrics"], 1)

    def test_uses_latest_when_n_trials_none(self):
        """When n_trials=None, falls back to latest trials_* folder."""
        from lg_cotrain.optuna_per_experiment import export_metrics_from_studies

        with tempfile.TemporaryDirectory() as tmp:
            storage = Path(tmp) / "studies"
            results_root = Path(tmp) / "results"

            # Two trial counts; trials_20 has the newer best_full_metrics
            self._write_study(storage, "test_event", 5, 1, 10,
                              _make_result(test_macro_f1=0.70))
            self._write_study(storage, "test_event", 5, 1, 20,
                              _make_result(test_macro_f1=0.85))

            counts = export_metrics_from_studies(
                storage_dir=str(storage),
                results_root=str(results_root),
                n_trials=None,
                events=["test_event"],
                budgets=[5],
                seed_sets=[1],
            )

            self.assertEqual(counts["written"], 1)
            metrics_path = results_root / "test_event" / "5_set1" / "metrics.json"
            saved = json.loads(metrics_path.read_text())
            self.assertEqual(saved["test_macro_f1"], 0.85)


if __name__ == "__main__":
    unittest.main(verbosity=2)
