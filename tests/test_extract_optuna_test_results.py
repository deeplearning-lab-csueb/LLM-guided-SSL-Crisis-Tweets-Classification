"""Tests for extract_optuna_test_results.py."""

import json
import tempfile
import unittest
from pathlib import Path

from extract_optuna_test_results import (
    collect_all_test_metrics,
    extract_results,
    parse_study_log,
)

MOCK_STUDY_LOG = """\
2026-02-22 18:29:12,841 - lg_cotrain - INFO - === Optuna study: test_event budget=5 seed=1 ===
2026-02-22 18:29:12,842 - lg_cotrain - INFO - --- Trial 1/3 ---
2026-02-22 18:29:35,112 - lg_cotrain - INFO - Starting LG-CoTrain: event=test_event, budget=5, seed_set=1
2026-02-22 18:32:03,503 - lg_cotrain - INFO - Phase 1 epoch 7/7: mean_prob1=0.1063, mean_prob2=0.1143
2026-02-22 18:42:14,530 - lg_cotrain - INFO - Test error rate: 27.52%, Test macro-F1: 0.6061, Test ECE: 0.1851
2026-02-22 18:42:14,538 - lg_cotrain - INFO - --- Trial 1/3 done: dev_macro_f1=0.6076 ---
2026-02-22 18:42:14,538 - lg_cotrain - INFO - --- Trial 2/3 ---
2026-02-22 18:54:31,965 - lg_cotrain - INFO - Test error rate: 29.16%, Test macro-F1: 0.6079, Test ECE: 0.1533
2026-02-22 18:54:31,981 - lg_cotrain - INFO - --- Trial 2/3 done: dev_macro_f1=0.6306 ---
2026-02-22 18:54:31,981 - lg_cotrain - INFO - --- Trial 3/3 ---
2026-02-22 19:12:53,345 - lg_cotrain - INFO - Test error rate: 26.69%, Test macro-F1: 0.6013, Test ECE: 0.1243
2026-02-22 19:12:53,351 - lg_cotrain - INFO - --- Trial 3/3 done: dev_macro_f1=0.6051 ---
"""

MOCK_BEST_PARAMS = {
    "event": "test_event",
    "budget": 5,
    "seed_set": 1,
    "status": "done",
    "best_params": {
        "lr": 0.00015,
        "batch_size": 16,
        "cotrain_epochs": 17,
        "finetune_patience": 6,
        "weight_decay": 0.098,
        "warmup_ratio": 0.15,
    },
    "best_value": 0.6306,
    "n_trials": 3,
    "continued_from": None,
    "trials": [
        {
            "number": 0,
            "state": "COMPLETE",
            "params": {"lr": 3.3e-05, "batch_size": 8, "cotrain_epochs": 6,
                       "finetune_patience": 8, "weight_decay": 0.016, "warmup_ratio": 0.28},
            "dev_macro_f1": 0.6076,
            "duration_seconds": 781.7,
        },
        {
            "number": 1,
            "state": "COMPLETE",
            "params": {"lr": 0.00015, "batch_size": 16, "cotrain_epochs": 17,
                       "finetune_patience": 6, "weight_decay": 0.098, "warmup_ratio": 0.15},
            "dev_macro_f1": 0.6306,
            "duration_seconds": 737.4,
        },
        {
            "number": 2,
            "state": "COMPLETE",
            "params": {"lr": 1.4e-05, "batch_size": 64, "cotrain_epochs": 14,
                       "finetune_patience": 4, "weight_decay": 0.081, "warmup_ratio": 0.076},
            "dev_macro_f1": 0.6051,
            "duration_seconds": 1101.4,
        },
    ],
}


class TestParseStudyLog(unittest.TestCase):
    """Tests for parse_study_log()."""

    def test_parses_all_trials(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(MOCK_STUDY_LOG)
            f.flush()
            result = parse_study_log(f.name)

        self.assertEqual(len(result), 3)
        # Trial 1 in log -> 0-indexed key 0
        self.assertIn(0, result)
        self.assertIn(1, result)
        self.assertIn(2, result)

    def test_correct_values(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(MOCK_STUDY_LOG)
            f.flush()
            result = parse_study_log(f.name)

        # Trial 1 (0-indexed: 0)
        self.assertAlmostEqual(result[0]["test_error_rate"], 27.52)
        self.assertAlmostEqual(result[0]["test_macro_f1"], 0.6061)
        self.assertAlmostEqual(result[0]["test_ece"], 0.1851)

        # Trial 2 (0-indexed: 1)
        self.assertAlmostEqual(result[1]["test_error_rate"], 29.16)
        self.assertAlmostEqual(result[1]["test_macro_f1"], 0.6079)
        self.assertAlmostEqual(result[1]["test_ece"], 0.1533)

    def test_missing_test_line(self):
        """Trial with no test metrics line should be omitted."""
        log_content = """\
2026-02-22 18:29:12,842 - lg_cotrain - INFO - --- Trial 1/2 ---
2026-02-22 18:42:14,530 - lg_cotrain - INFO - Test error rate: 27.52%, Test macro-F1: 0.6061, Test ECE: 0.1851
2026-02-22 18:42:14,538 - lg_cotrain - INFO - --- Trial 1/2 done: dev_macro_f1=0.6076 ---
2026-02-22 18:42:14,538 - lg_cotrain - INFO - --- Trial 2/2 ---
2026-02-22 18:54:31,981 - lg_cotrain - INFO - --- Trial 2/2 done: dev_macro_f1=0.0302 ---
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(log_content)
            f.flush()
            result = parse_study_log(f.name)

        self.assertEqual(len(result), 1)
        self.assertIn(0, result)
        self.assertNotIn(1, result)

    def test_empty_log(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("")
            f.flush()
            result = parse_study_log(f.name)

        self.assertEqual(result, {})


class TestCollectAllTestMetrics(unittest.TestCase):
    """Tests for collect_all_test_metrics()."""

    def test_single_trials_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            trials_dir = exp_dir / "trials_10"
            trials_dir.mkdir()
            (trials_dir / "study.log").write_text(MOCK_STUDY_LOG)

            result = collect_all_test_metrics(exp_dir)

        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[1]["test_macro_f1"], 0.6079)

    def test_continued_study(self):
        """Metrics from multiple trials_* folders are merged."""
        log_part1 = """\
2026-01-01 00:00:00,000 - lg_cotrain - INFO - --- Trial 1/2 ---
2026-01-01 00:10:00,000 - lg_cotrain - INFO - Test error rate: 30.00%, Test macro-F1: 0.5500, Test ECE: 0.2000
2026-01-01 00:10:00,000 - lg_cotrain - INFO - --- Trial 1/2 done: dev_macro_f1=0.5600 ---
2026-01-01 00:10:00,000 - lg_cotrain - INFO - --- Trial 2/2 ---
2026-01-01 00:20:00,000 - lg_cotrain - INFO - Test error rate: 28.00%, Test macro-F1: 0.5800, Test ECE: 0.1800
2026-01-01 00:20:00,000 - lg_cotrain - INFO - --- Trial 2/2 done: dev_macro_f1=0.5900 ---
"""
        log_part2 = """\
2026-01-02 00:00:00,000 - lg_cotrain - INFO - --- Trial 3/4 ---
2026-01-02 00:10:00,000 - lg_cotrain - INFO - Test error rate: 25.00%, Test macro-F1: 0.6200, Test ECE: 0.1500
2026-01-02 00:10:00,000 - lg_cotrain - INFO - --- Trial 3/4 done: dev_macro_f1=0.6300 ---
2026-01-02 00:10:00,000 - lg_cotrain - INFO - --- Trial 4/4 ---
2026-01-02 00:20:00,000 - lg_cotrain - INFO - Test error rate: 26.00%, Test macro-F1: 0.6100, Test ECE: 0.1600
2026-01-02 00:20:00,000 - lg_cotrain - INFO - --- Trial 4/4 done: dev_macro_f1=0.6000 ---
"""
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp)
            (exp_dir / "trials_2").mkdir()
            (exp_dir / "trials_2" / "study.log").write_text(log_part1)
            (exp_dir / "trials_4").mkdir()
            (exp_dir / "trials_4" / "study.log").write_text(log_part2)

            result = collect_all_test_metrics(exp_dir)

        # Should have all 4 trials (0, 1, 2, 3)
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0]["test_macro_f1"], 0.5500)
        self.assertAlmostEqual(result[2]["test_macro_f1"], 0.6200)


class TestExtractResults(unittest.TestCase):
    """Tests for extract_results()."""

    def _create_experiment(self, base_dir, event, budget, seed_set, n_trials,
                           best_params_data, study_log_content):
        exp_dir = base_dir / event / f"{budget}_set{seed_set}" / f"trials_{n_trials}"
        exp_dir.mkdir(parents=True)
        (exp_dir / "best_params.json").write_text(json.dumps(best_params_data))
        (exp_dir / "study.log").write_text(study_log_content)

    def test_extract_single_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._create_experiment(
                base, "test_event", 5, 1, 3,
                MOCK_BEST_PARAMS, MOCK_STUDY_LOG,
            )
            results = extract_results(str(base), n_trials=3)

        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r["event"], "test_event")
        self.assertEqual(r["budget"], 5)
        self.assertEqual(r["seed_set"], 1)
        self.assertEqual(r["best_trial"], 1)  # trial with dev_macro_f1=0.6306
        self.assertAlmostEqual(r["dev_macro_f1"], 0.6306)
        # Best trial (0-indexed 1) test metrics from log
        self.assertAlmostEqual(r["test_macro_f1"], 0.6079)
        self.assertAlmostEqual(r["test_error_rate"], 29.16)
        self.assertAlmostEqual(r["test_ece"], 0.1533)

    def test_extract_multiple_experiments(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            # Experiment 1
            self._create_experiment(
                base, "event_a", 5, 1, 3,
                MOCK_BEST_PARAMS, MOCK_STUDY_LOG,
            )

            # Experiment 2 — different event
            bp2 = dict(MOCK_BEST_PARAMS)
            bp2["event"] = "event_b"
            bp2["budget"] = 10
            bp2["seed_set"] = 2
            self._create_experiment(
                base, "event_b", 10, 2, 3,
                bp2, MOCK_STUDY_LOG,
            )

            results = extract_results(str(base), n_trials=3)

        self.assertEqual(len(results), 2)
        events = [r["event"] for r in results]
        self.assertIn("event_a", events)
        self.assertIn("event_b", events)

    def test_latest_trials_when_n_trials_not_specified(self):
        """When n_trials is None, should pick the latest trials_N folder."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            # Create trials_3 and trials_5
            self._create_experiment(
                base, "test_event", 5, 1, 3,
                MOCK_BEST_PARAMS, MOCK_STUDY_LOG,
            )

            bp5 = dict(MOCK_BEST_PARAMS)
            bp5["n_trials"] = 5
            bp5["best_value"] = 0.7000
            # Add 2 more trials
            bp5["trials"] = list(MOCK_BEST_PARAMS["trials"]) + [
                {"number": 3, "state": "COMPLETE", "params": {}, "dev_macro_f1": 0.6500, "duration_seconds": 500},
                {"number": 4, "state": "COMPLETE", "params": {}, "dev_macro_f1": 0.7000, "duration_seconds": 500},
            ]
            bp5["best_params"] = {"lr": 0.001}

            exp_dir = base / "test_event" / "5_set1" / "trials_5"
            exp_dir.mkdir(parents=True)
            (exp_dir / "best_params.json").write_text(json.dumps(bp5))
            (exp_dir / "study.log").write_text(MOCK_STUDY_LOG)

            results = extract_results(str(base))

        # Should pick trials_5 (latest)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["n_trials"], 5)

    def test_skips_nonexistent_n_trials(self):
        """When n_trials specified but folder doesn't exist, skip experiment."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._create_experiment(
                base, "test_event", 5, 1, 3,
                MOCK_BEST_PARAMS, MOCK_STUDY_LOG,
            )
            # Ask for trials_20 which doesn't exist
            results = extract_results(str(base), n_trials=20)

        self.assertEqual(len(results), 0)

    def test_missing_test_metrics_sets_none(self):
        """If test metrics not found in log for best trial, fields are None."""
        # Log with no test metrics for trial 2 (which is the best)
        log_content = """\
2026-01-01 00:00:00,000 - lg_cotrain - INFO - --- Trial 1/2 ---
2026-01-01 00:10:00,000 - lg_cotrain - INFO - Test error rate: 30.00%, Test macro-F1: 0.5500, Test ECE: 0.2000
2026-01-01 00:10:00,000 - lg_cotrain - INFO - --- Trial 1/2 done: dev_macro_f1=0.5600 ---
2026-01-01 00:10:00,000 - lg_cotrain - INFO - --- Trial 2/2 ---
2026-01-01 00:20:00,000 - lg_cotrain - INFO - --- Trial 2/2 done: dev_macro_f1=0.7000 ---
"""
        bp = dict(MOCK_BEST_PARAMS)
        bp["best_value"] = 0.7000
        bp["trials"] = [
            {"number": 0, "state": "COMPLETE", "params": {}, "dev_macro_f1": 0.5600, "duration_seconds": 500},
            {"number": 1, "state": "COMPLETE", "params": {}, "dev_macro_f1": 0.7000, "duration_seconds": 500},
        ]
        bp["n_trials"] = 2

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._create_experiment(base, "test_event", 5, 1, 2, bp, log_content)
            results = extract_results(str(base), n_trials=2)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["best_trial"], 1)
        self.assertIsNone(results[0]["test_macro_f1"])
        self.assertIsNone(results[0]["test_error_rate"])
        self.assertIsNone(results[0]["test_ece"])


if __name__ == "__main__":
    unittest.main()
