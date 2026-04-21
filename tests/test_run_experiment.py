"""Tests for run_experiment.py — CLI entry point.

Pure-Python tests: no torch/numpy/transformers required.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "/workspace")


def _make_result(event="test_event", budget=5, seed_set=1):
    """Build a result dict matching trainer.run() output."""
    return {
        "event": event,
        "budget": budget,
        "seed_set": seed_set,
        "test_error_rate": 40.0,
        "test_macro_f1": 0.50,
        "test_per_class_f1": [0.5] * 10,
        "dev_error_rate": 39.0,
        "dev_macro_f1": 0.51,
        "lambda1_mean": 0.7,
        "lambda1_std": 0.1,
        "lambda2_mean": 0.5,
        "lambda2_std": 0.1,
    }


class TestSingleExperimentMode(unittest.TestCase):
    """Test that --event X --budget Y --seed-set Z runs a single experiment."""

    def test_single_experiment_calls_run_all(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment", "--event", "test_event",
            "--budget", "5", "--seed-set", "1",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                main()

        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(args[0], "test_event")
        self.assertEqual(kwargs["budgets"], [5])
        self.assertEqual(kwargs["seed_sets"], [1])

    def test_pseudo_label_source_forwarded_single(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment", "--event", "test_event",
            "--budget", "5", "--seed-set", "1",
            "--pseudo-label-source", "llama-3",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                main()

        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["pseudo_label_source"], "llama-3")

    def test_output_folder_overrides_results_root(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment", "--event", "test_event",
            "--budget", "5", "--seed-set", "1",
            "--output-folder", "/custom/output",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                main()

        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["results_root"], "/custom/output")


class TestBatchMode(unittest.TestCase):
    """Test that multiple events/budgets/seeds triggers batch mode."""

    def test_multiple_events_uses_run_all(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment", "--events", "event_a", "event_b",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                with patch("lg_cotrain.run_experiment.format_summary_table",
                           return_value=""):
                    main()

        self.assertEqual(mock_run.call_count, 2)
        events_called = [call[0][0] for call in mock_run.call_args_list]
        self.assertEqual(events_called, ["event_a", "event_b"])

    def test_single_event_all_budgets_uses_batch(self):
        """--event X without --budget runs all budgets x all seeds."""
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment", "--event", "test_event",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                with patch("lg_cotrain.run_experiment.format_summary_table",
                           return_value=""):
                    main()

        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        # budgets=None means use all defaults
        self.assertIsNone(kwargs["budgets"])
        self.assertIsNone(kwargs["seed_sets"])

    def test_pseudo_label_source_forwarded_batch(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment", "--event", "test_event",
            "--pseudo-label-source", "custom-model",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                with patch("lg_cotrain.run_experiment.format_summary_table",
                           return_value=""):
                    main()

        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["pseudo_label_source"], "custom-model")

    def test_output_folder_forwarded_batch(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment", "--event", "test_event",
            "--output-folder", "/results/run-2",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                with patch("lg_cotrain.run_experiment.format_summary_table",
                           return_value=""):
                    main()

        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["results_root"], "/results/run-2")

    def test_custom_budgets_forwarded(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment", "--event", "test_event",
            "--budgets", "5", "10",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                with patch("lg_cotrain.run_experiment.format_summary_table",
                           return_value=""):
                    main()

        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["budgets"], [5, 10])

    def test_custom_seed_sets_forwarded(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment", "--event", "test_event",
            "--seed-sets", "1", "3",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                with patch("lg_cotrain.run_experiment.format_summary_table",
                           return_value=""):
                    main()

        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["seed_sets"], [1, 3])


class TestCLIValidation(unittest.TestCase):
    """Test CLI argument validation."""

    def test_no_event_exits(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", ["run_experiment"]):
            with self.assertRaises(SystemExit) as ctx:
                main()
            self.assertNotEqual(ctx.exception.code, 0)

    def test_event_and_events_mutually_exclusive(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment", "--event", "a", "--events", "b",
        ]):
            with self.assertRaises(SystemExit):
                main()

    def test_default_hyperparameters_forwarded(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment", "--event", "test_event",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                with patch("lg_cotrain.run_experiment.format_summary_table",
                           return_value=""):
                    main()

        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["model_name"], "bert-base-uncased")
        self.assertEqual(kwargs["lr"], 2e-5)
        self.assertEqual(kwargs["batch_size"], 32)
        self.assertEqual(kwargs["pseudo_label_source"], "gpt-4o")

    def test_phase1_seed_strategy_default(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment", "--event", "test_event",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                with patch("lg_cotrain.run_experiment.format_summary_table",
                           return_value=""):
                    main()

        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["phase1_seed_strategy"], "last")

    def test_phase1_seed_strategy_best_forwarded(self):
        from lg_cotrain.run_experiment import main

        with patch("sys.argv", [
            "run_experiment", "--event", "test_event",
            "--phase1-seed-strategy", "best",
        ]):
            with patch("lg_cotrain.run_experiment.run_all_experiments") as mock_run:
                mock_run.return_value = [_make_result()]
                with patch("lg_cotrain.run_experiment.format_summary_table",
                           return_value=""):
                    main()

        _, kwargs = mock_run.call_args
        self.assertEqual(kwargs["phase1_seed_strategy"], "best")


if __name__ == "__main__":
    unittest.main(verbosity=2)
