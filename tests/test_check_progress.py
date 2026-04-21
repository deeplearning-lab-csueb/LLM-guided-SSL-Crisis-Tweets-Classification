"""Tests for check_progress.py standalone script."""

import os
import sys
import tempfile
import textwrap
import unittest
from io import StringIO
from pathlib import Path

# Ensure repo root is on sys.path so we can import the standalone script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from check_progress import parse_study_log, parse_timestamp, format_duration, print_progress


SAMPLE_LOG = textwrap.dedent("""\
    2026-01-15 10:00:00,000 - Optuna study: hurricane_harvey_2017 budget=50 seed=1 | target=10 trials
    2026-01-15 10:00:01,000 - --- Trial 1/10 ---
    2026-01-15 10:00:01,100 - === Phase 1: Weight Generation ===
    2026-01-15 10:00:02,000 - Phase 1 epoch 1/5
    2026-01-15 10:00:03,000 - Phase 1 epoch 2/5
    2026-01-15 10:00:04,000 - === Phase 2: Co-Training ===
    2026-01-15 10:00:05,000 - Phase 2 epoch 1/10
    2026-01-15 10:00:06,000 - === Phase 3: Fine-Tuning ===
    2026-01-15 10:00:07,000 - Phase 3 epoch 1/20
    2026-01-15 10:00:08,000 - === Final Evaluation ===
    2026-01-15 10:01:01,000 - --- Trial 1/10 done: dev_macro_f1=0.4500 ---
    2026-01-15 10:01:02,000 - --- Trial 2/10 ---
    2026-01-15 10:01:02,100 - === Phase 1: Weight Generation ===
    2026-01-15 10:02:02,000 - --- Trial 2/10 done: dev_macro_f1=0.5200 ---
""")


class TestParseTimestamp(unittest.TestCase):
    def test_valid_timestamp(self):
        ts = parse_timestamp("2026-01-15 10:00:00,123 - some message")
        self.assertIsNotNone(ts)
        self.assertEqual(ts.year, 2026)
        self.assertEqual(ts.hour, 10)

    def test_no_timestamp(self):
        self.assertIsNone(parse_timestamp("no timestamp here"))


class TestFormatDuration(unittest.TestCase):
    def test_seconds(self):
        self.assertEqual(format_duration(45), "45s")

    def test_minutes(self):
        self.assertEqual(format_duration(125), "2m 5s")

    def test_hours(self):
        self.assertEqual(format_duration(3725), "1h 2m")


class TestParseStudyLog(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.tmpdir, "study.log")
        with open(self.log_path, "w") as f:
            f.write(SAMPLE_LOG)

    def test_parses_event_info(self):
        info = parse_study_log(self.log_path)
        self.assertEqual(info["event"], "hurricane_harvey_2017")
        self.assertEqual(info["budget"], 50)
        self.assertEqual(info["seed"], 1)
        self.assertEqual(info["target_trials"], 10)

    def test_counts_completed_trials(self):
        info = parse_study_log(self.log_path)
        self.assertEqual(info["completed_trials"], 2)

    def test_tracks_best_f1(self):
        info = parse_study_log(self.log_path)
        self.assertAlmostEqual(info["best_f1"], 0.52)

    def test_records_trial_durations(self):
        info = parse_study_log(self.log_path)
        self.assertEqual(len(info["trial_durations"]), 2)
        # Trial 1: 10:00:01 -> 10:01:01 = 60s
        self.assertAlmostEqual(info["trial_durations"][0], 60.0)
        # Trial 2: 10:01:02 -> 10:02:02 = 60s
        self.assertAlmostEqual(info["trial_durations"][1], 60.0)

    def test_not_marked_done_when_incomplete(self):
        info = parse_study_log(self.log_path)
        # 2 of 10 trials done — should NOT be "done"
        self.assertNotEqual(info["current_phase"], "done")

    def test_empty_file(self):
        empty_path = os.path.join(self.tmpdir, "empty.log")
        with open(empty_path, "w") as f:
            f.write("")
        info = parse_study_log(empty_path)
        self.assertEqual(info["completed_trials"], 0)
        self.assertIsNone(info["event"])


class TestPrintProgressTotalStudies(unittest.TestCase):
    """Verify that total_studies is computed from the CLI parameters, not hardcoded."""

    def setUp(self):
        """Create a temp results dir with 2 study logs."""
        self.tmpdir = tempfile.mkdtemp()
        base = Path(self.tmpdir) / "optuna" / "per_experiment"

        # Study 1: hurricane_harvey_2017 budget=50 seed=1
        d1 = base / "hurricane_harvey_2017" / "50_set1" / "trials_10"
        d1.mkdir(parents=True)
        with open(d1 / "study.log", "w") as f:
            f.write(SAMPLE_LOG)

        # Study 2: kerala_floods_2018 budget=50 seed=1 (all 10 trials done)
        d2 = base / "kerala_floods_2018" / "50_set1" / "trials_10"
        d2.mkdir(parents=True)
        completed_log = textwrap.dedent("""\
            2026-01-15 10:00:00,000 - Optuna study: kerala_floods_2018 budget=50 seed=1 | target=10 trials
            2026-01-15 10:00:01,000 - --- Trial 1/10 ---
            2026-01-15 10:01:01,000 - --- Trial 1/10 done: dev_macro_f1=0.50 ---
            2026-01-15 10:01:02,000 - --- Trial 2/10 ---
            2026-01-15 10:02:02,000 - --- Trial 2/10 done: dev_macro_f1=0.55 ---
            2026-01-15 10:02:03,000 - --- Trial 3/10 ---
            2026-01-15 10:03:03,000 - --- Trial 3/10 done: dev_macro_f1=0.58 ---
            2026-01-15 10:03:04,000 - --- Trial 4/10 ---
            2026-01-15 10:04:04,000 - --- Trial 4/10 done: dev_macro_f1=0.60 ---
            2026-01-15 10:04:05,000 - --- Trial 5/10 ---
            2026-01-15 10:05:05,000 - --- Trial 5/10 done: dev_macro_f1=0.62 ---
            2026-01-15 10:05:06,000 - --- Trial 6/10 ---
            2026-01-15 10:06:06,000 - --- Trial 6/10 done: dev_macro_f1=0.63 ---
            2026-01-15 10:06:07,000 - --- Trial 7/10 ---
            2026-01-15 10:07:07,000 - --- Trial 7/10 done: dev_macro_f1=0.64 ---
            2026-01-15 10:07:08,000 - --- Trial 8/10 ---
            2026-01-15 10:08:08,000 - --- Trial 8/10 done: dev_macro_f1=0.65 ---
            2026-01-15 10:08:09,000 - --- Trial 9/10 ---
            2026-01-15 10:09:09,000 - --- Trial 9/10 done: dev_macro_f1=0.66 ---
            2026-01-15 10:09:10,000 - --- Trial 10/10 ---
            2026-01-15 10:10:10,000 - --- Trial 10/10 done: dev_macro_f1=0.67 ---
        """)
        with open(d2 / "study.log", "w") as f:
            f.write(completed_log)

    def _capture_output(self, **kwargs):
        """Run print_progress and capture stdout."""
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_progress(self.tmpdir, **kwargs)
        return buf.getvalue()

    def test_default_total_is_120(self):
        output = self._capture_output()
        # Default: 10 events × 4 budgets × 3 seeds = 120
        self.assertIn("of 120", output)

    def test_custom_num_events(self):
        output = self._capture_output(num_events=2)
        # 2 events × 4 budgets × 3 seeds = 24
        self.assertIn("of 24", output)

    def test_custom_all_dimensions(self):
        output = self._capture_output(num_events=2, num_budgets=1, num_seeds=1)
        # 2 × 1 × 1 = 2
        self.assertIn("of 2", output)

    def test_expected_trials_scales_with_total(self):
        output = self._capture_output(num_events=2, num_budgets=1, num_seeds=1)
        # 2 studies × 10 trials each = 20 expected trials
        self.assertIn("/ 20 completed", output)

    def test_not_started_count(self):
        # 2 logs exist; with num_events=2, num_budgets=1, num_seeds=1 -> total=2
        # not_started = 2 - 2 = 0
        output = self._capture_output(num_events=2, num_budgets=1, num_seeds=1)
        self.assertIn("0 not started", output)

        # With total=6, not_started = 6 - 2 = 4
        output = self._capture_output(num_events=2, num_budgets=1, num_seeds=3)
        self.assertIn("4 not started", output)


if __name__ == "__main__":
    unittest.main()
