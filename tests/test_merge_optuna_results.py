"""Tests for merge_optuna_results.py standalone script."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure repo root is on sys.path so we can import the standalone script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from merge_optuna_results import (
    ALL_EVENTS, BUDGETS, SEED_SETS, SEARCH_SPACE, TOTAL_STUDIES,
    merge_sources, generate_summary, _study_path,
)


def _make_best_params(event, budget, seed, n_trials=10, best_value=0.55):
    """Create a minimal best_params.json dict."""
    return {
        "event": event,
        "budget": budget,
        "seed_set": seed,
        "status": "done",
        "best_params": {
            "lr": 1e-4,
            "batch_size": 32,
            "cotrain_epochs": 10,
            "finetune_patience": 5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
        },
        "best_value": best_value,
        "n_trials": n_trials,
        "continued_from": None,
        "trials": [],
    }


def _write_study(base_dir, event, budget, seed, n_trials=10, best_value=0.55):
    """Write a best_params.json file into the expected directory structure."""
    trial_dir = _study_path(Path(base_dir), event, budget, seed, n_trials)
    trial_dir.mkdir(parents=True, exist_ok=True)
    data = _make_best_params(event, budget, seed, n_trials, best_value)
    with open(trial_dir / "best_params.json", "w") as f:
        json.dump(data, f)
    # Also write a dummy study.log
    with open(trial_dir / "study.log", "w") as f:
        f.write("dummy log\n")
    return data


class TestSummaryOnlyMode(unittest.TestCase):
    """Test generate_summary without any merge."""

    def test_generates_summary_from_existing(self):
        """Scan target with 3 studies, verify summary has correct counts."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            _write_study(target, "california_wildfires_2018", 5, 1)
            _write_study(target, "california_wildfires_2018", 5, 2)
            _write_study(target, "canada_wildfires_2016", 10, 3)

            summary, missing = generate_summary(target, n_trials=10)

            self.assertEqual(summary["completed"], 3)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(summary["total_studies"], TOTAL_STUDIES)
            self.assertEqual(summary["n_trials_per_study"], 10)
            self.assertEqual(len(summary["studies"]), 3)
            self.assertEqual(len(missing), TOTAL_STUDIES - 3)

    def test_writes_summary_file(self):
        """summary_{n}.json should be written to disk."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            _write_study(target, "hurricane_harvey_2017", 50, 1)

            generate_summary(target, n_trials=10)

            summary_path = target / "summary_10.json"
            self.assertTrue(summary_path.exists())
            with open(summary_path) as f:
                data = json.load(f)
            self.assertIn("studies", data)
            self.assertIn("search_space", data)

    def test_empty_target(self):
        """No results at all: summary has 0 completed, 120 missing."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            summary, missing = generate_summary(target, n_trials=10)

            self.assertEqual(summary["completed"], 0)
            self.assertEqual(len(summary["studies"]), 0)
            self.assertEqual(len(missing), TOTAL_STUDIES)


class TestSummaryFormat(unittest.TestCase):
    """Verify summary JSON matches run_all_studies() format."""

    def test_has_required_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            _write_study(target, "kerala_floods_2018", 25, 2)

            summary, _ = generate_summary(target, n_trials=10)

            self.assertIn("total_studies", summary)
            self.assertIn("completed", summary)
            self.assertIn("failed", summary)
            self.assertIn("n_trials_per_study", summary)
            self.assertIn("search_space", summary)
            self.assertIn("studies", summary)

    def test_study_entry_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            _write_study(target, "cyclone_idai_2019", 10, 1, best_value=0.72)

            summary, _ = generate_summary(target, n_trials=10)

            study = summary["studies"][0]
            self.assertEqual(study["event"], "cyclone_idai_2019")
            self.assertEqual(study["budget"], 10)
            self.assertEqual(study["seed_set"], 1)
            self.assertEqual(study["status"], "done")
            self.assertIsNotNone(study["best_params"])
            self.assertAlmostEqual(study["best_value"], 0.72)

    def test_search_space_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            _write_study(target, "kerala_floods_2018", 5, 1)

            summary, _ = generate_summary(target, n_trials=10)

            self.assertEqual(summary["search_space"], SEARCH_SPACE)


class TestMergeFromSource(unittest.TestCase):
    """Test merge_sources copying behavior."""

    def test_copies_from_source_to_target(self):
        with tempfile.TemporaryDirectory() as src_tmp, \
             tempfile.TemporaryDirectory() as dst_tmp:
            source = Path(src_tmp)
            target = Path(dst_tmp)

            _write_study(source, "hurricane_irma_2017", 5, 1)
            _write_study(source, "hurricane_irma_2017", 5, 2)

            stats = merge_sources([str(source)], target, n_trials=10)

            self.assertEqual(stats["copied"], 2)
            self.assertEqual(stats["skipped"], 0)

            # Verify files exist in target
            p1 = _study_path(target, "hurricane_irma_2017", 5, 1, 10) / "best_params.json"
            p2 = _study_path(target, "hurricane_irma_2017", 5, 2, 10) / "best_params.json"
            self.assertTrue(p1.exists())
            self.assertTrue(p2.exists())

    def test_copies_study_log(self):
        """study.log should also be copied alongside best_params.json."""
        with tempfile.TemporaryDirectory() as src_tmp, \
             tempfile.TemporaryDirectory() as dst_tmp:
            source = Path(src_tmp)
            target = Path(dst_tmp)

            _write_study(source, "kerala_floods_2018", 50, 3)

            merge_sources([str(source)], target, n_trials=10)

            log_path = _study_path(target, "kerala_floods_2018", 50, 3, 10) / "study.log"
            self.assertTrue(log_path.exists())

    def test_skips_existing_in_target(self):
        """If study already exists in target, skip (don't overwrite)."""
        with tempfile.TemporaryDirectory() as src_tmp, \
             tempfile.TemporaryDirectory() as dst_tmp:
            source = Path(src_tmp)
            target = Path(dst_tmp)

            # Write to both source and target with different values
            _write_study(source, "hurricane_harvey_2017", 10, 1, best_value=0.60)
            _write_study(target, "hurricane_harvey_2017", 10, 1, best_value=0.99)

            stats = merge_sources([str(source)], target, n_trials=10)

            self.assertEqual(stats["copied"], 0)
            self.assertEqual(stats["skipped"], 1)

            # Target should keep its original value
            p = _study_path(target, "hurricane_harvey_2017", 10, 1, 10) / "best_params.json"
            with open(p) as f:
                data = json.load(f)
            self.assertAlmostEqual(data["best_value"], 0.99)

    def test_merges_multiple_sources(self):
        """Merge from two sources with non-overlapping events."""
        with tempfile.TemporaryDirectory() as s1_tmp, \
             tempfile.TemporaryDirectory() as s2_tmp, \
             tempfile.TemporaryDirectory() as dst_tmp:
            s1 = Path(s1_tmp)
            s2 = Path(s2_tmp)
            target = Path(dst_tmp)

            _write_study(s1, "california_wildfires_2018", 5, 1)
            _write_study(s2, "kerala_floods_2018", 5, 1)

            stats = merge_sources([str(s1), str(s2)], target, n_trials=10)

            self.assertEqual(stats["copied"], 2)
            self.assertTrue(
                (_study_path(target, "california_wildfires_2018", 5, 1, 10) / "best_params.json").exists()
            )
            self.assertTrue(
                (_study_path(target, "kerala_floods_2018", 5, 1, 10) / "best_params.json").exists()
            )


class TestDryRun(unittest.TestCase):
    """Test --dry-run behavior."""

    def test_no_files_copied(self):
        with tempfile.TemporaryDirectory() as src_tmp, \
             tempfile.TemporaryDirectory() as dst_tmp:
            source = Path(src_tmp)
            target = Path(dst_tmp)

            _write_study(source, "hurricane_dorian_2019", 25, 3)

            stats = merge_sources([str(source)], target, n_trials=10, dry_run=True)

            self.assertEqual(stats["copied"], 1)  # counted but not actually copied
            p = _study_path(target, "hurricane_dorian_2019", 25, 3, 10) / "best_params.json"
            self.assertFalse(p.exists())


class TestMissingStudies(unittest.TestCase):
    """Test that missing studies are correctly reported."""

    def test_reports_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            # Write only 1 study out of 120
            _write_study(target, "california_wildfires_2018", 5, 1)

            summary, missing = generate_summary(target, n_trials=10)

            self.assertEqual(len(missing), TOTAL_STUDIES - 1)
            # The one we wrote should NOT be in missing
            self.assertNotIn(("california_wildfires_2018", 5, 1), missing)
            # Others should be missing
            self.assertIn(("california_wildfires_2018", 5, 2), missing)

    def test_no_missing_when_all_present(self):
        """If all 120 studies exist, missing list is empty."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            for event in ALL_EVENTS:
                for budget in BUDGETS:
                    for seed in SEED_SETS:
                        _write_study(target, event, budget, seed)

            summary, missing = generate_summary(target, n_trials=10)

            self.assertEqual(len(missing), 0)
            self.assertEqual(summary["completed"], TOTAL_STUDIES)


class TestConstants(unittest.TestCase):
    """Verify constants match expected values."""

    def test_total_studies(self):
        self.assertEqual(TOTAL_STUDIES, 120)

    def test_10_events(self):
        self.assertEqual(len(ALL_EVENTS), 10)

    def test_4_budgets(self):
        self.assertEqual(BUDGETS, [5, 10, 25, 50])

    def test_3_seeds(self):
        self.assertEqual(SEED_SETS, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
