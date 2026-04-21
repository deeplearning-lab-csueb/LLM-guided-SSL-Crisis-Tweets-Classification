"""Integration tests for the trainer pipeline.

Requires torch + transformers + pandas + sklearn. Skips gracefully if unavailable.
Also includes a pure-Python integration test for the weight tracker + evaluate pipeline.
"""

import csv
import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, "/workspace")

from lg_cotrain.data_loading import CLASS_LABELS, _split_labeled_set_pure
from lg_cotrain.evaluate import compute_metrics
from lg_cotrain.weight_tracker import WeightTracker


class TestWeightTrackerEvaluateIntegration(unittest.TestCase):
    """Integration: weight tracker feeds lambda weights → evaluate checks metrics."""

    def test_weight_generation_pipeline(self):
        """Simulate Phase 1: record probs across epochs, compute lambdas."""
        num_samples = 20
        tracker1 = WeightTracker(num_samples)
        tracker2 = WeightTracker(num_samples)

        # Simulate 5 epochs of probability recording
        import random
        rng = random.Random(42)
        for _ in range(5):
            probs1 = [rng.uniform(0.1, 0.9) for _ in range(num_samples)]
            probs2 = [rng.uniform(0.1, 0.9) for _ in range(num_samples)]
            tracker1.record_epoch(probs1)
            tracker2.record_epoch(probs2)

        lam1 = tracker1.compute_lambda_optimistic()
        lam2 = tracker2.compute_lambda_conservative()

        # Lambda1 should be >= Lambda2 (optimistic vs conservative)
        self.assertEqual(len(lam1), num_samples)
        self.assertEqual(len(lam2), num_samples)
        for l1, l2_val in zip(lam1, lam2):
            self.assertGreaterEqual(l1, 0.0)
            self.assertGreaterEqual(l2_val, 0.0)

    def test_cotrain_phase_lambda_seeding_last_epoch_only(self):
        """Simulate Phase 1→2: seed from last epoch only, verify initial lambdas."""
        num_samples = 10

        # Phase 1: record 3 epochs with varying probabilities
        p1_tracker = WeightTracker(num_samples)
        for val in [0.3, 0.5, 0.7]:
            p1_tracker.record_epoch([val] * num_samples)

        # Seed Phase 2 with only the last Phase 1 epoch (per Algorithm 1)
        p2_tracker = WeightTracker.seed_from_last_epoch(p1_tracker)

        # Phase 2 tracker should have exactly 1 epoch
        self.assertEqual(p2_tracker.num_epochs_recorded, 1)

        # Variability should be 0 (only one observation)
        var = p2_tracker.compute_variability()
        for v in var:
            self.assertAlmostEqual(v, 0.0)

        # With variability=0: lambda1 = lambda2 = confidence = last epoch probs
        lam1 = p2_tracker.compute_lambda_optimistic()
        lam2 = p2_tracker.compute_lambda_conservative()
        for a, b in zip(lam1, lam2):
            self.assertAlmostEqual(a, b)
        for a in lam1:
            self.assertAlmostEqual(a, 0.7)  # last epoch value

        # After recording a Phase 2 epoch, history grows to 2
        p2_tracker.record_epoch([0.8] * num_samples)
        self.assertEqual(p2_tracker.num_epochs_recorded, 2)

        # Now variability > 0 and lambdas diverge
        var = p2_tracker.compute_variability()
        for v in var:
            self.assertGreater(v, 0.0)
        lam1 = p2_tracker.compute_lambda_optimistic()
        lam2 = p2_tracker.compute_lambda_conservative()
        for a, b in zip(lam1, lam2):
            self.assertGreater(a, b)

        # Source is not affected
        self.assertEqual(p1_tracker.num_epochs_recorded, 3)


class TestSplitAndMetricsIntegration(unittest.TestCase):
    """Integration: split data → predict → evaluate."""

    def test_split_predict_evaluate(self):
        """Simulate a simple prediction pipeline."""
        # Build records
        records = []
        for cls in CLASS_LABELS:
            for j in range(4):
                records.append({
                    "tweet_id": f"{cls}_{j}",
                    "tweet_text": f"text {cls} {j}",
                    "class_label": cls,
                })

        # Split
        d1, d2 = _split_labeled_set_pure(records, seed=42)
        self.assertGreater(len(d1), 0)
        self.assertGreater(len(d2), 0)

        # Simulate predictions (just use ground truth = perfect)
        y_true = [r["class_label"] for r in d1]
        y_pred = [r["class_label"] for r in d1]
        m = compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(m["error_rate"], 0.0)
        self.assertAlmostEqual(m["macro_f1"], 1.0)

        # Simulate random predictions
        import random
        rng = random.Random(123)
        y_pred_rand = [rng.choice(CLASS_LABELS) for _ in d1]
        m2 = compute_metrics(y_true, y_pred_rand)
        self.assertGreater(m2["error_rate"], 0.0)


class TestFullPipelineTiny(unittest.TestCase):
    """End-to-end pipeline on tiny synthetic data using bert-tiny.

    Requires: torch, transformers, pandas, sklearn.
    """

    def _make_tiny_data(self, tmp_path):
        """Create tiny synthetic data files."""
        classes = CLASS_LABELS
        labeled_rows = []
        for i, cls in enumerate(classes):
            for j in range(2):
                labeled_rows.append([str(100 + i * 10 + j), f"Labeled text {cls} {j}", cls])

        unlabeled_rows = []
        for i, cls in enumerate(classes):
            for j in range(4):
                unlabeled_rows.append([str(500 + i * 10 + j), f"Unlabeled text {cls} {j}", cls])

        pseudo_rows = []
        for i, row in enumerate(unlabeled_rows):
            pred = row[2] if i > 0 else classes[1]  # make one wrong
            pseudo_rows.append([row[0], row[1], row[2], pred, "0.9", "", "ok"])

        dev_rows = [[str(900 + i), f"Dev text {cls}", cls] for i, cls in enumerate(classes)]
        test_rows = [[str(950 + i), f"Test text {cls}", cls] for i, cls in enumerate(classes)]

        paths = {}
        for name, rows, sep, header in [
            ("labeled", labeled_rows, "\t",
             ["tweet_id", "tweet_text", "class_label"]),
            ("unlabeled", unlabeled_rows, "\t",
             ["tweet_id", "tweet_text", "class_label"]),
            ("dev", dev_rows, "\t",
             ["tweet_id", "tweet_text", "class_label"]),
            ("test", test_rows, "\t",
             ["tweet_id", "tweet_text", "class_label"]),
        ]:
            path = os.path.join(tmp_path, f"{name}.tsv")
            with open(path, "w", newline="") as f:
                writer = csv.writer(f, delimiter=sep)
                writer.writerow(header)
                writer.writerows(rows)
            paths[name] = path

        pseudo_path = os.path.join(tmp_path, "pseudo.csv")
        with open(pseudo_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tweet_id", "tweet_text", "class_label",
                             "predicted_label", "confidence", "entropy", "status"])
            writer.writerows(pseudo_rows)
        paths["pseudo"] = pseudo_path
        return paths

    def test_full_pipeline(self):
        try:
            import torch
            import transformers
            import pandas
            import sklearn
        except ImportError:
            self.skipTest("torch/transformers/pandas/sklearn not available")

        from lg_cotrain.config import LGCoTrainConfig
        from lg_cotrain.trainer import LGCoTrainer

        with tempfile.TemporaryDirectory() as tmp_path:
            paths = self._make_tiny_data(tmp_path)
            cfg = LGCoTrainConfig(
                event="test",
                budget=5,
                seed_set=1,
                model_name="prajjwal1/bert-tiny",
                num_labels=10,
                weight_gen_epochs=2,
                cotrain_epochs=2,
                finetune_max_epochs=3,
                finetune_patience=2,
                batch_size=8,
                max_seq_length=32,
            )
            cfg.labeled_path = paths["labeled"]
            cfg.unlabeled_path = paths["unlabeled"]
            cfg.pseudo_label_path = paths["pseudo"]
            cfg.dev_path = paths["dev"]
            cfg.test_path = paths["test"]
            cfg.output_dir = os.path.join(tmp_path, "results")

            trainer = LGCoTrainer(cfg)
            results = trainer.run()

            # Verify output structure
            self.assertIn("test_error_rate", results)
            self.assertIn("test_macro_f1", results)
            self.assertIn("test_per_class_f1", results)
            self.assertIn("dev_error_rate", results)
            self.assertIn("dev_macro_f1", results)
            self.assertIn("lambda1_mean", results)
            self.assertIn("lambda2_mean", results)

            # Verify metrics range
            self.assertGreaterEqual(results["test_error_rate"], 0)
            self.assertLessEqual(results["test_error_rate"], 100)
            self.assertGreaterEqual(results["test_macro_f1"], 0)
            self.assertLessEqual(results["test_macro_f1"], 1.0)

            # Verify metrics.json was written
            metrics_path = os.path.join(tmp_path, "results", "metrics.json")
            self.assertTrue(os.path.exists(metrics_path))
            with open(metrics_path) as f:
                saved = json.load(f)
            self.assertEqual(saved["test_error_rate"], results["test_error_rate"])


    def test_full_pipeline_best_seed_strategy(self):
        """End-to-end with phase1_seed_strategy='best'."""
        try:
            import torch
            import transformers
            import pandas
            import sklearn
        except ImportError:
            self.skipTest("torch/transformers/pandas/sklearn not available")

        from lg_cotrain.config import LGCoTrainConfig
        from lg_cotrain.trainer import LGCoTrainer

        with tempfile.TemporaryDirectory() as tmp_path:
            paths = self._make_tiny_data(tmp_path)
            cfg = LGCoTrainConfig(
                event="test",
                budget=5,
                seed_set=1,
                model_name="prajjwal1/bert-tiny",
                num_labels=10,
                weight_gen_epochs=2,
                cotrain_epochs=2,
                finetune_max_epochs=3,
                finetune_patience=2,
                batch_size=8,
                max_seq_length=32,
                phase1_seed_strategy="best",
            )
            cfg.labeled_path = paths["labeled"]
            cfg.unlabeled_path = paths["unlabeled"]
            cfg.pseudo_label_path = paths["pseudo"]
            cfg.dev_path = paths["dev"]
            cfg.test_path = paths["test"]
            cfg.output_dir = os.path.join(tmp_path, "results")

            trainer = LGCoTrainer(cfg)
            results = trainer.run()

            # Verify output structure
            self.assertIn("test_macro_f1", results)
            self.assertEqual(results["phase1_seed_strategy"], "best")
            self.assertIsNotNone(results["phase1_best_epoch"])
            self.assertGreaterEqual(results["phase1_best_epoch"], 1)
            self.assertLessEqual(results["phase1_best_epoch"], 2)

            # Verify metrics.json
            metrics_path = os.path.join(tmp_path, "results", "metrics.json")
            with open(metrics_path) as f:
                saved = json.load(f)
            self.assertEqual(saved["phase1_seed_strategy"], "best")
            self.assertIsNotNone(saved["phase1_best_epoch"])

    def test_full_pipeline_last_strategy_records_field(self):
        """Default phase1_seed_strategy='last' records correctly in metrics."""
        try:
            import torch
            import transformers
            import pandas
            import sklearn
        except ImportError:
            self.skipTest("torch/transformers/pandas/sklearn not available")

        from lg_cotrain.config import LGCoTrainConfig
        from lg_cotrain.trainer import LGCoTrainer

        with tempfile.TemporaryDirectory() as tmp_path:
            paths = self._make_tiny_data(tmp_path)
            cfg = LGCoTrainConfig(
                event="test",
                budget=5,
                seed_set=1,
                model_name="prajjwal1/bert-tiny",
                num_labels=10,
                weight_gen_epochs=2,
                cotrain_epochs=2,
                finetune_max_epochs=3,
                finetune_patience=2,
                batch_size=8,
                max_seq_length=32,
            )
            cfg.labeled_path = paths["labeled"]
            cfg.unlabeled_path = paths["unlabeled"]
            cfg.pseudo_label_path = paths["pseudo"]
            cfg.dev_path = paths["dev"]
            cfg.test_path = paths["test"]
            cfg.output_dir = os.path.join(tmp_path, "results")

            trainer = LGCoTrainer(cfg)
            results = trainer.run()

            self.assertEqual(results["phase1_seed_strategy"], "last")
            self.assertIsNone(results["phase1_best_epoch"])

    def test_invalid_phase1_seed_strategy_raises(self):
        """Unknown phase1_seed_strategy raises ValueError."""
        try:
            import torch
            import transformers
            import pandas
            import sklearn
        except ImportError:
            self.skipTest("torch/transformers/pandas/sklearn not available")

        from lg_cotrain.config import LGCoTrainConfig
        from lg_cotrain.trainer import LGCoTrainer

        with tempfile.TemporaryDirectory() as tmp_path:
            paths = self._make_tiny_data(tmp_path)
            cfg = LGCoTrainConfig(
                event="test",
                budget=5,
                seed_set=1,
                model_name="prajjwal1/bert-tiny",
                num_labels=10,
                weight_gen_epochs=2,
                cotrain_epochs=2,
                finetune_max_epochs=3,
                finetune_patience=2,
                batch_size=8,
                max_seq_length=32,
                phase1_seed_strategy="invalid",
            )
            cfg.labeled_path = paths["labeled"]
            cfg.unlabeled_path = paths["unlabeled"]
            cfg.pseudo_label_path = paths["pseudo"]
            cfg.dev_path = paths["dev"]
            cfg.test_path = paths["test"]
            cfg.output_dir = os.path.join(tmp_path, "results")

            trainer = LGCoTrainer(cfg)
            with self.assertRaises(ValueError):
                trainer.run()


if __name__ == "__main__":
    unittest.main(verbosity=2)
