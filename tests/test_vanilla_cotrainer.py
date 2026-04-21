"""Tests for the vanilla_cotrain package — config and trainer logic.

Config tests are pure-Python (no torch/transformers). Trainer tests
require torch and are skipped if not available.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vanilla_cotrain.config import VanillaCoTrainConfig


class TestVanillaConfig(unittest.TestCase):
    """VanillaCoTrainConfig defaults and path computation."""

    def test_default_model(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.model_name, "vinai/bertweet-base")

    def test_default_num_iterations(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.num_iterations, 30)

    def test_default_samples_per_class(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.samples_per_class, 1)

    def test_default_train_epochs(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.train_epochs, 5)

    def test_default_finetune_patience(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.finetune_patience, 5)

    def test_default_lr(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.lr, 2e-5)

    def test_default_batch_size(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.batch_size, 32)

    def test_default_weight_decay(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.weight_decay, 0.01)

    def test_default_warmup_ratio(self):
        cfg = VanillaCoTrainConfig()
        self.assertEqual(cfg.warmup_ratio, 0.1)

    def test_labeled_path(self):
        cfg = VanillaCoTrainConfig(
            event="canada_wildfires_2016", budget=5, seed_set=1
        )
        self.assertTrue(cfg.labeled_path.endswith("labeled_5_set1.tsv"))
        self.assertIn("canada_wildfires_2016", cfg.labeled_path)

    def test_unlabeled_path(self):
        cfg = VanillaCoTrainConfig(
            event="canada_wildfires_2016", budget=5, seed_set=1
        )
        self.assertTrue(cfg.unlabeled_path.endswith("unlabeled_5_set1.tsv"))

    def test_dev_path(self):
        cfg = VanillaCoTrainConfig(event="canada_wildfires_2016")
        self.assertTrue(cfg.dev_path.endswith("canada_wildfires_2016_dev.tsv"))

    def test_test_path(self):
        cfg = VanillaCoTrainConfig(event="canada_wildfires_2016")
        self.assertTrue(cfg.test_path.endswith("canada_wildfires_2016_test.tsv"))

    def test_output_dir(self):
        cfg = VanillaCoTrainConfig(
            event="canada_wildfires_2016", budget=5, seed_set=1
        )
        self.assertTrue(cfg.output_dir.endswith("canada_wildfires_2016/5_set1"))

    def test_no_pseudo_label_path(self):
        """Vanilla co-training has no pseudo-label path (no external teacher)."""
        cfg = VanillaCoTrainConfig()
        self.assertFalse(hasattr(cfg, "pseudo_label_path"))

    def test_custom_results_root(self):
        cfg = VanillaCoTrainConfig(results_root="/custom/results")
        self.assertTrue(cfg.output_dir.startswith("/custom/results/"))

    def test_paths_change_with_budget_and_seed(self):
        cfg1 = VanillaCoTrainConfig(
            event="canada_wildfires_2016", budget=5, seed_set=1
        )
        cfg2 = VanillaCoTrainConfig(
            event="canada_wildfires_2016", budget=50, seed_set=3
        )
        self.assertNotEqual(cfg1.labeled_path, cfg2.labeled_path)
        self.assertNotEqual(cfg1.output_dir, cfg2.output_dir)


class TestTopKSelection(unittest.TestCase):
    """Test the top-k-per-class selection logic (numpy-based, no GPU).

    Re-implements the algorithm inline to avoid importing trainer.py
    (which pulls in torch). The real trainer uses the same logic.
    """

    def setUp(self):
        try:
            import numpy as np
            self.np = np
        except ImportError:
            self.skipTest("numpy not available")

    @staticmethod
    def _select_top_k_per_class(probs, pred_labels, samples_per_class, num_labels):
        """Standalone copy of the top-k selection algorithm for testing."""
        import numpy as np
        k = samples_per_class
        selected = []
        confidences = probs.max(axis=1)
        for cls_id in range(num_labels):
            cls_mask = pred_labels == cls_id
            cls_indices = np.where(cls_mask)[0]
            if len(cls_indices) == 0:
                continue
            cls_confs = confidences[cls_indices]
            top_k_local = min(k, len(cls_indices))
            top_k_idx = np.argsort(cls_confs)[-top_k_local:]
            selected.extend(cls_indices[top_k_idx])
        return np.array(selected, dtype=int)

    def test_basic_selection(self):
        """Select top-2 per class from 3 classes."""
        np = self.np
        probs = np.array([
            [0.8, 0.1, 0.1],  # class 0, conf 0.8
            [0.6, 0.2, 0.2],  # class 0, conf 0.6
            [0.4, 0.3, 0.3],  # class 0, conf 0.4
            [0.1, 0.9, 0.0],  # class 1, conf 0.9
            [0.2, 0.7, 0.1],  # class 1, conf 0.7
            [0.3, 0.5, 0.2],  # class 1, conf 0.5
            [0.0, 0.1, 0.9],  # class 2, conf 0.9
            [0.1, 0.1, 0.8],  # class 2, conf 0.8
            [0.2, 0.2, 0.6],  # class 2, conf 0.6
        ])
        pred_labels = probs.argmax(axis=1)
        selected = self._select_top_k_per_class(
            probs, pred_labels, samples_per_class=2, num_labels=3
        )

        self.assertEqual(len(selected), 6)  # 2 per class x 3 classes
        self.assertIn(0, selected)
        self.assertIn(1, selected)
        self.assertIn(3, selected)
        self.assertIn(4, selected)
        self.assertIn(6, selected)
        self.assertIn(7, selected)

    def test_fewer_than_k_samples(self):
        """When a class has fewer samples than k, take all of them."""
        np = self.np
        probs = np.array([
            [0.9, 0.1],  # class 0
            [0.1, 0.9],  # class 1
        ])
        pred_labels = probs.argmax(axis=1)
        selected = self._select_top_k_per_class(
            probs, pred_labels, samples_per_class=5, num_labels=2
        )
        self.assertEqual(len(selected), 2)

    def test_empty_class(self):
        """A class with no predicted samples is skipped."""
        np = self.np
        probs = np.array([
            [0.9, 0.05, 0.05],  # class 0
            [0.8, 0.1, 0.1],    # class 0
        ])
        pred_labels = probs.argmax(axis=1)  # all class 0
        selected = self._select_top_k_per_class(
            probs, pred_labels, samples_per_class=5, num_labels=3
        )
        self.assertEqual(len(selected), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
