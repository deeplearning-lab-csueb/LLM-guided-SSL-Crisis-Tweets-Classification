"""Tests for the supervised_baseline module."""

import unittest

from supervised_baseline.config import SupervisedBaselineConfig


class TestSupervisedBaselineConfig(unittest.TestCase):
    """Test SupervisedBaselineConfig defaults and path computation."""

    def test_default_model(self):
        cfg = SupervisedBaselineConfig()
        self.assertEqual(cfg.model_name, "vinai/bertweet-base")

    def test_default_max_epochs(self):
        cfg = SupervisedBaselineConfig()
        self.assertEqual(cfg.max_epochs, 100)

    def test_default_patience(self):
        cfg = SupervisedBaselineConfig()
        self.assertEqual(cfg.patience, 5)

    def test_default_batch_size(self):
        cfg = SupervisedBaselineConfig()
        self.assertEqual(cfg.batch_size, 32)

    def test_default_lr(self):
        cfg = SupervisedBaselineConfig()
        self.assertEqual(cfg.lr, 2e-5)

    def test_default_weight_decay(self):
        cfg = SupervisedBaselineConfig()
        self.assertEqual(cfg.weight_decay, 0.01)

    def test_default_warmup_ratio(self):
        cfg = SupervisedBaselineConfig()
        self.assertEqual(cfg.warmup_ratio, 0.1)

    def test_default_max_seq_length(self):
        cfg = SupervisedBaselineConfig()
        self.assertEqual(cfg.max_seq_length, 128)

    def test_default_device_is_none(self):
        cfg = SupervisedBaselineConfig()
        self.assertIsNone(cfg.device)

    def test_labeled_path(self):
        cfg = SupervisedBaselineConfig(
            event="hurricane_harvey_2017", budget=50, seed_set=2,
            data_root="/data",
        )
        self.assertIn("hurricane_harvey_2017", cfg.labeled_path)
        self.assertIn("labeled_50_set2.tsv", cfg.labeled_path)

    def test_unlabeled_path(self):
        cfg = SupervisedBaselineConfig(
            event="hurricane_harvey_2017", budget=50, seed_set=2,
            data_root="/data",
        )
        self.assertIn("unlabeled_50_set2.tsv", cfg.unlabeled_path)

    def test_dev_path(self):
        cfg = SupervisedBaselineConfig(
            event="hurricane_harvey_2017", data_root="/data",
        )
        self.assertIn("hurricane_harvey_2017_dev.tsv", cfg.dev_path)

    def test_test_path(self):
        cfg = SupervisedBaselineConfig(
            event="hurricane_harvey_2017", data_root="/data",
        )
        self.assertIn("hurricane_harvey_2017_test.tsv", cfg.test_path)

    def test_output_dir(self):
        cfg = SupervisedBaselineConfig(
            event="hurricane_harvey_2017", budget=25, seed_set=3,
            results_root="/results",
        )
        self.assertIn("hurricane_harvey_2017", cfg.output_dir)
        self.assertIn("25_set3", cfg.output_dir)

    def test_paths_change_with_budget_and_seed(self):
        cfg1 = SupervisedBaselineConfig(budget=5, seed_set=1)
        cfg2 = SupervisedBaselineConfig(budget=50, seed_set=3)
        self.assertNotEqual(cfg1.labeled_path, cfg2.labeled_path)
        self.assertNotEqual(cfg1.output_dir, cfg2.output_dir)

    def test_custom_results_root(self):
        cfg = SupervisedBaselineConfig(
            event="kaikoura_earthquake_2016", budget=10, seed_set=1,
            results_root="/custom/results",
        )
        self.assertTrue(cfg.output_dir.startswith("/custom/results"))


if __name__ == "__main__":
    unittest.main()
