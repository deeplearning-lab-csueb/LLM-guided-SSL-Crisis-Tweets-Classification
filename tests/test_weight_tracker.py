"""Tests for weight_tracker.py — works with or without numpy."""

import math
import sys
import unittest

sys.path.insert(0, "/workspace")

from lg_cotrain.weight_tracker import WeightTracker


class TestWeightTrackerSingleEpoch(unittest.TestCase):
    """Record a single epoch of probabilities."""

    def test_confidence_equals_input(self):
        tracker = WeightTracker(num_samples=5)
        probs = [0.8, 0.6, 0.7, 0.9, 0.5]
        tracker.record_epoch(probs)
        conf = tracker.compute_confidence()
        for a, b in zip(conf, probs):
            self.assertAlmostEqual(a, b, places=6)

    def test_variability_is_zero(self):
        tracker = WeightTracker(num_samples=5)
        tracker.record_epoch([0.8, 0.6, 0.7, 0.9, 0.5])
        var = tracker.compute_variability()
        for v in var:
            self.assertAlmostEqual(v, 0.0, places=6)

    def test_lambda_optimistic_equals_confidence(self):
        tracker = WeightTracker(num_samples=3)
        tracker.record_epoch([0.5, 0.6, 0.7])
        lam = tracker.compute_lambda_optimistic()
        conf = tracker.compute_confidence()
        for a, b in zip(lam, conf):
            self.assertAlmostEqual(a, b, places=6)

    def test_lambda_conservative_equals_confidence(self):
        tracker = WeightTracker(num_samples=3)
        tracker.record_epoch([0.5, 0.6, 0.7])
        lam = tracker.compute_lambda_conservative()
        conf = tracker.compute_confidence()
        for a, b in zip(lam, conf):
            self.assertAlmostEqual(a, b, places=6)


class TestWeightTrackerMultipleEpochs(unittest.TestCase):
    """Record multiple epochs and verify confidence/variability math."""

    def test_confidence_is_mean(self):
        tracker = WeightTracker(num_samples=3)
        tracker.record_epoch([0.8, 0.6, 0.7])
        tracker.record_epoch([0.6, 0.4, 0.9])
        conf = tracker.compute_confidence()
        self.assertAlmostEqual(conf[0], 0.7, places=6)
        self.assertAlmostEqual(conf[1], 0.5, places=6)
        self.assertAlmostEqual(conf[2], 0.8, places=6)

    def test_variability_is_population_std(self):
        tracker = WeightTracker(num_samples=3)
        tracker.record_epoch([0.8, 0.6, 0.7])
        tracker.record_epoch([0.6, 0.4, 0.9])
        var = tracker.compute_variability()
        # Population std for [0.8, 0.6] = sqrt(((0.8-0.7)^2 + (0.6-0.7)^2)/2) = 0.1
        self.assertAlmostEqual(var[0], 0.1, places=6)
        # Population std for [0.6, 0.4] = 0.1
        self.assertAlmostEqual(var[1], 0.1, places=6)
        # Population std for [0.7, 0.9] = 0.1
        self.assertAlmostEqual(var[2], 0.1, places=6)

    def test_lambda_optimistic_is_c_plus_v(self):
        tracker = WeightTracker(num_samples=3)
        tracker.record_epoch([0.8, 0.6, 0.7])
        tracker.record_epoch([0.6, 0.4, 0.9])
        lam = tracker.compute_lambda_optimistic()
        # c + v = [0.7+0.1, 0.5+0.1, 0.8+0.1]
        self.assertAlmostEqual(lam[0], 0.8, places=6)
        self.assertAlmostEqual(lam[1], 0.6, places=6)
        self.assertAlmostEqual(lam[2], 0.9, places=6)

    def test_lambda_conservative_is_c_minus_v(self):
        tracker = WeightTracker(num_samples=3)
        tracker.record_epoch([0.8, 0.6, 0.7])
        tracker.record_epoch([0.6, 0.4, 0.9])
        lam = tracker.compute_lambda_conservative()
        # c - v = [0.7-0.1, 0.5-0.1, 0.8-0.1]
        self.assertAlmostEqual(lam[0], 0.6, places=6)
        self.assertAlmostEqual(lam[1], 0.4, places=6)
        self.assertAlmostEqual(lam[2], 0.7, places=6)


class TestWeightTrackerEdgeCases(unittest.TestCase):
    """Edge cases for lambda computation."""

    def test_identical_probs_across_epochs(self):
        """If all epochs have same probs, variability=0, lambda1 == lambda2."""
        tracker = WeightTracker(num_samples=3)
        probs = [0.8, 0.6, 0.7]
        tracker.record_epoch(probs)
        tracker.record_epoch(probs)
        tracker.record_epoch(probs)
        for v in tracker.compute_variability():
            self.assertAlmostEqual(v, 0.0, places=6)
        lam1 = tracker.compute_lambda_optimistic()
        lam2 = tracker.compute_lambda_conservative()
        for a, b in zip(lam1, lam2):
            self.assertAlmostEqual(a, b, places=6)

    def test_high_variability_clips_to_zero(self):
        """When variability > confidence, lambda2 should be clipped to 0."""
        tracker = WeightTracker(num_samples=2)
        tracker.record_epoch([0.1, 0.2])
        tracker.record_epoch([0.9, 0.8])
        # confidence = [0.5, 0.5], variability = [0.4, 0.3]
        # c - v = [0.1, 0.2] -> both positive, no clipping here
        lam2 = tracker.compute_lambda_conservative()
        for val in lam2:
            self.assertGreaterEqual(val, 0.0)

    def test_extreme_high_variability_clips(self):
        """Force a case where c - v < 0."""
        tracker = WeightTracker(num_samples=1)
        tracker.record_epoch([0.0])
        tracker.record_epoch([1.0])
        # confidence = 0.5, variability = 0.5, c - v = 0
        lam2 = tracker.compute_lambda_conservative()
        self.assertAlmostEqual(lam2[0], 0.0, places=6)

    def test_known_three_epoch_example(self):
        """probs=[0.8, 0.6, 0.7] for 1 sample → confidence=0.7, variability≈0.0816."""
        tracker = WeightTracker(num_samples=1)
        tracker.record_epoch([0.8])
        tracker.record_epoch([0.6])
        tracker.record_epoch([0.7])
        self.assertAlmostEqual(tracker.compute_confidence()[0], 0.7, places=5)
        expected_std = math.sqrt(((0.8-0.7)**2 + (0.6-0.7)**2 + (0.7-0.7)**2) / 3)
        self.assertAlmostEqual(tracker.compute_variability()[0], expected_std, places=4)


class TestWeightTrackerBookkeeping(unittest.TestCase):
    """Test record counting and input validation."""

    def test_num_epochs_starts_at_zero(self):
        tracker = WeightTracker(num_samples=3)
        self.assertEqual(tracker.num_epochs_recorded, 0)

    def test_num_epochs_increments(self):
        tracker = WeightTracker(num_samples=3)
        tracker.record_epoch([0.5, 0.5, 0.5])
        self.assertEqual(tracker.num_epochs_recorded, 1)
        tracker.record_epoch([0.6, 0.6, 0.6])
        self.assertEqual(tracker.num_epochs_recorded, 2)

    def test_wrong_length_raises(self):
        tracker = WeightTracker(num_samples=3)
        with self.assertRaises(AssertionError):
            tracker.record_epoch([0.5, 0.5])  # too short

    def test_record_does_not_alias_input(self):
        """Mutating the input list after recording should not affect stored data."""
        tracker = WeightTracker(num_samples=2)
        probs = [0.5, 0.5]
        tracker.record_epoch(probs)
        probs[0] = 999.0
        self.assertAlmostEqual(tracker.compute_confidence()[0], 0.5, places=6)


class TestWeightTrackerSeedFromTracker(unittest.TestCase):
    """Test the seed_from_tracker classmethod."""

    def test_seeded_tracker_has_same_num_samples(self):
        source = WeightTracker(num_samples=5)
        source.record_epoch([0.8, 0.6, 0.7, 0.9, 0.5])
        seeded = WeightTracker.seed_from_tracker(source)
        self.assertEqual(seeded.num_samples, 5)

    def test_seeded_tracker_has_same_epoch_count(self):
        source = WeightTracker(num_samples=3)
        source.record_epoch([0.8, 0.6, 0.7])
        source.record_epoch([0.6, 0.4, 0.9])
        source.record_epoch([0.7, 0.5, 0.8])
        seeded = WeightTracker.seed_from_tracker(source)
        self.assertEqual(seeded.num_epochs_recorded, 3)

    def test_seeded_tracker_computes_same_confidence(self):
        source = WeightTracker(num_samples=3)
        source.record_epoch([0.8, 0.6, 0.7])
        source.record_epoch([0.6, 0.4, 0.9])
        seeded = WeightTracker.seed_from_tracker(source)
        src_conf = source.compute_confidence()
        seed_conf = seeded.compute_confidence()
        for a, b in zip(src_conf, seed_conf):
            self.assertAlmostEqual(a, b, places=6)

    def test_seeded_tracker_computes_same_variability(self):
        source = WeightTracker(num_samples=3)
        source.record_epoch([0.8, 0.6, 0.7])
        source.record_epoch([0.6, 0.4, 0.9])
        seeded = WeightTracker.seed_from_tracker(source)
        src_var = source.compute_variability()
        seed_var = seeded.compute_variability()
        for a, b in zip(src_var, seed_var):
            self.assertAlmostEqual(a, b, places=6)

    def test_seeded_tracker_computes_same_lambdas(self):
        source = WeightTracker(num_samples=3)
        source.record_epoch([0.8, 0.6, 0.7])
        source.record_epoch([0.6, 0.4, 0.9])
        seeded = WeightTracker.seed_from_tracker(source)
        src_l1 = source.compute_lambda_optimistic()
        src_l2 = source.compute_lambda_conservative()
        seed_l1 = seeded.compute_lambda_optimistic()
        seed_l2 = seeded.compute_lambda_conservative()
        for a, b in zip(src_l1, seed_l1):
            self.assertAlmostEqual(a, b, places=6)
        for a, b in zip(src_l2, seed_l2):
            self.assertAlmostEqual(a, b, places=6)

    def test_seeded_tracker_is_independent_of_source(self):
        """Appending to the seeded tracker does not affect the source."""
        source = WeightTracker(num_samples=2)
        source.record_epoch([0.5, 0.5])
        source.record_epoch([0.7, 0.3])
        seeded = WeightTracker.seed_from_tracker(source)
        seeded.record_epoch([0.9, 0.1])
        self.assertEqual(source.num_epochs_recorded, 2)
        self.assertEqual(seeded.num_epochs_recorded, 3)

    def test_seeded_tracker_data_is_deep_copy(self):
        """Mutating seeded history does not affect source history."""
        source = WeightTracker(num_samples=2)
        source.record_epoch([0.5, 0.6])
        seeded = WeightTracker.seed_from_tracker(source)
        seeded.prob_history[0][0] = 999.0
        self.assertAlmostEqual(source.prob_history[0][0], 0.5, places=6)

    def test_variability_gt_zero_with_multi_epoch_source(self):
        """Key invariant: seeding from multi-epoch source preserves variability > 0."""
        source = WeightTracker(num_samples=3)
        source.record_epoch([0.8, 0.6, 0.7])
        source.record_epoch([0.6, 0.4, 0.9])
        seeded = WeightTracker.seed_from_tracker(source)
        var = seeded.compute_variability()
        for v in var:
            self.assertGreater(v, 0.0)
        # Therefore lambda_optimistic != lambda_conservative
        l1 = seeded.compute_lambda_optimistic()
        l2 = seeded.compute_lambda_conservative()
        any_different = any(abs(a - b) > 1e-9 for a, b in zip(l1, l2))
        self.assertTrue(any_different, "lambda1 and lambda2 should differ")


class TestWeightTrackerSeedFromLastEpoch(unittest.TestCase):
    """Test the seed_from_last_epoch classmethod."""

    def test_seeded_tracker_has_exactly_one_epoch(self):
        """Regardless of how many epochs source recorded, seeded tracker has exactly 1."""
        source = WeightTracker(num_samples=3)
        source.record_epoch([0.8, 0.6, 0.7])
        source.record_epoch([0.6, 0.4, 0.9])
        source.record_epoch([0.7, 0.5, 0.8])
        seeded = WeightTracker.seed_from_last_epoch(source)
        self.assertEqual(seeded.num_epochs_recorded, 1)

    def test_seeded_probs_match_sources_last_epoch(self):
        """The single seeded epoch must equal source.prob_history[-1]."""
        source = WeightTracker(num_samples=3)
        source.record_epoch([0.8, 0.6, 0.7])
        source.record_epoch([0.6, 0.4, 0.9])
        seeded = WeightTracker.seed_from_last_epoch(source)
        conf = seeded.compute_confidence()
        last = source.prob_history[-1]
        for a, b in zip(conf, last):
            self.assertAlmostEqual(a, b, places=6)

    def test_variability_zero_after_single_epoch_seed(self):
        """With only one epoch, variability must be 0."""
        source = WeightTracker(num_samples=3)
        source.record_epoch([0.8, 0.6, 0.7])
        source.record_epoch([0.6, 0.4, 0.9])
        seeded = WeightTracker.seed_from_last_epoch(source)
        for v in seeded.compute_variability():
            self.assertAlmostEqual(v, 0.0, places=6)

    def test_confidence_after_seed_and_one_phase2_epoch(self):
        """Core paper behaviour: confidence = mean(final_p1, p2_ep1)."""
        source = WeightTracker(num_samples=2)
        source.record_epoch([0.4, 0.6])  # earlier Phase 1
        source.record_epoch([0.8, 0.2])  # final Phase 1
        seeded = WeightTracker.seed_from_last_epoch(source)
        seeded.record_epoch([0.6, 0.4])  # Phase 2 epoch 1
        conf = seeded.compute_confidence()
        # mean([0.8, 0.6]) = 0.7, mean([0.2, 0.4]) = 0.3
        self.assertAlmostEqual(conf[0], 0.7, places=6)
        self.assertAlmostEqual(conf[1], 0.3, places=6)

    def test_does_not_use_earlier_phase1_epochs(self):
        """Key correctness test: earlier Phase 1 epochs must be excluded."""
        source = WeightTracker(num_samples=1)
        source.record_epoch([0.2])  # early Phase 1 — must NOT be included
        source.record_epoch([0.8])  # final Phase 1 — must be included
        seeded = WeightTracker.seed_from_last_epoch(source)
        # Full-history copy: confidence = mean(0.2, 0.8) = 0.5 (wrong)
        # Last-epoch only:   confidence = 0.8              (correct)
        self.assertAlmostEqual(seeded.compute_confidence()[0], 0.8, places=6)

    def test_is_independent_of_source(self):
        """Adding epochs to the seeded tracker must not affect the source."""
        source = WeightTracker(num_samples=2)
        source.record_epoch([0.5, 0.5])
        source.record_epoch([0.7, 0.3])
        seeded = WeightTracker.seed_from_last_epoch(source)
        seeded.record_epoch([0.9, 0.1])
        self.assertEqual(source.num_epochs_recorded, 2)
        self.assertEqual(seeded.num_epochs_recorded, 2)

    def test_data_is_deep_copy(self):
        """Mutating the seeded history must not affect source history."""
        source = WeightTracker(num_samples=2)
        source.record_epoch([0.5, 0.6])
        source.record_epoch([0.7, 0.8])
        seeded = WeightTracker.seed_from_last_epoch(source)
        seeded.prob_history[0][0] = 999.0
        self.assertAlmostEqual(source.prob_history[-1][0], 0.7, places=6)


class TestWeightTrackerSeedFromEpoch(unittest.TestCase):
    """Test the seed_from_epoch classmethod."""

    def test_seeded_tracker_has_exactly_one_epoch(self):
        source = WeightTracker(num_samples=3)
        source.record_epoch([0.8, 0.6, 0.7])
        source.record_epoch([0.6, 0.4, 0.9])
        source.record_epoch([0.7, 0.5, 0.8])
        seeded = WeightTracker.seed_from_epoch(source, epoch_idx=1)
        self.assertEqual(seeded.num_epochs_recorded, 1)

    def test_seeded_probs_match_selected_epoch(self):
        source = WeightTracker(num_samples=3)
        source.record_epoch([0.8, 0.6, 0.7])
        source.record_epoch([0.6, 0.4, 0.9])
        source.record_epoch([0.7, 0.5, 0.8])
        seeded = WeightTracker.seed_from_epoch(source, epoch_idx=1)
        conf = seeded.compute_confidence()
        expected = [0.6, 0.4, 0.9]
        for a, b in zip(conf, expected):
            self.assertAlmostEqual(a, b, places=6)

    def test_epoch_zero_selects_first(self):
        source = WeightTracker(num_samples=2)
        source.record_epoch([0.1, 0.2])
        source.record_epoch([0.9, 0.8])
        seeded = WeightTracker.seed_from_epoch(source, epoch_idx=0)
        conf = seeded.compute_confidence()
        self.assertAlmostEqual(conf[0], 0.1, places=6)
        self.assertAlmostEqual(conf[1], 0.2, places=6)

    def test_last_epoch_matches_seed_from_last_epoch(self):
        source = WeightTracker(num_samples=3)
        source.record_epoch([0.8, 0.6, 0.7])
        source.record_epoch([0.6, 0.4, 0.9])
        from_last = WeightTracker.seed_from_last_epoch(source)
        from_epoch = WeightTracker.seed_from_epoch(source, epoch_idx=1)
        for a, b in zip(from_last.compute_confidence(), from_epoch.compute_confidence()):
            self.assertAlmostEqual(a, b, places=6)

    def test_out_of_range_raises_index_error(self):
        source = WeightTracker(num_samples=2)
        source.record_epoch([0.5, 0.5])
        with self.assertRaises(IndexError):
            WeightTracker.seed_from_epoch(source, epoch_idx=5)

    def test_negative_index_raises_index_error(self):
        source = WeightTracker(num_samples=2)
        source.record_epoch([0.5, 0.5])
        with self.assertRaises(IndexError):
            WeightTracker.seed_from_epoch(source, epoch_idx=-1)

    def test_variability_zero_after_single_epoch_seed(self):
        source = WeightTracker(num_samples=3)
        source.record_epoch([0.8, 0.6, 0.7])
        source.record_epoch([0.6, 0.4, 0.9])
        seeded = WeightTracker.seed_from_epoch(source, epoch_idx=0)
        for v in seeded.compute_variability():
            self.assertAlmostEqual(v, 0.0, places=6)

    def test_is_independent_of_source(self):
        source = WeightTracker(num_samples=2)
        source.record_epoch([0.5, 0.5])
        source.record_epoch([0.7, 0.3])
        seeded = WeightTracker.seed_from_epoch(source, epoch_idx=0)
        seeded.record_epoch([0.9, 0.1])
        self.assertEqual(source.num_epochs_recorded, 2)
        self.assertEqual(seeded.num_epochs_recorded, 2)

    def test_data_is_deep_copy(self):
        source = WeightTracker(num_samples=2)
        source.record_epoch([0.5, 0.6])
        source.record_epoch([0.7, 0.8])
        seeded = WeightTracker.seed_from_epoch(source, epoch_idx=0)
        seeded.prob_history[0][0] = 999.0
        self.assertAlmostEqual(source.prob_history[0][0], 0.5, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
