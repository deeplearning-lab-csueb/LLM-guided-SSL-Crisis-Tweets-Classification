"""Tests for data_loading.py — label encoder and split logic testable without pandas."""

import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, "/workspace")

from lg_cotrain.data_loading import (
    CLASS_LABELS,
    build_label_encoder,
    detect_event_classes,
    split_labeled_set,
    _split_labeled_set_pure,
)


class TestClassLabels(unittest.TestCase):
    """CLASS_LABELS constant is correct."""

    def test_count(self):
        self.assertEqual(len(CLASS_LABELS), 10)

    def test_sorted(self):
        self.assertEqual(CLASS_LABELS, sorted(CLASS_LABELS))

    def test_expected_values(self):
        expected = {
            "caution_and_advice",
            "displaced_people_and_evacuations",
            "infrastructure_and_utility_damage",
            "injured_or_dead_people",
            "missing_or_found_people",
            "not_humanitarian",
            "other_relevant_information",
            "requests_or_urgent_needs",
            "rescue_volunteering_or_donation_effort",
            "sympathy_and_support",
        }
        self.assertEqual(set(CLASS_LABELS), expected)


class TestBuildLabelEncoder(unittest.TestCase):
    """build_label_encoder produces consistent alphabetical mapping."""

    def test_label2id_length(self):
        label2id, _ = build_label_encoder()
        self.assertEqual(len(label2id), 10)

    def test_id2label_length(self):
        _, id2label = build_label_encoder()
        self.assertEqual(len(id2label), 10)

    def test_alphabetical_order(self):
        label2id, id2label = build_label_encoder()
        for i, lbl in enumerate(sorted(CLASS_LABELS)):
            self.assertEqual(label2id[lbl], i)
            self.assertEqual(id2label[i], lbl)

    def test_roundtrip(self):
        label2id, id2label = build_label_encoder()
        for label in CLASS_LABELS:
            self.assertEqual(id2label[label2id[label]], label)

    def test_caution_and_advice_is_zero(self):
        """First alphabetically should be id 0."""
        label2id, _ = build_label_encoder()
        self.assertEqual(label2id["caution_and_advice"], 0)

    def test_custom_labels_subset(self):
        """build_label_encoder with a custom subset produces matching-size mappings."""
        subset = ["not_humanitarian", "sympathy_and_support"]
        label2id, id2label = build_label_encoder(labels=subset)
        self.assertEqual(len(label2id), 2)
        self.assertEqual(len(id2label), 2)
        self.assertEqual(label2id["not_humanitarian"], 0)
        self.assertEqual(label2id["sympathy_and_support"], 1)
        self.assertEqual(id2label[0], "not_humanitarian")
        self.assertEqual(id2label[1], "sympathy_and_support")

    def test_custom_labels_roundtrip(self):
        """Roundtrip works for custom label list."""
        subset = ["caution_and_advice", "infrastructure_and_utility_damage", "not_humanitarian"]
        label2id, id2label = build_label_encoder(labels=subset)
        for label in subset:
            self.assertEqual(id2label[label2id[label]], label)

    def test_default_labels_unchanged(self):
        """Calling with labels=None still uses all 10 CLASS_LABELS."""
        label2id, _ = build_label_encoder(labels=None)
        self.assertEqual(len(label2id), 10)


class TestDetectEventClasses(unittest.TestCase):
    """Test detect_event_classes with list-of-dicts (pure-Python path)."""

    def test_single_list(self):
        records = [
            {"class_label": "not_humanitarian"},
            {"class_label": "caution_and_advice"},
            {"class_label": "not_humanitarian"},
        ]
        result = detect_event_classes(records)
        self.assertEqual(result, ["caution_and_advice", "not_humanitarian"])

    def test_union_of_multiple_lists(self):
        list1 = [{"class_label": "caution_and_advice"}]
        list2 = [{"class_label": "not_humanitarian"}]
        list3 = [{"class_label": "sympathy_and_support"}]
        result = detect_event_classes(list1, list2, list3)
        self.assertEqual(result, [
            "caution_and_advice", "not_humanitarian", "sympathy_and_support",
        ])

    def test_overlapping_lists(self):
        list1 = [
            {"class_label": "caution_and_advice"},
            {"class_label": "not_humanitarian"},
        ]
        list2 = [
            {"class_label": "not_humanitarian"},
            {"class_label": "sympathy_and_support"},
        ]
        result = detect_event_classes(list1, list2)
        self.assertEqual(result, [
            "caution_and_advice", "not_humanitarian", "sympathy_and_support",
        ])

    def test_result_is_sorted(self):
        records = [
            {"class_label": "sympathy_and_support"},
            {"class_label": "caution_and_advice"},
            {"class_label": "not_humanitarian"},
        ]
        result = detect_event_classes(records)
        self.assertEqual(result, sorted(result))

    def test_empty_input(self):
        result = detect_event_classes([])
        self.assertEqual(result, [])


class TestSplitLabeledSetPure(unittest.TestCase):
    """Test the pure-Python split function with list-of-dicts."""

    def _make_records(self, per_class=2):
        """Create list-of-dict records with per_class samples per class."""
        records = []
        for cls in CLASS_LABELS:
            for j in range(per_class):
                records.append({
                    "tweet_id": f"{cls}_{j}",
                    "tweet_text": f"Text about {cls} {j}",
                    "class_label": cls,
                })
        return records

    def test_disjoint_union(self):
        """D_l1 and D_l2 together cover all records without overlap."""
        records = self._make_records(per_class=2)
        d1, d2 = _split_labeled_set_pure(records, seed=42)
        ids1 = {r["tweet_id"] for r in d1}
        ids2 = {r["tweet_id"] for r in d2}
        all_ids = {r["tweet_id"] for r in records}
        self.assertEqual(ids1 | ids2, all_ids)
        self.assertEqual(len(ids1 & ids2), 0)

    def test_each_class_in_both_halves(self):
        """Each class should be represented in both splits."""
        records = self._make_records(per_class=2)
        d1, d2 = _split_labeled_set_pure(records, seed=42)
        classes1 = {r["class_label"] for r in d1}
        classes2 = {r["class_label"] for r in d2}
        for cls in CLASS_LABELS:
            self.assertIn(cls, classes1)
            self.assertIn(cls, classes2)

    def test_deterministic_same_seed(self):
        """Same seed produces same split."""
        records = self._make_records(per_class=4)
        d1a, d2a = _split_labeled_set_pure(records, seed=42)
        d1b, d2b = _split_labeled_set_pure(records, seed=42)
        self.assertEqual(
            [r["tweet_id"] for r in d1a],
            [r["tweet_id"] for r in d1b],
        )
        self.assertEqual(
            [r["tweet_id"] for r in d2a],
            [r["tweet_id"] for r in d2b],
        )

    def test_different_seed_different_split(self):
        """Different seeds produce different splits."""
        records = self._make_records(per_class=4)
        d1a, _ = _split_labeled_set_pure(records, seed=42)
        d1b, _ = _split_labeled_set_pure(records, seed=99)
        ids_a = [r["tweet_id"] for r in d1a]
        ids_b = [r["tweet_id"] for r in d1b]
        self.assertNotEqual(ids_a, ids_b)

    def test_odd_class_size(self):
        """Odd class count: first half gets extra sample."""
        records = self._make_records(per_class=3)  # 3 per class = odd
        d1, d2 = _split_labeled_set_pure(records, seed=42)
        # Each class: d1 gets 2, d2 gets 1
        for cls in CLASS_LABELS:
            count1 = sum(1 for r in d1 if r["class_label"] == cls)
            count2 = sum(1 for r in d2 if r["class_label"] == cls)
            self.assertEqual(count1, 2)
            self.assertEqual(count2, 1)

    def test_even_class_size(self):
        """Even class count: both halves get equal samples."""
        records = self._make_records(per_class=4)
        d1, d2 = _split_labeled_set_pure(records, seed=42)
        for cls in CLASS_LABELS:
            count1 = sum(1 for r in d1 if r["class_label"] == cls)
            count2 = sum(1 for r in d2 if r["class_label"] == cls)
            self.assertEqual(count1, 2)
            self.assertEqual(count2, 2)

    def test_single_sample_per_class(self):
        """1 per class: first half gets the sample, second half is empty for that class."""
        records = self._make_records(per_class=1)  # 1 per class
        d1, d2 = _split_labeled_set_pure(records, seed=42)
        # Each class in d1 has 1 sample, d2 has 0
        self.assertEqual(len(d1), 10)
        self.assertEqual(len(d2), 0)


class TestLoadTsvIO(unittest.TestCase):
    """Test load_tsv with actual files on disk."""

    def _write_tsv(self, path, rows):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["tweet_id", "tweet_text", "class_label"])
            for row in rows:
                writer.writerow(row)

    def test_load_tsv_reads_correctly(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            self.skipTest("pandas not available")
        from lg_cotrain.data_loading import load_tsv

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            path = f.name
        try:
            self._write_tsv(path, [
                ["1", "hello world", "caution_and_advice"],
                ["2", "test tweet", "not_humanitarian"],
            ])
            df = load_tsv(path)
            self.assertEqual(len(df), 2)
            self.assertIn("tweet_id", df.columns)
            self.assertIn("tweet_text", df.columns)
            self.assertIn("class_label", df.columns)
            self.assertEqual(df.iloc[0]["tweet_id"], "1")
        finally:
            os.unlink(path)

    def test_load_tsv_rejects_bad_columns(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            self.skipTest("pandas not available")
        from lg_cotrain.data_loading import load_tsv

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            path = f.name
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["id", "text"])  # wrong columns
            writer.writerow(["1", "hello"])
        try:
            with self.assertRaises(ValueError):
                load_tsv(path)
        finally:
            os.unlink(path)


class TestLoadPseudoLabelsIO(unittest.TestCase):
    """Test load_pseudo_labels with actual CSV files."""

    def test_load_pseudo_labels(self):
        try:
            import pandas  # noqa: F401
        except ImportError:
            self.skipTest("pandas not available")
        from lg_cotrain.data_loading import load_pseudo_labels

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
            writer = csv.writer(f)
            writer.writerow(["tweet_id", "tweet_text", "class_label",
                             "predicted_label", "confidence", "entropy", "status"])
            writer.writerow(["1", "hello", "caution_and_advice",
                             "not_humanitarian", "0.9", "", "ok"])
        try:
            df = load_pseudo_labels(path)
            self.assertEqual(len(df), 1)
            self.assertIn("predicted_label", df.columns)
            self.assertIn("confidence", df.columns)
            self.assertEqual(df.iloc[0]["tweet_id"], "1")
        finally:
            os.unlink(path)


class TestBuildDLG(unittest.TestCase):
    """Test build_d_lg join logic."""

    def test_full_match(self):
        try:
            import pandas as pd
            from lg_cotrain.data_loading import build_d_lg
        except ImportError:
            self.skipTest("pandas not available")

        df_unl = pd.DataFrame({
            "tweet_id": ["1", "2", "3"],
            "tweet_text": ["a", "b", "c"],
            "class_label": ["x", "y", "z"],
        })
        df_pseudo = pd.DataFrame({
            "tweet_id": ["1", "2", "3"],
            "tweet_text": ["a", "b", "c"],
            "predicted_label": ["x", "z", "z"],
            "confidence": [0.9, 0.8, 0.7],
        })
        merged = build_d_lg(df_unl, df_pseudo)
        self.assertEqual(len(merged), 3)
        self.assertIn("predicted_label", merged.columns)
        self.assertIn("class_label", merged.columns)
        self.assertIn("confidence", merged.columns)
        self.assertNotIn("tweet_text_pseudo", merged.columns)

    def test_partial_match(self):
        try:
            import pandas as pd
            from lg_cotrain.data_loading import build_d_lg
        except ImportError:
            self.skipTest("pandas not available")

        df_unl = pd.DataFrame({
            "tweet_id": ["1", "2", "3"],
            "tweet_text": ["a", "b", "c"],
            "class_label": ["x", "y", "z"],
        })
        df_pseudo = pd.DataFrame({
            "tweet_id": ["1", "3"],  # missing tweet_id "2"
            "tweet_text": ["a", "c"],
            "predicted_label": ["x", "z"],
            "confidence": [0.9, 0.7],
        })
        merged = build_d_lg(df_unl, df_pseudo)
        self.assertEqual(len(merged), 2)



if __name__ == "__main__":
    unittest.main(verbosity=2)
