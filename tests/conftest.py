"""Shared pytest fixtures for LG-CoTrain tests.

These fixtures require pandas. When running with unittest (no pandas),
individual test files handle their own data setup.
"""

import os
import sys

sys.path.insert(0, "/workspace")

try:
    import pandas as pd
    import pytest

    from lg_cotrain.config import LGCoTrainConfig
    from lg_cotrain.data_loading import CLASS_LABELS, build_label_encoder

    CLASSES = CLASS_LABELS

    @pytest.fixture
    def label2id():
        l2i, _ = build_label_encoder()
        return l2i

    @pytest.fixture
    def id2label():
        _, i2l = build_label_encoder()
        return i2l

    @pytest.fixture
    def sample_labeled_df():
        rows = []
        for i, cls in enumerate(CLASSES):
            for j in range(2):
                rows.append({
                    "tweet_id": str(1000 + i * 2 + j),
                    "tweet_text": f"Sample tweet about {cls} number {j}",
                    "class_label": cls,
                })
        return pd.DataFrame(rows)

    @pytest.fixture
    def sample_unlabeled_df():
        rows = []
        for i, cls in enumerate(CLASSES):
            for j in range(4):
                rows.append({
                    "tweet_id": str(2000 + i * 4 + j),
                    "tweet_text": f"Unlabeled tweet about {cls} number {j}",
                    "class_label": cls,
                })
        return pd.DataFrame(rows)

    @pytest.fixture
    def sample_pseudo_df(sample_unlabeled_df):
        df = sample_unlabeled_df.copy()
        df["predicted_label"] = df["class_label"]
        df.loc[0, "predicted_label"] = CLASSES[1]
        df.loc[5, "predicted_label"] = CLASSES[0]
        df["confidence"] = 0.85
        df["entropy"] = ""
        df["status"] = "ok"
        return df

    @pytest.fixture
    def tmp_tsv_files(sample_labeled_df, sample_unlabeled_df, sample_pseudo_df, tmp_path):
        labeled_path = tmp_path / "labeled_5_set1.tsv"
        unlabeled_path = tmp_path / "unlabeled_5_set1.tsv"
        pseudo_path = tmp_path / "pseudo_pred.csv"
        dev_path = tmp_path / "dev.tsv"
        test_path = tmp_path / "test.tsv"

        sample_labeled_df.to_csv(labeled_path, sep="\t", index=False)
        sample_unlabeled_df.to_csv(unlabeled_path, sep="\t", index=False)
        sample_pseudo_df.to_csv(pseudo_path, index=False)
        sample_unlabeled_df.head(8).to_csv(dev_path, sep="\t", index=False)
        sample_unlabeled_df.tail(8).to_csv(test_path, sep="\t", index=False)

        return {
            "labeled": str(labeled_path),
            "unlabeled": str(unlabeled_path),
            "pseudo": str(pseudo_path),
            "dev": str(dev_path),
            "test": str(test_path),
        }

    @pytest.fixture
    def default_config(tmp_tsv_files, tmp_path):
        cfg = LGCoTrainConfig(
            event="test_event", budget=5, seed_set=1,
            data_root=str(tmp_path), results_root=str(tmp_path / "results"),
        )
        cfg.labeled_path = tmp_tsv_files["labeled"]
        cfg.unlabeled_path = tmp_tsv_files["unlabeled"]
        cfg.pseudo_label_path = tmp_tsv_files["pseudo"]
        cfg.dev_path = tmp_tsv_files["dev"]
        cfg.test_path = tmp_tsv_files["test"]
        cfg.output_dir = str(tmp_path / "results" / "test_event" / "5_set1")
        return cfg

except ImportError:
    # No pandas/pytest available — tests use unittest directly
    pass
