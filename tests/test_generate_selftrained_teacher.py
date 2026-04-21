"""Tests for lg_cotrain.generate_selftrained_teacher.

These are pure stdlib tests that don't require torch / transformers - they
exercise path computation, CLI argument parsing, the grid-building logic
inside generate_all(), and the integration with LGCoTrainConfig's
PER_SPLIT_SOURCES branch. The actual model training (which requires torch
and runs on a GPU) is mocked out in the few tests that need it.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from lg_cotrain.generate_selftrained_teacher import (
    ALL_EVENTS,
    DEFAULT_BUDGETS,
    DEFAULT_SEEDS,
    _output_path,
    generate_all,
    main,
)
from lg_cotrain.config import LGCoTrainConfig, PER_SPLIT_SOURCES


class TestConstants(unittest.TestCase):
    """Module-level constants are correct."""

    def test_all_events_has_10(self):
        self.assertEqual(len(ALL_EVENTS), 10)

    def test_all_events_alphabetical(self):
        self.assertEqual(ALL_EVENTS, sorted(ALL_EVENTS))

    def test_default_budgets(self):
        self.assertEqual(DEFAULT_BUDGETS, [5, 10, 25, 50])

    def test_default_seeds(self):
        self.assertEqual(DEFAULT_SEEDS, [1, 2, 3])

    def test_all_events_are_strings(self):
        for ev in ALL_EVENTS:
            self.assertIsInstance(ev, str)
            self.assertGreater(len(ev), 0)


class TestOutputPath(unittest.TestCase):
    """The _output_path helper builds the right per-cell path."""

    def test_basic_shape(self):
        p = _output_path(Path("/data"), "canada_wildfires_2016", 5, 1)
        self.assertEqual(
            str(p),
            "/data/pseudo-labelled/self-trained/canada_wildfires_2016/labeled_5_set1_pseudo.csv",
        )

    def test_different_budget(self):
        p = _output_path(Path("/data"), "canada_wildfires_2016", 50, 1)
        self.assertTrue(str(p).endswith("labeled_50_set1_pseudo.csv"))

    def test_different_seed(self):
        p = _output_path(Path("/data"), "canada_wildfires_2016", 5, 3)
        self.assertTrue(str(p).endswith("labeled_5_set3_pseudo.csv"))

    def test_different_event(self):
        p = _output_path(Path("/data"), "hurricane_harvey_2017", 5, 1)
        self.assertIn("hurricane_harvey_2017", str(p))

    def test_uses_self_trained_subdir(self):
        p = _output_path(Path("/data"), "any_event", 5, 1)
        self.assertIn("/pseudo-labelled/self-trained/", str(p))


class TestConfigIntegration(unittest.TestCase):
    """The output path matches what LGCoTrainConfig builds for self-trained.

    This is the critical contract: if these don't match, the LG-CoTrain
    pipeline won't find the teacher pseudo-labels.
    """

    def test_path_matches_config_for_one_cell(self):
        """The teacher script writes to exactly the path config.py looks up."""
        # Where the teacher script will write:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            generated = _output_path(tmp_path, "canada_wildfires_2016", 5, 1)

            # Where config.py will look for it (with self-trained source):
            cfg = LGCoTrainConfig(
                event="canada_wildfires_2016",
                budget=5,
                seed_set=1,
                pseudo_label_source="self-trained",
                data_root=str(tmp_path),
            )

            self.assertEqual(str(generated), cfg.pseudo_label_path)

    def test_path_matches_config_for_all_combinations(self):
        """Every combination in the grid must produce matching paths."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for event in ["canada_wildfires_2016", "hurricane_irma_2017", "kerala_floods_2018"]:
                for budget in [5, 10, 25, 50]:
                    for seed in [1, 2, 3]:
                        generated = _output_path(tmp_path, event, budget, seed)
                        cfg = LGCoTrainConfig(
                            event=event,
                            budget=budget,
                            seed_set=seed,
                            pseudo_label_source="self-trained",
                            data_root=str(tmp_path),
                        )
                        self.assertEqual(
                            str(generated), cfg.pseudo_label_path,
                            f"Mismatch for {event}/b={budget}/s={seed}",
                        )

    def test_self_trained_is_in_per_split_sources(self):
        self.assertIn("self-trained", PER_SPLIT_SOURCES)


class TestGenerateAllSkipsExisting(unittest.TestCase):
    """generate_all() should skip cells whose output CSV already exists,
    unless --force is passed."""

    def _setup_with_one_existing(self, tmp_path: Path):
        """Create a single existing teacher CSV in the data dir."""
        existing = _output_path(tmp_path, "canada_wildfires_2016", 5, 1)
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text("tweet_id,tweet_text,predicted_label,confidence\n")
        return existing

    def test_skips_existing_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            existing = self._setup_with_one_existing(tmp_path)

            # Mock _worker so we never actually train anything
            with patch(
                "lg_cotrain.generate_selftrained_teacher._worker"
            ) as mock_worker:
                mock_worker.return_value = {"status": "ok", "cell": "x", "mean_confidence": 0.5, "n_unlabeled": 100}
                results = generate_all(
                    events=["canada_wildfires_2016"],
                    budgets=[5],
                    seed_sets=[1, 2, 3],
                    data_root=tmp_path,
                    num_gpus=1,
                    force=False,
                )
                # Worker should be called twice (for set 2 and set 3, NOT set 1)
                self.assertEqual(mock_worker.call_count, 2)

            # Existing file should be unchanged
            self.assertTrue(existing.exists())

    def test_force_regenerates_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._setup_with_one_existing(tmp_path)

            with patch(
                "lg_cotrain.generate_selftrained_teacher._worker"
            ) as mock_worker:
                mock_worker.return_value = {"status": "ok", "cell": "x", "mean_confidence": 0.5, "n_unlabeled": 100}
                results = generate_all(
                    events=["canada_wildfires_2016"],
                    budgets=[5],
                    seed_sets=[1, 2, 3],
                    data_root=tmp_path,
                    num_gpus=1,
                    force=True,
                )
                # All 3 should run when force=True
                self.assertEqual(mock_worker.call_count, 3)


class TestGenerateAllGrid(unittest.TestCase):
    """generate_all() builds the right grid from the inputs."""

    def test_grid_size_with_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            with patch(
                "lg_cotrain.generate_selftrained_teacher._worker"
            ) as mock_worker:
                mock_worker.return_value = {"status": "ok", "cell": "x", "mean_confidence": 0.5, "n_unlabeled": 100}
                generate_all(
                    events=["canada_wildfires_2016", "hurricane_harvey_2017"],
                    budgets=[5, 50],
                    seed_sets=[1, 2, 3],
                    data_root=tmp_path,
                    num_gpus=1,
                )
                # 2 events x 2 budgets x 3 seeds = 12 cells
                self.assertEqual(mock_worker.call_count, 12)

    def test_full_committed_grid(self):
        """The committed Stage 3 grid: 10 events x 2 budgets x 3 seeds = 60."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            with patch(
                "lg_cotrain.generate_selftrained_teacher._worker"
            ) as mock_worker:
                mock_worker.return_value = {"status": "ok", "cell": "x", "mean_confidence": 0.5, "n_unlabeled": 100}
                generate_all(
                    events=ALL_EVENTS,
                    budgets=[5, 50],
                    seed_sets=[1, 2, 3],
                    data_root=tmp_path,
                    num_gpus=1,
                )
                self.assertEqual(mock_worker.call_count, 60)

    def test_full_grid(self):
        """The full 4-budget grid: 10 events x 4 budgets x 3 seeds = 120."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            with patch(
                "lg_cotrain.generate_selftrained_teacher._worker"
            ) as mock_worker:
                mock_worker.return_value = {"status": "ok", "cell": "x", "mean_confidence": 0.5, "n_unlabeled": 100}
                generate_all(
                    events=ALL_EVENTS,
                    budgets=DEFAULT_BUDGETS,
                    seed_sets=DEFAULT_SEEDS,
                    data_root=tmp_path,
                    num_gpus=1,
                )
                self.assertEqual(mock_worker.call_count, 120)


class TestCLI(unittest.TestCase):
    """The CLI parses arguments and forwards them to generate_all()."""

    def test_help_flag(self):
        with self.assertRaises(SystemExit) as ctx:
            with patch("sys.argv", ["prog", "--help"]):
                main()
        self.assertEqual(ctx.exception.code, 0)

    def test_default_events_is_all(self):
        with patch(
            "lg_cotrain.generate_selftrained_teacher.generate_all"
        ) as mock_run:
            mock_run.return_value = []
            with patch("sys.argv", ["prog", "--budgets", "5", "--seed-sets", "1"]):
                main()
            kwargs = mock_run.call_args.kwargs
            self.assertEqual(kwargs["events"], ALL_EVENTS)

    def test_explicit_events_subset(self):
        with patch(
            "lg_cotrain.generate_selftrained_teacher.generate_all"
        ) as mock_run:
            mock_run.return_value = []
            with patch("sys.argv", [
                "prog", "--events", "canada_wildfires_2016", "kaikoura_earthquake_2016",
                "--budgets", "5",
                "--seed-sets", "1",
            ]):
                main()
            kwargs = mock_run.call_args.kwargs
            self.assertEqual(
                kwargs["events"],
                ["canada_wildfires_2016", "kaikoura_earthquake_2016"],
            )

    def test_events_all_keyword_expands(self):
        with patch(
            "lg_cotrain.generate_selftrained_teacher.generate_all"
        ) as mock_run:
            mock_run.return_value = []
            with patch("sys.argv", ["prog", "--events", "all", "--budgets", "5", "--seed-sets", "1"]):
                main()
            kwargs = mock_run.call_args.kwargs
            self.assertEqual(kwargs["events"], ALL_EVENTS)

    def test_default_budgets(self):
        with patch(
            "lg_cotrain.generate_selftrained_teacher.generate_all"
        ) as mock_run:
            mock_run.return_value = []
            with patch("sys.argv", ["prog", "--seed-sets", "1"]):
                main()
            kwargs = mock_run.call_args.kwargs
            self.assertEqual(kwargs["budgets"], DEFAULT_BUDGETS)

    def test_default_seed_sets(self):
        with patch(
            "lg_cotrain.generate_selftrained_teacher.generate_all"
        ) as mock_run:
            mock_run.return_value = []
            with patch("sys.argv", ["prog", "--budgets", "5"]):
                main()
            kwargs = mock_run.call_args.kwargs
            self.assertEqual(kwargs["seed_sets"], DEFAULT_SEEDS)

    def test_force_flag_forwarded(self):
        with patch(
            "lg_cotrain.generate_selftrained_teacher.generate_all"
        ) as mock_run:
            mock_run.return_value = []
            with patch("sys.argv", ["prog", "--budgets", "5", "--force"]):
                main()
            kwargs = mock_run.call_args.kwargs
            self.assertTrue(kwargs["force"])

    def test_force_flag_default_false(self):
        with patch(
            "lg_cotrain.generate_selftrained_teacher.generate_all"
        ) as mock_run:
            mock_run.return_value = []
            with patch("sys.argv", ["prog", "--budgets", "5"]):
                main()
            kwargs = mock_run.call_args.kwargs
            self.assertFalse(kwargs["force"])

    def test_model_name_default(self):
        with patch(
            "lg_cotrain.generate_selftrained_teacher.generate_all"
        ) as mock_run:
            mock_run.return_value = []
            with patch("sys.argv", ["prog", "--budgets", "5"]):
                main()
            kwargs = mock_run.call_args.kwargs
            self.assertEqual(kwargs["model_name"], "vinai/bertweet-base")

    def test_custom_model_name(self):
        with patch(
            "lg_cotrain.generate_selftrained_teacher.generate_all"
        ) as mock_run:
            mock_run.return_value = []
            with patch("sys.argv", ["prog", "--budgets", "5", "--model-name", "bert-base-uncased"]):
                main()
            kwargs = mock_run.call_args.kwargs
            self.assertEqual(kwargs["model_name"], "bert-base-uncased")

    def test_num_gpus_forwarded(self):
        with patch(
            "lg_cotrain.generate_selftrained_teacher.generate_all"
        ) as mock_run:
            mock_run.return_value = []
            with patch("sys.argv", ["prog", "--budgets", "5", "--num-gpus", "4"]):
                main()
            kwargs = mock_run.call_args.kwargs
            self.assertEqual(kwargs["num_gpus"], 4)


class TestTrainAndPredictDatasetContract(unittest.TestCase):
    """Regression tests for train_and_predict_one_cell.

    A previous bug passed a pandas DataFrame as the first arg of TweetDataset
    (which expects a list of strings). Type-checked fine but blew up at
    runtime with KeyError on every cell because pandas interpreted DataLoader
    indices as column labels. We catch that here via static AST inspection
    of the function body - no torch / transformers / pandas needed.

    We also assert that gold class_labels (which the unlabeled TSVs carry,
    since the unlabeled split is held-out from the original labeled dataset)
    are never forwarded to model(...) at inference.
    """

    def _function_source(self):
        import inspect
        from lg_cotrain.generate_selftrained_teacher import train_and_predict_one_cell
        return inspect.getsource(train_and_predict_one_cell)

    def _function_ast(self):
        import ast
        return ast.parse(self._function_source())

    def test_tweet_dataset_first_arg_is_not_a_dataframe(self):
        """TweetDataset(...) must not be called with a `df_*` variable as
        the first positional arg. The constructor signature is
        (texts, labels, tokenizer, max_length) - passing a DataFrame here
        causes KeyError at every dataloader fetch."""
        import ast
        tree = self._function_ast()
        tweet_dataset_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "TweetDataset"
        ]
        self.assertGreaterEqual(
            len(tweet_dataset_calls), 2,
            "Expected at least 2 TweetDataset calls (train + unlabeled)",
        )
        for call in tweet_dataset_calls:
            self.assertGreaterEqual(len(call.args), 1, "TweetDataset called with no args")
            first = call.args[0]
            if isinstance(first, ast.Name):
                self.assertFalse(
                    first.id.startswith("df_"),
                    f"TweetDataset called with DataFrame `{first.id}` as first arg "
                    f"- must pass a list of texts (use df.tolist()).",
                )

    def test_inference_loop_does_not_forward_labels_to_model(self):
        """The inference forward pass must call model(input_ids=..., attention_mask=...)
        only - never with labels=. Gold class_labels live in the dataset for
        well-typed batch construction, but they must not leak into the model."""
        import ast
        tree = self._function_ast()
        # Find the `with torch.no_grad():` block (the inference loop) and
        # assert any model(...) call inside it has no `labels=` kwarg.
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                for item in node.items:
                    expr = item.context_expr
                    if (
                        isinstance(expr, ast.Call)
                        and isinstance(expr.func, ast.Attribute)
                        and expr.func.attr == "no_grad"
                    ):
                        # Walk the body for model(...) calls
                        for inner in ast.walk(node):
                            if (
                                isinstance(inner, ast.Call)
                                and isinstance(inner.func, ast.Name)
                                and inner.func.id == "model"
                            ):
                                kwarg_names = {kw.arg for kw in inner.keywords}
                                self.assertNotIn(
                                    "labels", kwarg_names,
                                    "Gold class_labels must NOT be forwarded to model() "
                                    "at inference - that would leak labels into the "
                                    "teacher's predictions.",
                                )

    def test_unknown_unlabeled_classes_have_fallback(self):
        """The unlabeled set may contain classes not present in the small labeled
        split (event_classes is detected from df_labeled only). The script must
        encode those with a fallback rather than raising KeyError."""
        src = self._function_source()
        # The fix uses a conditional `if c in label2id else 0`. Don't pin the
        # exact spelling - just require there's a guard against unknown classes
        # in the unlabeled encoding line.
        self.assertIn("if c in label2id", src,
                      "Missing fallback for unlabeled-set classes that aren't in label2id")


if __name__ == "__main__":
    unittest.main(verbosity=2)
