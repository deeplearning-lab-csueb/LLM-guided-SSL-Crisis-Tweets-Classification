"""Configuration dataclass for LG-CoTrain experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Pseudo-label sources that produce a *different* pseudo-label file per
# (budget, seed_set) combination, rather than one shared file per event.
# This is the case when the pseudo-labels come from a model trained on the
# small labeled split (e.g. the self-trained teacher used in Ablation V).
# For these sources, the pseudo-label file is named
# ``labeled_{budget}_set{seed_set}_pseudo.csv`` instead of the standard
# ``{event}_train_pred.csv``. The directory layout is otherwise unchanged.
PER_SPLIT_SOURCES = frozenset({"self-trained", "self-trained-top-p", "self-trained-optuna", "self-trained-top-p-optuna"})


@dataclass
class LGCoTrainConfig:
    # Experiment identifiers
    event: str = "canada_wildfires_2016"
    budget: int = 5
    seed_set: int = 1

    # Pseudo-label source directory name (under data/pseudo-labelled/)
    pseudo_label_source: str = "gpt-4o"

    # Model
    model_name: str = "bert-base-uncased"
    num_labels: int = 10
    max_seq_length: int = 128

    # Phase 1: Weight generation
    weight_gen_epochs: int = 7

    # Phase 2: Co-training
    cotrain_epochs: int = 10

    # Phase 3: Fine-tuning
    finetune_max_epochs: int = 100
    finetune_patience: int = 5
    # Phase 3 early stopping strategy: "baseline" | "no_early_stopping" | "per_class_patience"
    #                                  | "weighted_macro_f1" | "balanced_dev" | "scaled_threshold"
    stopping_strategy: str = "baseline"
    # Phase 1 → Phase 2 seeding: "last" (default, per Algorithm 1) | "best" (best ensemble dev F1)
    phase1_seed_strategy: str = "last"

    # Optimization
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # Save per-sample lambda weights after Phase 2 (for analysis only, no effect on training)
    save_lambda_details: bool = False

    # Device override: "cuda:0", "cuda:1", "cpu", or None (auto-detect)
    device: Optional[str] = None

    # Paths (base) — default to sibling directories of this package, which
    # resolves correctly on any OS regardless of where the repo is cloned.
    data_root: str = field(
        default_factory=lambda: str(Path(__file__).parent.parent / "data")
    )
    results_root: str = field(
        default_factory=lambda: str(Path(__file__).parent.parent / "results")
    )

    # Auto-computed paths (set in __post_init__)
    labeled_path: str = field(init=False, default="")
    unlabeled_path: str = field(init=False, default="")
    pseudo_label_path: str = field(init=False, default="")
    dev_path: str = field(init=False, default="")
    test_path: str = field(init=False, default="")
    output_dir: str = field(init=False, default="")

    def __post_init__(self):
        original_dir = Path(self.data_root) / "original" / self.event
        pseudo_dir = Path(self.data_root) / "pseudo-labelled" / self.pseudo_label_source / self.event

        self.labeled_path = str(
            original_dir / f"labeled_{self.budget}_set{self.seed_set}.tsv"
        )
        self.unlabeled_path = str(
            original_dir / f"unlabeled_{self.budget}_set{self.seed_set}.tsv"
        )
        # For sources where the pseudo-labels are produced by a model trained
        # on the small labeled split (e.g. the self-trained teacher used in
        # Ablation V), the pseudo-label file is per-(budget, seed_set), not
        # shared across the event. For all other sources (gpt-4o, llama-3,
        # etc.) the file is the standard one per event.
        if self.pseudo_label_source in PER_SPLIT_SOURCES:
            self.pseudo_label_path = str(
                pseudo_dir / f"labeled_{self.budget}_set{self.seed_set}_pseudo.csv"
            )
        else:
            self.pseudo_label_path = str(
                pseudo_dir / f"{self.event}_train_pred.csv"
            )
        self.dev_path = str(original_dir / f"{self.event}_dev.tsv")
        self.test_path = str(original_dir / f"{self.event}_test.tsv")
        self.output_dir = str(
            Path(self.results_root) / self.event / f"{self.budget}_set{self.seed_set}"
        )
