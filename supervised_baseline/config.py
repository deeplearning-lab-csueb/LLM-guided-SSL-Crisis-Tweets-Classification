"""Configuration dataclass for supervised-only baseline experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SupervisedBaselineConfig:
    # Experiment identifiers
    event: str = "canada_wildfires_2016"
    budget: int = 5
    seed_set: int = 1

    # Model
    model_name: str = "vinai/bertweet-base"
    num_labels: int = 10
    max_seq_length: int = 128

    # Training
    max_epochs: int = 100
    patience: int = 5

    # Optimization (match LG-CoTrain defaults)
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # Device override: "cuda:0", "cuda:1", "cpu", or None (auto-detect)
    device: Optional[str] = None

    # Paths (base)
    data_root: str = field(
        default_factory=lambda: str(Path(__file__).parent.parent / "data")
    )
    results_root: str = field(
        default_factory=lambda: str(Path(__file__).parent.parent / "results")
    )

    # Auto-computed paths (set in __post_init__)
    labeled_path: str = field(init=False, default="")
    unlabeled_path: str = field(init=False, default="")
    dev_path: str = field(init=False, default="")
    test_path: str = field(init=False, default="")
    output_dir: str = field(init=False, default="")

    def __post_init__(self):
        original_dir = Path(self.data_root) / "original" / self.event

        self.labeled_path = str(
            original_dir / f"labeled_{self.budget}_set{self.seed_set}.tsv"
        )
        # Unlabeled path needed only for class detection
        self.unlabeled_path = str(
            original_dir / f"unlabeled_{self.budget}_set{self.seed_set}.tsv"
        )
        self.dev_path = str(original_dir / f"{self.event}_dev.tsv")
        self.test_path = str(original_dir / f"{self.event}_test.tsv")
        self.output_dir = str(
            Path(self.results_root) / self.event / f"{self.budget}_set{self.seed_set}"
        )
