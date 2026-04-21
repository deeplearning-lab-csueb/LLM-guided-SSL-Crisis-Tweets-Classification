"""CLI entry point for running LG-CoTrain experiments.

Supports single experiments or batch mode with multiple events/budgets/seeds.
"""

import argparse
from pathlib import Path

from .config import LGCoTrainConfig
from .run_all import BUDGETS, SEED_SETS, format_summary_table, run_all_experiments

# Default roots resolve to sibling directories of the lg_cotrain package, so
# they work correctly on any OS regardless of where the repo is cloned.
_DEFAULT_DATA_ROOT = str(Path(__file__).parent.parent / "data")
_DEFAULT_RESULTS_ROOT = str(Path(__file__).parent.parent / "results")


def main():
    parser = argparse.ArgumentParser(description="LG-CoTrain experiment runner")

    # Event(s) — supports both --event (single) and --events (multiple)
    event_group = parser.add_mutually_exclusive_group(required=True)
    event_group.add_argument(
        "--event", type=str, dest="events", nargs=1,
        help="Single disaster event name",
    )
    event_group.add_argument(
        "--events", type=str, nargs="+",
        help="One or more disaster event names",
    )

    # Budget(s)
    budget_group = parser.add_mutually_exclusive_group()
    budget_group.add_argument(
        "--budget", type=int, dest="budgets", nargs=1,
        help="Single budget value",
    )
    budget_group.add_argument(
        "--budgets", type=int, nargs="+", default=None,
        help="One or more budget values (default: all [5, 10, 25, 50])",
    )

    # Seed set(s)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed-set", type=int, dest="seed_sets", nargs=1,
        help="Single seed set",
    )
    seed_group.add_argument(
        "--seed-sets", type=int, nargs="+", default=None,
        help="One or more seed sets (default: all [1, 2, 3])",
    )

    # Pseudo-label source and output
    parser.add_argument(
        "--pseudo-label-source", type=str, default="gpt-4o",
        help="Pseudo-label directory name under data/pseudo-labelled/ (default: gpt-4o)",
    )
    parser.add_argument(
        "--output-folder", type=str, default=None,
        help="Output folder for results (overrides --results-root). "
             "E.g. results/gpt-4o-1st-run",
    )

    # Model hyperparameters
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--weight-gen-epochs", type=int, default=7)
    parser.add_argument("--cotrain-epochs", type=int, default=10)
    parser.add_argument("--finetune-max-epochs", type=int, default=100)
    parser.add_argument("--finetune-patience", type=int, default=5)
    parser.add_argument(
        "--stopping-strategy", type=str, default="baseline",
        choices=[
            "baseline", "no_early_stopping", "per_class_patience",
            "weighted_macro_f1", "balanced_dev", "scaled_threshold",
        ],
        help="Phase 3 early stopping strategy (default: baseline)",
    )
    parser.add_argument(
        "--phase1-seed-strategy", type=str, default="last",
        choices=["last", "best"],
        help="Phase 1→2 seeding strategy: 'last' (default, per Algorithm 1) or 'best' (best ensemble dev F1)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=128)

    # Paths
    parser.add_argument("--data-root", type=str, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=str, default=_DEFAULT_RESULTS_ROOT)

    # Parallel execution
    parser.add_argument(
        "--num-gpus", type=int, default=1,
        help="Number of GPUs for parallel execution (default: 1 = sequential)",
    )

    args = parser.parse_args()

    events = args.events
    budgets = args.budgets
    seed_sets = args.seed_sets
    results_root = args.output_folder or args.results_root

    # Common hyperparameter kwargs
    hyperparams = dict(
        pseudo_label_source=args.pseudo_label_source,
        model_name=args.model_name,
        weight_gen_epochs=args.weight_gen_epochs,
        cotrain_epochs=args.cotrain_epochs,
        finetune_max_epochs=args.finetune_max_epochs,
        finetune_patience=args.finetune_patience,
        stopping_strategy=args.stopping_strategy,
        phase1_seed_strategy=args.phase1_seed_strategy,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_seq_length=args.max_seq_length,
    )

    # Run experiments for each event via run_all_experiments
    # (handles both single and multi budget/seed cases uniformly)
    for event in events:
        all_results = run_all_experiments(
            event,
            budgets=budgets,
            seed_sets=seed_sets,
            num_gpus=args.num_gpus,
            data_root=args.data_root,
            results_root=results_root,
            **hyperparams,
        )

        # Single experiment: show inline results
        if (
            budgets is not None and len(budgets) == 1
            and seed_sets is not None and len(seed_sets) == 1
        ):
            result = all_results[0]
            if result is not None:
                print(f"\nFinal Results:")
                print(f"  Test Error Rate: {result['test_error_rate']:.2f}%")
                print(f"  Test Macro-F1:   {result['test_macro_f1']:.4f}")
        else:
            print()
            print(format_summary_table(
                all_results, event, budgets=budgets, seed_sets=seed_sets,
            ))


if __name__ == "__main__":
    main()
