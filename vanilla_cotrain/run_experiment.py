"""CLI entry point for running Vanilla Co-Training experiments.

Usage::

    # Single cell
    python -m vanilla_cotrain.run_experiment \\
        --event canada_wildfires_2016 --budget 5 --seed-set 1

    # Full grid (10 events x 4 budgets x 3 seeds)
    python -m vanilla_cotrain.run_experiment \\
        --events california_wildfires_2018 canada_wildfires_2016 ... \\
        --num-gpus 2 --output-folder results/bertweet/ablation/vanilla-cotrain/baseline

    # Specific budgets
    python -m vanilla_cotrain.run_experiment \\
        --events canada_wildfires_2016 --budgets 5 50 --seed-sets 1 2 3
"""

import argparse
import json
import logging
import time
from pathlib import Path

from .config import VanillaCoTrainConfig

_DEFAULT_DATA_ROOT = str(Path(__file__).parent.parent / "data")
_DEFAULT_RESULTS_ROOT = str(Path(__file__).parent.parent / "results")

BUDGETS = [5, 10, 25, 50]
SEED_SETS = [1, 2, 3]

logger = logging.getLogger("vanilla_cotrain")


def _run_one(config: VanillaCoTrainConfig):
    """Run a single vanilla co-training experiment. Returns result dict or None."""
    from .trainer import VanillaCoTrainer

    metrics_path = Path(config.output_dir) / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            result = json.load(f)
        logger.info(
            f"Skipping {config.event}/{config.budget}_set{config.seed_set} "
            f"(metrics.json exists, test_macro_f1={result.get('test_macro_f1', '?')})"
        )
        return result

    try:
        trainer = VanillaCoTrainer(config)
        return trainer.run()
    except Exception as e:
        logger.error(
            f"FAILED {config.event}/{config.budget}_set{config.seed_set}: {e}",
            exc_info=True,
        )
        return None


def _run_one_on_device(args_tuple):
    """Wrapper for parallel execution: unpack (config_kwargs, device) and run."""
    config_kwargs, device_str = args_tuple
    config_kwargs["device"] = device_str
    config = VanillaCoTrainConfig(**config_kwargs)
    return _run_one(config)


def run_all_experiments(
    event,
    *,
    budgets=None,
    seed_sets=None,
    num_gpus=1,
    model_name="vinai/bertweet-base",
    num_iterations=30,
    samples_per_class=1,
    train_epochs=5,
    finetune_max_epochs=100,
    finetune_patience=5,
    batch_size=32,
    lr=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_seq_length=128,
    data_root=None,
    results_root=None,
    _on_experiment_done=None,
):
    """Run all budget x seed_set combinations for one event.

    Returns a list of result dicts (or None for failed experiments).
    Cells whose metrics.json already exists are loaded and skipped.
    """
    budgets = budgets or BUDGETS
    seed_sets = seed_sets or SEED_SETS
    data_root = data_root or _DEFAULT_DATA_ROOT
    results_root = results_root or _DEFAULT_RESULTS_ROOT

    common_kwargs = dict(
        event=event,
        model_name=model_name,
        num_iterations=num_iterations,
        samples_per_class=samples_per_class,
        train_epochs=train_epochs,
        finetune_max_epochs=finetune_max_epochs,
        finetune_patience=finetune_patience,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        max_seq_length=max_seq_length,
        data_root=data_root,
        results_root=results_root,
    )

    # Build list of (config_kwargs, budget, seed) for all cells
    cells = []
    for budget in budgets:
        for seed_set in seed_sets:
            cells.append((budget, seed_set))

    results = []

    if num_gpus <= 1:
        # Sequential execution
        for budget, seed_set in cells:
            config = VanillaCoTrainConfig(
                budget=budget, seed_set=seed_set, **common_kwargs
            )
            result = _run_one(config)
            results.append(result)
            if _on_experiment_done:
                status = "done" if result else "failed"
                _on_experiment_done(event, budget, seed_set, status)
    else:
        # Parallel execution across GPUs
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        from concurrent.futures import ProcessPoolExecutor

        tasks = []
        for i, (budget, seed_set) in enumerate(cells):
            device_str = f"cuda:{i % num_gpus}"
            kw = dict(budget=budget, seed_set=seed_set, **common_kwargs)
            tasks.append((kw, device_str))

        with ProcessPoolExecutor(
            max_workers=num_gpus, mp_context=ctx
        ) as executor:
            for result in executor.map(_run_one_on_device, tasks):
                results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Vanilla Co-Training experiment runner"
    )

    event_group = parser.add_mutually_exclusive_group(required=True)
    event_group.add_argument(
        "--event", type=str, dest="events", nargs=1,
        help="Single disaster event name",
    )
    event_group.add_argument(
        "--events", type=str, nargs="+",
        help="One or more disaster event names",
    )

    budget_group = parser.add_mutually_exclusive_group()
    budget_group.add_argument(
        "--budget", type=int, dest="budgets", nargs=1,
    )
    budget_group.add_argument(
        "--budgets", type=int, nargs="+", default=None,
    )

    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed-set", type=int, dest="seed_sets", nargs=1,
    )
    seed_group.add_argument(
        "--seed-sets", type=int, nargs="+", default=None,
    )

    parser.add_argument(
        "--output-folder", type=str, default=None,
        help="Output folder for results",
    )

    # Model and co-training hyperparameters
    parser.add_argument("--model-name", type=str, default="vinai/bertweet-base")
    parser.add_argument("--num-iterations", type=int, default=30)
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--finetune-max-epochs", type=int, default=100)
    parser.add_argument("--finetune-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=128)

    # Paths
    parser.add_argument("--data-root", type=str, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=str, default=_DEFAULT_RESULTS_ROOT)

    # Parallel execution
    parser.add_argument("--num-gpus", type=int, default=1)

    args = parser.parse_args()

    results_root = args.output_folder or args.results_root

    hyperparams = dict(
        model_name=args.model_name,
        num_iterations=args.num_iterations,
        samples_per_class=args.samples_per_class,
        train_epochs=args.train_epochs,
        finetune_max_epochs=args.finetune_max_epochs,
        finetune_patience=args.finetune_patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_seq_length=args.max_seq_length,
    )

    for event in args.events:
        print(f"\n{'='*70}")
        print(f"Event: {event}")
        print(f"{'='*70}")

        all_results = run_all_experiments(
            event,
            budgets=args.budgets,
            seed_sets=args.seed_sets,
            num_gpus=args.num_gpus,
            data_root=args.data_root,
            results_root=results_root,
            **hyperparams,
        )

        completed = sum(1 for r in all_results if r is not None)
        print(f"\n{event}: {completed}/{len(all_results)} cells completed")


# Allow `python -m vanilla_cotrain.run_experiment`
if __name__ == "__main__":
    main()
