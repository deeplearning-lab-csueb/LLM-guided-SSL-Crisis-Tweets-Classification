"""Batch runner: execute all 12 (budget x seed_set) experiments for one event."""

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

from .config import LGCoTrainConfig

BUDGETS = [5, 10, 25, 50]
SEED_SETS = [1, 2, 3]

# Default roots resolve to sibling directories of the lg_cotrain package, so
# they work correctly on any OS regardless of where the repo is cloned.
_DEFAULT_DATA_ROOT = str(Path(__file__).parent.parent / "data")
_DEFAULT_RESULTS_ROOT = str(Path(__file__).parent.parent / "results")

logger = logging.getLogger("lg_cotrain")


def run_all_experiments(
    event,
    *,
    budgets=None,
    seed_sets=None,
    num_gpus=1,
    pseudo_label_source="gpt-4o",
    model_name="bert-base-uncased",
    weight_gen_epochs=7,
    cotrain_epochs=10,
    finetune_max_epochs=100,
    finetune_patience=5,
    stopping_strategy="baseline",
    phase1_seed_strategy="last",
    batch_size=32,
    lr=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_seq_length=128,
    data_root=None,
    results_root=None,
    _trainer_cls=None,
    _on_experiment_done=None,
):
    """Run all budget x seed_set combinations for *event*.

    Returns a list of result dicts (or ``None`` for failed experiments).
    Experiments whose ``metrics.json`` already exists are loaded and skipped.

    Args:
        budgets: List of budgets to run. Defaults to BUDGETS ([5, 10, 25, 50]).
        seed_sets: List of seed sets to run. Defaults to SEED_SETS ([1, 2, 3]).
        pseudo_label_source: Pseudo-label directory name (default "gpt-4o").
        num_gpus: Number of GPUs for parallel execution (default 1 = sequential).

    If *_on_experiment_done* is provided, it is called after each experiment
    with ``(event, budget, seed_set, status)`` where *status* is one of
    ``"done"``, ``"skipped"``, or ``"failed"``.
    """
    budgets = budgets if budgets is not None else BUDGETS
    seed_sets = seed_sets if seed_sets is not None else SEED_SETS
    data_root = data_root if data_root is not None else _DEFAULT_DATA_ROOT
    results_root = results_root if results_root is not None else _DEFAULT_RESULTS_ROOT

    # Common config kwargs for building experiment configs
    _common_kwargs = dict(
        pseudo_label_source=pseudo_label_source,
        model_name=model_name,
        weight_gen_epochs=weight_gen_epochs,
        cotrain_epochs=cotrain_epochs,
        finetune_max_epochs=finetune_max_epochs,
        finetune_patience=finetune_patience,
        stopping_strategy=stopping_strategy,
        phase1_seed_strategy=phase1_seed_strategy,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        max_seq_length=max_seq_length,
        data_root=data_root,
        results_root=results_root,
    )

    # Parallel path: num_gpus > 1 and no custom trainer class
    if num_gpus > 1 and _trainer_cls is None:
        return _run_all_parallel(
            event,
            budgets=budgets,
            seed_sets=seed_sets,
            num_gpus=num_gpus,
            common_kwargs=_common_kwargs,
            _on_experiment_done=_on_experiment_done,
        )

    # Sequential path (original behavior)
    if _trainer_cls is None:
        from .trainer import LGCoTrainer  # lazy — avoids torch import at module level

        _trainer_cls = LGCoTrainer

    all_results = []
    total = len(budgets) * len(seed_sets)
    completed = skipped = failed = 0
    start_time = time.time()

    for budget in budgets:
        for seed_set in seed_sets:
            idx = completed + skipped + failed + 1
            metrics_path = (
                Path(results_root) / event / f"{budget}_set{seed_set}" / "metrics.json"
            )

            # Resume: reuse existing results
            if metrics_path.exists():
                with open(metrics_path) as f:
                    result = json.load(f)
                all_results.append(result)
                skipped += 1
                print(
                    f"[{idx}/{total}] budget={budget}, seed={seed_set}"
                    f" -- SKIPPED (exists)"
                )
                if _on_experiment_done is not None:
                    _on_experiment_done(event, budget, seed_set, "skipped")
                continue

            print(f"[{idx}/{total}] budget={budget}, seed={seed_set} -- starting...")
            config = LGCoTrainConfig(
                event=event,
                budget=budget,
                seed_set=seed_set,
                **_common_kwargs,
            )

            try:
                trainer = _trainer_cls(config)
                result = trainer.run()
                all_results.append(result)
                completed += 1
                print(
                    f"[{idx}/{total}] budget={budget}, seed={seed_set}"
                    f" -- done (macro_f1={result['test_macro_f1']:.4f})"
                )
                if _on_experiment_done is not None:
                    _on_experiment_done(event, budget, seed_set, "done")
            except Exception as e:
                logger.error(
                    f"Experiment budget={budget}, seed={seed_set} failed: {e}"
                )
                all_results.append(None)
                failed += 1
                if _on_experiment_done is not None:
                    _on_experiment_done(event, budget, seed_set, "failed")

            # Free GPU memory between experiments
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    elapsed = time.time() - start_time
    print(
        f"\nBatch complete: {completed} ran, {skipped} skipped, {failed} failed"
        f" ({elapsed:.1f}s total)"
    )
    return all_results


def _run_all_parallel(
    event,
    *,
    budgets,
    seed_sets,
    num_gpus,
    common_kwargs,
    _on_experiment_done=None,
):
    """Parallel dispatch path for run_all_experiments."""
    from .parallel import run_experiments_parallel

    results_root = common_kwargs["results_root"]
    all_results = []
    total = len(budgets) * len(seed_sets)
    completed = skipped = failed = 0
    start_time = time.time()

    # Separate skipped (resume) from pending experiments
    pre_results = {}  # (budget, seed_set) -> result
    experiment_configs = []
    pending_keys = []  # track (budget, seed_set) order for pending

    for budget in budgets:
        for seed_set in seed_sets:
            metrics_path = (
                Path(results_root) / event / f"{budget}_set{seed_set}" / "metrics.json"
            )
            if metrics_path.exists():
                with open(metrics_path) as f:
                    result = json.load(f)
                pre_results[(budget, seed_set)] = result
                skipped += 1
                print(
                    f"budget={budget}, seed={seed_set} -- SKIPPED (exists)"
                )
                if _on_experiment_done is not None:
                    _on_experiment_done(event, budget, seed_set, "skipped")
            else:
                experiment_configs.append(dict(
                    event=event,
                    budget=budget,
                    seed_set=seed_set,
                    **common_kwargs,
                ))
                pending_keys.append((budget, seed_set))

    # Run pending experiments in parallel
    if experiment_configs:
        print(
            f"Running {len(experiment_configs)} experiments in parallel "
            f"across {num_gpus} GPUs..."
        )
        parallel_results = run_experiments_parallel(
            experiment_configs,
            num_gpus=num_gpus,
            on_experiment_done=_on_experiment_done,
        )
    else:
        parallel_results = []

    # Build lookup from parallel results
    parallel_lookup = {}
    for (budget, seed_set), outcome in zip(pending_keys, parallel_results):
        parallel_lookup[(budget, seed_set)] = outcome
        if outcome["status"] == "done":
            completed += 1
        else:
            failed += 1

    # Merge in original order
    for budget in budgets:
        for seed_set in seed_sets:
            if (budget, seed_set) in pre_results:
                all_results.append(pre_results[(budget, seed_set)])
            else:
                outcome = parallel_lookup[(budget, seed_set)]
                all_results.append(outcome["result"])

    elapsed = time.time() - start_time
    print(
        f"\nBatch complete: {completed} ran, {skipped} skipped, {failed} failed"
        f" ({elapsed:.1f}s total)"
    )
    return all_results


def format_summary_table(all_results, event, budgets=None, seed_sets=None):
    """Return a formatted summary table grouped by budget."""
    budgets = budgets if budgets is not None else BUDGETS
    seed_sets = seed_sets if seed_sets is not None else SEED_SETS

    # Build lookup: (budget, seed_set) -> result dict
    lookup = {}
    for r in all_results:
        if r is not None:
            lookup[(r["budget"], r["seed_set"])] = r

    lines = []
    lines.append(f"=== Results for {event} ===")
    lines.append("")

    # Header
    seed_hdrs = "".join(f"  Seed {s:<13}" for s in seed_sets)
    lines.append(f"{'Budget':>6}  {seed_hdrs}  {'Mean':>8}  {'Std':>8}")

    sub_cells = "".join(f"  {'ErrR%':>6} {'MacF1':>6}" for _ in seed_sets)
    lines.append(f"{'':>6}  {sub_cells}  {'ErrR%':>8}  {'MacF1':>8}")
    lines.append("-" * len(lines[-1]))

    for budget in budgets:
        err_rates = []
        macro_f1s = []
        cells = ""
        for seed_set in seed_sets:
            r = lookup.get((budget, seed_set))
            if r is not None:
                cells += f"  {r['test_error_rate']:>6.2f} {r['test_macro_f1']:>6.4f}"
                err_rates.append(r["test_error_rate"])
                macro_f1s.append(r["test_macro_f1"])
            else:
                cells += f"  {'N/A':>6} {'N/A':>6}"

        # Mean ± std
        if len(err_rates) >= 2:
            e_mean = statistics.mean(err_rates)
            e_std = statistics.stdev(err_rates)
            f_mean = statistics.mean(macro_f1s)
            f_std = statistics.stdev(macro_f1s)
            agg = f"  {e_mean:>5.2f}+/-{e_std:<5.2f}  {f_mean:.4f}+/-{f_std:.4f}"
        elif len(err_rates) == 1:
            agg = f"  {err_rates[0]:>8.2f}  {macro_f1s[0]:>8.4f}"
        else:
            agg = f"  {'N/A':>8}  {'N/A':>8}"

        lines.append(f"{budget:>6}  {cells}{agg}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run all (budget x seed_set) experiments for one event"
    )
    parser.add_argument(
        "--event", type=str, required=True,
        help="Disaster event name, e.g. canada_wildfires_2016",
    )
    parser.add_argument(
        "--budgets", type=int, nargs="*", default=None,
        help="Budgets to run (default: all [5, 10, 25, 50])",
    )
    parser.add_argument(
        "--seed-sets", type=int, nargs="*", default=None,
        help="Seed sets to run (default: all [1, 2, 3])",
    )
    parser.add_argument(
        "--pseudo-label-source", type=str, default="gpt-4o",
        help="Pseudo-label directory name (default: gpt-4o)",
    )
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--data-root", type=str, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=str, default=_DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--num-gpus", type=int, default=1,
        help="Number of GPUs for parallel execution (default: 1 = sequential)",
    )

    args = parser.parse_args()

    all_results = run_all_experiments(
        args.event,
        budgets=args.budgets,
        seed_sets=args.seed_sets,
        num_gpus=args.num_gpus,
        pseudo_label_source=args.pseudo_label_source,
        model_name=args.model_name,
        weight_gen_epochs=args.weight_gen_epochs,
        cotrain_epochs=args.cotrain_epochs,
        finetune_max_epochs=args.finetune_max_epochs,
        finetune_patience=args.finetune_patience,
        stopping_strategy=args.stopping_strategy,
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_length=args.max_seq_length,
        data_root=args.data_root,
        results_root=args.results_root,
    )

    print()
    print(format_summary_table(all_results, args.event,
                               budgets=args.budgets, seed_sets=args.seed_sets))


if __name__ == "__main__":
    main()
