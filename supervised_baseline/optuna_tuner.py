"""Per-experiment Optuna hyperparameter tuning for the supervised baseline.

Runs separate Optuna studies -- one per (event, budget, seed_set) -- to find
experiment-specific optimal hyperparameters for supervised-only fine-tuning.
Studies run in parallel across GPUs using ProcessPoolExecutor.

Results are saved as JSON files (no database) under ``trials_{n}/`` subfolders.
Incremental scaling is built-in: running with a higher ``n_trials`` continues
from the latest previous run, replaying earlier trials into the TPE sampler
so only the delta needs to execute.

Usage::

    # Run on 2 priority events
    python -m supervised_baseline.optuna_tuner --n-trials 10 --num-gpus 2 \\
        --events hurricane_maria_2017 hurricane_dorian_2019

    # Extend to all events (already-completed studies are skipped)
    python -m supervised_baseline.optuna_tuner --n-trials 10 --num-gpus 2
"""

import argparse
import json
import logging
import multiprocessing as mp
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Reuse generic helpers from the LG-CoTrain tuner.
from lg_cotrain.optuna_per_experiment import (
    _close_temp_file_handlers,
    _find_latest_trials,
    _setup_study_logging,
)

logger = logging.getLogger("supervised_baseline")

ALL_EVENTS = [
    "california_wildfires_2018",
    "canada_wildfires_2016",
    "cyclone_idai_2019",
    "hurricane_dorian_2019",
    "hurricane_florence_2018",
    "hurricane_harvey_2017",
    "hurricane_irma_2017",
    "hurricane_maria_2017",
    "kaikoura_earthquake_2016",
    "kerala_floods_2018",
]

BUDGETS = [5, 10, 25, 50]
SEED_SETS = [1, 2, 3]

# Canonical search space -- 3 hyperparameters for supervised fine-tuning.
SEARCH_SPACE = {
    "lr": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
    "batch_size": {"type": "categorical", "choices": [8, 16, 32, 64]},
    "max_epochs": {"type": "int", "low": 20, "high": 100},
}


def _build_distributions():
    """Build Optuna distribution objects from SEARCH_SPACE.

    Returns a dict of ``{param_name: optuna.distributions.*Distribution}``.
    Requires optuna to be importable.
    """
    import optuna.distributions as dist

    distributions = {}
    for name, spec in SEARCH_SPACE.items():
        if spec["type"] == "float":
            distributions[name] = dist.FloatDistribution(
                low=spec["low"], high=spec["high"], log=spec.get("log", False),
            )
        elif spec["type"] == "int":
            distributions[name] = dist.IntDistribution(
                low=spec["low"], high=spec["high"],
            )
        elif spec["type"] == "categorical":
            distributions[name] = dist.CategoricalDistribution(
                choices=spec["choices"],
            )
    return distributions


def _replay_trials_into_study(study, trials_info: List[dict]):
    """Replay previously saved trial records into an Optuna study."""
    from optuna.trial import create_trial, TrialState

    distributions = _build_distributions()

    for t_info in trials_info:
        if t_info.get("state", "COMPLETE") != "COMPLETE":
            continue
        frozen = create_trial(
            state=TrialState.COMPLETE,
            params=t_info["params"],
            distributions=distributions,
            values=[t_info["dev_macro_f1"]],
        )
        study.add_trial(frozen)


def create_supervised_objective(
    event: str,
    budget: int,
    seed_set: int,
    device: Optional[str] = None,
    data_root: str = "/workspace/data",
    model_name: Optional[str] = None,
    _trainer_cls=None,
):
    """Return an Optuna objective function for a single supervised-baseline experiment.

    The returned objective:
    1. Samples 3 hyperparameters (lr, batch_size, max_epochs) from the trial.
    2. Trains a single BERTweet on the labeled set via SupervisedTrainer.
    3. Attaches full ``trainer.run()`` result to ``trial.user_attrs["full_metrics"]``
       (metadata only -- never read by the sampler, so no test-set leakage).
    4. Returns ``dev_macro_f1``.
    """

    def objective(trial):
        lr = trial.suggest_float(
            "lr", SEARCH_SPACE["lr"]["low"], SEARCH_SPACE["lr"]["high"],
            log=SEARCH_SPACE["lr"].get("log", False),
        )
        batch_size = trial.suggest_categorical(
            "batch_size", SEARCH_SPACE["batch_size"]["choices"],
        )
        max_epochs = trial.suggest_int(
            "max_epochs",
            SEARCH_SPACE["max_epochs"]["low"],
            SEARCH_SPACE["max_epochs"]["high"],
        )

        with tempfile.TemporaryDirectory() as tmp_results:
            from .config import SupervisedBaselineConfig

            config_kwargs = dict(
                event=event,
                budget=budget,
                seed_set=seed_set,
                lr=lr,
                batch_size=batch_size,
                max_epochs=max_epochs,
                device=device,
                data_root=data_root,
                results_root=tmp_results,
            )
            if model_name is not None:
                config_kwargs["model_name"] = model_name

            config = SupervisedBaselineConfig(**config_kwargs)

            if _trainer_cls is None:
                from .trainer import SupervisedTrainer
                trainer = SupervisedTrainer(config)
            else:
                trainer = _trainer_cls(config)

            try:
                result = trainer.run()
            finally:
                _close_temp_file_handlers(logging.getLogger("supervised_baseline"))

        # Persist full metrics (test metrics, per-class F1, hyperparams, etc.)
        # so we can later materialize a canonical metrics.json without rerun.
        try:
            trial.set_user_attr("full_metrics", result)
        except Exception:
            pass

        return result["dev_macro_f1"]

    return objective


def run_single_study(
    event: str,
    budget: int,
    seed_set: int,
    n_trials: int,
    device: Optional[str] = None,
    storage_dir: str = "/workspace/results/bertweet/supervised/optuna-tuned",
    data_root: str = "/workspace/data",
    model_name: Optional[str] = None,
    on_trial_done: Optional[Callable] = None,
    _trainer_cls=None,
) -> dict:
    """Run one Optuna study for a single (event, budget, seed_set).

    Creates an in-memory Optuna study, runs ``n_trials`` trials, saves
    ``best_params.json`` under ``trials_{n_trials}/``.

    Incremental support: if a previous ``trials_k/best_params.json`` exists
    with ``k < n_trials``, those ``k`` trials are replayed into the new study
    and only ``n_trials - k`` new trials are executed. If ``k >= n_trials``,
    the study is skipped.
    """
    experiment_dir = Path(storage_dir) / event / f"{budget}_set{seed_set}"
    output_dir = experiment_dir / f"trials_{n_trials}"
    best_params_path = output_dir / "best_params.json"

    if best_params_path.exists():
        with open(best_params_path) as f:
            return json.load(f)

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=f"{event}_b{budget}_s{seed_set}",
        direction="maximize",
    )

    previous_trials = []
    previous_n = 0
    latest = _find_latest_trials(experiment_dir)
    if latest is not None:
        previous_n, prev_data = latest
        if previous_n >= n_trials:
            return prev_data
        previous_trials = prev_data.get("trials", [])
        _replay_trials_into_study(study, previous_trials)
        logger.info(
            f"Replayed {len(previous_trials)} trials from trials_{previous_n} "
            f"into study for {event} b={budget} s={seed_set}"
        )

    new_trials_needed = n_trials - len(previous_trials)

    output_dir.mkdir(parents=True, exist_ok=True)
    study_log_path = str(output_dir / "study.log")
    study_fh = _setup_study_logging(study_log_path)
    # _setup_study_logging attaches to "lg_cotrain" logger; also attach to ours
    # so trainer log messages (from SupervisedTrainer) land in the same file.
    sup_logger = logging.getLogger("supervised_baseline")
    sup_logger.addHandler(study_fh)
    sup_logger.setLevel(logging.INFO)
    logger.info(
        f"=== Optuna study: {event} budget={budget} seed={seed_set} | "
        f"target={n_trials} trials, {len(previous_trials)} replayed, "
        f"{new_trials_needed} new ==="
    )

    base_objective = create_supervised_objective(
        event=event,
        budget=budget,
        seed_set=seed_set,
        device=device,
        data_root=data_root,
        model_name=model_name,
        _trainer_cls=_trainer_cls,
    )

    def objective_with_callback(trial):
        logger.info(f"--- Trial {trial.number + 1}/{n_trials} ---")
        dev_f1 = base_objective(trial)
        logger.info(
            f"--- Trial {trial.number + 1}/{n_trials} done: dev_macro_f1={dev_f1:.4f} ---"
        )
        if on_trial_done is not None:
            on_trial_done(trial.number, n_trials, dev_f1)
        return dev_f1

    study.optimize(objective_with_callback, n_trials=new_trials_needed)

    study_fh.close()
    logging.getLogger("lg_cotrain").removeHandler(study_fh)
    sup_logger.removeHandler(study_fh)

    trials_info = []
    for t in study.trials:
        trial_info = {
            "number": t.number,
            "state": t.state.name,
            "params": t.params,
        }
        if t.value is not None:
            trial_info["dev_macro_f1"] = round(t.value, 6)
        if t.datetime_start and t.datetime_complete:
            trial_info["duration_seconds"] = round(
                (t.datetime_complete - t.datetime_start).total_seconds(), 1
            )
        full_metrics = t.user_attrs.get("full_metrics") if hasattr(t, "user_attrs") else None
        if full_metrics is not None:
            trial_info["full_metrics"] = full_metrics
        trials_info.append(trial_info)

    best_full_metrics = None
    try:
        best_full_metrics = study.best_trial.user_attrs.get("full_metrics")
    except Exception:
        best_full_metrics = None

    result = {
        "event": event,
        "budget": budget,
        "seed_set": seed_set,
        "status": "done",
        "best_params": study.best_params,
        "best_value": round(study.best_value, 6),
        "best_full_metrics": best_full_metrics,
        "n_trials": len(study.trials),
        "continued_from": previous_n if previous_n > 0 else None,
        "trials": trials_info,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(best_params_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def _run_study_worker(kwargs: dict) -> dict:
    """Worker function: run one Optuna study in a child process."""
    event = kwargs["event"]
    budget = kwargs["budget"]
    seed_set = kwargs["seed_set"]

    try:
        return run_single_study(
            event=event,
            budget=budget,
            seed_set=seed_set,
            n_trials=kwargs["n_trials"],
            device=kwargs.get("device"),
            storage_dir=kwargs["storage_dir"],
            data_root=kwargs["data_root"],
            model_name=kwargs.get("model_name"),
        )
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {e}"
        error_tb = traceback.format_exc()
        logging.getLogger("supervised_baseline").error(
            f"Optuna study {event} budget={budget} seed={seed_set} failed: {error_msg}"
        )
        return {
            "event": event,
            "budget": budget,
            "seed_set": seed_set,
            "status": "failed",
            "best_params": None,
            "best_value": None,
            "n_trials": 0,
            "trials": [],
            "error": error_msg,
            "traceback": error_tb,
        }
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def run_all_studies(
    events: Optional[List[str]] = None,
    budgets: Optional[List[int]] = None,
    seed_sets: Optional[List[int]] = None,
    n_trials: int = 10,
    num_gpus: int = 1,
    storage_dir: str = "/workspace/results/bertweet/supervised/optuna-tuned",
    data_root: str = "/workspace/data",
    model_name: Optional[str] = None,
    on_study_done: Optional[Callable] = None,
) -> List[dict]:
    """Run per-experiment Optuna studies for all combinations.

    Studies whose ``trials_{n_trials}/best_params.json`` already exists are
    skipped. Studies with fewer previous trials continue incrementally.
    Pending studies run in parallel across GPUs.
    """
    events = events if events is not None else ALL_EVENTS
    budgets = budgets if budgets is not None else BUDGETS
    seed_sets = seed_sets if seed_sets is not None else SEED_SETS

    total = len(events) * len(budgets) * len(seed_sets)
    start_time = time.time()

    pre_results: Dict[Tuple[str, int, int], dict] = {}
    pending_configs: List[dict] = []
    pending_keys: List[Tuple[str, int, int]] = []
    skipped = 0

    for event in events:
        for budget in budgets:
            for seed_set in seed_sets:
                best_params_path = (
                    Path(storage_dir) / event / f"{budget}_set{seed_set}"
                    / f"trials_{n_trials}" / "best_params.json"
                )
                if best_params_path.exists():
                    with open(best_params_path) as f:
                        result = json.load(f)
                    pre_results[(event, budget, seed_set)] = result
                    skipped += 1
                    print(
                        f"  {event} budget={budget} seed={seed_set}"
                        f" -- SKIPPED (trials_{n_trials} exists)"
                    )
                    if on_study_done is not None:
                        on_study_done(event, budget, seed_set, "skipped")
                else:
                    pending_configs.append(dict(
                        event=event,
                        budget=budget,
                        seed_set=seed_set,
                        n_trials=n_trials,
                        storage_dir=storage_dir,
                        data_root=data_root,
                        model_name=model_name,
                    ))
                    pending_keys.append((event, budget, seed_set))

    print(
        f"\nSupervised Optuna per-experiment: {total} total, {skipped} skipped, "
        f"{len(pending_configs)} pending"
    )

    if pending_configs:
        if num_gpus > 1:
            parallel_results = _run_studies_parallel(
                pending_configs, pending_keys, num_gpus, on_study_done,
            )
        else:
            parallel_results = _run_studies_sequential(
                pending_configs, pending_keys, on_study_done,
            )
    else:
        parallel_results = {}

    all_results = []
    completed = failed = 0
    for event in events:
        for budget in budgets:
            for seed_set in seed_sets:
                key = (event, budget, seed_set)
                if key in pre_results:
                    all_results.append(pre_results[key])
                elif key in parallel_results:
                    result = parallel_results[key]
                    all_results.append(result)
                    if result["status"] == "done":
                        completed += 1
                    else:
                        failed += 1

    summary = {
        "total_studies": total,
        "completed": completed + skipped,
        "failed": failed,
        "n_trials_per_study": n_trials,
        "model_name": model_name,
        "method": "supervised",
        "search_space": {
            "lr": "1e-5 to 1e-3 (log-uniform)",
            "batch_size": [8, 16, 32, 64],
            "max_epochs": "20 to 100",
        },
        "studies": [
            {
                "event": r["event"],
                "budget": r["budget"],
                "seed_set": r["seed_set"],
                "status": r["status"],
                "best_params": r.get("best_params"),
                "best_value": r.get("best_value"),
            }
            for r in all_results
        ],
    }
    summary_path = Path(storage_dir) / f"summary_{n_trials}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - start_time
    print(
        f"\nAll studies complete: {completed} ran, {skipped} skipped, "
        f"{failed} failed ({elapsed:.1f}s total)"
    )
    print(f"Summary saved to: {summary_path}")

    return all_results


def _run_studies_parallel(
    pending_configs: List[dict],
    pending_keys: List[Tuple[str, int, int]],
    num_gpus: int,
    on_study_done: Optional[Callable],
) -> Dict[Tuple[str, int, int], dict]:
    """Dispatch studies across GPUs with dynamic round-robin assignment."""
    print(
        f"Running {len(pending_configs)} Optuna studies in parallel "
        f"across {num_gpus} GPUs..."
    )

    ctx = mp.get_context("spawn")
    results_map: Dict[int, dict] = {}
    config_queue = list(range(len(pending_configs)))

    with ProcessPoolExecutor(
        max_workers=num_gpus, mp_context=ctx
    ) as executor:
        active: Dict = {}

        for gpu_id in range(min(num_gpus, len(config_queue))):
            idx = config_queue.pop(0)
            pending_configs[idx]["device"] = f"cuda:{gpu_id}"
            future = executor.submit(_run_study_worker, pending_configs[idx])
            active[future] = (idx, gpu_id)

        while active:
            for future in as_completed(active):
                idx, gpu_id = active.pop(future)
                key = pending_keys[idx]

                try:
                    result = future.result()
                except Exception as e:
                    cfg = pending_configs[idx]
                    result = {
                        "event": cfg["event"],
                        "budget": cfg["budget"],
                        "seed_set": cfg["seed_set"],
                        "status": "failed",
                        "best_params": None,
                        "best_value": None,
                        "n_trials": 0,
                        "trials": [],
                    }
                    logger.error(f"Process-level failure for study {idx}: {e}")

                results_map[idx] = result

                status = result["status"]
                best_val = result.get("best_value")
                val_str = f" (best_dev_f1={best_val:.4f})" if best_val else ""
                print(
                    f"  {key[0]} budget={key[1]} seed={key[2]}"
                    f" -- {status}{val_str}"
                )
                if status == "failed" and result.get("error"):
                    print(f"    ERROR: {result['error']}")

                if on_study_done is not None:
                    on_study_done(key[0], key[1], key[2], status)

                if config_queue:
                    next_idx = config_queue.pop(0)
                    pending_configs[next_idx]["device"] = f"cuda:{gpu_id}"
                    new_future = executor.submit(
                        _run_study_worker, pending_configs[next_idx]
                    )
                    active[new_future] = (next_idx, gpu_id)

                break

    return {pending_keys[i]: results_map[i] for i in range(len(pending_configs))}


def _run_studies_sequential(
    pending_configs: List[dict],
    pending_keys: List[Tuple[str, int, int]],
    on_study_done: Optional[Callable],
) -> Dict[Tuple[str, int, int], dict]:
    """Run studies sequentially (num_gpus=1)."""
    results = {}
    for idx, (cfg, key) in enumerate(zip(pending_configs, pending_keys)):
        print(
            f"  [{idx + 1}/{len(pending_configs)}] "
            f"{key[0]} budget={key[1]} seed={key[2]} -- starting..."
        )
        result = _run_study_worker(cfg)
        results[key] = result

        status = result["status"]
        best_val = result.get("best_value")
        val_str = f" (best_dev_f1={best_val:.4f})" if best_val else ""
        print(
            f"  [{idx + 1}/{len(pending_configs)}] "
            f"{key[0]} budget={key[1]} seed={key[2]}"
            f" -- {status}{val_str}"
        )
        if status == "failed" and result.get("error"):
            print(f"    ERROR: {result['error']}")

        if on_study_done is not None:
            on_study_done(key[0], key[1], key[2], status)

    return results


def load_best_params(
    storage_dir: str = "/workspace/results/bertweet/supervised/optuna-tuned",
    events: Optional[List[str]] = None,
    budgets: Optional[List[int]] = None,
    seed_sets: Optional[List[int]] = None,
    n_trials: Optional[int] = None,
) -> Dict[Tuple[str, int, int], dict]:
    """Load all best_params.json files into a dict keyed by (event, budget, seed_set)."""
    events = events if events is not None else ALL_EVENTS
    budgets = budgets if budgets is not None else BUDGETS
    seed_sets = seed_sets if seed_sets is not None else SEED_SETS

    results = {}
    for event in events:
        for budget in budgets:
            for seed_set in seed_sets:
                experiment_dir = (
                    Path(storage_dir) / event / f"{budget}_set{seed_set}"
                )
                if n_trials is not None:
                    path = experiment_dir / f"trials_{n_trials}" / "best_params.json"
                    if path.exists():
                        with open(path) as f:
                            results[(event, budget, seed_set)] = json.load(f)
                else:
                    latest = _find_latest_trials(experiment_dir)
                    if latest is not None:
                        _, data = latest
                        results[(event, budget, seed_set)] = data

    return results


def main():
    """CLI entry point: ``python -m supervised_baseline.optuna_tuner``."""
    parser = argparse.ArgumentParser(
        description="Per-experiment Optuna hyperparameter tuner for the "
        "supervised-only BERTweet baseline. Runs separate studies per "
        "(event, budget, seed) to find experiment-specific optimal "
        "hyperparameters (lr, batch_size, max_epochs). Supports incremental "
        "scaling: running with a higher --n-trials continues from previous runs.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=10,
        help="Number of Optuna trials per study (default: 10)",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1,
        help="Number of GPUs for parallel study execution (default: 1)",
    )
    parser.add_argument(
        "--events", type=str, nargs="*", default=None,
        help="Events to tune (default: all 10)",
    )
    parser.add_argument(
        "--budgets", type=int, nargs="*", default=None,
        help="Budgets to tune (default: all [5, 10, 25, 50])",
    )
    parser.add_argument(
        "--seed-sets", type=int, nargs="*", default=None,
        help="Seed sets to tune (default: all [1, 2, 3])",
    )
    parser.add_argument(
        "--data-root", type=str,
        default=str(Path(__file__).parent.parent / "data"),
    )
    parser.add_argument(
        "--storage-dir", type=str,
        default=str(Path(__file__).parent.parent / "results" / "bertweet"
                    / "supervised" / "optuna-tuned"),
        help="Directory for storing study results",
    )
    parser.add_argument(
        "--model-name", type=str, default="vinai/bertweet-base",
        help="HuggingFace model name (default: vinai/bertweet-base)",
    )

    args = parser.parse_args()

    run_all_studies(
        events=args.events,
        budgets=args.budgets,
        seed_sets=args.seed_sets,
        n_trials=args.n_trials,
        num_gpus=args.num_gpus,
        storage_dir=args.storage_dir,
        data_root=args.data_root,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
