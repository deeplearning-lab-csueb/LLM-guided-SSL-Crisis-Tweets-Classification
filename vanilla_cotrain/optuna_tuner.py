"""Per-experiment Optuna hyperparameter tuner for Vanilla Co-Training.

Modeled after lg_cotrain/optuna_per_experiment.py but adapted for
VanillaCoTrainConfig. Tunes 7 parameters per (event, budget, seed) study.

Search space:
    lr              : 1e-5 to 1e-3 (log-uniform)
    batch_size      : [8, 16, 32, 64]
    train_epochs    : 3 to 10 (epochs per co-training iteration)
    samples_per_class : [1, 5, 10] (samples exchanged per class per iteration)
    finetune_patience : 4 to 10
    weight_decay    : 0.0 to 0.1
    warmup_ratio    : 0.0 to 0.3

Fixed: num_iterations=30 (capped by unlabeled pool size).

Usage::

    python -m vanilla_cotrain.optuna_tuner \\
        --n-trials 10 --num-gpus 2 \\
        --events hurricane_harvey_2017 \\
        --budgets 5 \\
        --storage-dir results/bertweet/ablation/vanilla-cotrain/optuna
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

logger = logging.getLogger("vanilla_cotrain")

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

SEARCH_SPACE = {
    "lr": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
    "batch_size": {"type": "categorical", "choices": [8, 16, 32, 64]},
    "train_epochs": {"type": "int", "low": 3, "high": 10},
    "samples_per_class": {"type": "categorical", "choices": [1, 5, 10]},
    "finetune_patience": {"type": "int", "low": 4, "high": 10},
    "weight_decay": {"type": "float", "low": 0.0, "high": 0.1},
    "warmup_ratio": {"type": "float", "low": 0.0, "high": 0.3},
}

NUM_ITERATIONS = 30  # fixed


def _close_temp_file_handlers(logger_obj):
    """Close FileHandlers pointing at temp directories."""
    for handler in logger_obj.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            try:
                path = handler.baseFilename
                if "tmp" in path.lower() or "temp" in path.lower():
                    handler.close()
                    logger_obj.removeHandler(handler)
            except Exception:
                pass


def _setup_study_logging(log_path: str) -> logging.FileHandler:
    """Add a FileHandler for the study log."""
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logging.getLogger("vanilla_cotrain").addHandler(fh)
    # Also capture lg_cotrain logs (trainer uses that logger)
    logging.getLogger("lg_cotrain").addHandler(fh)
    return fh


def _find_latest_trials(experiment_dir: Path):
    """Find the trials_* subfolder with the highest trial count."""
    if not experiment_dir.exists():
        return None
    best_n = 0
    best_data = None
    for d in experiment_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("trials_"):
            continue
        try:
            n = int(d.name.split("_", 1)[1])
        except ValueError:
            continue
        bp = d / "best_params.json"
        if bp.exists() and n > best_n:
            best_n = n
            with open(bp, encoding="utf-8") as f:
                best_data = json.load(f)
    return (best_n, best_data) if best_data is not None else None


def _replay_trials_into_study(study, trials_info):
    """Replay previous trials into an Optuna study for TPE warm-start."""
    import optuna
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

    for t in trials_info:
        if t.get("state") != "COMPLETE" or "dev_macro_f1" not in t:
            continue
        study.add_trial(
            optuna.trial.create_trial(
                params=t["params"],
                distributions=distributions,
                values=[t["dev_macro_f1"]],
                state=optuna.trial.TrialState.COMPLETE,
            )
        )


def create_objective(
    event: str,
    budget: int,
    seed_set: int,
    device: Optional[str] = None,
    data_root: str = "/workspace/data",
    model_name: str = "vinai/bertweet-base",
):
    """Return an Optuna objective function for a single vanilla co-training experiment."""

    def objective(trial):
        lr = trial.suggest_float(
            "lr", SEARCH_SPACE["lr"]["low"], SEARCH_SPACE["lr"]["high"],
            log=SEARCH_SPACE["lr"].get("log", False),
        )
        batch_size = trial.suggest_categorical(
            "batch_size", SEARCH_SPACE["batch_size"]["choices"],
        )
        train_epochs = trial.suggest_int(
            "train_epochs",
            SEARCH_SPACE["train_epochs"]["low"],
            SEARCH_SPACE["train_epochs"]["high"],
        )
        samples_per_class = trial.suggest_categorical(
            "samples_per_class", SEARCH_SPACE["samples_per_class"]["choices"],
        )
        finetune_patience = trial.suggest_int(
            "finetune_patience",
            SEARCH_SPACE["finetune_patience"]["low"],
            SEARCH_SPACE["finetune_patience"]["high"],
        )
        weight_decay = trial.suggest_float(
            "weight_decay",
            SEARCH_SPACE["weight_decay"]["low"],
            SEARCH_SPACE["weight_decay"]["high"],
        )
        warmup_ratio = trial.suggest_float(
            "warmup_ratio",
            SEARCH_SPACE["warmup_ratio"]["low"],
            SEARCH_SPACE["warmup_ratio"]["high"],
        )

        with tempfile.TemporaryDirectory() as tmp_results:
            from .config import VanillaCoTrainConfig
            from .trainer import VanillaCoTrainer

            config = VanillaCoTrainConfig(
                event=event,
                budget=budget,
                seed_set=seed_set,
                model_name=model_name,
                num_iterations=NUM_ITERATIONS,
                samples_per_class=samples_per_class,
                train_epochs=train_epochs,
                finetune_patience=finetune_patience,
                lr=lr,
                batch_size=batch_size,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                device=device,
                data_root=data_root,
                results_root=tmp_results,
            )

            trainer = VanillaCoTrainer(config)
            result = trainer.run()

            try:
                trial.set_user_attr("full_metrics", result)
            except Exception:
                pass

            _close_temp_file_handlers(logging.getLogger("lg_cotrain"))
            _close_temp_file_handlers(logging.getLogger("vanilla_cotrain"))

        return result["dev_macro_f1"]

    return objective


def run_single_study(
    event: str,
    budget: int,
    seed_set: int,
    n_trials: int,
    device: Optional[str] = None,
    storage_dir: str = "results/bertweet/ablation/vanilla-cotrain/optuna",
    data_root: str = "/workspace/data",
    model_name: str = "vinai/bertweet-base",
) -> dict:
    """Run one Optuna study for a single (event, budget, seed_set)."""
    experiment_dir = Path(storage_dir) / event / f"{budget}_set{seed_set}"
    output_dir = experiment_dir / f"trials_{n_trials}"
    best_params_path = output_dir / "best_params.json"

    if best_params_path.exists():
        with open(best_params_path, encoding="utf-8") as f:
            return json.load(f)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=f"vanilla_{event}_b{budget}_s{seed_set}",
        direction="maximize",
    )

    # Check for previous trials
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
            f"Replayed {len(previous_trials)} trials from trials_{previous_n}"
        )

    new_trials_needed = n_trials - len(previous_trials)

    output_dir.mkdir(parents=True, exist_ok=True)
    study_log_path = str(output_dir / "study.log")
    study_fh = _setup_study_logging(study_log_path)
    logger.info(
        f"=== Optuna study: {event} budget={budget} seed={seed_set} | "
        f"target={n_trials} trials, {len(previous_trials)} replayed, "
        f"{new_trials_needed} new ==="
    )

    base_objective = create_objective(
        event=event,
        budget=budget,
        seed_set=seed_set,
        device=device,
        data_root=data_root,
        model_name=model_name,
    )

    def objective_with_callback(trial):
        logger.info(f"--- Trial {trial.number + 1}/{n_trials} ---")
        dev_f1 = base_objective(trial)
        logger.info(
            f"--- Trial {trial.number + 1}/{n_trials} done: "
            f"dev_macro_f1={dev_f1:.4f} ---"
        )
        return dev_f1

    study.optimize(objective_with_callback, n_trials=new_trials_needed)

    study_fh.close()
    logging.getLogger("vanilla_cotrain").removeHandler(study_fh)
    logging.getLogger("lg_cotrain").removeHandler(study_fh)

    # Build result
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
        pass

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

    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def _run_study_worker(kwargs: dict) -> dict:
    """Worker function for parallel execution."""
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
            model_name=kwargs.get("model_name", "vinai/bertweet-base"),
        )
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(
            f"Optuna study {event} budget={budget} seed={seed_set} "
            f"failed: {error_msg}"
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
            "traceback": traceback.format_exc(),
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
    storage_dir: str = "results/bertweet/ablation/vanilla-cotrain/optuna",
    data_root: str = "/workspace/data",
    model_name: str = "vinai/bertweet-base",
) -> List[dict]:
    """Run per-experiment Optuna studies for all combinations."""
    events = events if events is not None else ALL_EVENTS
    budgets = budgets if budgets is not None else BUDGETS
    seed_sets = seed_sets if seed_sets is not None else SEED_SETS

    total = len(events) * len(budgets) * len(seed_sets)

    # Separate skipped from pending
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
                    skipped += 1
                    print(
                        f"  {event} budget={budget} seed={seed_set}"
                        f" -- SKIPPED (trials_{n_trials} exists)"
                    )
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
        f"\nOptuna per-experiment: {total} total, {skipped} skipped, "
        f"{len(pending_configs)} pending"
    )

    if not pending_configs:
        print("Nothing to do.")
        return []

    if num_gpus > 1:
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

                    if config_queue:
                        next_idx = config_queue.pop(0)
                        pending_configs[next_idx]["device"] = f"cuda:{gpu_id}"
                        new_future = executor.submit(
                            _run_study_worker, pending_configs[next_idx]
                        )
                        active[new_future] = (next_idx, gpu_id)
                    break
    else:
        print(f"Running {len(pending_configs)} Optuna studies sequentially...")
        for idx, cfg in enumerate(pending_configs):
            key = pending_keys[idx]
            print(
                f"  [{idx + 1}/{len(pending_configs)}] "
                f"{key[0]} budget={key[1]} seed={key[2]} -- starting..."
            )
            result = _run_study_worker(cfg)
            status = result["status"]
            best_val = result.get("best_value")
            val_str = f" (best_dev_f1={best_val:.4f})" if best_val else ""
            print(
                f"  [{idx + 1}/{len(pending_configs)}] "
                f"{key[0]} budget={key[1]} seed={key[2]}"
                f" -- {status}{val_str}"
            )


def main():
    """CLI entry point: python -m vanilla_cotrain.optuna_tuner"""
    parser = argparse.ArgumentParser(
        description="Per-experiment Optuna tuner for Vanilla Co-Training.",
    )
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--events", type=str, nargs="*", default=None)
    parser.add_argument("--budgets", type=int, nargs="*", default=None)
    parser.add_argument("--seed-sets", type=int, nargs="*", default=None)
    parser.add_argument("--data-root", type=str, default="/workspace/data")
    parser.add_argument(
        "--storage-dir", type=str,
        default="results/bertweet/ablation/vanilla-cotrain/optuna",
    )
    parser.add_argument("--model-name", type=str, default="vinai/bertweet-base")

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
