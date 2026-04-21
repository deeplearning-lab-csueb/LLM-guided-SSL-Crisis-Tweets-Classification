"""Per-experiment Optuna hyperparameter tuning for LG-CoTrain.

Runs 120 separate Optuna studies — one for each (event, budget, seed_set)
combination — to find experiment-specific optimal hyperparameters.  Studies
run in parallel across multiple GPUs using ProcessPoolExecutor.

Results are saved as JSON files (no database) under ``trials_{n}/``
subfolders.  **Incremental scaling** is built-in: running with a higher
``n_trials`` automatically continues from the latest previous run, replaying
earlier trials into the TPE sampler so only the delta needs to execute.

Usage::

    # First run: 10 trials each
    python -m lg_cotrain.optuna_per_experiment --n-trials 10 --num-gpus 2

    # Later: scale to 20 trials (only 10 new trials per study)
    python -m lg_cotrain.optuna_per_experiment --n-trials 20 --num-gpus 2

    # Subset
    python -m lg_cotrain.optuna_per_experiment --n-trials 10 --events hurricane_harvey_2017
"""

import argparse
import json
import logging
import multiprocessing as mp
import re
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("lg_cotrain")

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

# Canonical search space — used by the objective function and by
# _replay_trials_into_study() to reconstruct Optuna distributions.
SEARCH_SPACE = {
    "lr": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
    "batch_size": {"type": "categorical", "choices": [8, 16, 32, 64]},
    "cotrain_epochs": {"type": "int", "low": 5, "high": 20},
    "finetune_patience": {"type": "int", "low": 4, "high": 10},
    "weight_decay": {"type": "float", "low": 0.0, "high": 0.1},
    "warmup_ratio": {"type": "float", "low": 0.0, "high": 0.3},
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


def _find_latest_trials(experiment_dir: Path) -> Optional[Tuple[int, dict]]:
    """Find the ``trials_*`` subfolder with the highest trial count.

    Parameters
    ----------
    experiment_dir : Path
        e.g. ``results/optuna/per_experiment/hurricane_harvey_2017/50_set1``

    Returns
    -------
    ``(n_trials, data_dict)`` from the best_params.json of the highest
    trial-count folder, or ``None`` if no completed trials exist.
    """
    if not experiment_dir.is_dir():
        return None

    best_n = 0
    best_data = None

    for child in experiment_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.match(r"^trials_(\d+)$", child.name)
        if match is None:
            continue
        n = int(match.group(1))
        bp_path = child / "best_params.json"
        if bp_path.exists() and n > best_n:
            best_n = n
            with open(bp_path) as f:
                best_data = json.load(f)

    if best_data is None:
        return None
    return (best_n, best_data)


def _replay_trials_into_study(study, trials_info: List[dict]):
    """Replay previously saved trial records into an Optuna study.

    This feeds the TPE sampler with historical observations so it can
    make informed suggestions for new trials.

    Parameters
    ----------
    study : optuna.Study
        An in-memory study (typically freshly created).
    trials_info : list of dict
        Trial records as saved in ``best_params.json["trials"]``.
        Each must have ``"number"``, ``"params"``, and ``"dev_macro_f1"`` keys.
    """
    from optuna.trial import create_trial, TrialState
    from datetime import datetime

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


def _setup_study_logging(log_path: str):
    """Set up a persistent log file for an Optuna study.

    Replaces any existing FileHandlers on the ``lg_cotrain`` logger with
    a single handler that writes to *log_path* (inside the project's
    results folder, not a temp directory).  This ensures:

    1. All trials within a study log to the same persistent file.
    2. ``tempfile.TemporaryDirectory`` cleanup doesn't conflict with
       open file handles (Windows PermissionError fix).
    3. The user can ``tail -f`` the log to monitor live progress.

    Returns the created FileHandler so it can be closed after the study.
    """
    lgr = logging.getLogger("lg_cotrain")

    # Remove any existing FileHandlers (e.g. from a previous study or
    # from setup_logging pointing at a temp dir)
    for h in lgr.handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.close()
            lgr.removeHandler(h)

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    lgr.addHandler(fh)

    # Ensure a StreamHandler exists (setup_logging's if-not-handlers
    # guard may have been skipped after handler removal)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in lgr.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        lgr.addHandler(ch)

    lgr.setLevel(logging.INFO)
    return fh


def _close_temp_file_handlers(logger_instance):
    """Close and remove FileHandlers pointing at temp directories.

    Called after each trial so ``tempfile.TemporaryDirectory`` cleanup
    can delete the temp dir on Windows.  The persistent study log handler
    (set up by ``_setup_study_logging``) is left untouched because its
    path is in the project results folder, not a temp dir.
    """
    for handler in logger_instance.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            # Only remove handlers whose file is inside a temp directory
            handler_path = getattr(handler, 'baseFilename', '')
            if tempfile.gettempdir() in handler_path:
                handler.close()
                logger_instance.removeHandler(handler)


def create_per_experiment_objective(
    event: str,
    budget: int,
    seed_set: int,
    device: Optional[str] = None,
    data_root: str = "/workspace/data",
    pseudo_label_source: str = "gpt-4o",
    model_name: Optional[str] = None,
    weight_gen_epochs: Optional[int] = None,
    _trainer_cls=None,
):
    """Return an Optuna objective function for a single experiment.

    Each call to the returned function:

    1. Samples 6 hyperparameters from the Optuna trial.
    2. Runs the full 3-phase pipeline for the given experiment.
    3. Attaches the full ``trainer.run()`` result dict to ``trial.user_attrs``
       under the key ``"full_metrics"`` so the test-set numbers (which are
       computed inside the trial as a side effect of the final evaluation
       phase) survive the temp-dir cleanup.  This is metadata-only — Optuna's
       sampler and ``best_trial`` selection never read ``user_attrs``, so it
       does NOT leak test info into the search.
    4. Returns ``dev_macro_f1`` (no test-set leakage).

    Parameters
    ----------
    event, budget, seed_set : experiment identifiers
    device : GPU device string (e.g. "cuda:0") or None for auto-detect
    data_root : base directory for input data
    pseudo_label_source : pseudo-label directory name
    model_name : HuggingFace model name (e.g. "bert-base-uncased",
        "vinai/bertweet-base").  If ``None`` (default), uses the
        ``LGCoTrainConfig`` default.
    _trainer_cls : override trainer class (for testing with mocks)
    """

    def objective(trial):
        # Sample hyperparameters using the canonical search space
        lr = trial.suggest_float(
            "lr", SEARCH_SPACE["lr"]["low"], SEARCH_SPACE["lr"]["high"],
            log=SEARCH_SPACE["lr"].get("log", False),
        )
        batch_size = trial.suggest_categorical(
            "batch_size", SEARCH_SPACE["batch_size"]["choices"],
        )
        cotrain_epochs = trial.suggest_int(
            "cotrain_epochs",
            SEARCH_SPACE["cotrain_epochs"]["low"],
            SEARCH_SPACE["cotrain_epochs"]["high"],
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

        # Use a temp directory so trial outputs don't pollute real results
        with tempfile.TemporaryDirectory() as tmp_results:
            from .config import LGCoTrainConfig

            config_kwargs = dict(
                event=event,
                budget=budget,
                seed_set=seed_set,
                lr=lr,
                batch_size=batch_size,
                cotrain_epochs=cotrain_epochs,
                finetune_patience=finetune_patience,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                device=device,
                data_root=data_root,
                results_root=tmp_results,
                pseudo_label_source=pseudo_label_source,
            )
            if model_name is not None:
                config_kwargs["model_name"] = model_name
            if weight_gen_epochs is not None:
                config_kwargs["weight_gen_epochs"] = weight_gen_epochs

            config = LGCoTrainConfig(**config_kwargs)

            if _trainer_cls is None:
                from .trainer import LGCoTrainer
                trainer = LGCoTrainer(config)
            else:
                trainer = _trainer_cls(config)

            result = trainer.run()

            # Persist the full result on the trial as metadata.  This lets us
            # later materialize a canonical metrics.json from the best trial
            # without re-running the experiment.  set_user_attr is write-only
            # from Optuna's perspective: the sampler and best_trial selection
            # never read user_attrs, so this does NOT leak test info into the
            # hyperparameter search.
            try:
                trial.set_user_attr("full_metrics", result)
            except Exception:
                # If trial doesn't support set_user_attr (e.g. some mock
                # objects in tests), continue silently — the dev_macro_f1
                # return value is what Optuna actually needs.
                pass

            # Close FileHandlers pointing at the temp dir before cleanup.
            # On Windows, open handles prevent directory deletion.
            # The persistent study log handler is left untouched.
            _close_temp_file_handlers(logging.getLogger("lg_cotrain"))

        return result["dev_macro_f1"]

    return objective


def run_single_study(
    event: str,
    budget: int,
    seed_set: int,
    n_trials: int,
    device: Optional[str] = None,
    storage_dir: str = "/workspace/results/bert-base/optuna/per_experiment",
    data_root: str = "/workspace/data",
    pseudo_label_source: str = "gpt-4o",
    model_name: Optional[str] = None,
    weight_gen_epochs: Optional[int] = None,
    on_trial_done: Optional[Callable] = None,
    _trainer_cls=None,
) -> dict:
    """Run one Optuna study for a single (event, budget, seed_set).

    Creates an in-memory Optuna study, runs *n_trials* trials, saves
    ``best_params.json`` under ``trials_{n_trials}/``.

    **Incremental support**: If a previous ``trials_k/best_params.json``
    exists with ``k < n_trials``, those *k* trials are replayed into the
    new study (feeding the TPE sampler) and only ``n_trials - k`` new
    trials are executed.  If ``k >= n_trials``, the study is skipped.

    Parameters
    ----------
    on_trial_done : callable, optional
        Called after each *new* trial with ``(trial_number, n_trials, dev_f1)``.

    Returns
    -------
    dict with keys: event, budget, seed_set, status, best_params, best_value,
    n_trials, trials.
    """
    experiment_dir = Path(storage_dir) / event / f"{budget}_set{seed_set}"
    output_dir = experiment_dir / f"trials_{n_trials}"
    best_params_path = output_dir / "best_params.json"

    # Skip if exact trial count already done
    if best_params_path.exists():
        with open(best_params_path) as f:
            return json.load(f)

    import optuna  # lazy — not needed at module level

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=f"{event}_b{budget}_s{seed_set}",
        direction="maximize",
    )

    # Check for previous trials to continue from
    previous_trials = []
    previous_n = 0
    latest = _find_latest_trials(experiment_dir)
    if latest is not None:
        previous_n, prev_data = latest
        if previous_n >= n_trials:
            # Already have enough trials — return the previous results
            return prev_data
        previous_trials = prev_data.get("trials", [])
        _replay_trials_into_study(study, previous_trials)
        logger.info(
            f"Replayed {len(previous_trials)} trials from trials_{previous_n} "
            f"into study for {event} b={budget} s={seed_set}"
        )

    new_trials_needed = n_trials - len(previous_trials)

    # Set up persistent study log in the project results folder.
    # This file survives across trials and lets you monitor progress with
    # e.g.  tail -f results/optuna/per_experiment/{event}/{budget}_set{seed}/study.log
    output_dir.mkdir(parents=True, exist_ok=True)
    study_log_path = str(output_dir / "study.log")
    study_fh = _setup_study_logging(study_log_path)
    logger.info(
        f"=== Optuna study: {event} budget={budget} seed={seed_set} | "
        f"target={n_trials} trials, {len(previous_trials)} replayed, "
        f"{new_trials_needed} new ==="
    )

    # Wrap objective with callback
    base_objective = create_per_experiment_objective(
        event=event,
        budget=budget,
        seed_set=seed_set,
        device=device,
        data_root=data_root,
        pseudo_label_source=pseudo_label_source,
        model_name=model_name,
        weight_gen_epochs=weight_gen_epochs,
        _trainer_cls=_trainer_cls,
    )

    def objective_with_callback(trial):
        logger.info(f"--- Trial {trial.number + 1}/{n_trials} ---")
        dev_f1 = base_objective(trial)
        logger.info(f"--- Trial {trial.number + 1}/{n_trials} done: dev_macro_f1={dev_f1:.4f} ---")
        if on_trial_done is not None:
            on_trial_done(trial.number, n_trials, dev_f1)
        return dev_f1

    study.optimize(objective_with_callback, n_trials=new_trials_needed)

    # Close the persistent study log handler
    study_fh.close()
    logging.getLogger("lg_cotrain").removeHandler(study_fh)

    # Build result — include ALL trials (replayed + new)
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
        # If the trial saved full metrics via set_user_attr (new in this
        # commit), persist them in the JSON so we can later materialize a
        # canonical metrics.json without re-running the experiment.
        # Replayed trials (from previous best_params.json) won't have this.
        full_metrics = t.user_attrs.get("full_metrics") if hasattr(t, "user_attrs") else None
        if full_metrics is not None:
            trial_info["full_metrics"] = full_metrics
        trials_info.append(trial_info)

    # Extract best trial's full metrics (test_macro_f1, test_error_rate,
    # test_ece, test_per_class_f1, lambda values, etc.) from user_attrs.
    # study.best_trial is selected by the highest returned objective value
    # (dev_macro_f1) — never by user_attrs — so this does NOT leak test
    # info into the search.
    best_full_metrics = None
    try:
        best_full_metrics = study.best_trial.user_attrs.get("full_metrics")
    except Exception:
        # No completed trials, or best_trial is unavailable for some reason
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

    # Save to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(best_params_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def _run_study_worker(kwargs: dict) -> dict:
    """Worker function: run one Optuna study in a child process.

    Receives a plain dict (picklable), imports inside the function.
    Same pattern as ``parallel._run_single_experiment``.
    """
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
            pseudo_label_source=kwargs.get("pseudo_label_source", "gpt-4o"),
            model_name=kwargs.get("model_name"),
            weight_gen_epochs=kwargs.get("weight_gen_epochs"),
        )
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {e}"
        error_tb = traceback.format_exc()
        logging.getLogger("lg_cotrain").error(
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
    n_trials: int = 15,
    num_gpus: int = 1,
    storage_dir: str = "/workspace/results/bert-base/optuna/per_experiment",
    data_root: str = "/workspace/data",
    pseudo_label_source: str = "gpt-4o",
    model_name: Optional[str] = None,
    weight_gen_epochs: Optional[int] = None,
    on_study_done: Optional[Callable] = None,
) -> List[dict]:
    """Run per-experiment Optuna studies for all combinations.

    Studies whose ``trials_{n_trials}/best_params.json`` already exists
    are skipped.  Studies with fewer previous trials will continue
    incrementally.  Pending studies run in parallel across GPUs.

    Parameters
    ----------
    on_study_done : callable, optional
        Called after each study with ``(event, budget, seed_set, status)``.

    Returns
    -------
    List of result dicts in original (event, budget, seed_set) order.
    """
    events = events if events is not None else ALL_EVENTS
    budgets = budgets if budgets is not None else BUDGETS
    seed_sets = seed_sets if seed_sets is not None else SEED_SETS

    total = len(events) * len(budgets) * len(seed_sets)
    start_time = time.time()

    # Separate skipped from pending
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
                        pseudo_label_source=pseudo_label_source,
                        model_name=model_name,
                        weight_gen_epochs=weight_gen_epochs,
                    ))
                    pending_keys.append((event, budget, seed_set))

    print(
        f"\nOptuna per-experiment: {total} total, {skipped} skipped, "
        f"{len(pending_configs)} pending"
    )

    # Run pending studies
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

    # Merge in original order
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

    # Write summary_{n_trials}.json
    summary = {
        "total_studies": total,
        "completed": completed + skipped,
        "failed": failed,
        "n_trials_per_study": n_trials,
        "model_name": model_name,
        "search_space": {
            "lr": "1e-5 to 1e-3 (log-uniform)",
            "batch_size": [8, 16, 32, 64],
            "cotrain_epochs": "5 to 20",
            "finetune_patience": "4 to 10",
            "weight_decay": "0.0 to 0.1",
            "warmup_ratio": "0.0 to 0.3",
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
    """Dispatch studies across GPUs using ProcessPoolExecutor.

    GPU assignment is done dynamically: when a study on cuda:X finishes,
    the next pending study is assigned to cuda:X.  This ensures GPUs stay
    balanced even when studies have very different runtimes.
    """
    print(
        f"Running {len(pending_configs)} Optuna studies in parallel "
        f"across {num_gpus} GPUs..."
    )

    ctx = mp.get_context("spawn")
    results_map: Dict[int, dict] = {}
    config_queue = list(range(len(pending_configs)))  # indices

    with ProcessPoolExecutor(
        max_workers=num_gpus, mp_context=ctx
    ) as executor:
        active: Dict = {}   # future -> (config_idx, gpu_id)

        # Seed the pool: one task per GPU
        for gpu_id in range(min(num_gpus, len(config_queue))):
            idx = config_queue.pop(0)
            pending_configs[idx]["device"] = f"cuda:{gpu_id}"
            future = executor.submit(_run_study_worker, pending_configs[idx])
            active[future] = (idx, gpu_id)

        while active:
            # Wait for the next completion
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

                # Submit next study to the same GPU that just freed up
                if config_queue:
                    next_idx = config_queue.pop(0)
                    pending_configs[next_idx]["device"] = f"cuda:{gpu_id}"
                    new_future = executor.submit(
                        _run_study_worker, pending_configs[next_idx]
                    )
                    active[new_future] = (next_idx, gpu_id)

                # Process one completion at a time so the freed GPU
                # gets its next task immediately
                break

    # Build lookup by key
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
    storage_dir: str = "/workspace/results/bert-base/optuna/per_experiment",
    events: Optional[List[str]] = None,
    budgets: Optional[List[int]] = None,
    seed_sets: Optional[List[int]] = None,
    n_trials: Optional[int] = None,
) -> Dict[Tuple[str, int, int], dict]:
    """Load all best_params.json files into a dict.

    Parameters
    ----------
    n_trials : int, optional
        If specified, load from ``trials_{n_trials}/best_params.json``.
        If ``None`` (default), load from the latest (highest trial-count)
        ``trials_*/`` subfolder for each experiment.

    Returns
    -------
    Dict keyed by ``(event, budget, seed_set)`` with values being the
    full result dict (including ``best_params``, ``best_value``, etc.).
    """
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


def export_metrics_from_studies(
    storage_dir: str,
    results_root: str,
    n_trials: Optional[int] = None,
    events: Optional[List[str]] = None,
    budgets: Optional[List[int]] = None,
    seed_sets: Optional[List[int]] = None,
    overwrite: bool = False,
) -> Dict[str, int]:
    """Materialize canonical metrics.json files from Optuna best-trial metrics.

    Walks ``storage_dir`` looking for ``best_params.json`` files, extracts the
    ``best_full_metrics`` field (saved by the trial-level
    ``set_user_attr("full_metrics", ...)`` hook), and writes it to
    ``results_root/{event}/{budget}_set{seed_set}/metrics.json``.

    This is the "shortcut" that lets us skip a separate Phase B2 final-run
    grid: the best Optuna trial was already trained end-to-end with the best
    hyperparameters, and training is bit-exactly deterministic on a single
    machine, so the trial's metrics are byte-identical to what a separate
    re-run would produce.

    Parameters
    ----------
    storage_dir : str
        Path to the per_experiment Optuna storage tree, e.g.
        ``results/bertweet/optuna/wge7``.
    results_root : str
        Path to the canonical results tree, e.g.
        ``results/bertweet/gpt-4o/optuna-tuned/wge7``.  Each cell will be
        written to ``{results_root}/{event}/{budget}_set{seed}/metrics.json``.
    n_trials : int, optional
        Read from ``trials_{n_trials}/best_params.json``.  If ``None``
        (default), uses the latest available trial count for each cell.
    events, budgets, seed_sets : optional iterables to filter the grid.
    overwrite : bool, default ``False``
        If ``False``, skip cells whose metrics.json already exists.  If
        ``True``, overwrite them.

    Returns
    -------
    dict with counts: ``{"written": N, "skipped": N, "missing_metrics": N,
    "missing_study": N}``.
    """
    events = events if events is not None else ALL_EVENTS
    budgets = budgets if budgets is not None else BUDGETS
    seed_sets = seed_sets if seed_sets is not None else SEED_SETS

    storage_dir = Path(storage_dir)
    results_root = Path(results_root)

    counts = {
        "written": 0,
        "skipped": 0,
        "missing_metrics": 0,
        "missing_study": 0,
    }

    for event in events:
        for budget in budgets:
            for seed_set in seed_sets:
                experiment_dir = storage_dir / event / f"{budget}_set{seed_set}"

                # Resolve which best_params.json to read
                if n_trials is not None:
                    bp_path = experiment_dir / f"trials_{n_trials}" / "best_params.json"
                    if not bp_path.exists():
                        counts["missing_study"] += 1
                        continue
                    with open(bp_path) as f:
                        data = json.load(f)
                else:
                    latest = _find_latest_trials(experiment_dir)
                    if latest is None:
                        counts["missing_study"] += 1
                        continue
                    _, data = latest

                best_full_metrics = data.get("best_full_metrics")
                if best_full_metrics is None:
                    # The Optuna study was run before this commit, or the
                    # best trial was a replayed trial that did not have
                    # full_metrics in user_attrs.
                    counts["missing_metrics"] += 1
                    continue

                # Write canonical metrics.json
                out_path = (
                    results_root / event / f"{budget}_set{seed_set}" / "metrics.json"
                )
                if out_path.exists() and not overwrite:
                    counts["skipped"] += 1
                    continue

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(best_full_metrics, f, indent=2)
                counts["written"] += 1

    return counts


def main():
    """CLI entry point: ``python -m lg_cotrain.optuna_per_experiment``."""
    parser = argparse.ArgumentParser(
        description="Per-experiment Optuna hyperparameter tuner for LG-CoTrain. "
        "Runs 120 separate studies (one per event/budget/seed combination) to "
        "find experiment-specific optimal hyperparameters.  Supports incremental "
        "scaling: running with a higher --n-trials continues from previous runs.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=15,
        help="Number of Optuna trials per study (default: 15)",
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
        "--data-root", type=str, default="/workspace/data",
    )
    parser.add_argument(
        "--storage-dir", type=str,
        default="/workspace/results/bert-base/optuna/per_experiment",
        help="Directory for storing study results (default: results/optuna/per_experiment)",
    )
    parser.add_argument(
        "--pseudo-label-source", type=str, default="gpt-4o",
        help="Pseudo-label directory name (default: gpt-4o)",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="HuggingFace model name (e.g. 'bert-base-uncased', "
             "'vinai/bertweet-base'). If omitted, uses LGCoTrainConfig default.",
    )
    parser.add_argument(
        "--weight-gen-epochs", type=int, default=None,
        help="Phase 1 weight generation epochs (default: use LGCoTrainConfig default of 7). "
             "Set to 10 to match the original LG-CoTrain authors' implementation.",
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
        pseudo_label_source=args.pseudo_label_source,
        model_name=args.model_name,
        weight_gen_epochs=args.weight_gen_epochs,
    )


if __name__ == "__main__":
    main()
