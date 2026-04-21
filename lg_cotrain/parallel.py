"""Multi-GPU parallel experiment execution for LG-CoTrain."""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Optional

logger = logging.getLogger("lg_cotrain")


def _run_single_experiment(config_kwargs: dict) -> dict:
    """Worker function: run one experiment in a child process.

    Args:
        config_kwargs: Dict of keyword arguments for LGCoTrainConfig.

    Returns:
        Dict with keys: event, budget, seed_set, status, result.
    """
    from .config import LGCoTrainConfig
    from .trainer import LGCoTrainer

    config = LGCoTrainConfig(**config_kwargs)
    event = config.event
    budget = config.budget
    seed_set = config.seed_set

    try:
        trainer = LGCoTrainer(config)
        result = trainer.run()
        return {
            "event": event,
            "budget": budget,
            "seed_set": seed_set,
            "status": "done",
            "result": result,
        }
    except Exception as e:
        logging.getLogger("lg_cotrain").error(
            f"Experiment {event} budget={budget} seed={seed_set} failed: {e}"
        )
        return {
            "event": event,
            "budget": budget,
            "seed_set": seed_set,
            "status": "failed",
            "result": None,
        }
    finally:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def run_experiments_parallel(
    experiment_configs: List[dict],
    num_gpus: int = 2,
    on_experiment_done: Optional[Callable] = None,
) -> List[dict]:
    """Run multiple experiments in parallel across GPUs.

    Args:
        experiment_configs: List of dicts, each containing keyword
            arguments for LGCoTrainConfig.  The ``device`` key will be
            set automatically based on round-robin GPU assignment.
        num_gpus: Number of GPUs to use for parallel execution.
        on_experiment_done: Optional callback(event, budget, seed_set, status)
            invoked in the parent process after each experiment completes.

    Returns:
        List of outcome dicts (with keys event, budget, seed_set, status,
        result), in the same order as *experiment_configs*.
    """
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")

    ctx = mp.get_context("spawn")
    results_map: dict = {}  # index -> outcome
    config_queue = list(range(len(experiment_configs)))  # indices

    with ProcessPoolExecutor(
        max_workers=num_gpus, mp_context=ctx
    ) as executor:
        active: dict = {}  # future -> (config_idx, gpu_id)

        # Seed the pool: one task per GPU
        for gpu_id in range(min(num_gpus, len(config_queue))):
            idx = config_queue.pop(0)
            experiment_configs[idx]["device"] = f"cuda:{gpu_id}"
            future = executor.submit(
                _run_single_experiment, experiment_configs[idx]
            )
            active[future] = (idx, gpu_id)

        while active:
            for future in as_completed(active):
                idx, gpu_id = active.pop(future)

                try:
                    outcome = future.result()
                except Exception as e:
                    cfg = experiment_configs[idx]
                    outcome = {
                        "event": cfg.get("event", "?"),
                        "budget": cfg.get("budget", "?"),
                        "seed_set": cfg.get("seed_set", "?"),
                        "status": "failed",
                        "result": None,
                    }
                    logger.error(
                        f"Process-level failure for experiment {idx}: {e}"
                    )

                results_map[idx] = outcome

                if on_experiment_done is not None:
                    on_experiment_done(
                        outcome["event"],
                        outcome["budget"],
                        outcome["seed_set"],
                        outcome["status"],
                    )

                # Submit next experiment to the same GPU
                if config_queue:
                    next_idx = config_queue.pop(0)
                    experiment_configs[next_idx]["device"] = f"cuda:{gpu_id}"
                    new_future = executor.submit(
                        _run_single_experiment, experiment_configs[next_idx]
                    )
                    active[new_future] = (next_idx, gpu_id)

                break  # process one at a time to reassign GPU immediately

    # Return results in original order
    return [results_map[i] for i in range(len(experiment_configs))]
