#!/usr/bin/env python3
"""generate_sweep.py

Generate a W&B sweep configuration YAML for the co-training project.

This generator follows the sweep specification in cotrain_notes.md:

- Sweep: lr, num_epochs, epoch_patience
- Fixed (but tracked): dataset, hf_model_id_short, plm_id, metric_combination,
  seed, pseudo_label_dir, event, lbcl, data_dir, cuda_devices
- Runs a wrapper script (default: run_sweep_wrapper.py)

IMPORTANT:
W&B sweep "command" templates only reliably substitute the special placeholders
(${env}, ${interpreter}, ${program}, ${args}). To ensure *all* parameters are
passed correctly, this generator uses `${args}` and defines the full argument
set under `parameters:` (fixed params use `value:`).
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

import yaml


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a W&B sweep YAML for co-training hyperparameter tuning."
    )

    # Required per the notes
    p.add_argument("--event", type=str, required=True, help="Event name")
    p.add_argument("--lbcl", type=str, required=True, help="Labeled count per class")

    # Output
    p.add_argument(
        "--output",
        type=str,
        default="sweep.yaml",
        help="Output YAML file (default: sweep.yaml)",
    )

    # Entrypoint to execute under the sweep
    p.add_argument(
        "--program",
        type=str,
        default="run_sweep_wrapper.py",
        help="Entrypoint script that the sweep runs (default: run_sweep_wrapper.py)",
    )

    # Fixed (but configurable) parameters from the notes
    p.add_argument("--dataset", type=str, default="humaid", help="Dataset name")
    p.add_argument(
        "--hf_model_id_short",
        type=str,
        default="N/A",
        help="Short HF model id (default: N/A)",
    )
    p.add_argument(
        "--plm_id",
        nargs='+',
        default=["clip", "bert-tweet", "roberta-base", "bert-base", "deberta-base", "roberta-large"],
        help="Backbone PLM id(s)"
    )
    p.add_argument(
        "--metric_combination",
        type=str,
        default="cv",
        help="Metric combination (default: cv)",
    )

    # In the notes, --setup_local_logging is part of the standard run command.
    # W&B `${args}` cannot emit a bare flag reliably for argparse store_true,
    # so we include it explicitly by default and allow disabling.
    p.add_argument(
        "--no_setup_local_logging",
        action="store_false",
        dest="setup_local_logging",
        help="Disable --setup_local_logging in the generated command",
    )
    p.set_defaults(setup_local_logging=True)

    p.add_argument("--seed", type=int, default=1234, help="Random seed")
    p.add_argument(
        "--pseudo_label_dir",
        type=str,
        default="anh_4o",
        help="Directory containing LLM pseudo labels",
    )
    p.add_argument("--data_dir", type=str, default="../../data", help="Base data directory")
    p.add_argument(
        "--cuda_devices",
        type=str,
        default="0",
        help="CUDA devices (comma-separated), e.g., 0,1",
    )

    # Sweep behavior
    p.add_argument(
        "--method",
        type=str,
        default="grid",
        choices=["grid", "random", "bayes"],
        help="W&B sweep method (default: grid)",
    )
    p.add_argument(
        "--metric_name",
        type=str,
        default="avg_f1",
        help="Metric name to optimize (default: avg_f1)",
    )
    p.add_argument(
        "--metric_goal",
        type=str,
        default="maximize",
        choices=["maximize", "minimize"],
        help="Metric goal (default: maximize)",
    )


    return p.parse_args()


def generate_sweep_yaml(
    *,
    program: str,
    event: str,
    lbcl: str,
    dataset: str,
    hf_model_id_short: str,
    plm_id: List[str],
    metric_combination: str,
    setup_local_logging: bool,
    seed: int,
    pseudo_label_dir: str,
    data_dir: str,
    cuda_devices: str,
    method: str,
    metric_name: str,
    metric_goal: str,

) -> Dict[str, Any]:
    """Build a W&B sweep configuration dict."""

    # Tunable hyperparameters (the ones marked with '?' in the notes)
    parameters: Dict[str, Any] = {
        "lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3},
        "num_epochs": {"distribution": "int_uniform", "min": 5, "max": 20},
        "epoch_patience": {"distribution": "int_uniform", "min": 3, "max": 10},
        "weight_decay": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
        "max_grad_norm": {"distribution": "uniform", "min": 1, "max": 10},
        "batch_size": {"distribution": "int_uniform", "min": 8, "max": 64},

    }

    if len(plm_id) > 1:
        parameters["plm_id"] = {"values": plm_id, "distribution": "categorical"}
    else:
        parameters["plm_id"] = {"value": plm_id[0]}

    # Fixed parameters that should still be tracked (and passed) via `${args}`
    parameters.update({
        "dataset": {"value": dataset},
        "hf_model_id_short": {"value": hf_model_id_short},
        "metric_combination": {"value": metric_combination},
        "seed": {"value": seed},
        "pseudo_label_dir": {"value": pseudo_label_dir},
        "event": {"value": event},
        "lbcl": {"value": lbcl},
        "data_dir": {"value": data_dir},
        "cuda_devices": {"value": cuda_devices},
    })

    # Critical fix:
    # Use `${args}` so W&B reliably passes *all* parameters/values.
    command: List[str] = ["${env}", "${interpreter}", "${program}"]
    if setup_local_logging:
        command.append("--setup_local_logging")
    command.append("${args}")

    sweep_config: Dict[str, Any] = {
        "program": program,
        "method": method,
        "metric": {"name": metric_name, "goal": metric_goal},
        "parameters": parameters,
        "command": command,
    }
    return sweep_config


def main() -> None:
    args = _parse_args()

    sweep = generate_sweep_yaml(
        program=args.program,
        event=args.event,
        lbcl=args.lbcl,
        dataset=args.dataset,
        hf_model_id_short=args.hf_model_id_short,
        plm_id=args.plm_id,
        metric_combination=args.metric_combination,
        setup_local_logging=args.setup_local_logging,
        seed=args.seed,
        pseudo_label_dir=args.pseudo_label_dir,
        data_dir=args.data_dir,
        cuda_devices=args.cuda_devices,
        method=args.method,
        metric_name=args.metric_name,
        metric_goal=args.metric_goal,

    )

    with open(args.output, "w", encoding="utf-8") as f:
        # Keep YAML readable and stable
        yaml.safe_dump(sweep, f, sort_keys=False)

    print(f"Sweep configuration saved to {args.output}")


if __name__ == "__main__":
    main()

# python generate_sweep.py --event <event> --lbcl <lbcl> --output <event_lbcl_sweep.yaml> --no_setup_local_logging --cuda_devices <cuda_devices> --method <method> --metric_name <metric_name>