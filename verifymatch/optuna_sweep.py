import argparse, os, re, shlex, subprocess, json, pathlib, sys, time
import optuna

# --- helpers ---------------------------------------------------------------
def write_text(path: pathlib.Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def tail(text: str, n: int = 200) -> str:
    lines = text.splitlines()[-n:]
    return "\n".join(lines)

def run_cmd(cmd_list):
    """Run a command and return (rc, stdout)."""
    p = subprocess.run(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
    )
    return p.returncode, p.stdout

# Parse metrics: prefer macro-F1; fallback to accuracy
MACRO_F1_RES = [
    re.compile(r"macro[-_\s]?f1\s*=\s*([0-9.]+)", re.I),
    re.compile(r"f1[_\s-]?macro\s*=\s*([0-9.]+)", re.I),
    re.compile(r"macro\s*f1\s*[:=]\s*([0-9.]+)", re.I),
]
ACC_RE = re.compile(r"accuracy\s*=\s*([0-9.]+)", re.I)

def parse_macro_f1(text: str):
    for rx in MACRO_F1_RES:
        m = rx.search(text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None

def parse_accuracy(text: str):
    m = ACC_RE.search(text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None

# --- main ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # paths / fixed args passed through to train.py
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--train_py", required=True, help="Path to train.py")
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--task", required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--max_seq_length", type=int, default=256)

    # data (SSL expects labeled + unlabeled; dev used for model selection; test optional)
    ap.add_argument("--labeled_train_path", required=True)
    ap.add_argument("--unlabeled_train_path", required=True)
    ap.add_argument("--dev_path", required=True)
    ap.add_argument("--test_path", required=True)

    # study controls
    ap.add_argument("--study_name", default="verify_match_hpft")
    ap.add_argument("--storage", default=None, help="e.g., sqlite:///vmatch_optuna.db")
    ap.add_argument("--n_trials", type=int, default=20)        # small dataset → fewer trials
    ap.add_argument("--timeout", type=int, default=None)       # seconds
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="./vmatch_hpft_runs")

    args = ap.parse_args()
    pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)

    base_cmd = [
        args.python, args.train_py,
        "--ssl", "--mixup",
        "--do_train",
        "--model", args.model,
        "--task", args.task,
        "--device", str(args.device),
        "--max_seq_length", str(args.max_seq_length),
        "--labeled_train_path", args.labeled_train_path,
        "--unlabeled_train_path", args.unlabeled_train_path,
        "--dev_path", args.dev_path,
        "--pseudo_label_by_normalized",
    ]

    def objective(trial: optuna.Trial):
        # Compact search space for 1–6.5k rows
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        batch = trial.suggest_categorical("batch_size", [16, 32])
        u_batch = trial.suggest_categorical("unlabeled_batch_size", [32, 64])
        epochs = trial.suggest_int("epochs", 2, 4)
        weight_decay = trial.suggest_categorical("weight_decay", [0.0, 0.01])
        T = trial.suggest_categorical("T", [0.3, 0.5, 0.7])
        mixup_w = trial.suggest_categorical("mixup_loss_weight", [0.5, 1.0, 1.5])

        trial_dir = pathlib.Path(args.outdir) / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        ckpt = str(trial_dir / "model.pt")
        out_pred = str(trial_dir / "preds.json")
        train_log_path = trial_dir / "train.log"
        eval_log_path = trial_dir / "eval.log"

        # 1) TRAIN
        train_cmd = base_cmd + [
            "--epochs", str(epochs),
            "--batch_size", str(batch),
            "--unlabeled_batch_size", str(u_batch),
            "--learning_rate", str(lr),
            "--weight_decay", str(weight_decay),
            "--T", str(T),
            "--mixup_loss_weight", str(mixup_w),
            "--ckpt_path", ckpt,
            "--output_path", out_pred,  # produced on evaluate
        ]
        trial.set_user_attr("train_cmd", " ".join(shlex.quote(x) for x in train_cmd))
        trial.set_user_attr("train_log_path", str(train_log_path))
        rc, train_log = run_cmd(train_cmd)
        write_text(train_log_path, train_log)

        if rc != 0:
            print("\n===== TRAIN FAILED (rc={}) =====".format(rc))
            print("CMD:", " ".join(shlex.quote(x) for x in train_cmd))
            print("---- train.log (last 200 lines) ----")
            print(tail(train_log, 200))
            print("---- full log:", train_log_path, "----\n")
            raise optuna.TrialPruned(f"train.py failed (rc={rc}) — see {train_log_path}")

        # 2) EVALUATE on DEV (by pointing test_path at dev_path)
        eval_cmd = [
            args.python, args.train_py,
            "--ssl", "--mixup", "--do_evaluate",
            "--model", args.model, "--task", args.task,
            "--device", str(args.device),
            "--max_seq_length", str(args.max_seq_length),
            "--ckpt_path", ckpt,
            "--test_path", args.dev_path,
            "--batch_size", str(batch),
            "--output_path", out_pred,
        ]
        trial.set_user_attr("eval_cmd", " ".join(shlex.quote(x) for x in eval_cmd))
        trial.set_user_attr("eval_log_path", str(eval_log_path))
        rc2, eval_log = run_cmd(eval_cmd)
        write_text(eval_log_path, eval_log)

        if rc2 != 0:
            print("\n===== EVAL FAILED (rc={}) =====".format(rc2))
            print("CMD:", " ".join(shlex.quote(x) for x in eval_cmd))
            print("---- eval.log (last 200 lines) ----")
            print(tail(eval_log, 200))
            print("---- full log:", eval_log_path, "----\n")
            raise optuna.TrialPruned(f"evaluation failed (rc={rc2}) — see {eval_log_path}")

        # --- objective: macro-F1 (fallback to accuracy) ---
        f1 = parse_macro_f1(eval_log)
        if f1 is not None:
            objective_value = f1
        else:
            acc = parse_accuracy(eval_log)
            objective_value = acc if acc is not None else 0.0

        trial.report(objective_value, step=0)
        return objective_value

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(3, args.n_trials // 5))

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True if args.storage else False,
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    """
    Create that "best" model by re-running train.py with the best params on
    labeled + unlabeled data, and evaluating on test.

    Save the files that would be in a trial directory, but in: {outdir}/best_params/
    Create a file called "best_cfg.json" with:
    [best] cfg=({config}) f1=({test_macro_f1}) acc=({test_accuracy})
    """

    # Save best params snapshot
    best = {
        "value": study.best_value,
        "params": study.best_params,
        "user_attrs": study.best_trial.user_attrs,
        "number": study.best_trial.number,
    }
    best_path = pathlib.Path(args.outdir) / "best_params.json"
    write_text(best_path, json.dumps(best, indent=2))
    print("\nBest macro-F1/obj:", best["value"])
    print("Best params:", best["params"])
    print("Saved:", best_path)

    # === Re-train using the best params and evaluate on TEST ===
    best_dir = pathlib.Path(args.outdir) / "best_params"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = str(best_dir / "model.pt")
    best_pred = str(best_dir / "preds.json")
    best_train_log = best_dir / "train.log"
    best_eval_log = best_dir / "eval.log"

    # Build TRAIN command with best params
    bp = study.best_params
    train_best_cmd = [
        args.python, args.train_py,
        "--ssl", "--mixup", "--do_train",
        "--model", args.model, "--task", args.task,
        "--device", str(args.device), "--max_seq_length", str(args.max_seq_length),
        "--labeled_train_path", args.labeled_train_path,
        "--unlabeled_train_path", args.unlabeled_train_path,
        "--dev_path", args.dev_path,
        "--pseudo_label_by_normalized",
        "--epochs", str(bp.get("epochs")),
        "--batch_size", str(bp.get("batch_size")),
        "--unlabeled_batch_size", str(bp.get("unlabeled_batch_size")),
        "--learning_rate", str(bp.get("learning_rate")),
        "--weight_decay", str(bp.get("weight_decay")),
        "--T", str(bp.get("T")),
        "--mixup_loss_weight", str(bp.get("mixup_loss_weight")),
        "--ckpt_path", best_ckpt,
        "--output_path", best_pred,
    ]
    rc_train_best, log_train_best = run_cmd(train_best_cmd)
    write_text(best_train_log, log_train_best)
    if rc_train_best != 0:
        print("\n===== BEST TRAIN FAILED (rc={}) =====".format(rc_train_best))
        print("CMD:", " ".join(shlex.quote(x) for x in train_best_cmd))
        print("---- train.log (last 200 lines) ----")
        print(tail(log_train_best, 200))
        print("---- full log:", best_train_log, "----\n")
        raise SystemExit(1)

    # Build EVAL command on TEST
    eval_best_cmd = [
        args.python, args.train_py,
        "--ssl", "--mixup", "--do_evaluate",
        "--model", args.model, "--task", args.task,
        "--device", str(args.device), "--max_seq_length", str(args.max_seq_length),
        "--ckpt_path", best_ckpt,
        "--test_path", args.test_path,  # <-- evaluate on TEST here
        "--batch_size", str(bp.get("batch_size")),
        "--output_path", best_pred,
    ]
    rc_eval_best, log_eval_best = run_cmd(eval_best_cmd)
    write_text(best_eval_log, log_eval_best)
    if rc_eval_best != 0:
        print("\n===== BEST EVAL FAILED (rc={}) =====".format(rc_eval_best))
        print("CMD:", " ".join(shlex.quote(x) for x in eval_best_cmd))
        print("---- eval.log (last 200 lines) ----")
        print(tail(log_eval_best, 200))
        print("---- full log:", best_eval_log, "----\n")
        raise SystemExit(1)

    # Parse TEST metrics
    test_f1 = parse_macro_f1(log_eval_best)
    test_acc = parse_accuracy(log_eval_best)
    test_f1 = 0.0 if test_f1 is None else test_f1
    test_acc = 0.0 if test_acc is None else test_acc

    # Write best_cfg.json with the requested single-line format
    cfg_line = f"[best] cfg=({json.dumps(bp)}) f1=({test_f1}) acc=({test_acc})\n"
    write_text(best_dir / "best_cfg.json", cfg_line)

    print(f"\nWrote best artifacts to: {best_dir}")
    print(f"Test macro-F1: {test_f1:.4f} | Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
