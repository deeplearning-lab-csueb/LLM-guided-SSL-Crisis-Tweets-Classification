"""Generate self-trained teacher pseudo-labels for Ablation V.

For each (event, budget, seed_set) cell, this script:

1. Loads the labeled split (data/original/{event}/labeled_{budget}_set{seed}.tsv)
2. Trains a small BERTweet (or other) classifier on it for ``--epochs`` epochs
3. Runs the trained classifier over the unlabeled split
4. Writes per-cell pseudo-label CSVs to
   ``data/pseudo-labelled/self-trained/{event}/labeled_{budget}_set{seed}_pseudo.csv``

The output file name and directory match what ``LGCoTrainConfig`` expects when
``pseudo_label_source="self-trained"`` (see ``PER_SPLIT_SOURCES`` in
``lg_cotrain/config.py``).

Usage::

    python -m lg_cotrain.generate_selftrained_teacher \\
        --events all \\
        --budgets 5 50 \\
        --seed-sets 1 2 3 \\
        --num-gpus 2

Resume support: cells whose output CSV already exists are skipped unless
``--force`` is passed. So you can run this incrementally - first for
``--budgets 5 50`` (the committed grid), and later for ``--budgets 10 25``
(the optional middle budgets).

This is a small standalone script - it deliberately does NOT live inside
the LGCoTrain trainer. It only trains one model per cell (no co-training,
no Phase 2/3), so it is much faster than a full LG-CoTrain run.
"""

import argparse
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

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

DEFAULT_BUDGETS = [5, 10, 25, 50]
DEFAULT_SEEDS = [1, 2, 3]


def _output_path(data_root: Path, event: str, budget: int, seed_set: int) -> Path:
    """Compute the per-cell pseudo-label output path.

    Must match what ``LGCoTrainConfig.__post_init__`` builds when
    ``pseudo_label_source == "self-trained"``.
    """
    return (
        data_root
        / "pseudo-labelled"
        / "self-trained"
        / event
        / f"labeled_{budget}_set{seed_set}_pseudo.csv"
    )


def train_and_predict_one_cell(
    event: str,
    budget: int,
    seed_set: int,
    data_root: Path,
    model_name: str = "vinai/bertweet-base",
    epochs: int = 10,
    lr: float = 2e-5,
    batch_size: int = 32,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_seq_length: int = 128,
    device: Optional[str] = None,
) -> dict:
    """Train one teacher and write its pseudo-labels.

    Returns a dict with ``status``, ``cell``, ``output_path``, ``n_unlabeled``,
    and (on success) summary stats from the inference pass.
    """
    import torch
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    from lg_cotrain.data_loading import (
        CLASS_LABELS,
        load_tsv,
        TweetDataset,
        detect_event_classes,
        build_label_encoder,
    )

    cell = f"{event}/{budget}_set{seed_set}"

    # Resolve device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    # Set seeds for reproducibility within the cell
    seed = budget * 1000 + seed_set
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    original_dir = data_root / "original" / event
    labeled_path = original_dir / f"labeled_{budget}_set{seed_set}.tsv"
    unlabeled_path = original_dir / f"unlabeled_{budget}_set{seed_set}.tsv"

    if not labeled_path.exists():
        return {"status": "missing_labeled", "cell": cell, "path": str(labeled_path)}
    if not unlabeled_path.exists():
        return {"status": "missing_unlabeled", "cell": cell, "path": str(unlabeled_path)}

    # Load data and detect the event's class subset (matches what LG-CoTrain does)
    df_labeled = load_tsv(str(labeled_path))
    df_unlabeled = load_tsv(str(unlabeled_path))
    event_classes = detect_event_classes(df_labeled)
    label2id, id2label = build_label_encoder(event_classes)
    num_labels = len(event_classes)

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    model.to(device_obj)

    # Datasets
    train_texts = df_labeled["tweet_text"].tolist()
    train_labels = [label2id[c] for c in df_labeled["class_label"].tolist()]
    train_ds = TweetDataset(train_texts, train_labels, tokenizer, max_seq_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Optimizer + linear LR schedule with warmup
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device_obj)
            attention_mask = batch["attention_mask"].to(device_obj)
            labels = batch["labels"].to(device_obj)

            optimizer.zero_grad()
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += out.loss.item()
            n_batches += 1

    # Inference over the unlabeled set.
    #
    # The unlabeled TSV ships with gold class_labels (it is the held-out portion
    # of the original labeled dataset), but we MUST NOT let those leak into
    # either training or inference for this teacher script:
    #   - Training: we only call train_loader (which iterates the labeled split).
    #     The unlabeled set is never used to update model weights.
    #   - Inference: we deliberately only forward (input_ids, attention_mask) to
    #     the model. The "labels" tensor in the batch is constructed from the
    #     gold class_label so the dataset stays well-typed, but it is never
    #     passed to model(...) and never read by the prediction logic below.
    # The gold labels also do not appear in the output CSV - only the model's
    # predicted_label and softmax confidence are written.
    model.eval()
    unlabeled_texts = df_unlabeled["tweet_text"].tolist()
    # Some unlabeled rows may carry classes that aren't in the small labeled
    # split (event_classes is detected from df_labeled only). Fall back to 0
    # for those - the value is never used by the model.
    unlabeled_labels = [
        label2id[c] if c in label2id else 0
        for c in df_unlabeled["class_label"].tolist()
    ]
    unlabeled_ds = TweetDataset(unlabeled_texts, unlabeled_labels, tokenizer, max_seq_length)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size * 2, shuffle=False)

    all_pred_labels: List[str] = []
    all_confidences: List[float] = []

    with torch.no_grad():
        for batch in unlabeled_loader:
            input_ids = batch["input_ids"].to(device_obj)
            attention_mask = batch["attention_mask"].to(device_obj)
            # NOTE: batch["labels"] is intentionally NOT forwarded to the model.
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(out.logits, dim=-1)
            confs, preds = probs.max(dim=-1)
            for p, c in zip(preds.cpu().numpy(), confs.cpu().numpy()):
                all_pred_labels.append(id2label[int(p)])
                all_confidences.append(float(c))

    # Build the output DataFrame in the same schema as the existing GPT-4o
    # pseudo-label CSVs (tweet_id, tweet_text, predicted_label, confidence).
    out_df = pd.DataFrame({
        "tweet_id": df_unlabeled["tweet_id"].astype(str).values,
        "tweet_text": df_unlabeled["tweet_text"].values,
        "predicted_label": all_pred_labels,
        "confidence": all_confidences,
    })

    output_path = _output_path(data_root, event, budget, seed_set)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    # Free GPU memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "status": "ok",
        "cell": cell,
        "output_path": str(output_path),
        "n_labeled": len(df_labeled),
        "n_unlabeled": len(df_unlabeled),
        "num_classes": num_labels,
        "mean_confidence": round(float(np.mean(all_confidences)), 4),
    }


def _worker(kwargs: dict) -> dict:
    """Multiprocessing worker - re-imports inside the child process."""
    try:
        return train_and_predict_one_cell(**kwargs)
    except Exception as e:
        import traceback
        return {
            "status": "failed",
            "cell": f"{kwargs['event']}/{kwargs['budget']}_set{kwargs['seed_set']}",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def generate_all(
    events: List[str],
    budgets: List[int],
    seed_sets: List[int],
    data_root: Path,
    model_name: str = "vinai/bertweet-base",
    epochs: int = 10,
    num_gpus: int = 1,
    force: bool = False,
    **train_kwargs,
) -> List[dict]:
    """Generate teacher pseudo-labels for the given grid.

    Cells whose output CSV already exists are skipped unless ``force=True``.
    Multi-GPU dispatch via ``ProcessPoolExecutor`` (spawn context, one worker
    per GPU). For single-GPU or CPU runs (``num_gpus=1``), runs sequentially
    in the parent process.
    """
    grid: List[dict] = []
    skipped: List[dict] = []

    for event in events:
        for budget in budgets:
            for seed_set in seed_sets:
                output_path = _output_path(data_root, event, budget, seed_set)
                cell = f"{event}/{budget}_set{seed_set}"

                if output_path.exists() and not force:
                    skipped.append({
                        "status": "skipped",
                        "cell": cell,
                        "output_path": str(output_path),
                    })
                    print(f"  SKIP   {cell}  (already exists at {output_path.relative_to(data_root)})")
                    continue

                grid.append({
                    "event": event,
                    "budget": budget,
                    "seed_set": seed_set,
                    "data_root": data_root,
                    "model_name": model_name,
                    "epochs": epochs,
                    **train_kwargs,
                })

    print()
    print(f"Total cells in grid: {len(grid) + len(skipped)}")
    print(f"  Skipped (already done): {len(skipped)}")
    print(f"  To run:                 {len(grid)}")
    print()

    if not grid:
        print("Nothing to do.")
        return skipped

    results: List[dict] = list(skipped)

    if num_gpus <= 1:
        # Sequential
        device = "cuda:0" if num_gpus == 1 else "cpu"
        for i, kwargs in enumerate(grid, 1):
            kwargs["device"] = device
            print(f"  [{i}/{len(grid)}] training teacher for {kwargs['event']}/{kwargs['budget']}_set{kwargs['seed_set']} ...")
            t0 = time.time()
            r = _worker(kwargs)
            elapsed = time.time() - t0
            r["elapsed_sec"] = round(elapsed, 1)
            results.append(r)
            if r["status"] == "ok":
                print(f"    OK   {r['cell']}  ({elapsed:.1f}s, mean_conf={r['mean_confidence']:.3f}, n_unlabeled={r['n_unlabeled']})")
            else:
                print(f"    FAIL {r['cell']}  -> {r.get('error', r['status'])}")
        return results

    # Multi-GPU parallel
    print(f"Dispatching {len(grid)} teacher trainings across {num_gpus} GPUs (spawn context)...")
    ctx = mp.get_context("spawn")
    queue = list(range(len(grid)))

    with ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
        active: dict = {}
        for gpu_id in range(min(num_gpus, len(queue))):
            idx = queue.pop(0)
            grid[idx]["device"] = f"cuda:{gpu_id}"
            future = executor.submit(_worker, grid[idx])
            active[future] = (idx, gpu_id)

        n_done = 0
        while active:
            for future in as_completed(active):
                idx, gpu_id = active.pop(future)
                try:
                    r = future.result()
                except Exception as e:
                    r = {
                        "status": "failed",
                        "cell": f"{grid[idx]['event']}/{grid[idx]['budget']}_set{grid[idx]['seed_set']}",
                        "error": str(e),
                    }
                results.append(r)
                n_done += 1
                if r["status"] == "ok":
                    print(f"  [{n_done}/{len(grid)}] OK   {r['cell']}  (mean_conf={r['mean_confidence']:.3f}, n_unlabeled={r['n_unlabeled']})")
                else:
                    print(f"  [{n_done}/{len(grid)}] FAIL {r['cell']}  -> {r.get('error', r['status'])}")

                # Submit next from queue
                if queue:
                    next_idx = queue.pop(0)
                    grid[next_idx]["device"] = f"cuda:{gpu_id}"
                    new_future = executor.submit(_worker, grid[next_idx])
                    active[new_future] = (next_idx, gpu_id)
                break  # process one at a time so we can immediately re-dispatch

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate self-trained teacher pseudo-labels for Ablation V. "
            "Trains one classifier per (event, budget, seed_set) on the small "
            "labeled split and writes per-cell pseudo-label CSVs."
        )
    )
    parser.add_argument(
        "--events", type=str, nargs="*", default=None,
        help="Events to process. Pass 'all' or omit for all 10 events.",
    )
    parser.add_argument(
        "--budgets", type=int, nargs="*", default=None,
        help="Budgets to process (default: 5 10 25 50).",
    )
    parser.add_argument(
        "--seed-sets", type=int, nargs="*", default=None,
        help="Seed sets to process (default: 1 2 3).",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Path to the data/ directory (default: sibling of lg_cotrain/).",
    )
    parser.add_argument(
        "--model-name", type=str, default="vinai/bertweet-base",
        help="HuggingFace model name for the teacher (default: vinai/bertweet-base).",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument(
        "--num-gpus", type=int, default=1,
        help="Number of GPUs for parallel teacher training (default: 1 = sequential).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate teachers even if the output CSV already exists.",
    )
    args = parser.parse_args()

    # Resolve events
    if args.events is None or args.events == ["all"]:
        events = list(ALL_EVENTS)
    else:
        events = args.events
    budgets = args.budgets if args.budgets is not None else DEFAULT_BUDGETS
    seed_sets = args.seed_sets if args.seed_sets is not None else DEFAULT_SEEDS

    if args.data_root is None:
        data_root = Path(__file__).parent.parent / "data"
    else:
        data_root = Path(args.data_root)

    # Configure stdout logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info(
        f"generate_selftrained_teacher: events={len(events)}, budgets={budgets}, "
        f"seed_sets={seed_sets}, model={args.model_name}, num_gpus={args.num_gpus}"
    )

    start = time.time()
    results = generate_all(
        events=events,
        budgets=budgets,
        seed_sets=seed_sets,
        data_root=data_root,
        model_name=args.model_name,
        epochs=args.epochs,
        num_gpus=args.num_gpus,
        force=args.force,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_seq_length=args.max_seq_length,
    )
    elapsed = time.time() - start

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    n_failed = sum(1 for r in results if r["status"] == "failed")

    print()
    print("=" * 70)
    print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  ok:      {n_ok}")
    print(f"  skipped: {n_skipped}")
    print(f"  failed:  {n_failed}")
    if n_failed:
        print()
        print("Failed cells:")
        for r in results:
            if r["status"] == "failed":
                print(f"  {r['cell']}: {r.get('error', '?')}")


if __name__ == "__main__":
    main()
