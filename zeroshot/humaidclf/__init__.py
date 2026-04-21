"""
humaidclf: Zero-shot tweet classification pipeline for HumAID-style labels
using OpenAI Chat Completions + Structured Outputs and the Batch API.

This package exposes:
- I/O helpers (load TSVs, create per-run directories)
- Prompt assets (label order, system prompt, user message builder)
- Batch workflow (sync sanity-check, build requests.jsonl, submit/poll/download/parse)
- Evaluation utilities (macro-F1, analysis charts & confusion matrices)
- High-level runner (one-call end-to-end or resume)
"""

# =========================
# IO utilities
# =========================
from .io import (
    load_tsv,        # Read a TSV and normalize columns to: tweet_id, tweet_text, class_label (optional)
    plan_run_dirs,   # Create a timestamped run directory layout (requests.jsonl / outputs.jsonl / predictions.csv / analysis/)
)

# =========================
# Prompt assets
# =========================
from .prompts import (
    LABELS,          # Canonical label order (matches your README & structured output enum)
    SYSTEM_PROMPT,   # Short system prompt to enforce “one label + JSON schema” behavior
    make_user_message,  # Helper to build the user message from rules + tweet text (optional convenience)
)

# =========================
# Batch (OpenAI API workflow)
# =========================
from .batch import (
    sync_test_sample,       # Small synchronous sanity check on N examples (catches prompt/schema issues early)
    build_requests_jsonl_S, # Build requests.jsonl for the Batch API (one request per tweet)
    upload_file_for_batch,  # POST /files (purpose="batch") — upload the requests.jsonl
    create_batch,           # POST /batches — create a batch job for /v1/chat/completions
    get_batch,              # GET /batches/{id} — retrieve batch status/metadata
    wait_for_batch,         # Poll GET /batches/{id} until completed/failed/cancelled
    download_file_content,  # GET /files/{id}/content — download batch output JSONL
    parse_outputs_S_to_df,  # Parse outputs.jsonl back into a DataFrame (re-attach tweet_text/class_label when provided)
)

# =========================
# Evaluation utilities
# =========================
from .eval import (
    macro_f1,                       # Macro-averaged F1 over available truth labels
    analyze_and_export_mistakes,    # Save mistakes.csv and charts (counts + row-norm confusion, per-class metrics, summary)
)

# =========================
# Token calculations
# =========================
from .budget import (
    get_token_encoder,
    estimate_request_tokens,
    estimate_dataset_tokens,
    build_token_index,
    shard_dataset_by_tokens,
)

# =========================
# API key switcher
# =========================
from .batch import (
    set_api_key_env, set_api_key_value, get_active_api_key_label,
    use_api_key_env, use_api_key_value,
)

# =========================
# High-level runners
# =========================
from .runner import (
    run_experiment,     # End-to-end: load → dry-run → build → submit → wait → download → parse → (optional) analysis
    resume_experiment,  # Resume a submitted run using batch_meta.json (download/parse/analyze after it completes)
)

# =========================
# Stratified splitting (label-balanced shards)
# =========================
# Utility to split a DataFrame into K shards while preserving class ratios.
# This is kept separate from the runner so it can be reused in analysis/ablations.
try:
    from .stratify import (
        stratified_k_shards,  # Returns a list of (df_shard, indices) with per-class proportions preserved
    )
except Exception:
    # Optional module; only exported if available
    pass

# =========================
# Sharded runner (token/size constrained orchestration)
# =========================
# Orchestrates multiple stratified shards end-to-end and then merges the shard predictions.
# Useful when token budgets / provider limits require slicing large events.
try:
    from .runner_sharded import (
        run_experiment_sharded,  # Same contract as run_experiment, but runs per-shard under the hood
    )
except Exception:
    # Optional module; only exported if available
    pass


__all__ = [
    # IO
    "load_tsv", "plan_run_dirs",

    # Prompts
    "LABELS", "SYSTEM_PROMPT", "make_user_message",

    # Batch
    "sync_test_sample", "build_requests_jsonl_S",
    "upload_file_for_batch", "create_batch", "get_batch", "wait_for_batch",
    "download_file_content", "parse_outputs_S_to_df",

    # Eval
    "macro_f1", "analyze_and_export_mistakes",

    # Runners
    "run_experiment", "resume_experiment",

    # Tokens
    "get_token_encoder",
    "estimate_request_tokens",
    "estimate_dataset_tokens",
    "build_token_index",
    "shard_dataset_by_tokens",

    # API key
    "set_api_key_env", "set_api_key_value", "get_active_api_key_label",
    "use_api_key_env", "use_api_key_value",
]

# Conditionally expose optional modules if present
if "stratified_k_shards" in globals():
    __all__.append("stratified_k_shards")

if "run_experiment_sharded" in globals():
    __all__.append("run_experiment_sharded")
