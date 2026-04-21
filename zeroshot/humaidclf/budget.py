# humaidclf/budget.py
"""
Token budgeting utilities for OpenAI Batch runs (filtered-labels setup).

Features:
- Estimate tokens per request:
    system + user (rules + tweet + ALLOWED LABELS) + response_format (DYNAMIC SCHEMA) + output allowance
- Estimate total tokens for a dataset (with sampling)
- Build a token index across datasets (decide which fit under a cap)
- Optional sharding: split a TSV into token-budgeted chunks

Notes:
- We now use the same dynamic schema as batch.py (_make_schema(labels)),
  where `labels` = the labels PRESENT in the dataset's ground truth (truth-only scope).
- This keeps budgeting aligned with the actual request payload sizes.
- Uses tiktoken if available; otherwise falls back to a rough character-based heuristic.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Dict, Optional
import math
import json
import pandas as pd

# Project imports
from .io import load_tsv
from .prompts import SYSTEM_PROMPT, filter_rules_by_labels
try:
    # Canonical label ordering (optional). If available, we intersect with it to
    # guard against stray/typo labels in raw data.
    from .prompts import LABELS as CANON_LABELS  # type: ignore
except Exception:
    CANON_LABELS = None

# Dynamic schema builder from batch.py (no longer use a global SCHEMA_S)
from .batch import _make_schema

# Optional dependency for accurate token counting
try:
    import tiktoken
except Exception:
    tiktoken = None


# ---------- Tokenizer helpers ----------

def get_token_encoder(model: str):
    """Return a tiktoken encoder for the model; fallback to cl100k_base, else None."""
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoder=None) -> int:
    """
    Count tokens using tiktoken (if provided) or a simple char-based heuristic.
    Heuristic assumes ~4 chars/token for English-like text.
    """
    if not text:
        return 0
    if encoder is not None:
        return len(encoder.encode(text))
    return max(1, math.ceil(len(text) / 4))


# ---------- Label utilities (truth-only scope) ----------

def _present_labels_from_df(df: pd.DataFrame) -> list[str]:
    """
    Extract labels that actually appear in ground truth for this dataset.
    - Clean NaN/whitespace
    - Optionally intersect with canonical LABELS if available
    - Deterministic order: preserve canonical order when possible; else sorted
    """
    if "class_label" not in df.columns:
        return []

    s = (
        df["class_label"]
        .astype(str).str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        .dropna()
    )
    present = list(dict.fromkeys(s.tolist()))  # unique, keep first-seen order

    if CANON_LABELS:
        # Preserve canonical order; keep only labels that appear in this dataset
        canon_set = set(CANON_LABELS)
        return [lbl for lbl in CANON_LABELS if lbl in present and lbl in canon_set]

    # No canonical list available → return deterministic sorted set
    return sorted(set(present))


def _labels_bulleted(labels: list[str]) -> str:
    """Format labels as a bullet list for inclusion in the user prompt estimate."""
    if not labels:
        return "(none)"
    return "\n".join(f"- {lbl}" for lbl in labels)


# ---------- Estimation per request & per dataset ----------

def estimate_request_tokens(
    model: str,
    tweet_text: str,
    rules_text: str,
    labels: list[str],
    max_output_tokens: int = 40,
) -> int:
    """
    Estimate tokens consumed per single request with filtered labels:
      system + user (rules + tweet + explicit ALLOWED LABELS list)
      + response_format (JSON schema built from CURRENT_LABELS)
      + output allowance (max_output_tokens)

    Parameters
    ----------
    model : str
        Model name for tokenizer selection (affects token counting).
    tweet_text : str
        The tweet content (single example).
    rules_text : str
        The zero-shot rules you inject in the prompt.
    labels : list[str]
        CURRENT_LABELS (truth-present labels for the dataset/event).
    max_output_tokens : int
        Reserved tokens for the model's JSON output (e.g., 40).

    Returns
    -------
    int : estimated token count for this single request.
    """
    enc = get_token_encoder(model)

    # (1) System prompt tokens
    sys_tokens = count_tokens(SYSTEM_PROMPT, enc)

    # (2) User prompt tokens — include ALLOWED LABELS explicitly (matches batch prompt shape)
    labels_block = _labels_bulleted(labels)
    user_msg = (
        "Classify this tweet into exactly ONE of the allowed labels.\n"
        f"Allowed labels: {", ".join(labels)}\n"
        f"Rules:\n{rules_text}\n"
        "Return JSON that conforms to the provided schema.\n"
        f'Tweet: """{tweet_text}"""'
    )
    user_tokens = count_tokens(user_msg, enc)

    # (3) response_format schema tokens — build the same dynamic schema used in batch.py
    schema_dict = _make_schema(labels)
    schema_tokens = count_tokens(json.dumps(schema_dict, ensure_ascii=False), enc)

    # (4) Output allowance — reserved response tokens (upper bound)
    return sys_tokens + user_tokens + schema_tokens + max_output_tokens


def estimate_dataset_tokens(
    tsv_path: str | Path,
    model: str,
    rules_text: str,
    sample_size: int = 200,
    max_output_tokens: int = 40,
) -> Dict[str, int]:
    """
    Estimate total token usage for a dataset by sampling up to `sample_size` rows.

    Strategy:
    - Load TSV and compute CURRENT_LABELS from truth (cleaned).
    - For up to `sample_size` sampled tweets, estimate per-request tokens using:
        system + user (with the explicit label list) + dynamic schema + output allowance.
    - Multiply the sample average by the dataset size.

    Returns
    -------
    dict: {"num_rows", "avg_req_tokens", "est_total_tokens"}
    """
    df = load_tsv(tsv_path)
    n = len(df)
    if n == 0:
        return {"num_rows": 0, "avg_req_tokens": 0, "est_total_tokens": 0}

    # CURRENT_LABELS (truth-only) — same basis the batch builder will use
    labels = _present_labels_from_df(df)

    sample_n = min(sample_size, n)
    samp = df.sample(sample_n, random_state=42)["tweet_text"].tolist()

    per_req = [
        estimate_request_tokens(model, t, rules_text, labels, max_output_tokens=max_output_tokens)
        for t in samp
    ]
    avg_tokens = int(sum(per_req) / len(per_req))
    total_est = int(avg_tokens * n)
    return {"num_rows": n, "avg_req_tokens": avg_tokens, "est_total_tokens": total_est}


def build_token_index(
    sources_df: pd.DataFrame,
    model: str,
    rules_text: str,
    batch_token_limit: int,
    safety_margin: float = 0.90,
    sample_size: int = 200,
    max_output_tokens: int = 40,
) -> pd.DataFrame:
    """
    Compute token estimates for each dataset entry in `sources_df`.

    Expected columns in sources_df:
      - event, split, tsv

    Adds columns:
      - num_rows, avg_req_tokens, est_total_tokens
      - fits_cap (<= batch_token_limit * safety_margin)
      - limit_used_% (relative to hard cap)
    """
    rows = []
    for _, r in sources_df.iterrows():
        stats = estimate_dataset_tokens(
            r["tsv"], model, rules_text,
            sample_size=sample_size, max_output_tokens=max_output_tokens
        )
        rows.append({
            "event": r["event"],
            "split": r["split"],
            "tsv": r["tsv"],
            **stats,
        })
    out = pd.DataFrame(rows)

    est_limit = int(batch_token_limit * safety_margin)
    out["fits_cap"] = out["est_total_tokens"] <= est_limit
    out["limit_used_%"] = (out["est_total_tokens"] / batch_token_limit * 100).round(1)

    return out.sort_values(["fits_cap", "est_total_tokens"], ascending=[False, True])


# ---------- Sharding utility (optional) ----------

def shard_dataset_by_tokens(
    tsv_path: str | Path,
    model: str,
    rules_text: str,
    target_token_budget: int,
    max_output_tokens: int = 40,
) -> list[pd.DataFrame]:
    """
    Split a TSV into a list of DataFrames where each shard's *estimated* token sum
    stays under target_token_budget (greedy, one pass).

    We estimate each row's request tokens using the SAME CURRENT_LABELS (truth-only)
    for the whole dataset, matching how batch.jsonl will be constructed.
    """
    df = load_tsv(tsv_path)
    if df.empty:
        return [df]

    # Compute CURRENT_LABELS once for this dataset
    labels = _present_labels_from_df(df)

    enc = get_token_encoder(model)

    # Precompute per-row token estimates using the current labels
    def _row_estimate(text: str) -> int:
        return estimate_request_tokens(
            model, text, rules_text, labels, max_output_tokens=max_output_tokens
        )

    est_tokens = df["tweet_text"].apply(_row_estimate)

    # Greedy one-pass sharding
    shards: list[pd.DataFrame] = []
    current_rows: list[int] = []
    current_budget = 0

    for idx, tok in est_tokens.items():
        if current_budget + tok > target_token_budget and current_rows:
            shards.append(df.loc[current_rows].copy())
            current_rows = []
            current_budget = 0
        current_rows.append(idx)
        current_budget += tok

    if current_rows:
        shards.append(df.loc[current_rows].copy())

    return shards
