# humaidclf/batch.py
"""
Batch & synchronous helpers for HumAID zero-shot classification.

Highlights
----------
- Dynamic schema per event (restricts labels to those PRESENT in the event).
- Request builders for OpenAI Batch (/v1/chat/completions).
- Synchronous 'dry-run' sanity check on a small sample.
- Parsers for outputs.jsonl (+ optional errors.jsonl).
- Patch pass: re-classify *missing* OR *blank/OOS* predictions synchronously.
- API key switchers (env-based or direct).

This module intentionally adds plenty of comments for clarity and future maintenance.
"""

from __future__ import annotations
import os, json, time, requests, math
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
import pandas as pd

from .prompts import SYSTEM_PROMPT, make_user_message, LABELS

# =============================================================================
# API base & dynamic headers
# =============================================================================

OPENAI_BASE = "https://api.openai.com/v1"

# Module-level state (mutated by key-switching helpers)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ACTIVE_KEY_LABEL = "OPENAI_API_KEY" if OPENAI_API_KEY else None

def _request_params(model: str, max_out_tokens: int) -> dict:
    """
    Return the correct request parameters for this model family.
    - gpt-5*/o4*/o3* use 'max_completion_tokens' and ignore temperature/top_p (fixed to defaults).
    - older chat models (gpt-4.1/4o/4o-mini/0613) use 'max_tokens' and accept temperature/top_p.
    """
    m = (model or "").lower()
    is_new = m.startswith(("gpt-5", "o4", "o3"))

    tok_field = "max_completion_tokens" if is_new else "max_tokens"
    params = {tok_field: max_out_tokens}

    if not is_new:
        # Only include tunables for families that support them
        params.update({"temperature": 0.0, "top_p": 1})

    return params

def _rebuild_headers():
    """(Re)build request headers from the current OPENAI_API_KEY."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "No OPENAI_API_KEY is set. "
            "Use set_api_key_env('OPENAI_API_KEY_1') or set_api_key_value(...)."
        )
    return (
        {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},  # H_JSON
        {"Authorization": f"Bearer {OPENAI_API_KEY}"},                                      # H_MULTI
    )

H_JSON, H_MULTI = _rebuild_headers() if OPENAI_API_KEY else ({}, {})

def set_api_key_env(varname: str) -> None:
    """
    Switch the active API key using an environment variable name, e.g.:
        set_api_key_env('OPENAI_API_KEY_1')
        set_api_key_env('OPENAI_API_KEY_2')
    """
    global OPENAI_API_KEY, ACTIVE_KEY_LABEL, H_JSON, H_MULTI
    val = os.environ.get(varname)
    if not val:
        raise KeyError(f"Environment variable '{varname}' is not set.")
    OPENAI_API_KEY = val.strip()
    ACTIVE_KEY_LABEL = varname
    H_JSON, H_MULTI = _rebuild_headers()

def set_api_key_value(key: str, label: str | None = None) -> None:
    """
    Switch the active API key using a literal key string (avoid committing keys to code!).
    Optionally pass a label for debugging/readability (e.g., 'TIER1' or 'ALT').
    """
    global OPENAI_API_KEY, ACTIVE_KEY_LABEL, H_JSON, H_MULTI
    if not key or not isinstance(key, str):
        raise ValueError("key must be a non-empty string.")
    OPENAI_API_KEY = key.strip()
    ACTIVE_KEY_LABEL = label or "<direct>"
    H_JSON, H_MULTI = _rebuild_headers()

def get_active_api_key_label() -> str | None:
    """Return the current active key label (env var name or custom label)."""
    return ACTIVE_KEY_LABEL

@contextmanager
def use_api_key_env(varname: str):
    """
    Temporarily switch to the key from `varname` inside a with-block.
    Restores the previous key on exit.
    """
    prev_key, prev_label, prev_H_JSON, prev_H_MULTI = OPENAI_API_KEY, ACTIVE_KEY_LABEL, H_JSON.copy(), H_MULTI.copy()
    set_api_key_env(varname)
    try:
        yield
    finally:
        globals()["OPENAI_API_KEY"] = prev_key
        globals()["ACTIVE_KEY_LABEL"] = prev_label
        globals()["H_JSON"] = prev_H_JSON
        globals()["H_MULTI"] = prev_H_MULTI

@contextmanager
def use_api_key_value(key: str, label: str | None = None):
    """
    Temporarily switch to a literal key inside a with-block.
    Restores the previous key on exit.
    """
    prev_key, prev_label, prev_H_JSON, prev_H_MULTI = OPENAI_API_KEY, ACTIVE_KEY_LABEL, H_JSON.copy(), H_MULTI.copy()
    set_api_key_value(key, label=label)
    try:
        yield
    finally:
        globals()["OPENAI_API_KEY"] = prev_key
        globals()["ACTIVE_KEY_LABEL"] = prev_label
        globals()["H_JSON"] = prev_H_JSON
        globals()["H_MULTI"] = prev_H_MULTI


# =============================================================================
# Label scope & schema
# =============================================================================

def _extract_present_labels(df: pd.DataFrame) -> list[str]:
    """
    Determine which labels actually appear in the event's ground truth, while
    preserving canonical ordering from LABELS (so charts and tables are stable).

    If your TSV has no 'class_label' column (pure inference), we fallback to full LABELS.
    """
    if "class_label" not in df.columns:
        return list(LABELS)

    clean = (
        df["class_label"]
          .astype(str)
          .str.strip()
          .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
          .dropna()
    )
    present = set(clean.tolist())
    # Preserve canonical order from LABELS
    current = [lbl for lbl in LABELS if lbl in present]
    if not current:
        # Fallback defensively to all labels to avoid empty enums
        return list(LABELS)
    return current

def _make_schema(labels: list[str]) -> dict:
    """Structured Outputs schema using ONLY the labels passed in."""
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": labels},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["label"],
        "additionalProperties": False
    }


# =============================================================================
# Synchronous quick test (sanity check)
# =============================================================================

def sync_test_sample(
    df: pd.DataFrame,
    n: int = 5,
    rules: str = "",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Zero-shot sanity check on a small sample.
    - Uses event-scoped labels (not just the sample) to build the schema.
    - Prints Macro-F1 (truth-only scope) if truth exists.
    """
    from .eval import macro_f1  # local import to avoid hard dep for report users

    CURRENT_LABELS = _extract_present_labels(df)
    test = df.sample(min(n, len(df)), random_state=seed).copy()

    # Single-label: predict locally (deterministic)
    if len(CURRENT_LABELS) == 1:
        only = CURRENT_LABELS[0]
        out = pd.DataFrame({
            "tweet_id": test["tweet_id"].astype(str),
            "tweet_text": test["tweet_text"],
            "class_label": test.get("class_label", ""),
            "predicted_label": only,
            "confidence": 1.0,
            "entropy": float("nan"),
        })
        try:
            print("Macro-F1 (tiny sample):", macro_f1(out))
        except Exception:
            pass
        return out

    rows = []
    schema = _make_schema(CURRENT_LABELS)

    for _, r in test.iterrows():
        user_msg = make_user_message(str(r["tweet_text"]), rules, CURRENT_LABELS)
        params = _request_params(model, 40)
        body = {
            "model": model,
            **params,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "tweet_label", "schema": schema},
            },
        }

        resp = requests.post(f"{OPENAI_BASE}/chat/completions", headers=H_JSON, json=body, timeout=60)
        if resp.status_code != 200:
            print(">>> API error body:", resp.text)
        resp.raise_for_status()
        choice = resp.json()["choices"][0]["message"]
        parsed = choice.get("parsed")
        if not parsed:
            content = choice.get("content", "")
            if isinstance(content, list):  # some providers return list-of-parts
                content = content[0].get("text", ""
                )
            parsed = json.loads(content) if content else {}

        pred = parsed.get("label", "")
        if pred is None:
            pred = ""
        pred = str(pred).strip()

        rows.append({
            "tweet_id": str(r["tweet_id"]),
            "tweet_text": r["tweet_text"],
            "class_label": r.get("class_label", ""),
            "predicted_label": pred,
            "confidence": parsed.get("confidence", None),
            "entropy": float("nan"),
        })

    out = pd.DataFrame(rows)
    try:
        print("Macro-F1 (tiny sample):", macro_f1(out))
    except Exception:
        pass
    return out


# =============================================================================
# Batch request builder (Mode S)
# =============================================================================

def build_requests_jsonl_S(
    df: pd.DataFrame,
    out_path: str,
    rules: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    labels_override: list[str] | None = None,
    allow_single_label_bypass: bool = True,
):
    """Zero-shot Batch JSONL builder targeting /v1/chat/completions."""
    # Use override (event-wide labels) if provided; else derive from this df
    CURRENT_LABELS = labels_override or _extract_present_labels(df)

    # Single-label: optionally bypass (sharded runs should pass allow_single_label_bypass=False)
    if allow_single_label_bypass and len(CURRENT_LABELS) == 1:
        with open(out_path, "w", encoding="utf-8") as f:
            pass
        return out_path

    schema = _make_schema(CURRENT_LABELS)

    with open(out_path, "w", encoding="utf-8") as f:
        params = _request_params(model, 40)
        for _, row in df.iterrows():
            tid = str(row["tweet_id"]).strip()
            text = str(row["tweet_text"] or "").replace("\r", " ").strip()
            user_msg = make_user_message(text, rules, CURRENT_LABELS)
            body = {
                "model": model,
                **params,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "tweet_label", "schema": schema},
                },
            }

            line = {
                "custom_id": f"tweet-{tid}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return out_path

# =============================================================================
# Batch submission helpers
# =============================================================================

def upload_file_for_batch(filepath: str) -> str:
    with open(filepath, "rb") as f:
        r = requests.post(
            f"{OPENAI_BASE}/files", headers=H_MULTI,
            files={"file": (os.path.basename(filepath), f)},
            data={"purpose": "batch"}, timeout=60
        )
    r.raise_for_status()
    return r.json()["id"]

def create_batch(input_file_id: str, endpoint="/v1/chat/completions", completion_window="24h") -> str:
    payload = {"input_file_id": input_file_id, "endpoint": endpoint, "completion_window": completion_window}
    r = requests.post(f"{OPENAI_BASE}/batches", headers=H_JSON, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["id"]

def get_batch(batch_id: str) -> dict:
    r = requests.get(f"{OPENAI_BASE}/batches/{batch_id}", headers=H_JSON, timeout=60)
    r.raise_for_status()
    return r.json()

def wait_for_batch(batch_id: str, poll_secs=20) -> dict:
    while True:
        info = get_batch(batch_id)
        status = info.get("status")
        print(f"[batch {batch_id}] status = {status}")
        if status in {"completed", "failed", "cancelled"}:
            return info
        time.sleep(poll_secs)

def download_file_content(file_id: str, out_path: str) -> str:
    r = requests.get(f"{OPENAI_BASE}/files/{file_id}/content", headers=H_JSON, timeout=300)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path


# =============================================================================
# Outputs parser (+ optional errors.jsonl)
# =============================================================================

def parse_outputs_S_to_df(
    outputs_jsonl_path: str,
    source_df: pd.DataFrame,
    *,
    errors_jsonl_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Parse provider outputs.jsonl and (optionally) errors.jsonl, and reattach fields
    (tweet_text, class_label) using source_df. Returns a DataFrame with:
        tweet_id, tweet_text, class_label, predicted_label, confidence, entropy, status

    status ∈ {"ok", "error"} here. Missing rows are handled by the patch pass later.
    """
    # Build lookup from source_df
    by_id = {
        str(r["tweet_id"]): {
            "tweet_text": str(r.get("tweet_text", "")),
            "class_label": str(r.get("class_label", "")),
        }
        for _, r in source_df.iterrows()
    }

    rows: List[Dict[str, Any]] = []

    # Success outputs
    with open(outputs_jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tid = rec.get("custom_id", "").replace("tweet-", "")
            choice = rec["response"]["body"]["choices"][0]["message"]
            parsed = choice.get("parsed")
            if not parsed:
                content = choice.get("content", "")
                if isinstance(content, list):
                    content = content[0].get("text", "")
                parsed = json.loads(content) if content else {}

            pred = parsed.get("label", "")
            if pred is None:
                pred = ""
            pred = str(pred).strip()

            rows.append({
                "tweet_id": tid,
                "tweet_text": by_id.get(tid, {}).get("tweet_text", ""),
                "class_label": by_id.get(tid, {}).get("class_label", ""),
                "predicted_label": pred,
                "confidence": parsed.get("confidence", None),
                "entropy": float("nan"),
                "status": "ok",
            })

    # Optional error file — add placeholders so patch pass knows what failed
    if errors_jsonl_path and os.path.exists(errors_jsonl_path):
        with open(errors_jsonl_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                tid = rec.get("custom_id", "").replace("tweet-", "")
                err = rec.get("error") or {}
                rows.append({
                    "tweet_id": tid,
                    "tweet_text": by_id.get(tid, {}).get("tweet_text", ""),
                    "class_label": by_id.get(tid, {}).get("class_label", ""),
                    "predicted_label": "",
                    "confidence": None,
                    "entropy": float("nan"),
                    "status": "error",
                    "error": err,
                })

    # Return in a stable column order
    cols = ["tweet_id", "tweet_text", "class_label", "predicted_label", "confidence", "entropy", "status"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


# =============================================================================
# Patch pass: fill missing / blank / OOS predictions synchronously
# =============================================================================

def retry_fill_missing_predictions(
    source_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    *,
    rules: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 40,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
    labels_override: list[str] | None = None,   # <-- NEW (supports sharded runs)
) -> pd.DataFrame:
    """
    Patch pass that fixes:
      • rows missing entirely from preds_df
      • rows present but with blank predicted_label
      • (optional) rows with OOS predicted_label (outside event label set)

    Returns a row-aligned DataFrame (one row per source_df row).

    labels_override:
      If provided, use this fixed event-level label list for the schema
      (important for sharded runs so every shard uses the same schema).
    """
    CURRENT_LABELS = labels_override or _extract_present_labels(source_df)
    schema = _make_schema(CURRENT_LABELS)

    # Build source lookup
    by_id = {
        str(r["tweet_id"]): {
            "tweet_text": str(r.get("tweet_text", "")),
            "class_label": str(r.get("class_label", "")),
        }
        for _, r in source_df.iterrows()
    }
    all_ids = list(by_id.keys())

    # Normalize preds_df
    preds_df = preds_df.copy()
    if "tweet_id" not in preds_df.columns:
        preds_df["tweet_id"] = ""
    preds_df["tweet_id"] = preds_df["tweet_id"].astype(str)
    preds_df["predicted_label"] = preds_df.get("predicted_label", "").astype(str).str.strip()
    if "status" not in preds_df.columns:
        preds_df["status"] = "ok"

    # Identify missing / empty / OOS
    have_ids = set(preds_df["tweet_id"])
    missing_ids = [tid for tid in all_ids if tid not in have_ids]

    mask_empty = (preds_df["predicted_label"] == "") | preds_df["predicted_label"].isna()
    empty_ids = preds_df.loc[mask_empty, "tweet_id"].tolist()

    # OOS set (should be rare due to schema, but guard anyway)
    oos_ids = []
    if CURRENT_LABELS:
        mask_oos = ~preds_df["predicted_label"].isin(CURRENT_LABELS)
        oos_ids = preds_df.loc[mask_oos, "tweet_id"].tolist()

    # Merge targets, keep order, de-dupe
    target_ids = list(dict.fromkeys(missing_ids + empty_ids + oos_ids))
    if not target_ids:
        return preds_df

    patched_rows: List[Dict[str, Any]] = []

    # Synchronously classify each target row with the SAME event-scoped schema
    for tid in target_ids:
        text = by_id.get(tid, {}).get("tweet_text", "")
        user_msg = make_user_message(text, rules, CURRENT_LABELS)
        params = _request_params(model, max_tokens)
        body = {
            "model": model,
            **params,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "tweet_label", "schema": schema},
            },
        }

        # Small retry loop for transient 5xx/errors
        for attempt in range(max_retries):
            try:
                resp = requests.post(f"{OPENAI_BASE}/chat/completions", headers=H_JSON, json=body, timeout=60)
                if resp.status_code != 200:
                    # Print once to help debugging
                    print(f"[patch] {tid} HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                choice = resp.json()["choices"][0]["message"]
                parsed = choice.get("parsed")
                if not parsed:
                    content = choice.get("content", "")
                    if isinstance(content, list):
                        content = content[0].get("text", "")
                    parsed = json.loads(content) if content else {}

                pred = parsed.get("label", "")
                if pred is None:
                    pred = ""
                pred = str(pred).strip()

                patched_rows.append({
                    "tweet_id": tid,
                    "tweet_text": by_id.get(tid, {}).get("tweet_text", ""),
                    "class_label": by_id.get(tid, {}).get("class_label", ""),
                    "predicted_label": pred,
                    "confidence": parsed.get("confidence", None),
                    "entropy": float("nan"),
                    "status": "patched_sync",
                })
                break
            except Exception as e:
                if attempt + 1 >= max_retries:
                    # Give up: keep as explicit failure so downstream can see it
                    patched_rows.append({
                        "tweet_id": tid,
                        "tweet_text": by_id.get(tid, {}).get("tweet_text", ""),
                        "class_label": by_id.get(tid, {}).get("class_label", ""),
                        "predicted_label": "",
                        "confidence": None,
                        "entropy": float("nan"),
                        "status": "error",
                        "error": {"reason": "patch_failed", "message": str(e)},
                    })
                else:
                    time.sleep(backoff_seconds)

    # Combine and prefer higher-priority statuses when deduping
    out = pd.concat([preds_df, pd.DataFrame(patched_rows)], ignore_index=True)

    # Priority: ok (2) > patched_sync (1) > error/missing (0)
    pri = {"ok": 2, "patched_sync": 1, "error": 0, "missing": 0}
    out["_pri"] = out["status"].map(pri).fillna(-1)
    out = (
        out.sort_values(["tweet_id", "_pri"])
           .drop_duplicates(subset=["tweet_id"], keep="last")
           .drop(columns=["_pri"])
           .reset_index(drop=True)
    )

    # Final guard: enforce same row count as source (if some source ids never appeared, add empty rows)
    final_have = set(out["tweet_id"])
    missing_after = [tid for tid in all_ids if tid not in final_have]
    if missing_after:
        fill = [{
            "tweet_id": tid,
            "tweet_text": by_id.get(tid, {}).get("tweet_text", ""),
            "class_label": by_id.get(tid, {}).get("class_label", ""),
            "predicted_label": "",
            "confidence": None,
            "entropy": float("nan"),
            "status": "missing",
        } for tid in missing_after]
        out = pd.concat([out, pd.DataFrame(fill)], ignore_index=True)

    # Keep a stable column order
    cols = ["tweet_id", "tweet_text", "class_label", "predicted_label", "confidence", "entropy", "status"]
    return out[cols]
