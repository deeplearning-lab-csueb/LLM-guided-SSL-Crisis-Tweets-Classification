# humaidclf/runner.py
# -----------------------------------------------------------------------------
# Orchestrates a full zero-shot run for HumAID:
#   TSV -> (optional) dry-run sanity check -> build JSONL -> (maybe bypass) submit batch
#   -> (optional) wait -> download/parse -> PATCH MISSING/BLANK/OOS -> save predictions
#   -> (optional) analysis
#
# NEW in this version:
#   • Preflight probe before submitting a batch (catches bad API key/model/format).
#   • Safe finalize: gracefully handle missing output_file_id; download errors.jsonl.
#   • Automatic synchronous fallback when a batch yields no outputs.
# -----------------------------------------------------------------------------

from __future__ import annotations
import json, os, itertools
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import requests
import pandas as pd

from .io import load_tsv, plan_run_dirs
from .batch import (    
    sync_test_sample,
    build_requests_jsonl_S,
    upload_file_for_batch,
    create_batch,
    wait_for_batch,
    download_file_content,
    parse_outputs_S_to_df,
    retry_fill_missing_predictions,
    # We’ll reuse batch module’s HTTP config via its helpers
)
from .prompts import SYSTEM_PROMPT, LABELS, make_user_message
from .batch import _make_schema, OPENAI_BASE, H_JSON  # use same base & headers
from .eval import macro_f1, analyze_and_export_mistakes


# =============================================================================
# Local helpers (runner-only)
# =============================================================================

def _present_labels_from_df(df: pd.DataFrame) -> list[str]:
    """
    Extract unique labels that appear in ground truth (cleaned).
    Sorting is only for determinism; downstream eval uses truth-only scope.
    """
    s = (
        df.get("class_label", pd.Series(dtype=object))
          .astype(str).str.strip()
          .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
          .dropna()
    )
    present_lower = {x.lower() for x in s.tolist()}
    return [l for l in LABELS if l.lower() in present_lower]

def _predict_single_label_event(df: pd.DataFrame, only_label: str) -> pd.DataFrame:
    """
    Fast path for single-label events: no API call is needed.
    Produces the same columns as parse_outputs_S_to_df() (minus 'status').
    """
    return pd.DataFrame({
        "tweet_id": df["tweet_id"].astype(str),
        "tweet_text": df["tweet_text"],
        "class_label": df.get("class_label", ""),
        "predicted_label": only_label,
        "confidence": 1.0,       # deterministic since there is no choice
        "entropy": float("nan"),
        "status": "ok",
    })

def _request_params_for(model: str, max_out_tokens: int) -> dict:
    """
    Return the correct param bundle for this model family.
    - New families (gpt-5*, o4*, o3*): use max_completion_tokens ONLY.
    - Classic families (gpt-4*, gpt-4o*): use max_tokens + temperature/top_p.
    """
    m = (model or "").lower()
    if m.startswith(("gpt-5", "o4", "o3")):
        return {"max_completion_tokens": 500} # max token = 500 for reasoning
    # classic
    return {"max_tokens": max_out_tokens, "temperature": 0.0, "top_p": 1}    

def _preflight_probe(model: str,
                     rules: str = "",
                     labels: list[str] | None = None,
                     timeout: int = 20) -> tuple[bool, str]:
    """
    Tiny /chat/completions call with a toy schema to validate:
      - API key present/valid
      - model name resolvable
      - Structured Outputs accepted
    Returns (ok, message). ok=False => caller should not submit batch.
    """
    try:
        labs = (labels or ["Request-Help", "Other"])
        if len(labs) < 2:
            labs = list({*(labs or []), "Other", "Request-Help"})[:2]

        schema = _make_schema(labs)
        user_msg = make_user_message("Preflight probe — classify this sentence.", rules, labs)

        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "tweet_label", "schema": schema},
            },
        }
        # add model-appropriate params
        body.update(_request_params_for(model, max_out_tokens=8))

        resp = requests.post(f"{OPENAI_BASE}/chat/completions", headers=H_JSON, json=body, timeout=timeout)
        if resp.status_code != 200:
            return (False, f"HTTP {resp.status_code}: {resp.text[:240]}")
        # Touch payload to ensure shape
        _ = resp.json()["choices"][0]["message"]
        return (True, "ok")
    except Exception as e:
        return (False, f"{type(e).__name__}: {e}")

def _summarize_errors_jsonl(errors_jsonl_path: str, n: int = 8) -> str:
    """Return a short multi-line summary pulled from errors.jsonl (first n lines)."""
    if not (errors_jsonl_path and os.path.exists(errors_jsonl_path)):
        return ""
    msgs = []
    with open(errors_jsonl_path, encoding="utf-8") as f:
        for line in itertools.islice(f, n):
            try:
                rec = json.loads(line)
                err = rec.get("error") or {}
                code = err.get("code") or err.get("type") or "no-code"
                msg  = (err.get("message") or "").strip().replace("\n", " ")
                msgs.append(f"- {code}: {msg[:180]}")
            except Exception:
                pass
    return "\n".join(msgs)


# =============================================================================
# Public API
# =============================================================================

def run_experiment(
    dataset_path: str,
    rules: str,
    model: str = "gpt-4o-mini",
    tag: str = "modeS",
    *,
    temperature: float = 0.0,
    dryrun_n: int = 20,
    poll_secs: int = 60,
    out_root: str = "runs",
    do_analysis: bool = True,
    analysis_subdir: str = "analysis",
    submit_only: bool = False,
) -> Tuple[Dict[str, Any], pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    End-to-end: load TSV -> dry-run -> preflight -> build JSONL -> (maybe bypass) submit
    -> (optionally wait) -> download & parse -> PATCH -> save -> (optional) analysis.

    Blocks while waiting for batch completion unless submit_only=True.
    """
    # -------------------------------------------------------------------------
    # 0) Load TSV
    # -------------------------------------------------------------------------
    df = load_tsv(dataset_path)

    # -------------------------------------------------------------------------
    # 1) Dry-run sanity check (small, synchronous)
    # -------------------------------------------------------------------------
    if dryrun_n and dryrun_n > 0:
        _ = sync_test_sample(
            df, n=dryrun_n, rules=rules, model=model,
            temperature=temperature, seed=42
        )

    # -------------------------------------------------------------------------
    # 1.1) Preflight probe (cheap, avoids submitting broken batches)
    # -------------------------------------------------------------------------
    labels_for_probe = _present_labels_from_df(df) or ["Other", "Request-Help"]
    ok, msg = _preflight_probe(model, rules, labels=labels_for_probe)
    if not ok:
        raise RuntimeError(
            f"[preflight] Model '{model}' failed with current API key. Details: {msg}"
        )

    # -------------------------------------------------------------------------
    # 2) Plan run dirs + build requests.jsonl
    # -------------------------------------------------------------------------
    plan = plan_run_dirs(dataset_path, out_root=out_root, model=model, tag=tag)

    build_requests_jsonl_S(
        df, plan["requests_jsonl"],
        rules=rules, model=model, temperature=temperature
    )

    # -------------------------------------------------------------------------
    # 2.1) Single-label BYPASS
    # -------------------------------------------------------------------------
    req_path = Path(plan["requests_jsonl"])
    if req_path.exists() and req_path.stat().st_size == 0:
        present = _present_labels_from_df(df)
        if len(present) != 1:
            raise RuntimeError(
                "build_requests_jsonl_S produced an empty JSONL but present label count != 1.\n"
                f"Detected labels (truth): {present}"
            )
        only_label = present[0]

        preds = _predict_single_label_event(df, only_label)

        Path(plan["predictions_csv"]).parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(plan["predictions_csv"], index=False)
        print("[single-label] Saved predictions to:", plan["predictions_csv"])
        try:
            print("[single-label] Macro-F1:", macro_f1(preds))
        except Exception:
            pass

        # Provenance for bypass mode
        meta = {"mode": "local_single_label", "only_label": only_label}
        with open(plan["batch_meta_json"], "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Optional analysis
        analysis_summary = None
        if do_analysis:
            charts_dir = Path(plan["dir"]) / analysis_subdir / "charts"
            mistakes_csv = Path(plan["dir"]) / analysis_subdir / "mistakes.csv"
            _, summary, _, _ = analyze_and_export_mistakes(
                pred_csv_path=str(plan["predictions_csv"]),
                out_mistakes_csv_path=str(mistakes_csv),
                charts_dir=str(charts_dir),
            )
            analysis_summary = summary

        return plan, preds, analysis_summary

    # -------------------------------------------------------------------------
    # 3) Submit batch (normal multi-label path)
    # -------------------------------------------------------------------------
    fid = upload_file_for_batch(str(plan["requests_jsonl"]))
    bid = create_batch(fid, endpoint="/v1/chat/completions", completion_window="24h")

    with open(plan["batch_meta_json"], "w", encoding="utf-8") as f:
        json.dump({"file_id": fid, "batch_id": bid}, f, indent=2)

    if submit_only:
        # Caller will resume later using resume_experiment()
        return plan, pd.DataFrame(), None

    # -------------------------------------------------------------------------
    # 4) Wait for completion, then download & parse if possible
    # -------------------------------------------------------------------------
    info = wait_for_batch(bid, poll_secs=poll_secs)
    status = info.get("status")
    print(f"[batch {bid}] final status = {status}")

    # Prefer robust handling: even when 'completed', output_file_id might be missing.
    out_file_id = info.get("output_file_id")
    err_file_id = info.get("error_file_id")

    outputs_jsonl_path: Optional[str] = None
    errors_jsonl_path: Optional[str] = None

    if out_file_id:
        outputs_jsonl_path = str(plan["outputs_jsonl"])
        download_file_content(out_file_id, outputs_jsonl_path)

    if err_file_id:
        errors_jsonl_path = str(Path(plan["dir"]) / "errors.jsonl")
        download_file_content(err_file_id, errors_jsonl_path)

    # If we have no outputs, summarize errors and FALL BACK to sync classification
    if not outputs_jsonl_path:
        if errors_jsonl_path and os.path.exists(errors_jsonl_path):
            print("[batch] No outputs.jsonl. Partial error summary:")
            print(_summarize_errors_jsonl(errors_jsonl_path))
        else:
            print("[batch] No outputs.jsonl and no errors.jsonl available.")

        # Fallback: classify everything synchronously using the same event-scoped schema
        print("[batch] Falling back to synchronous classification for this run…")
        empty = pd.DataFrame(columns=["tweet_id", "predicted_label"])
        preds = retry_fill_missing_predictions(
            source_df=df,
            preds_df=empty,
            rules=rules,
            model=model,
            temperature=temperature,
            max_tokens=40,
            labels_override=_present_labels_from_df(df) or None,
        )

        Path(plan["predictions_csv"]).parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(plan["predictions_csv"], index=False)
        print("Saved predictions (sync fallback) to:", plan["predictions_csv"])
        try:
            print("Macro-F1:", macro_f1(preds))
        except Exception:
            pass

        analysis_summary = None
        if do_analysis:
            charts_dir = Path(plan["dir"]) / analysis_subdir / "charts"
            mistakes_csv = Path(plan["dir"]) / analysis_subdir / "mistakes.csv"
            _, summary, _, _ = analyze_and_export_mistakes(
                pred_csv_path=str(plan["predictions_csv"]),
                out_mistakes_csv_path=str(mistakes_csv),
                charts_dir=str(charts_dir),
            )
            analysis_summary = summary

        return plan, preds, analysis_summary

    # -------------------------------------------------------------------------
    # 4.1) Parse + PATCH PASS (normal path with outputs)
    # -------------------------------------------------------------------------
    preds = parse_outputs_S_to_df(
        outputs_jsonl_path, df,
        errors_jsonl_path=errors_jsonl_path
    )

    # Row-align & clean up: fill missing/blank/OOS
    if (len(preds) != len(df)) or (preds["predicted_label"] == "").any() or (~preds["predicted_label"].isin(_present_labels_from_df(df))).any():
        preds = retry_fill_missing_predictions(
            source_df=df,
            preds_df=preds,
            rules=rules,
            model=model,
            temperature=temperature,
            max_tokens=40,
            max_retries=3,
            backoff_seconds=2.0,
            labels_override=_present_labels_from_df(df) or None,
        )

    # Persist final predictions (patched/aligned)
    Path(plan["predictions_csv"]).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(plan["predictions_csv"], index=False)

    print("Saved predictions to:", plan["predictions_csv"])
    try:
        print("Macro-F1:", macro_f1(preds))  # eval default scope='truth'
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # 5) Optional analysis artifacts
    # -------------------------------------------------------------------------
    analysis_summary = None
    if do_analysis:
        charts_dir = Path(plan["dir"]) / analysis_subdir / "charts"
        mistakes_csv = Path(plan["dir"]) / analysis_subdir / "mistakes.csv"
        _, summary, _, _ = analyze_and_export_mistakes(
            pred_csv_path=str(plan["predictions_csv"]),
            out_mistakes_csv_path=str(mistakes_csv),
            charts_dir=str(charts_dir),
        )
        analysis_summary = summary

    return plan, preds, analysis_summary


def resume_experiment(
    run_dir: str | Path,
    *,
    do_analysis: bool = True,
    analysis_subdir: str = "analysis",
) -> Tuple[Dict[str, Any], pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Resume a previously submitted run by reading batch_meta.json from <run_dir>
    and finishing the download/parse/patch/analysis steps.

    Also supports single-label bypass runs where batch_meta.json contains:
      {"mode": "local_single_label", "only_label": "<label>"}.
    """
    run_dir = Path(run_dir)
    with open(run_dir / "batch_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    plan = {
        "dir": run_dir,
        "requests_jsonl": run_dir / "requests.jsonl",
        "outputs_jsonl": run_dir / "outputs.jsonl",
        "predictions_csv": run_dir / "predictions.csv",
        "batch_meta_json": run_dir / "batch_meta.json",
    }

    # -------------------------------------------------------------------------
    # Single-label bypass resume
    # -------------------------------------------------------------------------
    if meta.get("mode") == "local_single_label":
        preds = pd.read_csv(plan["predictions_csv"])
        analysis_summary = None
        if do_analysis:
            charts_dir = run_dir / analysis_subdir / "charts"
            mistakes_csv = run_dir / analysis_subdir / "mistakes.csv"
            _, summary, _, _ = analyze_and_export_mistakes(
                pred_csv_path=str(plan["predictions_csv"]),
                out_mistakes_csv_path=str(mistakes_csv),
                charts_dir=str(charts_dir),
            )
            analysis_summary = summary
        return plan, preds, analysis_summary

    # -------------------------------------------------------------------------
    # Normal batch resume path
    # -------------------------------------------------------------------------
    bid = meta["batch_id"]

    info = wait_for_batch(bid, poll_secs=20)
    status = info.get("status")
    if status != "completed":
        raise RuntimeError(f"Batch ended with status='{status}'")

    out_id = info.get("output_file_id")
    err_id = info.get("error_file_id")

    outputs_jsonl_path: Optional[str] = None
    errors_jsonl_path: Optional[str] = None

    if out_id:
        outputs_jsonl_path = str(plan["outputs_jsonl"])
        download_file_content(out_id, outputs_jsonl_path)
    if err_id:
        errors_jsonl_path = str(run_dir / "errors.jsonl")
        download_file_content(err_id, errors_jsonl_path)

    if not outputs_jsonl_path:
        # In resume mode we can't rebuild full source df; save whatever we have
        raise RuntimeError("No outputs.jsonl available for resumed batch.")

    # Parse minimal + save
    src_shell = pd.DataFrame(columns=["tweet_id", "tweet_text", "class_label"])
    preds = parse_outputs_S_to_df(outputs_jsonl_path, src_shell, errors_jsonl_path=errors_jsonl_path)
    preds.to_csv(plan["predictions_csv"], index=False)

    analysis_summary = None
    if do_analysis:
        charts_dir = run_dir / analysis_subdir / "charts"
        mistakes_csv = run_dir / analysis_subdir / "mistakes.csv"
        _, summary, _, _ = analyze_and_export_mistakes(
            pred_csv_path=str(plan["predictions_csv"]),
            out_mistakes_csv_path=str(mistakes_csv),
            charts_dir=str(charts_dir),
        )
        analysis_summary = summary

    return plan, preds, analysis_summary
