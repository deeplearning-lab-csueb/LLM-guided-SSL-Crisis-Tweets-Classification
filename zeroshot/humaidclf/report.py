# humaidclf/report.py
"""
Reporting utilities for curated results with a top summary table and zoomable charts.

Directory layout you maintain manually:
  results/<event>/<split>/<model>/<run_name>/
    predictions.csv
    analysis/
      charts/
        confusion_matrix_counts.png
        confusion_matrix_row_normalized.png
        per_class_error_rate.png
        per_class_f1.png
        top_confusions.png   (optional)
        confusion_matrix.csv
        confusion_matrix_row_normalized.csv
        per_class_metrics.csv
        summary.json
      mistakes.csv

Then call:
  build_results_index("results", recompute=False)
to generate results/index.html.

What this page shows (per SPLIT section):
  1) Summary table (Event, Model, Run, Test size, Accuracy, Macro-F1, [Scope], [OOS preds]),
     with:
       • Clickable headers to sort
       • Best-run badges (Macro-F1, Accuracy) per (split, event)
       • OOS preds is clickable to show a breakdown of out-of-scope labels (label → count)
  2) Detail cards with embedded chart previews (click to zoom)
     • “view” link for OOS breakdown inside each card

Recompute mode (optional):
- Set recompute=True to ignore/outdate old summaries and recompute from predictions.csv
  using the current eval.py (truth-only scope, sklearn macro-F1, OOS counter).
- New artifacts are written to <run>/<recompute_subdir>/ by default (non-destructive).

OOS breakdown persistence:
- If summary.json includes `oos_breakdown`, we use it.
- Otherwise we compute it from predictions.csv by:
  • deriving the truth label set from non-null `class_label`
  • counting `predicted_label` values outside that set
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import shutil
import pandas as pd

# Prefer not to fail import if user calls report without eval installed in same env.
try:
    from .eval import analyze_and_export_mistakes
except Exception:
    analyze_and_export_mistakes = None  # type: ignore


# Images we try to embed in each run card (skipped if missing)
IMG_FILES = [
    "confusion_matrix_counts.png",
    "confusion_matrix_row_normalized.png",
    "per_class_error_rate.png",
    "per_class_f1.png",
    "top_confusions.png",  # optional
]


# ---------- Helpers for manual promotion (optional) ----------

def promote_run_to_results(run_dir: str | Path, results_root: str | Path, run_name: str | None = None) -> Path:
    """
    Copy one completed run into the curated results tree.

    Source (run_dir):
      runs/<event>/<split>/<model>/<timestamp-tag>/

    Destination (results_root):
      results/<event>/<split>/<model>/<run_name>/
        if run_name is None, the source folder name is used.

    We copy only 'predictions.csv' and 'analysis/' to keep results lean.
    """
    run_dir = Path(run_dir)
    results_root = Path(results_root)

    model = run_dir.parent.name
    split = run_dir.parent.parent.name
    event = run_dir.parent.parent.parent.name
    run_id = run_dir.name
    run_name = run_name or run_id

    dest = results_root / event / split / model / run_name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    src_pred = run_dir / "predictions.csv"
    src_analysis = run_dir / "analysis"
    if not src_pred.exists() or not src_analysis.exists():
        raise FileNotFoundError("Run directory must contain predictions.csv and analysis/")

    shutil.copy2(src_pred, dest / "predictions.csv")
    shutil.copytree(src_analysis, dest / "analysis", dirs_exist_ok=True)

    print("Promoted to:", dest)
    return dest


# ---------- Collector & (optional) recompute ----------

def _read_summary_if_exists(analysis_dir: Path) -> Optional[dict]:
    """Read summary.json from charts/ or analysis/ root, if present."""
    summary_json = analysis_dir / "charts" / "summary.json"
    if not summary_json.exists():
        summary_json = analysis_dir / "summary.json"
    if summary_json.exists():
        try:
            with open(summary_json, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _compute_oos_breakdown_from_preds(pred_csv: Path) -> List[Tuple[str, int]]:
    """
    Compute out-of-scope breakdown from predictions.csv by:
      - deriving the truth label set from non-null class_label
      - counting predicted_label values not in that set
    Returns list of (label, count) sorted by count desc.
    """
    if not pred_csv.exists():
        return []
    try:
        df = pd.read_csv(pred_csv)
    except Exception:
        return []

    truth = (
        df.get("class_label", pd.Series(dtype=object))
          .astype(str).str.strip()
          .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
          .dropna()
    )
    if truth.empty:
        return []

    truth_set = set(truth.tolist())

    preds = (
        df.get("predicted_label", pd.Series(dtype=object))
          .astype(str).str.strip()
          .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
          .dropna()
    )
    if preds.empty:
        return []

    oos = preds[~preds.isin(truth_set)]
    if oos.empty:
        return []

    counts = oos.value_counts().sort_values(ascending=False)
    return [(lbl, int(cnt)) for lbl, cnt in counts.items()]


def _recompute_analysis(run_dir: Path, target_subdir: str) -> Optional[Tuple[dict, Path]]:
    """
    Recompute analysis artifacts using current eval.py from predictions.csv.

    Writes to:
      <run_dir>/<target_subdir>/charts/
    Returns (summary_dict, charts_dir) on success, else None if eval is unavailable or predictions missing.
    """
    if analyze_and_export_mistakes is None:
        print(f"[recompute] eval.analyze_and_export_mistakes not available; skipping {run_dir}")
        return None

    pred_csv = run_dir / "predictions.csv"
    if not pred_csv.exists():
        print(f"[recompute] predictions.csv missing; skipping {run_dir}")
        return None

    charts_dir = run_dir / target_subdir / "charts"
    mistakes_csv = run_dir / target_subdir / "mistakes.csv"
    charts_dir.parent.mkdir(parents=True, exist_ok=True)

    _, summary, _, _ = analyze_and_export_mistakes(
        pred_csv_path=str(pred_csv),
        out_mistakes_csv_path=str(mistakes_csv),
        charts_dir=str(charts_dir),
        # rely on eval.py defaults: truth-only scope, sklearn macro-F1, etc.
    )

    # If the recomputed summary did not include the breakdown, compute and attach it
    if "oos_breakdown" not in summary:
        summary["oos_breakdown"] = _compute_oos_breakdown_from_preds(pred_csv)
        if "invalid_pred_outside_truth" not in summary:
            summary["invalid_pred_outside_truth"] = sum(c for _, c in summary["oos_breakdown"])
        # persist updated summary.json alongside other charts
        with open(charts_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary, charts_dir


def _maybe_collect_entry(event: str, split: str, model: str, run_path: Path,
                         recompute: bool, recompute_missing_only: bool, recompute_subdir: str) -> Optional[Dict]:
    """
    Collect a single run entry. Optionally recompute analysis and prefer recomputed outputs.
    """
    analysis_dir = run_path / "analysis"
    summary = _read_summary_if_exists(analysis_dir)

    # Decide whether to recompute
    need_recompute = False
    if recompute:
        need_recompute = True
        if recompute_missing_only and summary is not None:
            need_recompute = False  # already has summary.json; user asked "missing only"

    # If recompute requested OR no summary, try to recompute into <recompute_subdir>/
    recomputed_summary = None
    recomputed_charts_dir = None
    if need_recompute:
        rc = _recompute_analysis(run_path, recompute_subdir)
        if rc is not None:
            recomputed_summary, recomputed_charts_dir = rc
        else:
            if summary is None:
                return None  # nothing to display

    # Choose which summary/charts to use:
    use_summary = recomputed_summary or summary
    if use_summary is None:
        return None

    # Charts dir to show (prefer recomputed)
    charts_dir_path = recomputed_charts_dir if (recomputed_charts_dir and recomputed_charts_dir.exists()) else (
        (analysis_dir / "charts") if (analysis_dir / "charts").exists() else analysis_dir
    )

    # Build entry
    entry = {
        "event": event,
        "split": split,
        "model": model,
        "run_name": run_path.name,
        "dir": str(run_path),
        "predictions_csv": str(run_path / "predictions.csv"),
        "test_size": use_summary.get("num_total_with_truth", 0),
        "accuracy": use_summary.get("accuracy", 0.0),
        "macro_f1": use_summary.get("macro_f1", 0.0),
        "charts_dir": str(charts_dir_path),
        # Optional fields (safe if missing)
        "labels_scope": use_summary.get("labels_scope", None),
        "invalid_pred_outside_truth": use_summary.get("invalid_pred_outside_truth", 0),
        "oos_breakdown": use_summary.get("oos_breakdown", None),
    }

    # If oos_breakdown is absent, compute from predictions.csv (backfill)
    if not entry.get("oos_breakdown"):
        bc = _compute_oos_breakdown_from_preds(Path(entry["predictions_csv"]))
        entry["oos_breakdown"] = bc
        if not entry.get("invalid_pred_outside_truth"):
            entry["invalid_pred_outside_truth"] = sum(c for _, c in bc)

    return entry


def _collect_results(results_root: Path,
                     recompute: bool,
                     recompute_missing_only: bool,
                     recompute_subdir: str) -> List[Dict]:
    """
    Traverse results tree and collect entries.
    If recompute=True, re-run eval on predictions.csv and write to <recompute_subdir>/,
    preferring those artifacts in the UI.
    """
    entries: List[Dict] = []

    for event_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        event = event_dir.name
        for split_dir in sorted([p for p in event_dir.iterdir() if p.is_dir()]):
            split = split_dir.name
            for model_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
                model = model_dir.name

                # Case A: model/<run_name> subfolders
                found_any = False
                for sub in sorted([p for p in model_dir.iterdir() if p.is_dir()]):
                    entry = _maybe_collect_entry(
                        event, split, model, sub,
                        recompute=recompute,
                        recompute_missing_only=recompute_missing_only,
                        recompute_subdir=recompute_subdir,
                    )
                    if entry:
                        entries.append(entry)
                        found_any = True

                # Case B: directly under model (fallback)
                if not found_any:
                    entry = _maybe_collect_entry(
                        event, split, model, model_dir,
                        recompute=recompute,
                        recompute_missing_only=recompute_missing_only,
                        recompute_subdir=recompute_subdir,
                    )
                    if entry:
                        entries.append(entry)

    return entries


# ---------- Rendering ----------

def _render_summary_table(df: pd.DataFrame) -> str:
    """
    Render compact, sortable HTML table for a given split section:
      Event | Model | Run | Test size | Accuracy | Macro-F1 | [Scope] | [OOS preds]

    - Adds best-run badges per (split, event) for Macro-F1 and Accuracy (ties allowed).
    - Sortable by clicking column headers (numeric-aware).
    - OOS preds is clickable (opens modal with label→count breakdown).
    - NEW: rows are group-striped by event; models are shown as small colored pills.
    """
    tbl = df.copy()

    # Optional columns?
    has_scope = "labels_scope" in tbl.columns and tbl["labels_scope"].notna().any()
    has_oos_any = "invalid_pred_outside_truth" in tbl.columns

    # Format display columns; keep raw numeric copies for data-sort attributes
    tbl["Test size"] = tbl["test_size"].astype(int)
    tbl["Accuracy"] = tbl["accuracy"].map(lambda x: f"{x:.4f}")
    tbl["Macro-F1"] = tbl["macro_f1"].map(lambda x: f"{x:.4f}")

    # Best-run flags (within Event)
    event_max_acc = tbl.groupby("event")["accuracy"].transform("max")
    event_max_f1  = tbl.groupby("event")["macro_f1"].transform("max")
    tbl["_best_acc"] = (tbl["accuracy"] >= event_max_acc - 1e-12)
    tbl["_best_f1"]  = (tbl["macro_f1"] >= event_max_f1 - 1e-12)

    # Ensure oos_breakdown column exists (used as data payload for the modal)
    if "oos_breakdown" not in tbl.columns:
        tbl["oos_breakdown"] = None

    # Select FIRST using original names, then rename for display
    base_cols = ["event", "model", "run_name", "Test size", "Accuracy", "Macro-F1", "_best_acc", "_best_f1"]
    if has_scope:
        base_cols.append("labels_scope")
    if has_oos_any:
        base_cols += ["invalid_pred_outside_truth", "oos_breakdown"]
    tbl = tbl[base_cols].rename(columns={
        "event": "Event",
        "model": "Model",
        "run_name": "Run",
        "labels_scope": "Scope",
        "invalid_pred_outside_truth": "OOS preds",
    })

    # Header cells (with data-key for sorting)
    headers = [
        ("Event", "string"), ("Model", "string"), ("Run", "string"),
        ("Test size", "number"), ("Accuracy", "number"), ("Macro-F1", "number")
    ]
    if has_scope: headers.append(("Scope", "string"))
    if has_oos_any: headers.append(("OOS preds", "number"))
    head_cells = "".join([f"<th data-type='{t}'>{h}</th>" for h, t in headers])

    # Row builder with group striping by Event
    rows = []
    grp = 0
    prev_event = None

    for _, r in tbl.iterrows():
        if r["Event"] != prev_event:
            grp ^= 1  # flip 0 <-> 1 when event changes
            prev_event = r["Event"]

        # Model pill styles
        m = (r["Model"] or "").lower()
        if "5-mini" in m:
            pill_cls = "pill-5mini"
        elif "5-pro" in m:
            pill_cls = "pill-5-pro" 
        elif "5-nano" in m:
            pill_cls = "pill-5-nano"
        elif "5.1" in m:
            pill_cls = "pill-5-1"            
        elif "5" in m:
            pill_cls = "pill-5"              
        elif "4o-mini" in m:
            pill_cls = "pill-4omini"
        elif "4.1" in m or "gpt-4-1" in m:
            pill_cls = "pill-41"
        else:
            pill_cls = "pill-4o"        
        model_html = f"<span class='pill {pill_cls}'>{r['Model']}</span>"

        best_acc_badge = "<span class='badge badge-acc' title='Best Accuracy in Event'>best</span>" if r["_best_acc"] else ""
        best_f1_badge  = "<span class='badge badge-f1'  title='Best Macro-F1 in Event'>best</span>" if r["_best_f1"] else ""

        scope_td = f"<td class='scope'>{r['Scope']}</td>" if has_scope else ""

        # OOS cell: clickable only when > 0; attach breakdown JSON as data attribute
        oos_td = ""
        if has_oos_any:
            oos_val = int(r.get("OOS preds", 0))
            breakdown = r.get("oos_breakdown", None) or []
            title = f"{r['Event']} • {r['Model']} • {r['Run']}".strip(" •")
            rules = next((p for p in r['Run'].split('-') if p.upper().startswith('RULES')), '')
            data_attr = f" data-oos='{json.dumps(breakdown, ensure_ascii=False)}' data-title='{title}'"
            if oos_val > 0:
                oos_td = f"<td class='num bad' data-sort='{oos_val}'><button class='oos-btn'{data_attr} aria-label='View OOS breakdown'>{oos_val}</button></td>"
            else:
                oos_td = f"<td class='num' data-sort='0'>0</td>"

        rows.append(
            f"<tr class='grp-{grp}' title='{r['Event']}\n{r['Model']}\n{rules}'>"
            f"<td><strong>{r['Event']}</strong></td>"
            f"<td>{model_html}</td>"
            f"<td><code>{r['Run']}</code></td>"
            f"<td class='num' data-sort='{int(r['Test size'])}'>{int(r['Test size'])}</td>"
            f"<td class='num' data-sort='{float(r['Accuracy'])}'>{r['Accuracy']} {best_acc_badge}</td>"
            f"<td class='num' data-sort='{float(r['Macro-F1'])}'>{r['Macro-F1']} {best_f1_badge}</td>"
            f"{scope_td}{oos_td}"
            "</tr>"
        )

    return (
        "<table class='summary sortable'>"
        f"<thead><tr>{head_cells}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


# Example usages:
# build_results_index("results")
# build_results_index("results", out_html="results/index_recap.html", recompute=True, recompute_missing_only=True)
# build_results_index("results", out_html="results/index_fresh.html", recompute=True, recompute_missing_only=False)

def build_results_index(
    results_root: str | Path,
    out_html: str | Path = None,
    *,
    recompute: bool = False,
    recompute_missing_only: bool = True,
    recompute_subdir: str = "analysis_recomputed",
) -> pd.DataFrame:
    """
    Build an HTML index page summarizing curated results.

    Layout
    ------
      1) Summary by split:
         - For each split: one compact table (Event, Model, Run, Test size, Accuracy, Macro-F1, [Scope], [OOS preds])
      2) Details by split:
         - For each split: labeled section with per-run cards and embedded charts.

    Options
    -------
    recompute : bool
        If True, recompute analysis from predictions.csv for each run with current eval.py.
    recompute_missing_only : bool
        If True, only recompute when original summary.json is missing.
        Set False to force-recompute every run (new artifacts go to recompute_subdir).
    recompute_subdir : str
        Subdirectory name under each run to write recomputed artifacts (non-destructive).
    """
    results_root = Path(results_root)
    out_html = Path(out_html) if out_html else (results_root / "index.html")

    entries = _collect_results(
        results_root,
        recompute=recompute,
        recompute_missing_only=recompute_missing_only,
        recompute_subdir=recompute_subdir,
    )
    if not entries:
        out_html.write_text("<h2>No results found.</h2>", encoding="utf-8")
        return pd.DataFrame()

    # Sorted for stable grouping; badges computed per split/event later
    df = pd.DataFrame(entries).sort_values(
        ["split", "event", "model", "run_name"]
    ).reset_index(drop=True)

    def rel(p: Path | str) -> str:
        """Return a path relative to results_root with POSIX separators."""
        return str(Path(p).relative_to(results_root)).replace("\\", "/")

    split_order = list(df["split"].unique())

    # ---------- 1) SUMMARY BY SPLIT (tables only) ----------
    summary_sections: List[str] = []
    for split in split_order:
        df_split = df[df["split"] == split].copy()

        base_cols = ["event", "model", "run_name", "test_size", "accuracy", "macro_f1"]
        if "labels_scope" in df_split.columns:
            base_cols.append("labels_scope")
        if "invalid_pred_outside_truth" in df_split.columns:
            base_cols += ["invalid_pred_outside_truth", "oos_breakdown"]

        summary_table_html = _render_summary_table(
            df_split[base_cols].sort_values(["event", "model", "run_name"])
        )

        summary_sections.append(f"""
          <section class="split-summary">
            <h2>Split: <code>{split}</code></h2>
            {summary_table_html}
          </section>
        """)

    # ---------- 2) DETAILS BY SPLIT (cards only) ----------
    detail_sections: List[str] = []
    for split in split_order:
        df_split = df[df["split"] == split].copy()

        rows_html: List[str] = []

        # Per-event maxima for badges on the cards
        max_acc_by_event = df_split.groupby("event")["accuracy"].transform("max")
        max_f1_by_event  = df_split.groupby("event")["macro_f1"].transform("max")
        df_split = df_split.assign(
            _best_acc=(df_split["accuracy"] >= max_acc_by_event - 1e-12),
            _best_f1=(df_split["macro_f1"] >= max_f1_by_event - 1e-12),
        )

        for _, r in df_split.iterrows():
            charts = Path(r["charts_dir"])
            imgs = []
            for name in IMG_FILES:
                fp = charts / name
                if fp.exists():
                    imgs.append(
                        f'<div class="imgbox">'
                        f'  <img class="zoomable" src="{rel(fp)}" alt="{name}" '
                        f'       data-fullsrc="{rel(fp)}">'
                        f'</div>'
                    )
            imgs_html = "\n".join(imgs) if imgs else "<em>No charts found.</em>"

            run_label = f" — <code>{r['run_name']}</code>" if r["run_name"] else ""

            # Optional metric rows (only if present)
            scope = r.get("labels_scope", None)
            scope_row = f"<tr><th>Scope</th><td>{scope}</td></tr>" if pd.notna(scope) else ""

            # OOS row with inline "view" link if any
            oos_val = int(r.get("invalid_pred_outside_truth", 0))
            if oos_val > 0:
                oos_breakdown = r.get("oos_breakdown", []) or []
                data_attr = (
                    f" data-oos='{json.dumps(oos_breakdown, ensure_ascii=False)}'"
                    f" data-title='{r['event']} • {r['model']} • {r['run_name']}'"
                )
                oos_row = (
                    "<tr><th>OOS preds</th>"
                    f"<td class='num bad'>{oos_val} "
                    f"<a href='#' class='oos-link'{data_attr}>view</a></td></tr>"
                )
            else:
                oos_row = "<tr><th>OOS preds</th><td class='num'>0</td></tr>"

            # Badges
            best_acc_badge = (
                "<span class='badge badge-acc' title='Best Accuracy in Event'>best</span>"
                if r["_best_acc"] else ""
            )
            best_f1_badge = (
                "<span class='badge badge-f1'  title='Best Macro-F1 in Event'>best</span>"
                if r["_best_f1"] else ""
            )

            rows_html.append(f"""
            <section class="card">
              <div class="head">
                <div>
                  <h3>{r['event']} — {split}</h3>
                  <div class="sub">model: <code>{r['model']}</code>{run_label}</div>
                  <div class="sub path">{rel(r['dir'])}</div>
                </div>
                <table class="metrics">
                  <tr><th>Test size</th><td>{int(r['test_size'])}</td></tr>
                  <tr><th>Accuracy</th><td>{r['accuracy']:.4f} {best_acc_badge}</td></tr>
                  <tr><th>Macro-F1</th><td>{r['macro_f1']:.4f} {best_f1_badge}</td></tr>
                  {scope_row}
                  {oos_row}
                </table>
              </div>
              <div class="imgs">
                {imgs_html}
              </div>
            </section>
            """)

        detail_sections.append(f"""
          <section class="split-detail">
            <h2>Split: <code>{split}</code></h2>
            <div class="grid">
              {''.join(rows_html)}
            </div>
          </section>
        """)

    summary_block = "".join(summary_sections)
    details_block = "".join(detail_sections)

    # Assemble full page
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>HumAID Zero-shot Results</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
  h1 {{ margin-top: 0; }}
  h2 {{ margin: 18px 0 8px; }}
  .summary {{ width: 100%; border-collapse: collapse; margin-bottom: 18px; }}
  .summary th, .summary td {{ border: 1px solid #e5e7eb; padding: 8px 10px; }}
  .summary th {{ background: #f9fafb; text-align: left; cursor: pointer; }}
  .summary td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
  .card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }}
  .head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; }}
  .sub {{ color: #6b7280; font-size: 12px; }}
  .sub.path {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  table.metrics {{ border-collapse: collapse; }}
  table.metrics th {{ text-align: left; padding-right: 8px; color: #374151; }}
  table.metrics td {{ text-align: right; font-weight: 600; color: #111827; }}
  .imgs {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; margin-top: 12px; }}
  .imgbox {{ border: 1px solid #eee; border-radius: 8px; padding: 8px; background: #fafafa; }}
  .imgbox img {{ width: 100%; height: auto; display: block; cursor: zoom-in; }}

  /* Badges */
  .badge {{ display: inline-block; padding: 2px 6px; border-radius: 999px; font-size: 10px; margin-left: 6px; vertical-align: 1px; }}
  .badge-acc {{ background: #6efffa; color: #0c484a; border: 1px solid #0c484a; }}  
  .badge-f1  {{ background: #92fcaf; color: #0e3318; border: 1px solid #0e3318; }}  

  /* Optional visual cues */
  td.bad {{ color: #b91c1c; font-weight: 700; }}   /* red numeric for OOS preds > 0 */
  td.scope {{ color: #374151; font-variant: all-small-caps; letter-spacing: .02em; }}

  /* --- Group striping by event --- */
  tr.grp-0 {{ background: #fff7ed; }}
  tr.grp-1 {{ background: #eaf2ff;    }}
  tr.grp-0:hover, tr.grp-1:hover {{ background: #e6f25e; }}

  /* Left color bar per event (two alternating hues) */
  tr.grp-0 td:first-child {{ border-left: 4px solid #60a5fa; }} /* blue */
  tr.grp-1 td:first-child {{ border-left: 4px solid #34d399; }} /* green */

  /* Model pills */
  .pill {{ display:inline-block; padding:2px 6px; border-radius:999px; font-size:11px; line-height:1; }}
  .pill-41     {{ background:#9ab4fc; color:#132557; border:1px solid #132557; }}  
  .pill-4o     {{ background:#84e38d; color:#1d4021; border:1px solid #1d4021; }}  
  .pill-4omini {{ background:#eefa4d; color:#3d4011; border:1px solid #3d4011; }}
  .pill-5mini {{ background:#f6edfa; color:#5b0c7d; border:1px solid #5b0c7d; }}
  .pill-5 {{ background:#eed7f7; color:#5b0c7d; border:1px solid #5b0c7d; }}
  .pill-5-pro {{ background:#e5aefc; color:#5b0c7d; border:1px solid #5b0c7d; }}
  .pill-5-nano {{ background:#d784fa; color:#5b0c7d; border:1px solid #5b0c7d; }}
  .pill-5-1 {{ background:#bc20fa; color:#fff94a; border:1px solid #33074a; }}

  /* Modal (for images and OOS tables) */
  .modal {{
    position: fixed; inset: 0; display: none;
    background: rgba(0,0,0,0.7); z-index: 9999;
    align-items: center; justify-content: center;
    padding: 24px;
  }}
  .modal.open {{ display: flex; }}
  .modal .panel {{
    background: #fff; color: #111827; border-radius: 10px; max-width: 90vw; max-height: 90vh; overflow: auto;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4); padding: 16px 18px;
  }}
  .modal img {{
    max-width: 95vw; max-height: 85vh;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    border-radius: 8px; background: #fff;
  }}
  .modal .close {{
    position: absolute; top: 12px; right: 16px;
    font-size: 28px; color: #fff; cursor: pointer; user-select: none;
  }}

  .oos-table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
  .oos-table th, .oos-table td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }}
  .oos-table th {{ background: #f9fafb; }}
  .oos-table td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
</style>
</head>
<body>
  <h1>HumAID Zero-shot Results</h1>

  <section>
    <h2>Summary by split</h2>
    {summary_block}
  </section>

  <hr style="margin:24px 0; border:none; border-top:1px solid #e5e7eb;" />

  <section>
    <h2>Details by split</h2>
    {details_block}
  </section>

  <!-- Shared Modal -->
  <div id="imgModal" class="modal" aria-hidden="true">
    <span class="close" title="Close (Esc)">&times;</span>
    <div class="panel" id="modalPanel"></div>
  </div>

  <script>
  (function() {{
    const modal = document.getElementById('imgModal');
    const panel = document.getElementById('modalPanel');
    const closeBtn = modal.querySelector('.close');

    function openModalWithImage(src) {{
      panel.innerHTML = '<img alt="Preview" src="' + src + '">';
      modal.classList.add('open');
      modal.setAttribute('aria-hidden', 'false');
    }}

    function openModalWithOOS(title, rows) {{
      let html = '<h3 style="margin:0 0 8px 0;">Out-of-scope predictions</h3>';
      if (title) html += '<div style="color:#6b7280;margin-bottom:8px;">' + title + '</div>';
      if (!rows || rows.length === 0) {{
        html += '<em>No out-of-scope predictions.</em>';
      }} else {{
        html += '<table class="oos-table"><thead><tr><th>Label</th><th>Count</th></tr></thead><tbody>';
        for (const [lbl, cnt] of rows) {{
          html += '<tr><td><code>' + lbl + '</code></td><td class="num">' + cnt + '</td></tr>';
        }}
        html += '</tbody></table>';
      }}
      panel.innerHTML = html;
      modal.classList.add('open');
      modal.setAttribute('aria-hidden', 'false');
    }}

    function closeModal() {{
      modal.classList.remove('open');
      modal.setAttribute('aria-hidden', 'true');
      panel.innerHTML = '';
    }}

    document.addEventListener('click', function(e) {{
      const img = e.target.closest('img.zoomable');
      if (img) {{
        const full = img.getAttribute('data-fullsrc') || img.src;
        openModalWithImage(full);
        return;
      }}
      const btn = e.target.closest('.oos-btn');
      if (btn) {{
        try {{
          const data = JSON.parse(btn.getAttribute('data-oos') || '[]');
          const title = btn.getAttribute('data-title') || '';
          openModalWithOOS(title, data);
        }} catch (err) {{
          openModalWithOOS('', []);
        }}
        return;
      }}
      const link = e.target.closest('.oos-link');
      if (link) {{
        e.preventDefault();
        try {{
          const data = JSON.parse(link.getAttribute('data-oos') || '[]');
          const title = link.getAttribute('data-title') || '';
          openModalWithOOS(title, data);
        }} catch (err) {{
          openModalWithOOS('', []);
        }}
        return;
      }}
      if (e.target === modal || e.target === closeBtn) {{
        closeModal();
      }}
    }});

    document.addEventListener('keydown', function(e) {{
      if (e.key === 'Escape' && modal.classList.contains('open')) {{
        closeModal();
      }}
    }});
  }})();

  // Sortable tables: click header to toggle ascending/descending.
  (function() {{
    function compare(a, b, type) {{
      if (type === 'number') {{
        const x = parseFloat(a); const y = parseFloat(b);
        if (isNaN(x) && isNaN(y)) return 0;
        if (isNaN(x)) return -1;
        if (isNaN(y)) return 1;
        return x - y;
      }}
      return ('' + a).localeCompare(('' + b), undefined, {{ sensitivity: 'base' }});
    }}

    document.querySelectorAll('table.sortable').forEach(function(table) {{
      const thead = table.querySelector('thead');
      if (!thead) return;

      const directions = []; // per-column: 1 asc, -1 desc

      thead.addEventListener('click', function(e) {{
        const th = e.target.closest('th');
        if (!th) return;
        const idx = Array.prototype.indexOf.call(th.parentNode.children, th);
        const dtype = th.getAttribute('data-type') || 'string';
        directions[idx] = directions[idx] ? -directions[idx] : 1;

        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));

        rows.sort(function(r1, r2) {{
          const c1 = r1.children[idx];
          const c2 = r2.children[idx];
          const v1 = c1.getAttribute('data-sort') || c1.textContent.trim();
          const v2 = c2.getAttribute('data-sort') || c2.textContent.trim();
          return directions[idx] * compare(v1, v2, dtype);
        }});

        rows.forEach(function(r) {{ tbody.appendChild(r); }});
      }});
    }});
  }})();
  </script>

  <footer style="margin-top:24px; color:#6b7280; font-size:12px;">
    Generated automatically. Click any chart to zoom. Press ESC to close. Click headers to sort tables.
  </footer>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")
    return df

