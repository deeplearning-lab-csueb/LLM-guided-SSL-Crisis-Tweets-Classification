# eval.py
# Metrics + analysis utilities for HumAID zero-shot runs.
# Changes vs. previous version:
#   - Normalize labels once up front (strip/empty/NaN cleanup)
#   - Resolve label order from TRUTH ONLY by default (scope="truth")
#   - Standardize per-class error rate to 1 - recall (row-wise misclassification rate)
#   - Track and report predictions that fall outside the event’s truth label set

import pathlib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List
from sklearn.metrics import f1_score

# Try to import the canonical label order from the package; if unavailable, fall back later.
try:
    from .prompts import LABELS as CANON_LABELS
except Exception:
    CANON_LABELS = None


# ---------------------------
# Normalization / label scope
# ---------------------------
def _clean_label_series(s: pd.Series) -> pd.Series:
    """
    Normalize a label column for reliable comparisons:
    - coerce to string
    - strip whitespace
    - convert "", "nan", "None" to NA
    """
    s = s.astype(str).str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return s


def _resolve_label_order(
    truth: pd.Series,
    pred: pd.Series,
    label_order=None,
    scope: str = "truth",
):
    """
    Decide the label set / order for metrics & plots.

    scope:
      - 'truth'     -> labels present in TRUTH ONLY (recommended for HumAID events)
      - 'union'     -> labels present in truth ∪ pred
      - 'canonical' -> CANON_LABELS ∩ (truth ∪ pred)

    Priority:
      1) explicit label_order arg if provided (filtered to chosen scope)
      2) canonical list (CANON_LABELS) if scope='canonical'
      3) derived set per 'truth' or 'union'
    """
    truth = _clean_label_series(truth).dropna()
    pred  = _clean_label_series(pred).dropna()

    truth_set = set(truth.unique())
    union_set = truth_set | set(pred.unique())

    if label_order and len(label_order) > 0:
        base = truth_set if scope == "truth" else union_set
        return [l in base and l or None for l in label_order if l in base] or sorted(base)

    if scope == "canonical" and CANON_LABELS:
        return [l for l in CANON_LABELS if l in union_set] or sorted(union_set)

    if scope == "union":
        return sorted(union_set)

    # default: truth-only
    return sorted(truth_set)


# -----------
# Core metric
# -----------
from typing import Optional, List
import numpy as np
import pandas as pd

def macro_f1(
    df: pd.DataFrame,
    truth_col: str = "class_label",
    pred_col: str = "predicted_label",
    label_order: Optional[List[str]] = None,
    scope: str = "truth",  # default matches our evaluation policy
) -> float:
    """
    Macro-F1 on a DataFrame of predictions, using scikit-learn when available.

    - Normalizes labels (NaN/empty/whitespace).
    - Drops rows without truth or pred.
    - Label set defaults to TRUTH ONLY (scope='truth') so events with missing classes
      don't get penalized by hallucinated columns.
    - Uses sklearn.metrics.f1_score(average='macro', labels=<resolved labels>, zero_division=0).
    - Falls back to a manual computation if sklearn is unavailable.
    """
    sub = df.copy()
    sub[truth_col] = _clean_label_series(sub.get(truth_col, pd.Series(dtype=object)))
    sub[pred_col]  = _clean_label_series(sub.get(pred_col,  pd.Series(dtype=object)))
    sub = sub.dropna(subset=[truth_col, pred_col])

    if sub.empty:
        return float("nan")

    labels = _resolve_label_order(sub[truth_col], sub[pred_col], label_order, scope=scope)
    if not labels:
        return float("nan")

    y_true = sub[truth_col].to_numpy()
    y_pred = sub[pred_col].to_numpy()

    # Preferred: scikit-learn
    try:        
        return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    except Exception:
        # Fallback: manual computation (same logic as before)
        f1s = []
        for c in labels:
            tp = int(((sub[truth_col] == c) & (sub[pred_col] == c)).sum())
            fp = int(((sub[truth_col] != c) & (sub[pred_col] == c)).sum())
            fn = int(((sub[truth_col] == c) & (sub[pred_col] != c)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            f1s.append(f1)
        return float(sum(f1s) / len(f1s))

# -------------------------------
# Full eval + artifacts & charts
# -------------------------------
def analyze_and_export_mistakes(
    pred_csv_path: str,
    out_mistakes_csv_path: str,
    charts_dir: str | None = None,
    truth_col: str = "class_label",
    pred_col: str = "predicted_label",
    id_col: str = "tweet_id",
    text_col: str = "tweet_text",
    save_summary_json: bool = True,
    annotate_norm_cm: bool = True,   # show numbers on normalized heatmap
    label_order: list[str] | None = None,
    scope: str = "truth",            # default to truth-only metrics/plots
):
    """
    Loads predictions CSV, exports misclassified rows, computes metrics,
    and saves charts/tables.

    Saved artifacts (if charts_dir is provided):
      - confusion_matrix_counts.png
      - confusion_matrix_row_normalized.png
      - per_class_f1.png
      - per_class_error_rate.png
      - top_confusions.png   (if there are off-diagonal errors)
      - per_class_metrics.csv
      - confusion_matrix.csv
      - confusion_matrix_row_normalized.csv
      - summary.json         (if save_summary_json=True)

    Returns: (mistakes_df, summary_dict, per_class_df, conf_mat_df)
    """
    # ---------- Load ----------
    df = pd.read_csv(pred_csv_path)

    # Normalize labels early for consistent behavior everywhere
    df[truth_col] = _clean_label_series(df.get(truth_col, pd.Series(dtype=object)))
    df[pred_col]  = _clean_label_series(df.get(pred_col,  pd.Series(dtype=object)))

    # Keep only rows that have a ground-truth label; predictions can still be NA and will be dropped later
    df_eval = df.dropna(subset=[truth_col]).copy()

    # ---------- Mistakes export ----------
    # (Rows with NA predicted_label are considered mistakes; consistent with truth-only scope.)
    mistakes_df = df_eval.loc[df_eval[truth_col] != df_eval[pred_col]].copy()
    out_p = pathlib.Path(out_mistakes_csv_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    mistakes_df.to_csv(out_p, index=False)

    # ---------- Determine label order (truth-only by default) ----------
    labels = _resolve_label_order(df_eval[truth_col], df_eval[pred_col], label_order, scope=scope)
    if not labels:
        # Nothing to evaluate; return early
        empty_conf = pd.DataFrame()
        summary = {
            "num_total_with_truth": 0,
            "num_correct": 0,
            "num_incorrect": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "labels": [],
            "labels_scope": scope,
            "invalid_pred_outside_truth": 0,
        }
        return mistakes_df, summary, pd.DataFrame(), empty_conf

    # Count predictions that are OUTSIDE the truth label set (QA signal)
    truth_set = set(labels)
    invalid_pred_outside_truth = int((~df_eval[pred_col].isin(truth_set)).sum())

    # ---------- Confusion matrix (counts)
    # Restrict both axes to the chosen label set; predictions outside are excluded from the table.
    conf_mat_df = (
        pd.crosstab(df_eval[truth_col], df_eval[pred_col], dropna=False)
        .reindex(index=labels, columns=labels, fill_value=0)
    )

    # ---------- Per-class metrics ----------
    C = conf_mat_df.values
    tp = np.diag(C)
    support_true = C.sum(axis=1)  # row sums
    support_pred = C.sum(axis=0)  # col sums
    fp = support_pred - tp
    fn = support_true - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall    = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    f1        = np.divide(2 * precision * recall, precision + recall,
                          out=np.zeros_like(tp, dtype=float), where=(precision + recall) != 0)
    # Standard row-wise misclassification rate (aligns with row-normalized CM)
    error_rate = 1.0 - recall

    per_class_df = pd.DataFrame({
        "label": labels,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "support":   support_true.astype(int),
        "error_rate": error_rate,
    })
    # No sort here; 'labels' is already in the desired order.

    # ---------- Aggregates ----------
    total = C.sum()
    accuracy = (tp.sum() / total) if total else 0.0

    # Use the shared macro_f1 implementation (truth-only by default unless you override `scope`)
    macro = macro_f1(
        df_eval,
        truth_col=truth_col,
        pred_col=pred_col,
        label_order=label_order,
        scope=scope,
    )

    summary = {
        "num_total_with_truth": int(len(df_eval)),
        "num_correct": int(tp.sum()),
        "num_incorrect": int(len(mistakes_df)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro) if pd.notna(macro) else 0.0,
        "labels": labels,
        "labels_scope": scope,
        "invalid_pred_outside_truth": invalid_pred_outside_truth,  # QA counter
    }

    # ---------- Charts & Tables ----------
    if charts_dir:
        charts_dir = pathlib.Path(charts_dir)
        charts_dir.mkdir(parents=True, exist_ok=True)

        # (A) Confusion matrix (raw counts) with colorbar
        fig = plt.figure(figsize=(8 + 0.3 * len(labels), 6 + 0.3 * len(labels)))
        im = plt.imshow(conf_mat_df.values, interpolation="nearest")
        plt.title("Confusion Matrix (counts)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel("Count", rotation=270, labelpad=12)
        plt.tight_layout()
        plt.savefig(charts_dir / "confusion_matrix_counts.png", dpi=200)
        plt.close(fig)

        # (B) Confusion matrix (row-normalized to probabilities) with fixed [0,1] range + colorbar
        conf_norm = conf_mat_df.div(conf_mat_df.sum(axis=1).replace(0, 1), axis=0).fillna(0.0)
        fig = plt.figure(figsize=(8 + 0.3 * len(labels), 6 + 0.3 * len(labels)))
        im = plt.imshow(conf_norm.values, interpolation="nearest", vmin=0, vmax=1)
        plt.title("Confusion Matrix (row-normalized)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel("Proportion", rotation=270, labelpad=12)

        # Optional numeric annotations inside cells (format: 0.00)
        if annotate_norm_cm:
            vals = conf_norm.values
            for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    v = vals[i, j]
                    if v > 0:  # set to >=0 to annotate all cells
                        plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

        plt.tight_layout()
        plt.savefig(charts_dir / "confusion_matrix_row_normalized.png", dpi=200)
        plt.close(fig)

        # (C) Per-class F1 (in chosen order)
        fig = plt.figure(figsize=(max(8, 0.6 * len(labels)), 5))
        plt.bar(per_class_df["label"], per_class_df["f1"])
        plt.title("Per-class F1")
        plt.xticks(rotation=90)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(charts_dir / "per_class_f1.png", dpi=200)
        plt.close(fig)

        # (D) Per-class error rate (1 - recall)
        fig = plt.figure(figsize=(max(8, 0.6 * len(labels)), 5))
        plt.bar(per_class_df["label"], per_class_df["error_rate"])
        plt.title("Per-class Error Rate (1 - recall)")
        plt.xticks(rotation=90)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(charts_dir / "per_class_error_rate.png", dpi=200)
        plt.close(fig)

        # (E) Top confusions (off-diagonal counts) — ordered by magnitude
        pairs = [
            (labels[i], labels[j], int(C[i, j]))
            for i in range(len(labels)) for j in range(len(labels))
            if i != j and C[i, j] > 0
        ]
        pairs.sort(key=lambda x: x[2], reverse=True)
        top_k = pairs[:15]
        if top_k:
            fig = plt.figure(figsize=(10, max(4, 0.4 * len(top_k))))
            ylabels = [f"{t} \u2192 {p}" for t, p, _ in top_k]  # → arrow
            counts = [c for _, _, c in top_k]
            y = range(len(top_k))
            plt.barh(list(y), counts)
            plt.yticks(list(y), ylabels)
            plt.gca().invert_yaxis()
            plt.title("Top Confusions (off-diagonal)")
            plt.tight_layout()
            plt.savefig(charts_dir / "top_confusions.png", dpi=200)
            plt.close(fig)

        # Save numeric summaries/tables alongside figures
        per_class_df.to_csv(charts_dir / "per_class_metrics.csv", index=False)
        conf_mat_df.to_csv(charts_dir / "confusion_matrix.csv")  # counts
        conf_norm.to_csv(charts_dir / "confusion_matrix_row_normalized.csv")  # proportions
        if save_summary_json:
            with open(charts_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

    return mistakes_df, summary, per_class_df, conf_mat_df
