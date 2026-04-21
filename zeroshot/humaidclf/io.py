import pathlib, datetime
import pandas as pd

def load_tsv(path, id_col="tweet_id", text_col="tweet_text", label_col="class_label"):
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    rename = {}
    if id_col != "tweet_id":   rename[id_col] = "tweet_id"
    if text_col != "tweet_text": rename[text_col] = "tweet_text"
    if label_col and label_col in df.columns and label_col != "class_label":
        rename[label_col] = "class_label"
    df = df.rename(columns=rename)
    assert "tweet_id" in df.columns and "tweet_text" in df.columns, "Need tweet_id + tweet_text"
    if "class_label" not in df.columns:
        df["class_label"] = ""
    return df

def plan_run_dirs(dataset_path: str, out_root: str = "runs", model: str = "gpt-4o-mini", tag: str = ""):
    p = pathlib.Path(dataset_path)
    event = p.parent.name
    stem = p.stem
    split = stem.replace(event, "").strip("_") or "data"
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{ts}{('-'+tag) if tag else ''}"
    run_dir = pathlib.Path(out_root) / event / split / model / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "event": event, "split": split, "model": model, "run_id": run_id, "dir": run_dir,
        "requests_jsonl": run_dir / "requests.jsonl",
        "outputs_jsonl": run_dir / "outputs.jsonl",
        "predictions_csv": run_dir / "predictions.csv",
        "batch_meta_json": run_dir / "batch_meta.json",
    }
