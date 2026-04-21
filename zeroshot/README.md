# zeroshot — GPT-4o pseudo-label generation for HumAID

A self-contained sub-module of the [LG-CoTrain](../README.md) project. The `zeroshot/` folder contains the tooling we use to generate zero-shot predictions with GPT-4o on the HumAID disaster-tweet classification task. The train-split predictions are then consumed by the main LG-CoTrain pipeline as the `D_LG` set in Phase 2 co-training.

The sub-module is independent: it has its own Python package (`humaidclf/`), its own copy of the HumAID splits (`Dataset/`), and its own dependency list (`requirements.txt`). It does not import from `lg_cotrain/`, and the main LG-CoTrain pipeline does not import from `humaidclf/`. The two halves communicate by file: a `predictions.csv` file produced here is copied (or symlinked) into `data/pseudo-labelled/<source>/<event>/<event>_train_pred.csv` for use by the main pipeline.

---

## Role within LG-CoTrain

| Stage | What zeroshot does |
|---|---|
| Prompt selection | Run GPT-4o on the **dev** split with 3 candidate prompts and pick the best (NB11) |
| Test evaluation | Run GPT-4o on the **test** split with the chosen prompt to report the zero-shot upper bound (NB12) |
| Pseudo-label generation | Run GPT-4o on the **train** split with the chosen prompt; the resulting `predictions.csv` files are committed under `../data/pseudo-labelled/gpt-4o/` and consumed by the LG-CoTrain pipeline (NB13) |

---

## Folder layout

```
zeroshot/
  README.md                  this documentation
  requirements.txt           Python dependencies for the zeroshot tooling
  humaidclf/                 the OpenAI Batch API runner package (see below)
  rules/humaid_rules.py      RULES_1, RULES_2, RULES_3 prompt variants (RULES_BASELINE and RULES_4 are defined but unused)
  Dataset/HumAID/            local copy of the 10 HumAID events with train/dev/test TSV splits
  11_…dev_RULES1_2_3.ipynb   prompt-selection notebook (run first)
  12_…test_rules_1…ipynb     test-split evaluation with the chosen prompt
  13_…train_rules_1…ipynb    train-split run that produces the LG-CoTrain pseudo-labels
  results/{event}/{train,test,dev}/gpt-4o/   committed predictions + per-event analysis charts
```

### `humaidclf/` package

| File | Purpose |
|---|---|
| `io.py` | TSV loading and run-directory planning |
| `prompts.py` | Canonical label order, system prompt, dynamic per-event user message construction |
| `batch.py` | OpenAI Batch API integration: request building, submission, polling, parsing, sync fallback, multi-key rotation |
| `budget.py` | Token estimation, sample-based budgeting, stratified sharding for events that exceed the per-batch token cap |
| `eval.py` | Macro-F1, accuracy, confusion matrices, per-class metrics, mistake export |
| `report.py` | `promote_run_to_results()` and `build_results_index()` — helpers for promoting individual runs and rebuilding an HTML index (not used by the shipped notebooks) |
| `runner.py` | `run_experiment()` end-to-end: dry-run, build requests, submit batch, poll, download, parse, optional analysis |
| `runner_sharded.py` | `run_experiment_sharded()` for large events split into stratified shards |
| `stratify.py` | Stratified k-fold splitting that preserves class ratios |

### `rules/humaid_rules.py`

Three prompt variants compared in NB11:

- `RULES_1` — compact label list (chosen for the train/test runs based on dev results)
- `RULES_2` — medium-length, with PRIMARY INTENT framing
- `RULES_3` — detailed, with Definition / Include / Exclude clauses per label

`RULES_BASELINE` and `RULES_4` are also defined in the file but were not used in the shipped runs.

---

## Commands

### Install dependencies

```bash
pip install -r zeroshot/requirements.txt
```

The zeroshot sub-module has its own dependency list because it does not need PyTorch, transformers, or any of the main pipeline's heavyweight ML packages. It only needs `pandas`, `requests`, `python-dotenv`, `tiktoken`, and a few small support libraries.

### Run order

The notebooks are numbered to match the workflow:

1. **`11_…dev_RULES1_2_3.ipynb`** — runs GPT-4o on the dev split with RULES_1, RULES_2, RULES_3 across all 10 events, then compares Macro-F1 to pick the best prompt. Result: RULES_1 wins (range 0.601–0.613).
2. **`12_…test_rules_1…ipynb`** — runs GPT-4o on the test split with RULES_1 to report the zero-shot test result (Macro-F1 ≈ 0.641).
3. **`13_…train_rules_1…ipynb`** — runs GPT-4o on the train split with RULES_1 to produce the pseudo-labels consumed by the LG-CoTrain pipeline.

### Programmatic single-run example

```python
from dotenv import load_dotenv; load_dotenv()
from humaidclf import run_experiment
from rules import RULES_1

plan, preds, summary = run_experiment(
    dataset_path="Dataset/HumAID/canada_wildfires_2016/canada_wildfires_2016_train.tsv",
    rules=RULES_1,
    model="gpt-4o",
    tag="modeS-gpt-4o-RULES1-filtered",
    dryrun_n=20,
    poll_secs=300,
    out_root="results",      # writes under results/{event}/{split}/{model}/{timestamp-tag}/
    do_analysis=True,
)
```

### Use a pseudo-label run with the main LG-CoTrain pipeline

The pseudo-labels needed by the LG-CoTrain pipeline are already committed under [`../data/pseudo-labelled/gpt-4o/`](../data/pseudo-labelled/gpt-4o/). To regenerate them (or generate pseudo-labels with a different model), run NB13 with your model of choice and copy the resulting `predictions.csv` per event into the main data tree:

```bash
mkdir -p ../data/pseudo-labelled/gpt-4o/canada_wildfires_2016
cp results/canada_wildfires_2016/train/gpt-4o/<timestamp-tag>/predictions.csv \
   ../data/pseudo-labelled/gpt-4o/canada_wildfires_2016/canada_wildfires_2016_train_pred.csv
```

Then run LG-CoTrain with `--pseudo-label-source gpt-4o`. The CSV must contain `tweet_id`, `tweet_text`, `predicted_label`, and `confidence` columns; see `lg_cotrain/data_loading.py::load_pseudo_labels()`.

### API keys

Create a `.env` file at the repository root (or inside `zeroshot/` if running notebooks from there) with:

```
OPENAI_API_KEY_1=sk-...
OPENAI_API_KEY_2=sk-...
OPENAI_API_KEY=${OPENAI_API_KEY_1}
```

`humaidclf.batch` supports global key switching via `set_api_key_env("OPENAI_API_KEY_1")` and per-block switching via `with use_api_key_env("OPENAI_API_KEY_1"):`. The notebooks use the context-manager form so that different events can route to different API tier limits when needed.

---

## Notebook index

| Notebook | Model | Split | Rules | Purpose |
|---|---|---|---|---|
| `11_zeroshot_gpt-4o_humaid_dev_RULES1_2_3.ipynb` | gpt-4o | dev | RULES_1, RULES_2, RULES_3 | Prompt-selection step — pick the best prompt for train/test runs |
| `12_zeroshot_gpt-4o_humaid_test_rules_1_filtered_labels.ipynb` | gpt-4o | test | RULES_1 | Test-split evaluation with the chosen prompt (Macro-F1 ≈ 0.641) |
| `13_zeroshot_gpt-4o_humaid_train_rules_1_filtered_labels.ipynb` | gpt-4o | train | RULES_1 | **Generated the pseudo-labels currently used by LG-CoTrain** (under `../data/pseudo-labelled/gpt-4o/`) |

---

## See also

- Parent project: [`../README.md`](../README.md)
- Pseudo-label consumer: [`../lg_cotrain/data_loading.py`](../lg_cotrain/data_loading.py) (`load_pseudo_labels`, `build_d_lg`)
- Pseudo-labels currently in use: [`../data/pseudo-labelled/gpt-4o/`](../data/pseudo-labelled/gpt-4o/)
