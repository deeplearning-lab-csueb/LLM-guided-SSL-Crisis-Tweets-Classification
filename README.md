# LG-CoTrain — LLM-Guided Co-Training for Disaster Tweet Classification

This repository implements **LG-CoTrain** (Rahman & Caragea, 2025) for the 10-event **HumAID** disaster-tweet benchmark (Alam, Qazi, Imran & Ofli, 2021), together with a GPT-4o zero-shot baseline and three ablation pipelines (self-trained co-training, self-trained top-p, vanilla Blum & Mitchell co-training). A BERTweet supervised baseline is also included as code (`supervised_baseline/`); committed result artifacts for it are limited to the per-cell Optuna-tuned hyperparameters that the SG-CoTrain ablation depends on.

LG-CoTrain is a semi-supervised pipeline that combines a small set of human-labeled tweets with LLM-generated pseudo-labels (here, GPT-4o), using a 3-phase training procedure with two BERT models that exchange per-sample reliability weights.

---

## Table of Contents

- [Quick Start](#quick-start)
- [How LG-CoTrain Works](#how-lg-cotrain-works)
- [Reproducing the Results](#reproducing-the-results)
- [Repository Layout](#repository-layout)
- [Data Layout](#data-layout)
- [Class Labels and Disaster Events](#class-labels-and-disaster-events)
- [CLI Reference](#cli-reference)
- [Zero-shot sub-module (`zeroshot/`)](#zero-shot-sub-module-zeroshot)
- [Notebook Index](#notebook-index)
- [Hyperparameter Configurations](#hyperparameter-configurations)
- [Testing](#testing)
- [Design Decisions](#design-decisions)
- [References](#references)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r lg_cotrain/requirements.txt

# 2. Run one LG-CoTrain experiment (Kaikoura, 5 labels per class, seed 1)
python -m lg_cotrain.run_experiment \
    --event kaikoura_earthquake_2016 \
    --budget 5 \
    --seed-set 1 \
    --model-name vinai/bertweet-base \
    --pseudo-label-source gpt-4o
```

Results are written under `results/{backbone}/{model}/{strategy}/{experiment}/{event}/{budget}_set{seed}/metrics.json`.

---

## How LG-CoTrain Works

The pipeline has three phases. Two BERT models work together throughout, exchanging information about which pseudo-labels they trust.

### Phase 1 — Weight Generation

Two fresh BERT models are trained separately — Model 1 on D_l1, Model 2 on D_l2 (stratified halves of the small labeled set). After training (default 7 epochs), each model predicts softmax probabilities for every sample in the pseudo-labeled set (D_LG) using the **final epoch's** weights. These final-epoch probabilities seed Phase 2's `WeightTracker` (matching Algorithm 1 in Rahman & Caragea 2025).

### Phase 2 — Co-Training

Two **new** BERT models are initialized fresh and trained on D_LG using **weighted cross-entropy**:

- **Model 1**'s loss is weighted by **lambda2** (conservative weights from Model 2's tracker)
- **Model 2**'s loss is weighted by **lambda1** (optimistic weights from Model 1's tracker)

Each Phase 2 epoch, both models re-evaluate D_LG; the new probability observations are added to the tracker so confidence and variability (and thus lambda weights) are recomputed with growing history.

```
confidence  = mean of p(pseudo_label | x; theta) across recorded epochs
variability = std of the same

lambda_optimistic    (lambda1) = confidence + variability
lambda_conservative  (lambda2) = max(confidence - variability, 0)
```

### Phase 3 — Fine-Tuning

Each co-trained model fine-tunes on its respective labeled split (Model 1 on D_l1, Model 2 on D_l2) with early stopping on dev macro-F1 (patience = 5).

**Final evaluation** uses **ensemble prediction**: average the softmax probabilities from both models, then take the argmax.

---

## Reproducing the Results

Every result file referenced below is committed to this repo. All numbers in the table were verified by computing means directly from the included `metrics.json` / `best_params.json` files.

| Result | Headline numbers | Source in this repo | Notebook |
|---|---|---|---|
| **LG-CoTrain Macro-F1 / ECE on HumAID** (10 events × 4 budgets, 3 seeds, BERTweet + GPT-4o) | F1 0.608 / 0.619 / 0.631 / 0.645 at 5 / 10 / 25 / 50 labels per class;  ECE 0.174 / 0.160 / 0.122 / 0.108 | `results/bertweet/optuna/wge7/**/trials_10/best_params.json` (Optuna study artifacts) and the equivalent materialized re-runs in `results/bertweet/gpt-4o/optuna-tuned/wge7/` | NB15 |
| **Per-event LG-CoTrain Macro-F1** | full 10 events × 4 budgets matrix | same as above (per-event aggregation) | NB15 |
| **SG-CoTrain ablation** (LG-CoTrain pipeline + BERTweet self-trained pseudo-labels instead of GPT-4o), b=5, per event | average Macro-F1 0.396; per-event values in JSON | `results/bertweet/ablation/self-trained/optuna/{event}/5_set*/trials_10/best_params.json` | NB27 |
| **SG-CoTrain-Top variant** (top-50-per-class confidence-filtered self-trained pseudo-labels) | Macro-F1 ≈ 0.607 at b=50 | `results/bertweet/ablation/self-trained-top-p/baseline/` | NB19 |
| **Vanilla Blum & Mitchell co-training ablation** | full grid in JSON | `results/bertweet/ablation/vanilla-cotrain/baseline/` | NB29 |
| **GPT-4o zero-shot baseline** on HumAID (RULES1 prompt) | test Macro-F1 ≈ 0.641; train Macro-F1 ≈ 0.628 | `zeroshot/results/{event}/{train,test,dev}/gpt-4o/.../predictions.csv` | `zeroshot/12_…test_rules_1…ipynb`, `zeroshot/13_…train_rules_1…ipynb` |
| **Prompt-selection comparison** (3 candidate prompts on dev split) | RULES1/2/3 all in 0.601–0.613 Macro-F1 range; RULES1 chosen (simplest + highest) | `zeroshot/results/{event}/dev/gpt-4o/.../predictions.csv` for each prompt | `zeroshot/11_…dev_RULES1_2_3.ipynb` |
| **Per-class GPT-4o pseudo-label quality** | e.g. *Injured or dead people* 0.885, *Other relevant* 0.276 | computed in NB26 from the train-split predictions above | NB26 |

To re-aggregate any number from raw files:

```bash
python -c "
import json, glob
paths = glob.glob('results/bertweet/optuna/wge7/*/5_set*/trials_10/best_params.json')
f1s = [json.load(open(p))['best_full_metrics']['test_macro_f1'] for p in paths]
print('LG-CoTrain b=5 mean macro-F1:', sum(f1s)/len(f1s))
"
```

---

## Repository Layout

```
clean_release/
├── README.md                                  # this file
│
├── lg_cotrain/                                # main pipeline package
│   ├── config.py                              # LGCoTrainConfig dataclass + auto path computation
│   ├── data_loading.py                        # TSV/CSV loading, label encoding, D_LG construction
│   ├── evaluate.py                            # macro-F1, ECE, ensemble prediction
│   ├── model.py                               # BertClassifier wrapper
│   ├── trainer.py                             # LGCoTrainer — orchestrates the 3-phase pipeline
│   ├── weight_tracker.py                      # per-sample probability tracking + lambda weights
│   ├── utils.py                               # seed, EarlyStopping variants, device selection
│   ├── parallel.py                            # multi-GPU dispatch (ProcessPoolExecutor + spawn)
│   ├── run_experiment.py                      # CLI entry (single + batch)
│   ├── run_all.py                             # batch runner for one event x all budgets x all seeds
│   ├── optuna_tuner.py                        # global Optuna study (mean dev F1 across events)
│   ├── optuna_per_experiment.py               # 120 separate Optuna studies (per event/budget/seed)
│   ├── generate_selftrained_teacher.py        # produces self-trained pseudo-labels (SG-CoTrain ablation)
│   ├── filter_pseudo_labels.py                # top-p filtering for SG-CoTrain-Top variant
│   └── requirements.txt
│
├── supervised_baseline/                       # BERTweet supervised baseline package (used by NB27 to fine-tune the SG-CoTrain teacher)
├── vanilla_cotrain/                           # vanilla Blum & Mitchell co-training (extra ablation)
│
├── zeroshot/                                  # GPT-4o zero-shot baseline + pseudo-label generation
│   ├── README.md                              # full zeroshot sub-module documentation
│   ├── 11_zeroshot_gpt-4o_humaid_dev_RULES1_2_3.ipynb     # prompt selection on dev split
│   ├── 12_zeroshot_gpt-4o_humaid_test_rules_1_filtered_labels.ipynb   # test-split eval (RULES1)
│   ├── 13_zeroshot_gpt-4o_humaid_train_rules_1_filtered_labels.ipynb  # train-split run (produces the LG-CoTrain pseudo-labels)
│   ├── humaidclf/                             # OpenAI Batch API runner library
│   ├── rules/                                 # prompt definitions (RULES_1 chosen by NB11)
│   ├── Dataset/HumAID/                        # local HumAID copy used by zeroshot notebooks
│   ├── results/{event}/{train,test,dev}/gpt-4o/   # committed predictions + analysis
│   └── requirements.txt
│
├── Notebooks/                                 # 6 LG-CoTrain notebooks (see Notebook Index)
├── tests/                                     # pytest suite for the kept modules
├── data/
│   ├── original/{event}/                      # train/dev/test TSVs + labeled_{5,10,25,50}_set{1,2,3} subsets
│   └── pseudo-labelled/
│       ├── gpt-4o/                            # GPT-4o pseudo-labels (LG-CoTrain Table 4)
│       ├── self-trained-optuna/               # BERTweet teacher pseudo-labels (SG-CoTrain Table 5)
│       └── self-trained-top-p/                # top-50-per-class filtered (SG-CoTrain-Top mention)
│
├── results/bertweet/
│   ├── gpt-4o/
│   │   ├── defaults/wge7/                     # LG-CoTrain runs with reference-paper-default hyperparameters
│   │   └── optuna-tuned/wge7/                 # canonical materialized LG-CoTrain runs (Table 4 source)
│   ├── optuna/wge7/                           # LG-CoTrain Optuna study artifacts (best_params.json)
│   ├── supervised/
│   │   └── optuna-tuned/                      # per-cell supervised Optuna best_params (NB27 reads these to seed teacher training; not promoted as a standalone result)
│   └── ablation/
│       ├── self-trained/
│       │   ├── baseline/                      # SG-CoTrain runs with default HP (NB19)
│       │   └── optuna/                        # SG-CoTrain Optuna study artifacts (Table 5 source)
│       ├── self-trained-top-p/baseline/       # SG-CoTrain-Top runs (top-50-per-class confidence-filtered teacher pseudo-labels)
│       └── vanilla-cotrain/baseline/          # vanilla cotrain runs (extra ablation)
│
├── check_progress.py                          # Optuna progress checker (study.log scanner with ETA)
├── merge_optuna_results.py                    # merges per-experiment Optuna results across machines
└── extract_optuna_test_results.py             # extracts test_macro_f1 from per-experiment best_params.json
```

---

## Data Layout

```
data/
├── original/{event}/
│   ├── {event}_{train,dev,test}.tsv        # full splits — columns: tweet_id, tweet_text, class_label
│   ├── labeled_{5,10,25,50}_set{1,2,3}.tsv # human-labeled subsets
│   └── unlabeled_{5,10,25,50}_set{1,2,3}.tsv  # remaining unlabeled tweets
│
└── pseudo-labelled/{source}/{event}/
    └── {event}_train_pred.csv              # columns: tweet_id, tweet_text, predicted_label, confidence
```

The 4 budget levels (5, 10, 25, 50 labels per class) and 3 seed sets give 12 experiments per event and 120 total per source.

---

## Class Labels and Disaster Events

10 humanitarian categories (alphabetically sorted; defined in `lg_cotrain/data_loading.py::CLASS_LABELS`):

`caution_and_advice`, `displaced_people_and_evacuations`, `infrastructure_and_utility_damage`, `injured_or_dead_people`, `missing_or_found_people`, `not_humanitarian`, `other_relevant_information`, `requests_or_urgent_needs`, `rescue_volunteering_or_donation_effort`, `sympathy_and_support`.

Not all events contain every class. The pipeline calls `detect_event_classes()` to compute the per-event subset.

10 disaster events from the HumAID benchmark:

| Event | Type |
|---|---|
| `california_wildfires_2018` | Wildfire |
| `canada_wildfires_2016` | Wildfire |
| `cyclone_idai_2019` | Cyclone |
| `hurricane_dorian_2019` | Hurricane |
| `hurricane_florence_2018` | Hurricane |
| `hurricane_harvey_2017` | Hurricane |
| `hurricane_irma_2017` | Hurricane |
| `hurricane_maria_2017` | Hurricane |
| `kaikoura_earthquake_2016` | Earthquake |
| `kerala_floods_2018` | Flood |

---

## CLI Reference

### Single-event LG-CoTrain experiment

```bash
python -m lg_cotrain.run_experiment \
    --event kaikoura_earthquake_2016 \
    --budget 5 \
    --seed-set 1 \
    --model-name vinai/bertweet-base
```

### Batch mode (all 12 experiments for one event)

```bash
python -m lg_cotrain.run_experiment --event kaikoura_earthquake_2016
```

### Multiple events

```bash
python -m lg_cotrain.run_experiment \
    --events california_wildfires_2018 canada_wildfires_2016 cyclone_idai_2019
```

### Custom pseudo-label source (e.g., for SG-CoTrain ablation)

```bash
python -m lg_cotrain.run_experiment \
    --events hurricane_harvey_2017 \
    --pseudo-label-source self-trained-optuna \
    --output-folder results/bertweet/ablation/self-trained/baseline
```

### Per-experiment Optuna tuning (120 separate studies, multi-GPU)

```bash
python -m lg_cotrain.optuna_per_experiment --n-trials 10 --num-gpus 2

# Scale to 20 trials (continues from 10, only runs 10 new trials per study)
python -m lg_cotrain.optuna_per_experiment --n-trials 20 --num-gpus 2
```

Results land in `{storage-dir}/{event}/{budget}_set{seed}/trials_{n}/best_params.json`. Each trial count gets its own subfolder; previous results are never overwritten.

### Merge per-experiment Optuna results across machines

```bash
python merge_optuna_results.py --target results/bertweet/optuna/wge7 --n-trials 10
python merge_optuna_results.py --sources pc2_results/ pc3_results/ \
       --target results/bertweet/optuna/wge7 --n-trials 10
```

### Selected CLI options

| Option | Default |
|---|---|
| `--model-name` | `bert-base-uncased` (committed runs use `vinai/bertweet-base`) |
| `--pseudo-label-source` | `gpt-4o` |
| `--weight-gen-epochs` | `7` |
| `--cotrain-epochs` | `10` |
| `--finetune-max-epochs` | `100` |
| `--finetune-patience` | `5` |
| `--lr` | `2e-5` |
| `--batch-size` | `32` |
| `--weight-decay` | `0.01` |
| `--warmup-ratio` | `0.1` |
| `--max-seq-length` | `128` |
| `--num-gpus` | `1` (sequential) |
| `--data-root` | `data/` |
| `--results-root` | `results/` |

See `python -m lg_cotrain.run_experiment --help` for the full list.

---

## Zero-shot sub-module (`zeroshot/`)

The [`zeroshot/`](zeroshot/) folder is a self-contained sub-module that runs GPT-4o zero-shot classification on HumAID via the OpenAI Batch API. It serves two purposes for the LG-CoTrain pipeline:

1. **Generate the GPT-4o pseudo-labels** that LG-CoTrain consumes in Phase 2 (committed under [`data/pseudo-labelled/gpt-4o/`](data/pseudo-labelled/gpt-4o/)).
2. **Provide the zero-shot upper-bound baseline** that LG-CoTrain is compared against (test Macro-F1 ≈ 0.641 on HumAID with the RULES1 prompt).

The sub-module has its own Python package (`humaidclf/`), prompt rules (`rules/`), local HumAID copy (`Dataset/`), and lightweight dependency list (`zeroshot/requirements.txt` — no torch/transformers needed). It does not import from `lg_cotrain/` and the main pipeline does not import from `humaidclf/`; the two halves communicate by file (predictions CSVs).

**Workflow:** `11_…dev_RULES1_2_3.ipynb` (prompt selection on dev) → `12_…test…ipynb` (test eval) → `13_…train…ipynb` (train run that produces the pseudo-labels).

**For full details** — `humaidclf/` API, prompt definitions, OpenAI key setup, programmatic single-run example, and how to plug a different LLM in — see [`zeroshot/README.md`](zeroshot/README.md).

---

## Notebook Index

Six LG-CoTrain notebooks under `Notebooks/` plus three GPT-4o zero-shot notebooks under `zeroshot/`.

### LG-CoTrain pipeline (`Notebooks/`)

| Notebook | Purpose | Output tree |
|---|---|---|
| `14_bertweet_full_rerun.ipynb` | 120-cell BERTweet LG-CoTrain run with the LG-CoTrain reference paper's default hyperparameters | `results/bertweet/gpt-4o/defaults/wge7/` |
| `15_bertweet_optuna_retune.ipynb` | Per-cell Optuna re-tuning of LG-CoTrain on BERTweet (6-parameter search × 10 trials per cell × 120 cells), then materialized re-runs | `results/bertweet/optuna/wge7/` (study artifacts) + `results/bertweet/gpt-4o/optuna-tuned/wge7/` (re-runs) — **canonical LG-CoTrain result tree** |
| `19_ablation_studies.ipynb` | 3-way ablation with default hyperparameters (SG-CoTrain, SG-CoTrain-Top, vanilla cotrain) | `results/bertweet/ablation/{self-trained,self-trained-top-p,vanilla-cotrain}/baseline/` |
| `26_pseudolabel_quality_analysis.ipynb` | Per-class GPT-4o pseudo-label quality | analysis only — reads `data/pseudo-labelled/gpt-4o/` and computes per-class F1 against `data/original/{event}/{event}_train.tsv` |
| `27_selftrained_optuna_ablation.ipynb` | SG-CoTrain Optuna ablation — generates self-trained pseudo-labels (Step 1) and Optuna-tunes the LG-CoTrain pipeline on them (Step 2), end-to-end across all 4 budgets x 10 events x 3 seeds | `results/bertweet/ablation/self-trained/optuna/` |
| `29_vanilla_cotrain_optuna_ablation.ipynb` | Vanilla Blum & Mitchell co-training Optuna ablation | `results/bertweet/ablation/vanilla-cotrain/optuna/` |

### GPT-4o zero-shot baseline (`zeroshot/`)

Run order: NB11 first (compares 3 prompts on the dev split to pick the best one), then NB12 + NB13 with the chosen prompt (RULES1) on test and train splits.

| Notebook | Purpose | Output tree |
|---|---|---|
| `11_zeroshot_gpt-4o_humaid_dev_RULES1_2_3.ipynb` | **Prompt-selection step.** Runs GPT-4o zero-shot on the dev split with all 3 candidate prompts (RULES1, RULES2, RULES3) for all 10 events, then compares Macro-F1 to pick the best prompt for the train/test runs. RULES1 wins (range across prompts ≈ 0.601–0.613). | `zeroshot/results/{event}/dev/gpt-4o/.../predictions.csv` for each of the 3 prompts |
| `12_zeroshot_gpt-4o_humaid_test_rules_1_filtered_labels.ipynb` | GPT-4o zero-shot on test split, RULES1 prompt — test Macro-F1 ≈ 0.641 | `zeroshot/results/{event}/test/gpt-4o/.../predictions.csv` |
| `13_zeroshot_gpt-4o_humaid_train_rules_1_filtered_labels.ipynb` | GPT-4o zero-shot on train split, RULES1 prompt — train Macro-F1 ≈ 0.628; train predictions feed NB26's per-class breakdown | `zeroshot/results/{event}/train/gpt-4o/.../predictions.csv` |

All notebooks support **resume** — if interrupted, they skip cells whose `metrics.json` already exists.

---

## Hyperparameter Configurations

All studies use 10 trials per (event, budget, seed) combination with `dev_macro_f1` as the objective (no test-set leakage).

### LG-CoTrain (canonical, `lg_cotrain.optuna_per_experiment`)

| Parameter | Tuned? | Range / Value |
|---|---|---|
| `lr` | tuned | 1e-5 to 1e-3 (log) |
| `batch_size` | tuned | [8, 16, 32, 64] |
| `cotrain_epochs` | tuned | 5 to 20 |
| `finetune_patience` | tuned | 4 to 10 |
| `weight_decay` | tuned | 0.0 to 0.1 |
| `warmup_ratio` | tuned | 0.0 to 0.3 |
| `weight_gen_epochs` | fixed | 7 (LG-CoTrain reference paper default) |
| `model` | fixed | `vinai/bertweet-base` |
| `max_seq_length` | fixed | 128 |
| `stopping_strategy` | fixed | `baseline` (ensemble macro-F1 patience) |

### Self-trained ablation (NB27)

Same 6-parameter LG-CoTrain search space, but with `pseudo_label_source = self-trained-optuna`. The teacher BERTweet model is first fine-tuned on the labeled set using the per-cell supervised Optuna-tuned hyperparameters from `results/bertweet/supervised/optuna-tuned/` (committed in this repo); the teacher then generates pseudo-labels which feed Step 2.

### Self-trained-top-p ablation (NB19, defaults only)

Same as self-trained but the teacher's pseudo-labels are filtered to the top 50 most confident predictions per class via `lg_cotrain.filter_pseudo_labels`.

### Vanilla co-training (NB29, `vanilla_cotrain.optuna_tuner`)

Classic Blum & Mitchell (1998) co-training with no external pseudo-labels.

| Parameter | Tuned? | Range / Value |
|---|---|---|
| `lr` | tuned | 1e-5 to 1e-3 (log) |
| `batch_size` | tuned | [8, 16, 32, 64] |
| `train_epochs` | tuned | 3 to 10 |
| `samples_per_class` | tuned | [1, 5, 10] |
| `finetune_patience` | tuned | 4 to 10 |
| `weight_decay` | tuned | 0.0 to 0.1 |
| `warmup_ratio` | tuned | 0.0 to 0.3 |
| `num_iterations` | fixed | 30 |

### Supervised teacher hyperparameter search (`supervised_baseline.optuna_tuner`)

This is the per-cell Optuna search whose `best_params` NB27 consumes to fine-tune the BERTweet teacher in Step 1 of the SG-CoTrain ablation. The search is over a small 3-parameter space:

| Parameter | Tuned? | Range / Value |
|---|---|---|
| `lr` | tuned | 1e-5 to 1e-3 (log) |
| `batch_size` | tuned | [8, 16, 32, 64] |
| `max_epochs` | tuned | 20 to 100 |
| `weight_decay` | fixed | 0.01 |
| `warmup_ratio` | fixed | 0.1 |
| `patience` | fixed | 5 |

The 120 `best_params.json` files (one per event/budget/seed) are committed under `results/bertweet/supervised/optuna-tuned/`. They are a hard prerequisite for NB27. The standalone supervised classification numbers from this search are not promoted as a result of this repo.

---

## Output Format

Every run writes one `metrics.json` per (event, budget, seed):

```json
{
  "event": "kaikoura_earthquake_2016",
  "budget": 5,
  "seed_set": 1,
  "test_macro_f1": 0.6685,
  "test_ece": 0.151,
  "test_per_class_f1": [0.52, 0.41, 0.38, ...],
  "test_error_rate": 35.21,
  "dev_macro_f1": 0.5023,
  "dev_ece": 0.075,
  "stopping_strategy": "baseline",
  "phase1_seed_strategy": "last",
  "lambda1_mean": 0.7234,
  "lambda1_std": 0.1456,
  "lambda2_mean": 0.5891,
  "lambda2_std": 0.1823
}
```

For Optuna runs, additionally `trials_{n}/best_params.json` is written under each (event, budget, seed) directory:

```json
{
  "best_params": {"lr": 2.4e-5, "batch_size": 16, "cotrain_epochs": 14, ...},
  "best_full_metrics": {"test_macro_f1": 0.608, "test_ece": 0.174, ...},
  "n_trials": 10
}
```

The headline numbers in the [Reproducing the Results](#reproducing-the-results) table are aggregations of `best_full_metrics.test_macro_f1` (and `test_ece`) across 30 cells per budget.

---

## Testing

Run the full suite (requires `torch`, `transformers`, `scikit-learn`):

```bash
python -m pytest tests/ -v
```

Pure-Python tests that work without torch/transformers:

```bash
python -m unittest tests/test_config.py tests/test_weight_tracker.py tests/test_evaluate.py
```

---

## Design Decisions

- **Lazy imports**: `data_loading.py` imports `torch`/`transformers`/`pandas` lazily so that pure-Python modules (`config`, `weight_tracker`, `evaluate`) work without ML dependencies installed.
- **Cross-platform paths**: `LGCoTrainConfig` uses `pathlib.Path` for all data and result paths.
- **Configurable pseudo-label source**: `pseudo_label_source` (default `gpt-4o`) selects which directory under `data/pseudo-labelled/` to read, so the same code drives both LG-CoTrain (GPT-4o) and SG-CoTrain (self-trained) without modification.
- **4-level results hierarchy**: `results/{backbone}/{model}/{strategy}/{experiment}/{event}/{budget}_set{seed}/metrics.json` — keeps backbone, pseudo-label source, stopping strategy, and run name independently navigable.
- **Optuna incremental scaling**: per-experiment studies replay earlier trials into the TPE sampler, so running with `--n-trials 20` after `--n-trials 10` adds only 10 new trials per study; results live under `trials_{n}/` subfolders and earlier results are never overwritten.
- **Multi-GPU parallel execution**: `parallel.run_experiments_parallel()` uses `ProcessPoolExecutor` with `mp.get_context("spawn")` (required for CUDA). Cells are dispatched round-robin across `cuda:i` and stolen from the queue as each GPU finishes.
- **Resume support**: both batch runners and notebooks skip cells whose `metrics.json` already exists, so it's safe to restart after interruption.
- **AdamW + linear LR scheduler**: not specified in the LG-CoTrain reference paper. We use `weight_decay=0.01` and a linear schedule with 10% warmup spanning Phase 2 + Phase 3 (Phase 1 uses AdamW with no scheduler since those models are discarded).
- **Lambda-conservative clipping**: `lambda2 = max(confidence - variability, 0)`. The reference paper's Eq. 4 omits the clip; we add it to prevent negative weights from inverting the loss gradient.
- **Per-epoch lambda updates**: Algorithm 1 in the reference paper updates lambdas per mini-batch; we update once per epoch via a full evaluation pass over D_LG. This is more stable and is standard in semi-supervised learning.
- **Ensemble stopping criterion**: Phase 3 uses ensemble macro-F1 (averaged softmax of both models) as the dev-set early-stopping signal. Both models must independently exhaust patience before training ends. Using the ensemble is reasonable because the ensemble is the final evaluation artifact.

---

## References

**LG-CoTrain method:**

> Md Mezbaur Rahman and Cornelia Caragea. 2025. **LLM-Guided Co-Training for Text Classification**. *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 31092–31109, Suzhou, China.
> arXiv: <https://arxiv.org/abs/2509.16516> · ACL Anthology: <https://aclanthology.org/2025.emnlp-main.1583/>

**HumAID benchmark:**

> Firoj Alam, Umair Qazi, Muhammad Imran, Ferda Ofli. 2021. **HumAID: Human-Annotated Disaster Incidents Data from Twitter with Deep Learning Benchmarks**. *Proceedings of the International AAAI Conference on Web and Social Media*, Vol. 15, pp. 933–942.
