import wandb, copy, os

# ───────────────────────────────
# 5lb/cl RERUN ONLY - clean sweep
# ───────────────────────────────
BASE_SWEEP = {
    "name": "humaid_aum_mixup_st_sweep",
    "program": "run_aum_mixup_st.py",
    "method": "bayes",
    "metric": {
        "name": "dev_macro-F1",
        "goal": "maximize"
    },
    "parameters": {
        "sup_epochs": {
            "distribution": "int_uniform",
            "min": 10,
            "max": 25
        },
        "unsup_epochs": {
            "distribution": "int_uniform",
            "min": 10,
            "max": 25
        },
        "sup_batch_size": {
            "values": [16, 32]
        },
        "unsup_batch_size": {
            "values": [32, 64]
        },
        "alpha": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 1.0
        },
        "T": {
            "values": [3, 5, 7, 10]
        },
        "sample_size": {
            "values": [1800]
        },
        "unsup_size": {
            "values": [1000]
        },
        "temp_scaling": {
            "values": ["True", "False"]
        },
        "sample_scheme": {
            "value": "easy_bald_class_conf"
        },
        "N_base": {
            "value": 1
        },
        "label_smoothing": {
             "value": 0.1
         },
        "hidden_dropout_prob": {
            "value": 0.1
        },
         "attention_probs_dropout_prob": {
            "value": 0.1
        },
        "dense_dropout": {
            "value": 0.1
        },
        "pt_teacher_checkpoint": {"value": "vinai/bertweet-base"},
        "seq_len": {"value": 128},
        
        # placeholders to be overwritten
        "set_num": {"values": [1]},
        "event": {"value": "placeholder"},
        "lbcl": {"value": 5},
    }
}

ENTITY = "YOUR_WANDB_ENTITY"

EVENTS = [
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
SET_NUMS = [1, 2, 3]

# ───────────────────────────────
# Only 5lb/cl sweeps
# ───────────────────────────────
ids = []
for event in EVENTS:
    for set_num in SET_NUMS:
        sweep_cfg = copy.deepcopy(BASE_SWEEP)
        sweep_cfg["name"] = f"aum_mixup_{event}_5lbcl_set{set_num}_rerun"
        sweep_cfg["description"] = (
            f"AUM-ST-Mixup HPO RERUN for {event} (5lbcl) set{set_num}"
        )
        
        sweep_cfg["parameters"]["set_num"] = {"values": [set_num]}
        sweep_cfg["parameters"]["event"] = {"value": event}
        sweep_cfg["parameters"]["lbcl"] = {"value": 5}

        project = "humaid_aum_mixup_st_hpo"

        sweep_id = wandb.sweep(sweep=sweep_cfg, project=project, entity=ENTITY)
        ids.append(sweep_id)
        print(f"🌀 Created sweep: {event} 5lbcl set{set_num} → {sweep_id}")

print(f"Total sweeps created: {len(ids)}")

with open("aum_mixup_5lb_rerun_sweep_ids.txt", "w") as f:
    for sweep_id in ids:
        f.write(f"{sweep_id}\n")
print("✅ Saved sweep IDs to aum_mixup_5lb_rerun_sweep_ids.txt")
