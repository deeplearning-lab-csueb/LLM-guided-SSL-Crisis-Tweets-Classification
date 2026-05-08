import wandb, subprocess, copy, os

# ───────────────────────────────
# Core sweep configuration
# ───────────────────────────────
BASE_SWEEP = {
    "name": "humaid_ssl_sweep",
    "program": "train.py",
    "method": "bayes",
    "metric": {
        "name": "dev_macro-F1",   # your training script logs this at the end
        "goal": "maximize"
    },
    "parameters": {
        # === optimization hyperparameters ===
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 4e-5
        },
        "weight_decay": {
            "values": [0.0, 1e-2]
        },
        "batch_size": {
            "values": [16, 32]
        },
        "epochs": {
            "distribution": "int_uniform",
            "min": 12,
            "max": 20
        },

        # === semi-supervised control ===
        "T": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 0.6
        },
        "mixup_loss_weight": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.5
        },

        # === stability & regularization ===
        "label_smoothing": {"values": [0.3]},
        "max_grad_norm": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 2.0
        },
        # "th": REMOVED (Dead code, uses Oracle filtering)
        "pseudo_label_by_normalized": {"values": [False]},
        "unlabeled_batch_size": {"values": [32]},

        # === fixed metadata ===
        "task": {"value": "HumAID"},
        "model": {"value": "vinai/bertweet-base"},
        "max_seq_length": {"value": 128},
        
        # placeholders to be overwritten
        "set_num": {"values": [1]},
        "event": {"value": "placeholder"},
        "lbcl": {"value": 5},
    }
}

# ───────────────────────────────
# Fill in your events + lbcl sizes here
# ───────────────────────────────
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
LBCL_SIZES = [
    5, 
    10, 
    25, 
    50,
]
SET_NUMS = [1, 2, 3]

# ───────────────────────────────
# Sweep creation loop
# ───────────────────────────────
ids = []
for lbcl in LBCL_SIZES:
    for event in EVENTS:
        for set_num in SET_NUMS:
            sweep_cfg = copy.deepcopy(BASE_SWEEP)
            sweep_cfg["name"] = f"{event}_{lbcl}lbcl_set{set_num}"
            sweep_cfg["description"] = (
                f"Grid search for {event} ({lbcl}lbcl) set{set_num}"
            )
            
            # Inject fixed values for this specific sweep
            # Since it's a grid search for hyparams, these become single-value "grids"
            # forcing this sweep to only run for this specific dataset configuration
            sweep_cfg["parameters"]["set_num"] = {"values": [set_num]}
            sweep_cfg["parameters"]["event"] = {"value": event}
            sweep_cfg["parameters"]["lbcl"] = {"value": lbcl}

            project = f"humaid_vmatch_category_match_es2"

            sweep_id = wandb.sweep(sweep=sweep_cfg, project=project, entity=ENTITY)
            ids.append(sweep_id)
            print(f"🌀 Created sweep: {event} {lbcl}lbcl set{set_num} → {sweep_id}")

print(f"Total sweeps created: {len(ids)}")
print(f"ids: {ids}")

with open("sweep_ids.txt", "w") as f:
    for sweep_id in ids:
        f.write(f"{sweep_id}\n")
print("✅ Saved sweep IDs to sweep_ids.txt")
