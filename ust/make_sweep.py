import wandb, copy, os

# ───────────────────────────────
# Core sweep configuration
# ───────────────────────────────
BASE_SWEEP = {
    "name": "humaid_ust_sweep",
    "program": "run_ust.py",
    "method": "bayes",
    "metric": {
        "name": "dev_macro-F1",   # ust.py logs this key
        "goal": "maximize"
    },
    "parameters": {
        # === optimization hyperparameters ===
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 5e-5
        },
        "sup_epochs": {
            "distribution": "int_uniform",
            "min": 15,
            "max": 25
        },
        "unsup_epochs": {
            "distribution": "int_uniform",
            "min": 10,
            "max": 20
        },
        "sup_batch_size": {
            "values": [16, 32]
        },
        "unsup_batch_size": {
            "values": [32, 64] # Typically larger than sup_batch_size
        },

        # === UST specific hyperparameters ===
        "T": {
            "distribution": "int_uniform",
            "min": 10, 
            "max": 50
        },
        "alpha": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 1.0
        },
         "label_smoothing": {
             "values": [0.0, 0.1]
         },
         
        # === Dropout ===
        "hidden_dropout_prob": {
            "values": [0.1, 0.3]
        },
         "attention_probs_dropout_prob": {
            "values": [0.1, 0.3]
        },
        "dense_dropout": {
            "values": [0.1, 0.3, 0.5]
        },

        # === fixed metadata ===
        "pt_teacher_checkpoint": {"value": "vinai/bertweet-base"},
        "seq_len": {"value": 128},
        
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
            sweep_cfg["name"] = f"ust_{event}_{lbcl}lbcl_set{set_num}"
            sweep_cfg["description"] = (
                f"UST HPO for {event} ({lbcl}lbcl) set{set_num}"
            )
            
            # Inject fixed values for this specific sweep
            sweep_cfg["parameters"]["set_num"] = {"values": [set_num]}
            sweep_cfg["parameters"]["event"] = {"value": event}
            sweep_cfg["parameters"]["lbcl"] = {"value": lbcl}

            project = f"humaid_ust_hpo"

            sweep_id = wandb.sweep(sweep=sweep_cfg, project=project, entity=ENTITY)
            ids.append(sweep_id)
            print(f"🌀 Created sweep: {event} {lbcl}lbcl set{set_num} → {sweep_id}")

print(f"Total sweeps created: {len(ids)}")
print(f"ids: {ids}")

with open("ust_sweep_ids.txt", "w") as f:
    for sweep_id in ids:
        f.write(f"{sweep_id}\n")
print("✅ Saved sweep IDs to ust_sweep_ids.txt")
