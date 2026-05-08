import wandb, copy, os

# ───────────────────────────────
# Core sweep configuration
# ───────────────────────────────
BASE_SWEEP = {
    "name": "humaid_supervised_sweep",
    "program": "bert_ft.py",
    "method": "bayes",
    "metric": {
        "name": "eval_f1",   # bert_ft.py logs this key
        "goal": "maximize"
    },
    "parameters": {
        # === optimization hyperparameters ===
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-4
        },
        "epochs": {
            "distribution": "int_uniform",
            "min": 12,
            "max": 20
        },
        "batch_size": {
            "values": [8, 16, 32]
        },

        # === fixed metadata ===
        "model_name": {"value": "vinai/bertweet-base"},
        "max_length": {"value": 128},
        
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
            sweep_cfg["name"] = f"sup_{event}_{lbcl}lbcl_set{set_num}"
            sweep_cfg["description"] = (
                f"Supervised HPO for {event} ({lbcl}lbcl) set{set_num}"
            )
            
            # Inject fixed values for this specific sweep
            sweep_cfg["parameters"]["set_num"] = {"values": [set_num]}
            sweep_cfg["parameters"]["event"] = {"value": event}
            sweep_cfg["parameters"]["lbcl"] = {"value": lbcl}

            project = f"humaid_supervised_hpo"

            sweep_id = wandb.sweep(sweep=sweep_cfg, project=project, entity=ENTITY)
            ids.append(sweep_id)
            print(f"🌀 Created sweep: {event} {lbcl}lbcl set{set_num} → {sweep_id}")

print(f"Total sweeps created: {len(ids)}")
print(f"ids: {ids}")

with open("bertweet_sweep_ids.txt", "w") as f:
    for sweep_id in ids:
        f.write(f"{sweep_id}\n")
print("✅ Saved sweep IDs to bertweet_sweep_ids.txt")
