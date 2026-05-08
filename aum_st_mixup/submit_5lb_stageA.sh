#!/usr/bin/env bash
# Stage A pilot: 3 targeted 5lb/cl reruns with fixed post-hoc calibration policy.

set -euo pipefail

IMAGE="${IMAGE:-cahsi-cotrain:test}"
MAX_GPUS="${MAX_GPUS:-3}"
HP_FILE="${HP_FILE:-best_5lb_hps.json}"

# Fixed pilot targets for quick signal across different events/sets.
TARGETS=(
  "california_wildfires_2018:1"
  "hurricane_harvey_2017:2"
  "kerala_floods_2018:3"
)

if [[ ! -f "$HP_FILE" ]]; then
  echo "Missing $HP_FILE. Run: python extract_best_hps.py --output $HP_FILE"
  exit 1
fi

python3 - <<'PYEOF'
import json

hp_file = "best_5lb_hps.json"
targets = [
    ("california_wildfires_2018", 1),
    ("hurricane_harvey_2017", 2),
    ("kerala_floods_2018", 3),
]

jobs = json.load(open(hp_file, "r", encoding="utf-8"))
lookup = {(j["event"], int(j["set_num"])): j for j in jobs}

stage_a_jobs = []
for event, set_num in targets:
    key = (event, set_num)
    if key not in lookup:
        raise RuntimeError(f"Missing HP config for {event} set{set_num} in {hp_file}")

    j = lookup[key]
    hp = dict(j.get("hyperparameters", {}))
    hp.pop("temp_scaling", None)

    args = f"--event {event} --lbcl 5 --set_num {set_num}"
    for k, v in hp.items():
        args += f" --{k} {v}"

    # Stage A policy: always evaluate with post-hoc temp scaling enabled.
    args += " --temp_scaling True"

    idx = len(stage_a_jobs)
    with open(f"run_stageA_config_{idx}.args", "w", encoding="utf-8") as f:
        f.write(args)

    stage_a_jobs.append(
        {
            "idx": idx,
            "event": event,
            "set_num": set_num,
            "source_run_id": j.get("source_run_id", ""),
            "best_dev_f1": j.get("best_dev_f1", None),
            "args_file": f"run_stageA_config_{idx}.args",
        }
    )

with open("stageA_jobs.json", "w", encoding="utf-8") as f:
    json.dump(stage_a_jobs, f, indent=2)

print(f"Prepared {len(stage_a_jobs)} Stage A jobs.")
PYEOF

NUM_JOBS=$(python3 -c "import json; print(len(json.load(open('stageA_jobs.json'))))")
echo "Stage A pilot jobs: $NUM_JOBS"

launch_run() {
    local gpu_id=$1
    local job_idx=$2

    local cname="cahsi-aum-stageA-gpu${gpu_id}"
    local args_file="run_stageA_config_${job_idx}.args"
    local log_file="stageA_log_gpu${gpu_id}_job${job_idx}.txt"

    local event set_num
    event=$(python3 -c "import json; d=json.load(open('stageA_jobs.json')); print(d[$job_idx]['event'])")
    set_num=$(python3 -c "import json; d=json.load(open('stageA_jobs.json')); print(d[$job_idx]['set_num'])")

    echo "Launching Stage A: GPU${gpu_id} -> ${event} 5lbcl set${set_num}"

    docker rm -f "$cname" >/dev/null 2>&1 || true

    docker run -d \
        --gpus "device=${gpu_id}" \
        --ipc=host \
        --name "$cname" \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
        -e DEBUG="${DEBUG:-}" \
        -v "${HOME}/ssl:/workspace/ssl" \
        -v "/tmp/humaid_ssl:/workspace/ssl/artifacts" \
        "$IMAGE" \
        bash -c '
            ARGS_FILE="/workspace/ssl/AUM-ST-Mixup/'"$args_file"'"
            LOG_FILE="/workspace/ssl/AUM-ST-Mixup/'"$log_file"'"
            pip install wandb torchmetrics aum==1.0.2 matplotlib -q
            cd /workspace/ssl/AUM-ST-Mixup
            echo "=== Stage A args: $(cat $ARGS_FILE) ===" | tee "$LOG_FILE"
            python run_aum_mixup_st.py $(cat "$ARGS_FILE") >> "$LOG_FILE" 2>&1
            echo "=== Exit: $? ===" >> "$LOG_FILE"
        '
}

to_launch=$NUM_JOBS
if (( to_launch > MAX_GPUS )); then
  to_launch=$MAX_GPUS
fi

for (( i=0; i<to_launch; i++ )); do
    launch_run "$i" "$i"
done

echo "Stage A launch complete."
echo "Monitor with: docker ps --filter name=cahsi-aum-stageA"
echo "Logs are written to stageA_log_gpu*_job*.txt"
