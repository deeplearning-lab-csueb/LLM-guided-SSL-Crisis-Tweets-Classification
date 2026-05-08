#!/usr/bin/env bash
# 5LB/CL RERUN — 30 direct runs across 7 GPUs
# Uses best HP configs extracted from previous sweeps.

IMAGE="cahsi-cotrain:test"
MAX_GPUS=7
HP_FILE="best_5lb_hps.json"

if [[ ! -f "$HP_FILE" ]]; then
  echo "❌ Missing $HP_FILE — run extract_best_hps.py first"
  exit 1
fi

NUM_JOBS=$(python3 -c "import json; print(len(json.load(open('$HP_FILE'))))")
echo "📋 $NUM_JOBS direct runs to execute across $MAX_GPUS GPUs"

# ─────────────────────────────────────────────────────────────────────────────
# Build one config-arg file per job up front (avoids inline python in a loop)
# ─────────────────────────────────────────────────────────────────────────────
python3 - <<'PYEOF'
import json, sys

hp_file = "best_5lb_hps.json"
jobs = json.load(open(hp_file))

for idx, j in enumerate(jobs):
    hp   = j["hyperparameters"]
    args = f"--event {j['event']} --lbcl 5 --set_num {j['set_num']}"
    for k, v in hp.items():
        args += f" --{k} {v}"
    with open(f"run_config_{idx}.args", "w") as f:
        f.write(args)

print(f"Wrote {len(jobs)} config files.")
PYEOF

# ─────────────────────────────────────────────────────────────────────────────
# Launch one container per GPU.  Container reads its own .args file from the
# shared volume and writes its full log back there.
# ─────────────────────────────────────────────────────────────────────────────
launch_run() {
    local gpu_id=$1
    local job_idx=$2

    local cname="cahsi-aum-5lb-gpu${gpu_id}"
    local args_file="run_config_${job_idx}.args"
    local log_file="log_gpu${gpu_id}_job${job_idx}.txt"

    local event set_num
    event=$(python3 -c "import json; print(json.load(open('$HP_FILE'))[$job_idx]['event'])")
    set_num=$(python3 -c "import json; print(json.load(open('$HP_FILE'))[$job_idx]['set_num'])")

    echo "🚀 GPU${gpu_id} → ${event} 5lbcl set${set_num} (job ${job_idx})"

    # Remove stale container if present
    docker rm -f "$cname" >/dev/null 2>&1 || true

    docker run -d \
        --gpus "device=${gpu_id}" \
        --ipc=host \
        --name "$cname" \
        -e HF_TOKEN="${HF_TOKEN}" \
        -e WANDB_API_KEY="${WANDB_API_KEY}" \
        -e DEBUG="${DEBUG:-}" \
        -v "${HOME}/ssl:/workspace/ssl" \
        -v "/tmp/humaid_ssl:/workspace/ssl/artifacts" \
        "$IMAGE" \
        bash -c '
            ARGS_FILE="/workspace/ssl/AUM-ST-Mixup/'"$args_file"'"
            LOG_FILE="/workspace/ssl/AUM-ST-Mixup/'"$log_file"'"
            pip install wandb torchmetrics aum==1.0.2 matplotlib -q
            cd /workspace/ssl/AUM-ST-Mixup
            echo "=== Args: $(cat $ARGS_FILE) ===" | tee "$LOG_FILE"
            python run_aum_mixup_st.py $(cat "$ARGS_FILE") >> "$LOG_FILE" 2>&1
            echo "=== Exit: $? ===" >> "$LOG_FILE"
        '
}

# ─────────────────────────────────────────────────────────────────────────────
# Seed: start up to MAX_GPUS jobs
# ─────────────────────────────────────────────────────────────────────────────
next_job=0
for (( gpu=0; gpu<MAX_GPUS && next_job<NUM_JOBS; gpu++ )); do
    launch_run "$gpu" "$next_job"
    (( next_job++ ))
done

active=$(( next_job < MAX_GPUS ? next_job : MAX_GPUS ))
echo "📡 Monitoring... ($active active, $(( NUM_JOBS - next_job )) queued)"

# ─────────────────────────────────────────────────────────────────────────────
# Refill: as containers die, launch the next queued job on the freed GPU
# ─────────────────────────────────────────────────────────────────────────────
while read -r dead_name; do
    [[ "$dead_name" != cahsi-aum-5lb-gpu* ]] && continue

    # Extract the single GPU digit from the container name
    gpu_id="${dead_name#cahsi-aum-5lb-gpu}"
    echo "⚡ GPU${gpu_id} free (container ${dead_name} exited)"
    (( active-- ))

    if (( next_job < NUM_JOBS )); then
        launch_run "$gpu_id" "$next_job"
        (( next_job++ ))
        (( active++ ))
    fi

    echo "📊 Active: ${active} | Launched: ${next_job}/${NUM_JOBS}"

    if (( active == 0 )); then
        echo "🎉 All 5lb/cl reruns finished!"
        exit 0
    fi
done < <(docker events --filter 'event=die' --format '{{.Actor.Attributes.name}}')
