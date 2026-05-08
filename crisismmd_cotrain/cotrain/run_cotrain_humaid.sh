#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
ENTITY="YOUR_WANDB_ENTITY"
PROJECT="lg-cotrain-humaid"
IMAGE="cahsi-cotrain:test"
SWEEP_ID_FILE="sweep_ids.txt"

# Configurable GPU list
declare -a GPUS=(
  "0,1"
  "2,3"
  "4,5"
)
NUM_GPUS=${#GPUS[@]}

# ──────────────────────────────────────────────
# DATA & PATHS
# ──────────────────────────────────────────────
# User requested specific mounts
HOME_SSL_MOUNT="${HOME}/ssl:/workspace/ssl"

# ──────────────────────────────────────────────
# LOAD SWEEPS
# ──────────────────────────────────────────────
if [[ ! -f "$SWEEP_ID_FILE" ]]; then
  echo "❌ Missing $SWEEP_ID_FILE. Run initialize_sweeps.py first."
  exit 1
fi
mapfile -t SWEEP_IDS < "$SWEEP_ID_FILE"
TOTAL_SWEEPS=${#SWEEP_IDS[@]}

if (( TOTAL_SWEEPS == 0 )); then
  echo "❌ No sweep IDs found."
  exit 1
fi

echo "📋 Loaded $TOTAL_SWEEPS sweeps"
echo "🧠 Starting $NUM_GPUS GPU agents"
echo "───────────────────────────────────────────────"

# ──────────────────────────────────────────────
# FUNCTION TO LAUNCH AN AGENT CONTAINER
# ──────────────────────────────────────────────
launch_agent() {
    local gpu_idx=$1
    local sweep_idx=$2
    local gpu_id=${GPUS[$gpu_idx]}
    
    # Assign sweep ID round-robin
    local safe_sweep_idx=$(( sweep_idx % TOTAL_SWEEPS ))
    local sweep_id=${SWEEP_IDS[$safe_sweep_idx]}
    
    # Remove commas for the container name (e.g. "0,1" -> "01")
    local gpu_suffix="${gpu_id//,/}"
    local cname="cotrain-test-${gpu_suffix}"

    echo "🚀 Preparing Worker for GPU ${gpu_id} → Sweep ${sweep_id} (Job ${sweep_idx})"
    
    # Check if WANDB_API_KEY is set
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        echo "⚠️  WANDB_API_KEY is not set! The container will likely fail."
    fi

    # Remove existing container if it exists
    docker rm -f "${cname}" >/dev/null 2>&1 || true

    # Live Run
    docker run -d --gpus '"device='${gpu_id}'"' \
      -v ${HOME_SSL_MOUNT} \
      -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
      --name "${cname}" \
      "${IMAGE}" \
      bash -c '
        cd /workspace/ssl/llm-co-training-crisismmd-main/cotrain && \
        wandb agent --count 5 '${ENTITY}'/'${PROJECT}'/'${sweep_id}'
      '
}

# ──────────────────────────────────────────────
# INITIAL LAUNCH
# ──────────────────────────────────────────────
CURRENT_JOB_IDX=0

for ((i=0; i<NUM_GPUS; i++)); do
  launch_agent "$i" "$CURRENT_JOB_IDX"
  ((CURRENT_JOB_IDX+=1))
done

# ──────────────────────────────────────────────
# EVENT-DRIVEN MONITOR: KEEP CONTAINERS ALIVE
# ──────────────────────────────────────────────
echo "📡 Watching for stopped containers..."

# Listen for 'die' events from our specific containers
docker events --filter 'event=die' --format '{{.Actor.Attributes.name}}' |
while read -r cname; do
  if [[ $cname == cotrain-test-* ]]; then
    # Extract GPU suffix from name "cotrain-test-<SUFFIX>"
    # e.g., "cotrain-test-01" -> "01"
    
    stopped_gpu_suffix="${cname##*-}"
    
    # Find the index of this GPU in our GPUS array
    worker_idx=-1
    for ((i=0; i<NUM_GPUS; i++)); do
      # We must compare against the sanitized version of the configured GPU ID
      # "0,1" -> "01"
      current_val="${GPUS[$i]}"
      current_suffix="${current_val//,/}"
      
      if [[ "$current_suffix" == "$stopped_gpu_suffix" ]]; then
        worker_idx=$i
        break
      fi
    done
    
    if [[ $worker_idx -ge 0 ]]; then
        echo "⚡ ${cname} stopped → restarting with next sweep..."
        
        # Launch next job
        launch_agent "$worker_idx" "$CURRENT_JOB_IDX"
        ((CURRENT_JOB_IDX+=1))
    else
        echo "⚠️ Could not map ${cname} to a worker index. Ignoring."
    fi
  fi
done

