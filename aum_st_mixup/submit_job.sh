#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
ENTITY="YOUR_WANDB_ENTITY"
PROJECT="humaid_aum_mixup_st_hpo"
IMAGE="cahsi/disaster-ssl:cuda12-py2.2"
MAX_GPUS=7
SWEEP_ID_FILE="aum_mixup_sweep_ids.txt"   # one sweep ID per line
INCREMENT=1                     # stride over (event×lbcl) combos
START_OFFSET=0                  # start combo index (0-based) before stride

# ──────────────────────────────────────────────
# EVENTS AND LBCL COMBOS
# ──────────────────────────────────────────────
declare -a EVENTS=(
  "california_wildfires_2018"
  "canada_wildfires_2016"
  "cyclone_idai_2019"
  "hurricane_dorian_2019"
  "hurricane_florence_2018"
  "hurricane_harvey_2017"
  "hurricane_irma_2017"
  "hurricane_maria_2017"
  "kaikoura_earthquake_2016"
  "kerala_floods_2018"
)
declare -a LBCLS=(5 10 25 50)
declare -a SET_NUMS=(1 2 3)

NUM_EVENTS=${#EVENTS[@]}     # 10
NUM_LBCLS=${#LBCLS[@]}       # 4
NUM_SETS=${#SET_NUMS[@]}     # 3
NUM_COMBOS=$(( NUM_LBCLS * NUM_EVENTS * NUM_SETS ))  # 120

# ──────────────────────────────────────────────
# LOAD SWEEPS
# ──────────────────────────────────────────────
if [[ ! -f "$SWEEP_ID_FILE" ]]; then
  echo "❌ Missing $SWEEP_ID_FILE"
  exit 1
fi
mapfile -t SWEEP_IDS < "$SWEEP_ID_FILE"
total_sweeps=${#SWEEP_IDS[@]}

# Require at least one sweep per (lbcl, event, set) combo
if (( total_sweeps < NUM_COMBOS )); then
  echo "❌ Need at least ${NUM_COMBOS} sweep IDs (have ${total_sweeps})."
  exit 1
fi

echo "📋 Loaded $total_sweeps sweeps"
echo "🧠 Starting $MAX_GPUS GPU agents (offset ${START_OFFSET}, increment ${INCREMENT})"
echo "───────────────────────────────────────────────"

# ──────────────────────────────────────────────
# FUNCTION TO LAUNCH AN AGENT CONTAINER
# ──────────────────────────────────────────────
launch_agent() {
    local gpu_id=$1
    local event=$2
    local lbcl=$3
    local set_num=$4
    local sweep_id=$5
    local cname="cahsi-aum-mixup-gpu${gpu_id}"

    echo "🚀 Launching ${cname} → ${event} ${lbcl}lbcl set${set_num} (sweep ${sweep_id})"
    docker run -d --gpus "device=${gpu_id}" \
      -e EVENT_NAME="${event}" \
      -e SET_NUM="${set_num}" \
      -e DEBUG="${DEBUG:-}" \
      -e HF_TOKEN="${HF_TOKEN}" \
      -e WANDB_API_KEY="${WANDB_API_KEY}" \
      -v ${HOME}/ssl:/workspace/ssl \
      -v /tmp/humaid_ssl:/workspace/ssl/artifacts \
      --name "${cname}" \
      "${IMAGE}" \
      bash -c '
        cd /workspace/ssl/AUM-ST-Mixup && \
        pip install wandb torchmetrics aum==1.0.2 matplotlib && \
        echo "[Agent '${gpu_id}'] Running sweep '${sweep_id}' ('${event}' '${lbcl}'lbcl set'${set_num}')" && \
        wandb agent --count 10 '${ENTITY}'/'${PROJECT}'/'${sweep_id}' && \
        echo "[Agent '${gpu_id}'] Sweep '${sweep_id}' finished."
    '
}

# Helper: compute (lbcl_idx, event_idx, set_idx) from a linear combo index
# make_sweep.py order:
# for lbcl in LBCL_SIZES:
#     for event in EVENTS:
#         for set_num in SET_NUMS:
compute_indices() {
  local combo_idx=$1
  # wrap combo index across all combos
  local wrapped_idx=$(( combo_idx % NUM_COMBOS ))

  # Inner loop: Set
  SET_IDX=$(( wrapped_idx % NUM_SETS ))
  local remaining=$(( wrapped_idx / NUM_SETS ))

  # Middle loop: Event
  EVENT_IDX=$(( remaining % NUM_EVENTS ))
  
  # Outer loop: LBCL
  LBCL_IDX=$(( remaining / NUM_EVENTS ))
}

# ──────────────────────────────────────────────
# INITIAL LAUNCH (fill GPUs)
# ──────────────────────────────────────────────
for ((gpu=0; gpu<MAX_GPUS; gpu++)); do
  combo_idx=$(( START_OFFSET + gpu * INCREMENT ))
  compute_indices "$combo_idx"
  
  event=${EVENTS[$EVENT_IDX]}
  lbcl=${LBCLS[$LBCL_IDX]}
  set_num=${SET_NUMS[$SET_IDX]}

  # 🔒 Enforced mapping: matches make_sweep.py order
  sweep_flat_idx=$(( LBCL_IDX * NUM_EVENTS * NUM_SETS + EVENT_IDX * NUM_SETS + SET_IDX ))

  if (( sweep_flat_idx >= total_sweeps )); then
    echo "❌ sweep index ${sweep_flat_idx} out of range (have ${total_sweeps})."
    exit 1
  fi
  sweep_id=${SWEEP_IDS[$sweep_flat_idx]}

  launch_agent "$gpu" "$event" "$lbcl" "$set_num" "$sweep_id"
done

# ──────────────────────────────────────────────
# EVENT-DRIVEN MONITOR: KEEP CONTAINERS ALIVE
# ──────────────────────────────────────────────
echo "📡 Watching for stopped containers..."

current_job=$MAX_GPUS  # next job index to run
active_agents=$MAX_GPUS

while read -r cname; do
  if [[ $cname == cahsi-aum-mixup-gpu* ]]; then
    gpu_id="${cname//[!0-9]/}"
    echo "⚡ ${cname} stopped (GPU ${gpu_id})"
    
    ((active_agents--))

    # Calculate next job
    combo_idx=$(( START_OFFSET + current_job * INCREMENT ))
    
    if (( combo_idx < NUM_COMBOS )); then
       compute_indices "$combo_idx"
       event=${EVENTS[$EVENT_IDX]}
       lbcl=${LBCLS[$LBCL_IDX]}
       set_num=${SET_NUMS[$SET_IDX]}

       sweep_flat_idx=$(( LBCL_IDX * NUM_EVENTS * NUM_SETS + EVENT_IDX * NUM_SETS + SET_IDX ))

       if (( sweep_flat_idx < total_sweeps )); then
         sweep_id=${SWEEP_IDS[$sweep_flat_idx]}
         
         echo "🔄 Restarting GPU ${gpu_id} with next sweep..."
         docker rm -f "$cname" >/dev/null 2>&1 || true
         launch_agent "$gpu_id" "$event" "$lbcl" "$set_num" "$sweep_id"
         
         ((active_agents++))
         ((current_job++))
       else
         echo "🛑 No more valid sweeps for this index path."
       fi
    else
       echo "🏁 All sweeps scheduled."
    fi

    echo "📊 Active agents: ${active_agents}"
    
    if (( active_agents == 0 )); then
      echo "🎉 All sweeps completed! Exiting."
      exit 0
    fi
  fi
done < <(docker events --filter 'event=die' --format '{{.Actor.Attributes.name}}')
