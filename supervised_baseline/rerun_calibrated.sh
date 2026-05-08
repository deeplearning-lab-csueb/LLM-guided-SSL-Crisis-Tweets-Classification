#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────
# CONFIG — Edit these for your server
# ──────────────────────────────────────────────
MAX_GPUS=7
IMAGE="cahsi/disaster-ssl:cuda12-py2.2"
SEED=42
PROJECT="humaid_supervised_calibrated"
JOBS_FILE="rerun_jobs.txt"
SSL_DIR="${HOME}/ssl"       # where the repo lives on the GPU server

# ──────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────
if [[ ! -f "$JOBS_FILE" ]]; then
  echo "❌ Missing $JOBS_FILE. Generate it first:"
  echo ""
  echo "   python extract_best_hps.py                  # from WandB"
  echo "   python extract_best_hps.py --use_defaults   # fallback (lr=2e-5, ep=15, bs=16)"
  echo ""
  exit 1
fi

total_jobs=$(wc -l < "$JOBS_FILE" | tr -d ' ')
jobs_per_gpu=$(( (total_jobs + MAX_GPUS - 1) / MAX_GPUS ))

echo "╔══════════════════════════════════════════════╗"
echo "║  Supervised Baseline Rerun (Calibrated)      ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Jobs: ${total_jobs}  |  GPUs: ${MAX_GPUS}  |  ~${jobs_per_gpu}/GPU          ║"
echo "║  Seed: ${SEED}   |  Project: ${PROJECT}  ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ──────────────────────────────────────────────
# PRE-ASSIGN JOBS TO GPUs (round-robin)
# ──────────────────────────────────────────────
declare -a GPU_QUEUES
for ((gpu=0; gpu<MAX_GPUS; gpu++)); do
  GPU_QUEUES[$gpu]=""
done

job_idx=0
while IFS=$'\t' read -r event lbcl set_num lr epochs bs; do
  gpu=$((job_idx % MAX_GPUS))
  GPU_QUEUES[$gpu]+="${event}|${lbcl}|${set_num}|${lr}|${epochs}|${bs}"$'\n'
  job_idx=$((job_idx + 1))
done < "$JOBS_FILE"

# ──────────────────────────────────────────────
# LAUNCH ONE WORKER PER GPU
# ──────────────────────────────────────────────
for ((gpu=0; gpu<MAX_GPUS; gpu++)); do
  (
    completed=0
    echo "${GPU_QUEUES[$gpu]}" | while IFS='|' read -r event lbcl set_num lr epochs bs; do
      [[ -z "$event" ]] && continue
      cname="cal${gpu}_${event:0:15}_${lbcl}_s${set_num}"

      echo "🚀 [GPU $gpu] ${event} ${lbcl}lbcl set${set_num} (lr=${lr} ep=${epochs} bs=${bs})"

      docker run --rm --gpus "device=$gpu" \
        -v "${SSL_DIR}:/workspace/ssl" \
        ${WANDB_API_KEY:+-e WANDB_API_KEY="${WANDB_API_KEY}"} \
        --name "$cname" \
        "$IMAGE" \
        bash -c "cd /workspace/ssl/supervised && python bert_ft.py \
          --event ${event} --lbcl ${lbcl} --set_num ${set_num} \
          --learning_rate ${lr} --epochs ${epochs} --batch_size ${bs} \
          --seed ${SEED} --project_name ${PROJECT}" \
      || echo "⚠️  [GPU $gpu] FAILED: ${event} ${lbcl}lbcl set${set_num}"

      completed=$((completed + 1))
      echo "✅ [GPU $gpu] ($completed done) ${event} ${lbcl}lbcl set${set_num}"
    done
  ) &
  echo "🔧 Worker started on GPU $gpu"
done

echo ""
echo "⏳ All workers launched. Waiting for completion..."
echo "   Monitor: docker ps --filter 'name=cal'"
echo "   Logs:    docker logs -f <container_name>"
echo ""
wait
echo ""
echo "🎉 All $total_jobs experiments complete!"
echo "   Results: WandB project '${PROJECT}'"
