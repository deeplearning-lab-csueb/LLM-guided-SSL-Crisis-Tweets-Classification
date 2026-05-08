#!/usr/bin/env bash
set -euo pipefail

# Rerun ONLY the 2 failed experiments from the original results:
#   1. Hurricane Maria 2017, 25 lb/class, Set 3   (original F1 = 0.0079)
#   2. Hurricane Dorian 2019, 50 lb/class, Set 3   (original F1 = 0.0214)
#
# Uses default HPs which work well at 25+ lb/class.
# Results go to the ORIGINAL project so they're alongside the other runs.

IMAGE="cahsi/disaster-ssl:cuda12-py2.2"
SSL_DIR="${HOME}/ssl"
PROJECT="humaid_supervised_hpo"

echo "🔧 Rerunning 2 failed experiments..."

# Job 1: Hurricane Maria 25lb Set 3 → GPU 0
docker run --rm --gpus "device=0" \
  -v "${SSL_DIR}:/workspace/ssl" \
  ${WANDB_API_KEY:+-e WANDB_API_KEY="${WANDB_API_KEY}"} \
  --name "fix_maria_25_s3" \
  "$IMAGE" \
  bash -c "cd /workspace/ssl/supervised && python bert_ft.py \
    --event hurricane_maria_2017 --lbcl 25 --set_num 3 \
    --learning_rate 2e-5 --epochs 15 --batch_size 16 \
    --seed 42 --project_name ${PROJECT}" &

# Job 2: Hurricane Dorian 50lb Set 3 → GPU 1
docker run --rm --gpus "device=1" \
  -v "${SSL_DIR}:/workspace/ssl" \
  ${WANDB_API_KEY:+-e WANDB_API_KEY="${WANDB_API_KEY}"} \
  --name "fix_dorian_50_s3" \
  "$IMAGE" \
  bash -c "cd /workspace/ssl/supervised && python bert_ft.py \
    --event hurricane_dorian_2019 --lbcl 50 --set_num 3 \
    --learning_rate 2e-5 --epochs 15 --batch_size 16 \
    --seed 42 --project_name ${PROJECT}" &

echo "⏳ Running on GPU 0 and GPU 1 in parallel (~5 min)..."
echo "   Monitor: docker ps --filter 'name=fix_'"
wait
echo "🎉 Done! Check WandB project '${PROJECT}' for updated results."
