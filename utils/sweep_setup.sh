#!/usr/bin/env bash
set -e  # Exit on error

# === Usage ===
# ./sweep_setup.sh <EVENT> <LBCL> <RUN_NUM>
# Example:
# ./sweep_setup.sh wildfire 25 42

EVENT=$1
LBCL=$2
RUN_NUM=$3
DEVICE_ID=$4

if [ -z "$EVENT" ] || [ -z "$LBCL" ] || [ -z "$RUN_NUM" ] || [ -z "$DEVICE_ID" ]; then
  echo "Usage: $0 <EVENT> <LBCL> <RUN_NUM> <DEVICE_ID>"
  exit 1
fi

TRAIN_FILE="./train.py"

if [ ! -f "$TRAIN_FILE" ]; then
  echo "Cannot find $TRAIN_FILE"
  exit 1
fi

echo "Updating train.py placeholders..."
sed -i "s|##EVENT|${EVENT}|g" "$TRAIN_FILE"
sed -i "s|##LBCL|${LBCL}|g" "$TRAIN_FILE"
sed -i "s|##RUN_NUM|${RUN_NUM}|g" "$TRAIN_FILE"
sed -i "s|##DEVICE_ID|${DEVICE_ID}|g" "$TRAIN_FILE"
echo "Updated $TRAIN_FILE with EVENT=$EVENT, LBCL=$LBCL, RUN_NUM=$RUN_NUM, DEVICE_ID=$DEVICE_ID"

# === Start W&B sweep ===
echo "Running W&B sweep..."
$(wandb sweep --project humaid_ssl sweep.yml 2>&1 | tail -n1 | sed 's/.*Run sweep agent with: //')
