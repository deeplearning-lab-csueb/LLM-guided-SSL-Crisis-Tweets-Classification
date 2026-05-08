#!/usr/bin/env bash
# Safe 5lb/cl backfill launcher.
#
# Goals:
# - Never clobber in-flight runs on protected targets.
# - Launch only on an explicit GPU set (for example, skip GPU 1 while a job is active).
# - Resume safely using simple state files.

set -euo pipefail

IMAGE="${IMAGE:-cahsi-cotrain:test}"
HP_FILE="${HP_FILE:-best_5lb_hps.json}"
HOST_SSL_ROOT="${HOST_SSL_ROOT:-${HOME}/ssl}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-/tmp/humaid_ssl}"
GPU_LIST="${GPU_LIST:-0,2,3,4,5,6}"

PROTECT_STAGEA="${PROTECT_STAGEA:-1}"
STAGEA_JOBS_FILE="${STAGEA_JOBS_FILE:-stageA_jobs.json}"
EXCLUDE_TARGETS="${EXCLUDE_TARGETS:-}"

STATE_DIR="${STATE_DIR:-safe_5lb_state}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_PREFIX="${LOG_PREFIX:-safe5lb}"

LABEL_SMOOTHING_OVERRIDE="${LABEL_SMOOTHING_OVERRIDE:-}"
EXTRA_CLI_ARGS="${EXTRA_CLI_ARGS:-}"
SKIP_FAILED_KEYS="${SKIP_FAILED_KEYS:-1}"

QUEUE_FILE="${STATE_DIR}/queue_${RUN_TAG}.json"
MAP_FILE="${STATE_DIR}/container_map_${RUN_TAG}.tsv"
RUNNING_KEYS_FILE="${STATE_DIR}/running_keys.txt"
COMPLETED_KEYS_FILE="${STATE_DIR}/completed_keys.txt"
FAILED_KEYS_FILE="${STATE_DIR}/failed_keys.txt"

if [[ ! -f "$HP_FILE" ]]; then
  echo "Missing $HP_FILE. Run extract_best_hps.py first."
  exit 1
fi

if [[ ! -d "$HOST_SSL_ROOT" ]]; then
  echo "Missing HOST_SSL_ROOT at $HOST_SSL_ROOT"
  exit 1
fi

mkdir -p "$STATE_DIR" "$ARTIFACTS_DIR"
touch "$RUNNING_KEYS_FILE" "$COMPLETED_KEYS_FILE" "$FAILED_KEYS_FILE" "$MAP_FILE"

append_unique_line() {
  local file="$1"
  local line="$2"
  grep -Fxq "$line" "$file" || echo "$line" >> "$file"
}

remove_exact_line() {
  local file="$1"
  local line="$2"
  if [[ ! -f "$file" ]]; then
    return 0
  fi
  awk -v needle="$line" '$0 != needle' "$file" > "${file}.tmp" || true
  mv "${file}.tmp" "$file"
}

queue_len() {
  python3 - <<PYEOF
import json
with open("$QUEUE_FILE", "r", encoding="utf-8") as f:
    q = json.load(f)
print(len(q))
PYEOF
}

queue_field() {
  local idx="$1"
  local field="$2"
  python3 - <<PYEOF
import json
with open("$QUEUE_FILE", "r", encoding="utf-8") as f:
    q = json.load(f)
print(q[$idx]["$field"])
PYEOF
}

gpu_is_busy() {
  local gpu_id="$1"
  docker ps --format '{{.Names}}' | grep -E "(^|-)gpu${gpu_id}($|[^0-9])" >/dev/null 2>&1
}

map_contains_container() {
  local cname="$1"
  awk -F '\t' -v n="$cname" '$1 == n {found=1} END{exit found?0:1}' "$MAP_FILE"
}

active_count() {
  awk 'NF > 0 {c+=1} END{print c+0}' "$MAP_FILE"
}

try_launch_on_any_free_gpu() {
  if (( next_job >= NUM_JOBS )); then
    return 0
  fi

  for gpu_raw in "${GPUS[@]}"; do
    local gpu_id
    gpu_id="${gpu_raw//[[:space:]]/}"
    [[ -z "$gpu_id" ]] && continue

    if gpu_is_busy "$gpu_id"; then
      continue
    fi

    if launch_job "$gpu_id" "$next_job"; then
      (( next_job += 1 ))
    fi

    if (( next_job >= NUM_JOBS )); then
      break
    fi
  done
}

launch_job() {
  local gpu_id="$1"
  local job_idx="$2"

  local key event set_num args_file_rel cname log_file
  key="$(queue_field "$job_idx" "key")"
  event="$(queue_field "$job_idx" "event")"
  set_num="$(queue_field "$job_idx" "set_num")"
  args_file_rel="$(queue_field "$job_idx" "args_file_rel")"

  cname="cahsi-aum-safe5lb-gpu${gpu_id}-q${job_idx}"
  log_file="${LOG_PREFIX}_${event}_set${set_num}_gpu${gpu_id}.txt"

  if docker ps -a --format '{{.Names}}' | grep -Fxq "$cname"; then
    if docker ps --format '{{.Names}}' | grep -Fxq "$cname"; then
      echo "Container name $cname already running; skipping launch."
      return 1
    fi
    docker rm "$cname" >/dev/null 2>&1 || true
  fi

  append_unique_line "$RUNNING_KEYS_FILE" "$key"

  echo "Launching job $job_idx on GPU $gpu_id -> $key"
  if ! docker run -d \
      --gpus "device=${gpu_id}" \
      --ipc=host \
      --name "$cname" \
      -e HF_TOKEN="${HF_TOKEN:-}" \
      -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
      -e DEBUG="${DEBUG:-}" \
      -v "${HOST_SSL_ROOT}:/workspace/ssl" \
      -v "${ARTIFACTS_DIR}:/workspace/ssl/artifacts" \
      "$IMAGE" \
      bash -c '
          ARGS_FILE="/workspace/ssl/AUM-ST-Mixup/'"$args_file_rel"'"
          LOG_FILE="/workspace/ssl/AUM-ST-Mixup/'"$log_file"'"
          pip install wandb torchmetrics aum==1.0.2 matplotlib -q
          cd /workspace/ssl/AUM-ST-Mixup
          echo "=== Safe args: $(cat $ARGS_FILE) ===" | tee "$LOG_FILE"
          python run_aum_mixup_st.py $(cat "$ARGS_FILE") >> "$LOG_FILE" 2>&1
          echo "=== Exit: $? ===" >> "$LOG_FILE"
      ' >/dev/null; then
    echo "Launch failed for $key on GPU $gpu_id"
    remove_exact_line "$RUNNING_KEYS_FILE" "$key"
    return 1
  fi

  echo -e "${cname}\t${gpu_id}\t${key}\t${job_idx}\t${log_file}" >> "$MAP_FILE"
  return 0
}

handle_container_exit() {
  local dead_name="$1"

  local row
  row="$(awk -F '\t' -v n="$dead_name" '$1 == n {print $0}' "$MAP_FILE")"
  if [[ -z "$row" ]]; then
    return 0
  fi

  local cname gpu_id key job_idx log_file
  IFS=$'\t' read -r cname gpu_id key job_idx log_file <<< "$row"

  local exit_code
  exit_code="$(docker inspect -f '{{.State.ExitCode}}' "$dead_name" 2>/dev/null || echo unknown)"

  awk -F '\t' -v n="$dead_name" '$1 != n' "$MAP_FILE" > "${MAP_FILE}.tmp" || true
  mv "${MAP_FILE}.tmp" "$MAP_FILE"
  remove_exact_line "$RUNNING_KEYS_FILE" "$key"

  if [[ "$exit_code" == "0" ]]; then
    append_unique_line "$COMPLETED_KEYS_FILE" "$key"
    echo "Completed: $key (GPU $gpu_id)"
  else
    append_unique_line "$FAILED_KEYS_FILE" "$key"
    echo "Failed: $key (GPU $gpu_id, exit=$exit_code). See $log_file"
  fi

  docker rm "$dead_name" >/dev/null 2>&1 || true

  if (( next_job < NUM_JOBS )); then
    if gpu_is_busy "$gpu_id"; then
      echo "GPU $gpu_id still busy; refill postponed"
      return 0
    fi
    if launch_job "$gpu_id" "$next_job"; then
      (( next_job += 1 ))
    fi
  fi
}

echo "Building safe queue at $QUEUE_FILE"
HP_FILE="$HP_FILE" \
PROTECT_STAGEA="$PROTECT_STAGEA" \
STAGEA_JOBS_FILE="$STAGEA_JOBS_FILE" \
EXCLUDE_TARGETS="$EXCLUDE_TARGETS" \
STATE_DIR="$STATE_DIR" \
RUN_TAG="$RUN_TAG" \
LABEL_SMOOTHING_OVERRIDE="$LABEL_SMOOTHING_OVERRIDE" \
EXTRA_CLI_ARGS="$EXTRA_CLI_ARGS" \
SKIP_FAILED_KEYS="$SKIP_FAILED_KEYS" \
RUNNING_KEYS_FILE="$RUNNING_KEYS_FILE" \
COMPLETED_KEYS_FILE="$COMPLETED_KEYS_FILE" \
FAILED_KEYS_FILE="$FAILED_KEYS_FILE" \
QUEUE_FILE="$QUEUE_FILE" \
python3 - <<'PYEOF'
import json
import os

hp_file = os.environ["HP_FILE"]
protect_stagea = os.environ.get("PROTECT_STAGEA", "1") == "1"
stagea_file = os.environ.get("STAGEA_JOBS_FILE", "stageA_jobs.json")
exclude_targets = os.environ.get("EXCLUDE_TARGETS", "")
state_dir = os.environ["STATE_DIR"]
run_tag = os.environ["RUN_TAG"]
ls_override_raw = os.environ.get("LABEL_SMOOTHING_OVERRIDE", "")
extra_cli_args = os.environ.get("EXTRA_CLI_ARGS", "").strip()
skip_failed = os.environ.get("SKIP_FAILED_KEYS", "1") == "1"
running_keys_file = os.environ["RUNNING_KEYS_FILE"]
completed_keys_file = os.environ["COMPLETED_KEYS_FILE"]
failed_keys_file = os.environ["FAILED_KEYS_FILE"]
queue_file = os.environ["QUEUE_FILE"]

def load_key_file(path):
    out = set()
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(line)
    return out

excluded = set()
excluded |= load_key_file(completed_keys_file)
excluded |= load_key_file(running_keys_file)
if skip_failed:
    excluded |= load_key_file(failed_keys_file)

for token in [t.strip() for t in exclude_targets.split(",") if t.strip()]:
    if ":" not in token:
        raise RuntimeError(f"Invalid EXCLUDE_TARGETS token: {token}. Expected event:set_num")
    event, set_str = token.rsplit(":", 1)
    excluded.add(f"{event}:{int(set_str)}")

if protect_stagea and os.path.exists(stagea_file):
    with open(stagea_file, "r", encoding="utf-8") as f:
        stagea = json.load(f)
    for j in stagea:
        excluded.add(f"{j['event']}:{int(j['set_num'])}")

if ls_override_raw:
    ls_override = float(ls_override_raw)
else:
    ls_override = None

with open(hp_file, "r", encoding="utf-8") as f:
    jobs = json.load(f)

queue = []
for idx, j in enumerate(jobs):
    event = j["event"]
    set_num = int(j["set_num"])
    key = f"{event}:{set_num}"
    if key in excluded:
        continue

    hp = dict(j.get("hyperparameters", {}))
    hp.pop("temp_scaling", None)
    if ls_override is not None:
        hp["label_smoothing"] = ls_override

    args = f"--event {event} --lbcl 5 --set_num {set_num}"
    for k, v in hp.items():
        args += f" --{k} {v}"
    args += " --temp_scaling True"
    if extra_cli_args:
        args += f" {extra_cli_args}"

    args_file_rel = f"{state_dir}/run_safe_{run_tag}_{len(queue)}.args"
    with open(args_file_rel, "w", encoding="utf-8") as f:
        f.write(args)

    queue.append({
        "source_idx": idx,
        "event": event,
        "set_num": set_num,
        "key": key,
        "args_file_rel": args_file_rel,
    })

with open(queue_file, "w", encoding="utf-8") as f:
    json.dump(queue, f, indent=2)

print(f"Prepared {len(queue)} jobs")
PYEOF

NUM_JOBS="$(queue_len)"
echo "Safe queue size: $NUM_JOBS"
if (( NUM_JOBS == 0 )); then
  echo "Nothing to run."
  exit 0
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
next_job=0

for gpu_raw in "${GPUS[@]}"; do
  gpu_id="${gpu_raw//[[:space:]]/}"
  [[ -z "$gpu_id" ]] && continue
  (( next_job >= NUM_JOBS )) && break

  if gpu_is_busy "$gpu_id"; then
    echo "Skipping busy GPU $gpu_id"
    continue
  fi

  if launch_job "$gpu_id" "$next_job"; then
    (( next_job += 1 ))
  fi
done

active="$(active_count)"
if (( active == 0 && next_job < NUM_JOBS )); then
  echo "No selected GPUs were free."
  echo "Pending jobs: $(( NUM_JOBS - next_job ))"
  echo "Tip: choose a different GPU_LIST or retry after active jobs finish."
  exit 1
fi

if (( active == 0 && next_job >= NUM_JOBS )); then
  echo "All jobs already handled during seed launch."
  exit 0
fi

echo "Monitoring safe containers..."
echo "Active: $active | Launched: $next_job/$NUM_JOBS"

while read -r dead_name; do
  handle_container_exit "$dead_name"
  try_launch_on_any_free_gpu
  active="$(active_count)"
  echo "Active: $active | Launched: $next_job/$NUM_JOBS"

  if (( next_job >= NUM_JOBS && active == 0 )); then
    echo "All safe queue jobs finished."
    exit 0
  fi
done < <(docker events --filter 'event=die' --format '{{.Actor.Attributes.name}}')
