#!/usr/bin/env bash
set -euo pipefail

# Cost-conscious GCP smoke run for Experiment 09 (ChunkGate-Lite).
# Defaults prioritize cheap/short iteration over max throughput.
#
# Usage:
#   bash scripts/gcp_exp09_l4_smoke.sh
#
# Optional overrides:
#   PROJECT=testingout-423013 ZONE=us-central1-a AUTO_SHUTDOWN_MINUTES=75 \
#   MAX_WALLCLOCK_SECONDS=900 ITERATIONS=600 TRAIN_BATCH_TOKENS=262144 \
#   DATA_SHARDS=1 DELETE_INSTANCE_AT_END=1 bash scripts/gcp_exp09_l4_smoke.sh

PROJECT="${PROJECT:-testingout-423013}"
ZONE="${ZONE:-us-central1-a}"
ZONE_CANDIDATES="${ZONE_CANDIDATES:-${ZONE} us-central1-b us-central1-c us-east4-a us-east4-c us-west1-a us-west1-b us-west1-c}"
INSTANCE="${INSTANCE:-exp09-l4-smoke-$(date +%m%d-%H%M%S)}"
RUN_ID="${RUN_ID:-exp09_l4_smoke_$(date +%Y%m%d_%H%M%S)}"

GPU_PROFILE="${GPU_PROFILE:-l4}" # l4 or t4
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}" # default for l4
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-7-cu128-ubuntu-2204-nvidia-570}"
BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-200}"

SPOT="${SPOT:-1}"
AUTO_SHUTDOWN_MINUTES="${AUTO_SHUTDOWN_MINUTES:-75}"
DELETE_INSTANCE_AT_END="${DELETE_INSTANCE_AT_END:-1}"

MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-900}"
ITERATIONS="${ITERATIONS:-600}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-262144}"
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
DATA_SHARDS="${DATA_SHARDS:-1}"

LOCAL_EXP_DIR="experiments/09-chunkgate-lite"
LOCAL_LOG_DIR="${LOCAL_EXP_DIR}/logs"
mkdir -p "${LOCAL_LOG_DIR}"
INSTANCE_CREATED=0
ACTIVE_ZONE=""

cleanup() {
  if [[ "${DELETE_INSTANCE_AT_END}" == "1" && "${INSTANCE_CREATED}" == "1" ]]; then
    echo "[cleanup] deleting instance ${INSTANCE} in ${ACTIVE_ZONE}"
    gcloud compute instances delete "${INSTANCE}" \
      --project "${PROJECT}" \
      --zone "${ACTIVE_ZONE}" \
      --quiet || true
  else
    echo "[cleanup] instance preserved or not created: ${INSTANCE}"
  fi
}
trap cleanup EXIT

echo "[info] project=${PROJECT} zone_candidates=${ZONE_CANDIDATES} instance=${INSTANCE}"
echo "[info] run_id=${RUN_ID} spot=${SPOT} auto_shutdown=${AUTO_SHUTDOWN_MINUTES}m"
echo "[info] max_wallclock_seconds=${MAX_WALLCLOCK_SECONDS} iterations=${ITERATIONS} gpu_profile=${GPU_PROFILE}"

if [[ "${GPU_PROFILE}" == "l4" ]]; then
  gpu_args=(--machine-type "${MACHINE_TYPE}")
elif [[ "${GPU_PROFILE}" == "t4" ]]; then
  gpu_args=(--machine-type "n1-standard-8" --accelerator "type=nvidia-tesla-t4,count=1")
else
  echo "[error] unsupported GPU_PROFILE=${GPU_PROFILE} (use l4 or t4)"
  exit 1
fi

echo "[step] creating VM (with zone fallback)"
for candidate_zone in ${ZONE_CANDIDATES}; do
  echo "[try] zone=${candidate_zone}"
  create_args=(
    compute instances create "${INSTANCE}"
    --project "${PROJECT}"
    --zone "${candidate_zone}"
    "${gpu_args[@]}"
    --image-project "${IMAGE_PROJECT}"
    --image-family "${IMAGE_FAMILY}"
    --boot-disk-size "${BOOT_DISK_SIZE_GB}GB"
    --boot-disk-type pd-ssd
    --maintenance-policy TERMINATE
  )
  if [[ "${SPOT}" == "1" ]]; then
    create_args+=(--provisioning-model SPOT --instance-termination-action DELETE)
  fi
  if gcloud "${create_args[@]}"; then
    ACTIVE_ZONE="${candidate_zone}"
    INSTANCE_CREATED=1
    echo "[ok] created in zone=${ACTIVE_ZONE}"
    break
  fi
  echo "[warn] create failed in zone=${candidate_zone}, trying next zone"
done

if [[ "${INSTANCE_CREATED}" != "1" ]]; then
  echo "[error] failed to create VM in all candidate zones"
  exit 1
fi

echo "[step] setting forced auto-shutdown on VM"
gcloud compute ssh "${INSTANCE}" \
  --project "${PROJECT}" \
  --zone "${ACTIVE_ZONE}" \
  --command "sudo shutdown -h +${AUTO_SHUTDOWN_MINUTES}" >/dev/null

echo "[step] copying Experiment 09 CUDA script"
gcloud compute scp "${LOCAL_EXP_DIR}/train_gpt.py" "${INSTANCE}:~/train_gpt_exp09.py" \
  --project "${PROJECT}" \
  --zone "${ACTIVE_ZONE}"

echo "[step] remote setup + training"
gcloud compute ssh "${INSTANCE}" \
  --project "${PROJECT}" \
  --zone "${ACTIVE_ZONE}" \
  --command "
set -euo pipefail
if [ ! -d ~/parameter-golf ]; then
  git clone https://github.com/openai/parameter-golf.git ~/parameter-golf
fi
cd ~/parameter-golf
python3 -m pip install --upgrade pip
python3 -m pip install sentencepiece huggingface-hub datasets tqdm
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards ${DATA_SHARDS}
cp ~/train_gpt_exp09.py ~/parameter-golf/train_gpt.py
RUN_ID=${RUN_ID} \
MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS} \
ITERATIONS=${ITERATIONS} \
TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS} \
TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN} \
VAL_LOSS_EVERY=${VAL_LOSS_EVERY} \
TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY} \
CHUNKGATE_ENABLE=1 \
CHUNKGATE_STRIDE=4 \
CHUNKGATE_INNER_LAYERS=2 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
"

echo "[step] collecting artifacts back to local"
gcloud compute scp "${INSTANCE}:~/parameter-golf/logs/${RUN_ID}.txt" "${LOCAL_LOG_DIR}/" \
  --project "${PROJECT}" \
  --zone "${ACTIVE_ZONE}" || true
gcloud compute scp "${INSTANCE}:~/parameter-golf/final_model.int8.ptz" "${LOCAL_EXP_DIR}/" \
  --project "${PROJECT}" \
  --zone "${ACTIVE_ZONE}" || true

echo "[done] smoke run complete. local log: ${LOCAL_LOG_DIR}/${RUN_ID}.txt"
