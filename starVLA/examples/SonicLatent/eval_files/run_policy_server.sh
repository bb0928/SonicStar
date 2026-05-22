#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${STARVLA_DIR}"

export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

###########################################################################################
# === Modify these for your environment ===
STARVLA_PYTHON="${STARVLA_PYTHON:-/home/user/miniconda3/envs/starVLA/bin/python}"
CKPT_PATH="${CKPT_PATH:-${STARVLA_DIR}/playground/Checkpoints/sonic_latent_scratch_frozen_vlm/checkpoints/steps_90000_pytorch_model.pt}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-10093}"
###########################################################################################

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT_PATH}" \
  --port "${PORT}" \
  --use_bf16
