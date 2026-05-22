#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

# Single-node default. Override these in your environment if needed.
unset NCCL_SOCKET_IFNAME
unset NCCL_IB_HCA
export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STARVLA_DISABLE_DEEPSPEED=1

###########################################################################################
# === Modify these for your environment ===
config_yaml="${CONFIG_YAML:-examples/SonicLatent/train_files/train_sonic_latent.yaml}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
# === End ===
###########################################################################################

if [[ -n "${wandb_api_key:-}" && -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY="${wandb_api_key}"
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-starvla_sonic_latent}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"

run_root_dir=$("${STARVLA_PYTHON}" - "${config_yaml}" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
print(cfg.run_root_dir)
PY
)
run_id=$("${STARVLA_PYTHON}" - "${config_yaml}" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
print(cfg.run_id)
PY
)
output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

"${STARVLA_PYTHON}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}"
