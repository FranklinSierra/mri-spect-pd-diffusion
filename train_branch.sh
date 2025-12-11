#!/bin/bash

# ---------------------------------------------------------
# USO:
#   ./train_branch.sh <nombre_rama> <gpu_id>
#
# EJEMPLO:
#   ./train_branch.sh exp_cond_deep 0
#   ./train_branch.sh exp_unet_sd 1
#   ./train_branch.sh exp_cond_unet 2
# ---------------------------------------------------------

BRANCH=$1
GPU_ID=$2

if [ -z "$BRANCH" ] || [ -z "$GPU_ID" ]; then
  echo "Uso: $0 <nombre_rama> <gpu_id>"
  exit 1
fi

echo "--------------------------------------------------------"
echo " Entrenando rama: $BRANCH en GPU: $GPU_ID"
echo "--------------------------------------------------------"

git checkout "$BRANCH" || exit 1
cd /workspace/projects/T1-SPECT-translation/IL-CLDM || exit 1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Usando GPU $GPU_ID"

python - <<'PY' | tee "logs_${BRANCH}.txt"
import main
main.train_LDM()
PY

echo "--------------------------------------------------------"
echo " Entrenamiento de rama $BRANCH FINALIZADO"
echo " Logs en: logs_${BRANCH}.txt"
echo "--------------------------------------------------------"
