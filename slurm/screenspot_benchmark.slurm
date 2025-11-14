#!/bin/bash
#SBATCH --job-name=screenspot_benchmark
#SBATCH --account=ingrai
#SBATCH --partition=hai
#SBATCH --nodes=1i
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/screenspot_benchmark_%j.out
#SBATCH --error=logs/screenspot_benchmark_%j.err

# End-to-end ScreenSpot-Pro benchmark job (dataset already cached):
#   1. Sync dataset from raw annotations/images
#   2. Run inference with Asanshay/websight-v2-grounded
#   3. Evaluate predictions (accuracy + distance stats)

set -euo pipefail

echo "========================================"
echo "ScreenSpot-Pro Benchmark Job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo "========================================"

WORKSPACE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${WORKSPACE_DIR}"

mkdir -p logs

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "[WARN] No virtual environment found; using system Python."
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${WORKSPACE_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

# Hardcoded configuration (edit as needed)
MODEL_PATH="Asanshay/websight-v2-grounded"
ADAPTER_PATH=""
DATA_DIR="/hai/users/a/s/asanshay/websight-v2/screenspot_pro"
RAW_DIR="${DATA_DIR}/raw"
RESULTS_DIR="${DATA_DIR}/results"
EXTRA_ARGS=""

mkdir -p "${RESULTS_DIR}"

PRED_FILE="${RESULTS_DIR}/screenspot_pro_${SLURM_JOB_ID}.jsonl"
METRICS_FILE="${RESULTS_DIR}/screenspot_pro_metrics_${SLURM_JOB_ID}.json"

echo ""
echo "=== Step 1: Sync dataset ==="
if [ ! -d "${RAW_DIR}" ]; then
    echo "[ERROR] Raw ScreenSpot-Pro snapshot missing at ${RAW_DIR}"
    exit 1
fi
python scripts/prepare_screenspot_pro.py \
    --output-dir "${DATA_DIR}" \
    --sync-from-raw "${RAW_DIR}" \
    --force-rebuild \
    --summary

DATA_JSONL="${DATA_DIR}/data.jsonl"
if [ ! -f "${DATA_JSONL}" ]; then
    echo "[ERROR] ScreenSpot dataset missing at ${DATA_JSONL}"
    exit 1
fi

echo ""
echo "=== Step 2: Run benchmark ==="
python scripts/run_screenspot_benchmark.py \
    --model-name-or-path "${MODEL_PATH}" \
    --output-dir "${DATA_DIR}" \
    --predictions "${PRED_FILE}" \
    --device cuda \
    --max-new-tokens 320 \
    ${ADAPTER_PATH:+--adapter-path "${ADAPTER_PATH}"} \
    ${EXTRA_ARGS}

if [ ! -f "${PRED_FILE}" ]; then
    echo "[ERROR] Predictions file not generated: ${PRED_FILE}"
    exit 1
fi

echo ""
echo "=== Step 3: Evaluate predictions ==="
python scripts/screenspot_eval.py \
    --predictions "${PRED_FILE}" \
    --dataset "${DATA_JSONL}" \
    --output-json "${METRICS_FILE}"

echo ""
echo "Artifacts:"
echo "  Predictions: ${PRED_FILE}"
echo "  Metrics: ${METRICS_FILE}"
echo "End time: $(date)"
echo "========================================"

