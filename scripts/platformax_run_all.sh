#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="${DATA_ROOT:-data/raw}"
NOISE_ROOT="${NOISE_ROOT:-data/noise}"
CONFIG_PATH="${CONFIG_PATH:-config/platform_gpu.yaml}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET_SNR_DB="${TARGET_SNR_DB:-5.0}"
SEED="${SEED:-42}"

NOISE_COUNT="${NOISE_COUNT:-$("$PYTHON_BIN" -c "import sys, yaml
value = 1
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    for doc in yaml.safe_load_all(f):
        data = (doc or {}).get('data')
        if isinstance(data, dict) and 'noise_count' in data:
            value = data['noise_count']
try:
    value = int(value)
except (TypeError, ValueError):
    value = 1
print(value)
" "$CONFIG_PATH")}"

case "$NOISE_COUNT" in
  1)
    NOISE_SUFFIX=""
    DEFAULT_MIX_DIR_NAME="合成声"
    DEFAULT_MANIFEST_DIR="manifests"
    ;;
  2|3)
    NOISE_SUFFIX="_${NOISE_COUNT}"
    DEFAULT_MIX_DIR_NAME="合成声${NOISE_SUFFIX}"
    DEFAULT_MANIFEST_DIR="manifests${NOISE_SUFFIX}"
    ;;
  *)
    echo "Unsupported NOISE_COUNT: $NOISE_COUNT (expected 1, 2, or 3)" >&2
    exit 1
    ;;
esac

MIX_DIR_NAME="${MIX_DIR_NAME:-$DEFAULT_MIX_DIR_NAME}"
MANIFEST_DIR="${MANIFEST_DIR:-$DEFAULT_MANIFEST_DIR}"

SUBJECTS_PATH="${SUBJECTS_PATH:-metadata/subjects.json}"
SYNTH_JSONL="${SYNTH_JSONL:-metadata/synthesized_mix_metadata${NOISE_SUFFIX}.jsonl}"
SYNTH_CSV="${SYNTH_CSV:-metadata/synthesized_mix_metadata${NOISE_SUFFIX}.csv}"
SNR_STATS_CSV="${SNR_STATS_CSV:-metadata/preprocess_snr_stats${NOISE_SUFFIX}.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/platformax/eval/metrics.csv}"
OUTPUT_WAV_DIR="${OUTPUT_WAV_DIR:-outputs/platformax/eval/enhanced_wavs}"
EVAL_SAVE_WAVS="${EVAL_SAVE_WAVS:-0}"
EVAL_CHECKPOINT_PATH="${EVAL_CHECKPOINT_PATH:-outputs/platformax/checkpoints/best_metric.pt}"

USE_D_VECTOR_RAW="$("$PYTHON_BIN" -c "import sys, yaml
value = True
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    for doc in yaml.safe_load_all(f):
        model = (doc or {}).get('model')
        if isinstance(model, dict) and 'use_d_vector' in model:
            value = model['use_d_vector']
normalized = str(value).strip().lower()
print(normalized if normalized else 'true')
" "$CONFIG_PATH")"

case "$USE_D_VECTOR_RAW" in
  1|true|TRUE|yes|YES|on|ON)
    USE_D_VECTOR=1
    ;;
  *)
    USE_D_VECTOR=0
    ;;
esac

echo "model.use_d_vector=$USE_D_VECTOR_RAW"
echo "noise_count=$NOISE_COUNT"
echo "mix_dir_name=$MIX_DIR_NAME"
echo "manifest_dir=$MANIFEST_DIR"

mkdir -p metadata splits "$MANIFEST_DIR" processed outputs/platformax

CHECK_ENV_ARGS=(
  --require-cuda
  --device "$DEVICE"
  --data-root "$DATA_ROOT"
  --noise-root "$NOISE_ROOT"
  --embedder-path pretrained/embedder.pt
)
if [[ "$USE_D_VECTOR" -eq 0 ]]; then
  CHECK_ENV_ARGS+=(--skip-embedder-check)
fi

"$PYTHON_BIN" scripts/check_platform_env.py "${CHECK_ENV_ARGS[@]}"

"$PYTHON_BIN" scripts/scan_dataset.py \
  --data-root "$DATA_ROOT" \
  --output "$SUBJECTS_PATH"

"$PYTHON_BIN" scripts/build_subject_splits.py \
  --subjects "$SUBJECTS_PATH" \
  --output-dir splits \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed "$SEED"

"$PYTHON_BIN" scripts/synthesize_mixed_snore.py \
  -c "$CONFIG_PATH" \
  --subjects "$SUBJECTS_PATH" \
  --noise-root "$NOISE_ROOT" \
  --noise-count "$NOISE_COUNT" \
  --output-subdir "$MIX_DIR_NAME" \
  --target-snr-db "$TARGET_SNR_DB" \
  --seed "$SEED" \
  --metadata-path "$SYNTH_JSONL" \
  --metadata-csv "$SYNTH_CSV"

"$PYTHON_BIN" scripts/preprocess_audio.py \
  -c "$CONFIG_PATH" \
  --subjects "$SUBJECTS_PATH" \
  --processed-root processed \
  --sample-rate 16000 \
  --vowel-seconds 1.0 \
  --noise-count "$NOISE_COUNT" \
  --mix-dir-name "$MIX_DIR_NAME" \
  --synthesis-metadata "$SYNTH_JSONL" \
  --snr-stats-csv "$SNR_STATS_CSV"

"$PYTHON_BIN" scripts/build_manifests.py \
  -c "$CONFIG_PATH" \
  --subjects "$SUBJECTS_PATH" \
  --splits-dir splits \
  --output-dir "$MANIFEST_DIR" \
  --processed-root processed \
  --noise-count "$NOISE_COUNT" \
  --mix-dir-name "$MIX_DIR_NAME" \
  --snr-stats-csv "$SNR_STATS_CSV"

if [[ "$USE_D_VECTOR" -eq 1 ]]; then
  "$PYTHON_BIN" scripts/precompute_vowel_embeddings.py \
    -c "$CONFIG_PATH" \
    --subjects "$SUBJECTS_PATH" \
    --processed-root processed \
    --device "$DEVICE"
else
  echo "Skipping vowel embedding precompute because model.use_d_vector=false"
fi

"$PYTHON_BIN" scripts/train_enhancement.py \
  -c "$CONFIG_PATH" \
  --noise-count "$NOISE_COUNT" \
  --device "$DEVICE"

case "$EVAL_SAVE_WAVS" in
  1|true|TRUE|yes|YES|on|ON)
    "$PYTHON_BIN" scripts/evaluate_enhancement.py \
      -c "$CONFIG_PATH" \
      --checkpoint-path "$EVAL_CHECKPOINT_PATH" \
      --noise-count "$NOISE_COUNT" \
      --output-csv "$OUTPUT_CSV" \
      --save-wavs-dir "$OUTPUT_WAV_DIR" \
      --device "$DEVICE"
    ;;
  *)
    "$PYTHON_BIN" scripts/evaluate_enhancement.py \
      -c "$CONFIG_PATH" \
      --checkpoint-path "$EVAL_CHECKPOINT_PATH" \
      --noise-count "$NOISE_COUNT" \
      --output-csv "$OUTPUT_CSV" \
      --no-save-wavs \
      --device "$DEVICE"
    ;;
esac

echo "Done."
echo "Latest checkpoint: outputs/platformax/checkpoints/latest.pt"
echo "Best loss checkpoint: outputs/platformax/checkpoints/best_loss.pt"
echo "Best metric checkpoint: outputs/platformax/checkpoints/best_metric.pt"
echo "Evaluation checkpoint: $EVAL_CHECKPOINT_PATH"
echo "Metrics CSV: $OUTPUT_CSV"
