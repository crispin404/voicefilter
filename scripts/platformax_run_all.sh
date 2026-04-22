#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="${DATA_ROOT:-data/raw}"
NOISE_ROOT="${NOISE_ROOT:-data/noise}"
CONFIG_PATH="${CONFIG_PATH:-config/platform_gpu.yaml}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET_SNR_DB="${TARGET_SNR_DB:-8.0}"
SEED="${SEED:-42}"
MIX_DIR_NAME="${MIX_DIR_NAME:-合成声}"

SUBJECTS_PATH="${SUBJECTS_PATH:-metadata/subjects.json}"
SYNTH_JSONL="${SYNTH_JSONL:-metadata/synthesized_mix_metadata.jsonl}"
SYNTH_CSV="${SYNTH_CSV:-metadata/synthesized_mix_metadata.csv}"
SNR_STATS_CSV="${SNR_STATS_CSV:-metadata/preprocess_snr_stats.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/platformax/eval/metrics.csv}"
OUTPUT_WAV_DIR="${OUTPUT_WAV_DIR:-outputs/platformax/eval/enhanced_wavs}"

mkdir -p metadata splits manifests processed outputs/platformax

"$PYTHON_BIN" scripts/check_platform_env.py \
  --require-cuda \
  --device "$DEVICE" \
  --data-root "$DATA_ROOT" \
  --noise-root "$NOISE_ROOT" \
  --embedder-path pretrained/embedder.pt

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
  --subjects "$SUBJECTS_PATH" \
  --noise-root "$NOISE_ROOT" \
  --output-subdir "$MIX_DIR_NAME" \
  --target-snr-db "$TARGET_SNR_DB" \
  --seed "$SEED" \
  --metadata-path "$SYNTH_JSONL" \
  --metadata-csv "$SYNTH_CSV"

"$PYTHON_BIN" scripts/preprocess_audio.py \
  --subjects "$SUBJECTS_PATH" \
  --processed-root processed \
  --sample-rate 16000 \
  --vowel-seconds 1.0 \
  --mix-dir-name "$MIX_DIR_NAME" \
  --synthesis-metadata "$SYNTH_JSONL" \
  --snr-stats-csv "$SNR_STATS_CSV"

"$PYTHON_BIN" scripts/build_manifests.py \
  --subjects "$SUBJECTS_PATH" \
  --splits-dir splits \
  --output-dir manifests \
  --processed-root processed \
  --mix-dir-name "$MIX_DIR_NAME" \
  --snr-stats-csv "$SNR_STATS_CSV"

"$PYTHON_BIN" scripts/precompute_vowel_embeddings.py \
  -c "$CONFIG_PATH" \
  --subjects "$SUBJECTS_PATH" \
  --processed-root processed \
  --output-dir processed/embeddings \
  --device "$DEVICE"

"$PYTHON_BIN" scripts/train_enhancement.py \
  -c "$CONFIG_PATH" \
  --device "$DEVICE"

"$PYTHON_BIN" scripts/evaluate_enhancement.py \
  -c "$CONFIG_PATH" \
  --checkpoint-path outputs/platformax/checkpoints/best.pt \
  --output-csv "$OUTPUT_CSV" \
  --save-wavs-dir "$OUTPUT_WAV_DIR" \
  --device "$DEVICE"

echo "Done."
echo "Best checkpoint: outputs/platformax/checkpoints/best.pt"
echo "Metrics CSV: $OUTPUT_CSV"
