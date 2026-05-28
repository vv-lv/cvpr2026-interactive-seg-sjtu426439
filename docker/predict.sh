#!/usr/bin/env bash
set -e
echo "[predict.sh] Starting inference …"

INPUT_DIR=/workspace/inputs
OUTPUT_DIR=/workspace/outputs

if [ ! -d "$INPUT_DIR" ]; then
  echo "[predict.sh] ERROR: $INPUT_DIR does not exist."
  exit 1
fi
mkdir -p "$OUTPUT_DIR"

for CASE_PATH in "$INPUT_DIR"/*.npz ; do
  CASE_FILE=$(basename "$CASE_PATH")
  echo "[predict.sh] -> Processing $CASE_FILE"
  python /workspace/predict.py \
      --case_path "$CASE_PATH" \
      --save_path "$OUTPUT_DIR/$CASE_FILE"
done

echo "[predict.sh] Inference completed."
