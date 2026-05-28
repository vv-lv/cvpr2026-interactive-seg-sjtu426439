#!/usr/bin/env bash
# Build Docker submission image for CVPR 2026 Interactive Track.
#
# Usage:
#   bash docker/build_docker.sh <attention_checkpoint_path> [team_name]
#
# Example:
#   bash docker/build_docker.sh experiments/v9_no_scale/best.pth sjtu_interactive

set -e

CKPT_PATH="${1:?Usage: $0 <attention_checkpoint.pth> [team_name]}"
TEAM_NAME="${2:-sjtu_interactive}"
DOCKER_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CKPT_PATH"
    exit 1
fi

echo "=== Building Docker submission ==="
echo "  Checkpoint: $CKPT_PATH"
echo "  Team name:  $TEAM_NAME"
echo "  Docker dir: $DOCKER_DIR"

# Copy checkpoint to docker build context
cp "$CKPT_PATH" "$DOCKER_DIR/attention_checkpoint.pth"
echo "  Copied checkpoint ($(du -h "$DOCKER_DIR/attention_checkpoint.pth" | cut -f1))"

# Build image
docker build -t "${TEAM_NAME}:latest" "$DOCKER_DIR"
echo "  Built image: ${TEAM_NAME}:latest"

# Show image size
docker images "${TEAM_NAME}:latest" --format "  Image size: {{.Size}}"

# Export
OUT="${DOCKER_DIR}/${TEAM_NAME}.tar.gz"
echo "  Exporting to ${OUT} ..."
docker save "${TEAM_NAME}:latest" | gzip > "$OUT"
echo "  Exported: $(du -h "$OUT" | cut -f1)"

# Cleanup temp checkpoint
rm -f "$DOCKER_DIR/attention_checkpoint.pth"

echo ""
echo "=== Done ==="
echo "To test locally:"
echo "  docker container run --gpus \"device=0\" -m 32G --name ${TEAM_NAME} --rm \\"
echo "    -v \$PWD/test_inputs/:/workspace/inputs/ \\"
echo "    -v \$PWD/test_outputs/:/workspace/outputs/ \\"
echo "    ${TEAM_NAME}:latest /bin/bash -c \"sh predict.sh\""
