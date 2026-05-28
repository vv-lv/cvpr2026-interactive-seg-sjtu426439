FROM nninteractive:latest

# Upgrade nnInteractive to v1.1.3 (needed for _build_network_input API)
RUN pip install --no-cache-dir nnInteractive==1.1.3

# Overwrite predict.py and predict.sh with our versions
COPY predict.py /workspace/predict.py
COPY predict.sh /workspace/predict.sh
COPY attention_inference.py /workspace/attention_inference.py

# Copy attention checkpoint (will be set at build time)
COPY attention_checkpoint.pth /workspace/attention_checkpoint.pth

# Environment variables for predict.py
ENV CHECKPOINT_DIR=/workspace/checkpoint
ENV ATTN_CKPT=/workspace/attention_checkpoint.pth

WORKDIR /workspace
