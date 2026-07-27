# =============================================================================
# GPU worker image for the BYO LLM Inference competition, with Ollama pre-baked.
# =============================================================================
# Why this exists: the stock codalab/codalab-legacy:gpu310 image cannot install
# Ollama at submission time --
#   - apt is unusable inside the worker container (no setuid/setgid
#     capabilities, /var/lib/apt not writable), and
#   - the image is Ubuntu 20.04 (focal), whose apt repos are EOL anyway, and
#   - downloading the 1.4 GB Ollama tarball on every submission wastes most of
#     the execution time budget.
#
# Ollama is fetched and extracted in a modern-Ubuntu build stage (so the EOL
# focal base never runs apt at all) and copied into /usr/local, the same layout
# the official install.sh produces (bin/ollama + lib/ollama). The starter kit's
# submission.py detects the preinstalled binary via shutil.which("ollama") and
# skips its download fallback entirely.
#
# Build & push (needs a Docker Hub account; replace <user>):
#   docker build --platform linux/amd64 -t <user>/codabench-gpu-ollama:0.32.0 docker/ollama/
#   docker push <user>/codabench-gpu-ollama:0.32.0
# Then set in competition.yaml:
#   docker_image: <user>/codabench-gpu-ollama:0.32.0
# Reference image published by this repo: moujar/codabench-gpu-ollama:0.32.0
# =============================================================================

# ---- Stage 1: fetch Ollama with tools the EOL base image lacks --------------
FROM ubuntu:24.04 AS ollama-fetch

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl zstd ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Pin the Ollama release for reproducible builds (bump deliberately).
ARG OLLAMA_VERSION=v0.32.0

RUN mkdir -p /opt/ollama && \
    curl -fsSL "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" \
      | zstd -d | tar -x -C /opt/ollama && \
    test -x /opt/ollama/bin/ollama

# ---- Stage 2: the Codabench GPU base, plus Ollama ---------------------------
FROM codalab/codalab-legacy:gpu310

# Same layout as the official installer: /usr/local/bin + /usr/local/lib/ollama
# (the binary looks for its runners in ../lib/ollama relative to itself).
COPY --from=ollama-fetch /opt/ollama/bin/ollama /usr/local/bin/ollama
COPY --from=ollama-fetch /opt/ollama/lib/ollama /usr/local/lib/ollama

# Python client preinstalled so submissions with an empty requirements.txt work.
# zstandard kept so the starter kit's download fallback also works on images
# built before it existed.
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
      ollama==0.6.2 zstandard==0.25.0

# Writable defaults for the worker (submissions may override via env).
ENV OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_MODELS=/tmp/ollama_home/models

RUN ollama --version
