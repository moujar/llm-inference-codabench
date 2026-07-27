# =============================================================================
# GPU worker image for the BYO LLM Inference competition, with vLLM pre-baked.
# =============================================================================
# Why this exists: a vLLM submission's requirements.txt pins `vllm==0.6.3`,
# which drags in its OWN torch build (torch 2.4.0+cu124) and replaces the
# torch 2.3.1 that ships in codalab/codalab-legacy:gpu310. That is several GB
# downloaded and reinstalled on EVERY submission, burning most of the execution
# budget. Baking vLLM into the image removes that cost entirely.
#
# The vLLM starter kit needs NO changes: its requirements.txt still lists
# `vllm==0.6.3`, and pip sees it already satisfied on this image, so the install
# step is a fast no-op. On the stock image the same requirements.txt still
# works -- it just pays the full download each run.
#
# IMPORTANT (untested build): vLLM's bundled torch must match the CUDA driver
# on your Codabench GPU workers (the base targets CUDA 12.4). Pin a vLLM release
# you have verified on your own queue before relying on it; this file ships a
# known-good pin but the maintainers cannot build/test it for your hardware.
#
# Build & push (needs a Docker Hub account; replace <user>):
#   docker build --platform linux/amd64 -t <user>/codabench-gpu-vllm:0.6.3 docker/vllm/
#   docker push <user>/codabench-gpu-vllm:0.6.3
# Then set in competition.yaml:
#   docker_image: <user>/codabench-gpu-vllm:0.6.3
# Reference image published by this repo: moujar/codabench-gpu-vllm:0.6.3
# =============================================================================

FROM codalab/codalab-legacy:gpu310

# Keep this in lockstep with starting_kit/vllm/requirements.txt and the image
# tag. Bumping vLLM usually bumps its bundled torch too -- re-verify on your
# GPU queue when you change it.
ARG VLLM_VERSION=0.6.3

# vLLM pulls the matching torch (cu124) build as a dependency; huggingface_hub
# is pinned to the same version the starter kit uses so model downloads behave
# identically to a from-scratch install.
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
      "vllm==${VLLM_VERSION}" \
      huggingface_hub==0.26.2

# Build-time canary: verify the wheels installed without importing torch/CUDA
# (importing would try to probe a GPU that the build host may not have).
RUN python3 -c "from importlib.metadata import version; \
print('vllm', version('vllm'), '| torch', version('torch'))"
