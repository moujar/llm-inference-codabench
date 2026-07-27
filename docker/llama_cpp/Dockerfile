# =============================================================================
# GPU worker image for the BYO LLM Inference competition, with a CUDA build of
# llama-cpp-python pre-baked.
# =============================================================================
# Why this exists: the llama.cpp starter kit sets `n_gpu_layers=-1` to offload
# the whole model to the GPU, but that only works if llama-cpp-python was
# COMPILED with CUDA. The wheel pip installs by default is CPU-only, so a plain
# `pip install llama-cpp-python` submission silently runs on CPU (slow) -- and
# compiling with CUDA at submission time needs nvcc + a toolchain the worker
# does not have and cannot apt-install. This image ships a prebuilt CUDA wheel.
#
# The llama.cpp starter kit needs NO changes: pip sees llama-cpp-python already
# satisfied on this image and skips the (CPU-only) reinstall, so `n_gpu_layers`
# offloading just works. On the stock image the same requirements.txt installs
# the CPU wheel and runs on CPU.
#
# ---------------------------------------------------------------------------
# IMPORTANT (untested build): compiling CUDA extensions is the most fragile
# image here. Two things must line up with the codalab base and your workers:
#   * glibc -- the base is Ubuntu 20.04 (focal, glibc 2.31), so the wheel is
#     built on a CUDA *devel* image that is ALSO 20.04. A wheel built on 22.04
#     (glibc 2.35) fails to load on the worker, same class of bug as the ollama
#     binary caveat.
#   * CUDA arch -- set CUDA_ARCHITECTURES to the compute capability of your GPU
#     workers (e.g. 75 for T4, 80 for A100, 86 for A10/3090, 89 for L4/4090).
#     The default below covers the common data-center cards; trim it to shrink
#     the wheel and speed up the build.
# Build/push and verify on your own GPU queue before relying on it.
# ---------------------------------------------------------------------------
#
# Build & push (needs a Docker Hub account; replace <user>):
#   docker build --platform linux/amd64 -t <user>/codabench-llama_cpp:0.1 docker/llama_cpp/
#   docker push <user>/codabench-llama_cpp:0.1
# Then set in competition.yaml:
#   docker_image: <user>/codabench-llama_cpp:0.1
# Reference image published by this repo: moujar/codabench-llama_cpp:0.1
# =============================================================================

# ---- Stage 1: build the CUDA wheel on a matching-glibc toolchain ------------
# nvidia/cuda *devel* images ship nvcc (the runtime images and the codalab base
# do not). Ubuntu 20.04 keeps glibc 2.31 in sync with the focal-era base.
FROM nvidia/cuda:12.4.1-devel-ubuntu20.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

# The codalab base is Python 3.10, but Ubuntu 20.04 ships 3.8 -- pull 3.10 from
# deadsnakes so the wheel's ABI tag (cp310) matches the target interpreter.
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common ca-certificates git build-essential cmake && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3.10-venv && \
    python3.10 -m ensurepip --upgrade && \
    rm -rf /var/lib/apt/lists/*

# Pin the llama-cpp-python release (this is independent of the image tag).
ARG LLAMA_CPP_PYTHON_VERSION=0.3.2
# Compute capabilities to compile for; see the arch note in the header.
ARG CUDA_ARCHITECTURES="75;80;86;89"

# Force a source build (--no-binary) with CUDA on, and emit a wheel we copy into
# the runtime image. GGML_CUDA replaced the old LLAMA_CUBLAS flag.
RUN CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
    FORCE_CMAKE=1 \
    python3.10 -m pip wheel --no-cache-dir --no-binary llama-cpp-python \
      "llama-cpp-python==${LLAMA_CPP_PYTHON_VERSION}" -w /wheels

# ---- Stage 2: the Codabench GPU base, plus the prebuilt CUDA wheel ----------
FROM codalab/codalab-legacy:gpu310

COPY --from=build /wheels /wheels

# Install the CUDA-enabled wheel (never rebuilds -- it is already compiled) plus
# the huggingface_hub pin the starter kit uses for `Llama.from_pretrained`.
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore \
      /wheels/llama_cpp_python-*.whl \
      huggingface_hub==0.26.2 && \
    rm -rf /wheels

# Build-time canary: import must succeed (it loads the CUDA-linked shared lib
# against the base image's CUDA 12.4 runtime; it does NOT init a GPU on import).
RUN python3 -c "import llama_cpp; print('llama_cpp', llama_cpp.__version__)"
