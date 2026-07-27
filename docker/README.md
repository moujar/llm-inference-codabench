# Custom worker images

The competition runs each submission inside a Docker image on a GPU worker. The
**stock** `codalab/codalab-legacy:gpu310` image (CUDA 12.4, Python 3.10, torch
2.3.1) is enough for **Transformers** submissions, but three backends benefit
from — or require — a custom image that pre-bakes their runtime. Each image
lives in its own subfolder here.

| Image (subfolder) | For backend | Why | Strictly required? |
| --- | --- | --- | --- |
| *(stock `gpu310`)* | Transformers | torch + CUDA already present | — |
| [`vllm/`](vllm/) → [moujar/codabench-gpu-vllm:0.6.3](https://hub.docker.com/repository/docker/moujar/codabench-gpu-vllm) | vLLM | vLLM installs its **own** multi-GB torch build on every run | Recommended |
| [`llama_cpp/`](llama_cpp/) → [moujar/codabench-llama_cpp:0.1](https://hub.docker.com/repository/docker/moujar/codabench-llama_cpp) | llama.cpp | GPU offload needs `llama-cpp-python` **compiled with CUDA**; the runtime worker can't compile it | Recommended |
| [`ollama/`](ollama/) → [moujar/codabench-gpu-ollama:0.32.0](https://hub.docker.com/repository/docker/moujar/codabench-gpu-ollama) | Ollama | Worker can't apt-install and would download a 1.4 GB tarball each run | **Required** |

> Reference images published by this repo: [moujar/codabench-gpu-vllm:0.6.3](https://hub.docker.com/repository/docker/moujar/codabench-gpu-vllm),
> [moujar/codabench-llama_cpp:0.1](https://hub.docker.com/repository/docker/moujar/codabench-llama_cpp),
> [moujar/codabench-gpu-ollama:0.32.0](https://hub.docker.com/repository/docker/moujar/codabench-gpu-ollama).
> They are the author's personal builds — **build and push your own** under a
> registry your workers can pull from (below). Don't depend on `moujar/*` in a
> competition you run.

## Why the stock image can't just install these at run time

On a Codabench worker the submission container runs **without setuid/setgid
capabilities** and with `/var/lib/apt` read-only, so `apt` cannot install
anything — and the base is Ubuntu 20.04 (focal), whose apt repos are EOL
anyway. Everything a submission needs must come from `pip` (which works) or be
**already in the image**. That is why:

* **Ollama** can't run its `install.sh` (needs apt-installed `zstd`) — the
  server binary must be baked in, or downloaded raw each run.
* **llama.cpp** can't compile a CUDA extension on the worker (no `nvcc`,
  no toolchain) — the CUDA wheel must be baked in, or it runs CPU-only.
* **vLLM** *can* pip-install at run time, but re-downloading its torch build
  every submission wastes most of the time budget.

## How the "bake it in" trick stays transparent to submissions

Every starter kit is written to work **with or without** its custom image:

* **Ollama** — `setup()` calls `shutil.which("ollama")` first; if the baked-in
  binary is found the 1.4 GB download is skipped, otherwise it falls back to
  downloading. See the header of `starting_kit/ollama/submission.py`.
* **vLLM / llama.cpp** — the package is listed in `requirements.txt`; on the
  custom image `pip install` finds it already satisfied and does nothing, on
  the stock image it installs normally (paying the download / CPU-only cost).

So a participant submits the **same two files** either way — the image only
changes how fast (and, for llama.cpp, whether on GPU) it runs.

## Build and publish

Requires a [Docker Hub](https://hub.docker.com) account (or any registry your
Codabench workers can pull from). Replace `<user>` with your username and run
from the **repo root**:

```bash
docker login

# vLLM
docker build --platform linux/amd64 -t <user>/codabench-gpu-vllm:0.6.3   docker/vllm/
docker push <user>/codabench-gpu-vllm:0.6.3

# llama.cpp  (slowest: compiles a CUDA extension)
docker build --platform linux/amd64 -t <user>/codabench-llama_cpp:0.1    docker/llama_cpp/
docker push <user>/codabench-llama_cpp:0.1

# Ollama
docker build --platform linux/amd64 -t <user>/codabench-gpu-ollama:0.32.0 docker/ollama/
docker push <user>/codabench-gpu-ollama:0.32.0
```

The image must be **public** (or pullable by the workers). Each build pulls the
~6 GB base image, so the first run of each takes a while. `--platform
linux/amd64` matters if you build on Apple Silicon — the workers are x86-64.

> **Heads-up — these builds are not CI-tested by this repo.** The vLLM and
> llama.cpp Dockerfiles were written to be correct but have not been built on
> the maintainer's hardware. In particular llama.cpp compiles CUDA and is
> sensitive to glibc and GPU compute capability. **Build, push, and run a real
> submission on your own GPU queue before trusting them.** Read each
> Dockerfile's header for the exact caveats and the build args to adjust
> (`VLLM_VERSION`, `OLLAMA_VERSION`, `LLAMA_CPP_PYTHON_VERSION`,
> `CUDA_ARCHITECTURES`).

## Wire an image into the competition

One image is active per competition. Set it in `competition.yaml`:

```yaml
docker_image: <user>/codabench-gpu-ollama:0.32.0
```

Re-upload the bundle (or edit the benchmark's settings in the Codabench UI),
then re-run affected submissions. Submissions still run on the same GPU queue.

**Only one `docker_image` at a time.** If your competition invites *multiple*
backends, either:

1. Pick the image for the backend you expect most (Transformers → stock is
   fine; otherwise pick that backend's image — the others still work, just
   slower / CPU for llama.cpp), or
2. Run **separate phases/competitions** per backend, each with its own
   `docker_image`, or
3. Build **one combined image** that layers vLLM + llama.cpp + Ollama on the
   base (largest image, but every backend runs at full speed).

## Updating a pinned version

Each image pins its runtime via a build arg for reproducibility (bump
deliberately, then re-tag to match):

```bash
docker build --platform linux/amd64 --build-arg OLLAMA_VERSION=v0.33.0 \
  -t <user>/codabench-gpu-ollama:0.33.0 docker/ollama/

docker build --platform linux/amd64 --build-arg VLLM_VERSION=0.7.0 \
  -t <user>/codabench-gpu-vllm:0.7.0 docker/vllm/

docker build --platform linux/amd64 --build-arg LLAMA_CPP_PYTHON_VERSION=0.3.4 \
  -t <user>/codabench-llama_cpp:0.2 docker/llama_cpp/
```

Keep the image's pinned version in step with the matching
`starting_kit/<backend>/requirements.txt` so a from-scratch install (on the
stock image) resolves to the same versions the baked image ships.
