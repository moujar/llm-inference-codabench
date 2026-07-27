# LLM Inference — Bring Your Own Backend (Codabench template)

An open-source **[Codabench](https://www.codabench.org/) competition template**
for LLM inference where participants **choose their own inference library**.
Clone it, swap in your own questions, and you have a working LLM challenge.

A submission is just two files — `requirements.txt` and `submission.py` — where
`submission.py` implements a tiny `Submission` contract (`setup` / `generate` /
`teardown`). The **ingestion** program installs the requirements and runs the
model on a hidden set of prompts; the **scoring** program evaluates the answers
and writes the leaderboard metrics.

Starter kits are provided for **Transformers**, **vLLM**, **Ollama**, and
**llama.cpp** — so a participant can bring any of them without you changing the
competition.

> **License:** [MIT](LICENSE) — use it, fork it, run your own challenge on it.
> Attribution appreciated but not required.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Repository layout](#repository-layout)
3. [The `Submission` contract](#the-submission-contract)
4. [Run it locally](#run-it-locally)
5. [Make it YOUR challenge](#make-it-your-challenge)
6. [Backends and Docker images](#backends-and-docker-images)
7. [Deploy to Codabench](#deploy-to-codabench)
8. [Contributing](#contributing)

---

## How it works

```text
participant submits                organizer's bundle runs on a GPU worker
┌──────────────────┐               ┌───────────────────────────────────────────┐
│ requirements.txt │──pip install─▶│ ingestion_program/                        │
│ submission.py    │──import ─────▶│   setup() ▸ generate(prompt)×N ▸ teardown │
└──────────────────┘               │        │ predictions.jsonl                 │
                                   │        ▼                                   │
        input_data/prompts.jsonl ─▶│ scoring_program/  ── scores.json ──▶ leaderboard
        reference_data/*  (hidden) │   accuracy · token-F1 · avg latency        │
                                   └───────────────────────────────────────────┘
```

1. **Ingestion** ([`ingestion_program/`](LLM-Inference-Competition/ingestion_program/))
   pip-installs the submission's `requirements.txt`, imports its `Submission`
   class, calls `setup()` once, then `generate(prompt)` for every prompt in
   `input_data/prompts.jsonl`, and writes `predictions.jsonl` (+ latency).
2. **Scoring** ([`scoring_program/`](LLM-Inference-Competition/scoring_program/))
   compares predictions to the hidden `reference_data/reference.jsonl` and
   reports **accuracy**, **token F1**, and **average latency**.

## Repository layout

```text
.
├── LICENSE                         # MIT
├── readme.md                       # this file
├── docker/                         # custom GPU worker images (one per backend)
│   ├── README.md                   #   when each is needed + build/push guide
│   ├── ollama/Dockerfile           #   gpu310 + Ollama server pre-baked
│   ├── vllm/Dockerfile             #   gpu310 + vLLM pre-baked
│   └── llama_cpp/Dockerfile        #   gpu310 + CUDA build of llama-cpp-python
└── LLM-Inference-Competition/  # the Codabench bundle (zip this to upload)
    ├── competition.yaml            # phases, tasks, leaderboard, docker_image
    ├── logo.png
    ├── pages/                      # overview / evaluation / terms (markdown)
    ├── input_data/prompts.jsonl    # prompts shown to the model (public here)
    ├── reference_data/reference.jsonl  # gold answers (KEEP HIDDEN in real use)
    ├── ingestion_program/          # runs the submission -> predictions.jsonl
    ├── scoring_program/            # predictions vs reference -> scores.json
    ├── sample_code_submission/     # default solution (Transformers)
    ├── sample_result_submission/   # example ingestion output
    └── starting_kit/               # ready-to-edit submissions, one per backend
        ├── transformers/  vllm/  ollama/  llama_cpp/
        └── README.md
```

## The `Submission` contract

Everything a participant writes implements this:

```python
class Submission:
    backend = "transformers"        # optional: shows on the leaderboard
    model_name = "my-model"         # optional: shows on the leaderboard

    def setup(self):
        """Download / load the model, start the backend. Called ONCE.
        Not counted toward latency — do the heavy lifting here."""

    def generate(self, prompt: str) -> str:
        """Return the model's answer to a single prompt. Must return a string;
        if it raises, the answer is recorded empty and the run continues."""

    def teardown(self):             # optional
        """Free resources / stop servers. Called once at the end."""
```

## Run it locally

No Codabench needed to test the flow. From the repo root:

```bash
cd LLM-Inference-Competition

# Run the sample (Transformers) submission end-to-end.
# Needs the sample deps (torch + transformers); a GPU is optional for the
# tiny 135M default model but recommended.
pip install -r sample_code_submission/requirements.txt
python3 ingestion_program/run_ingestion.py     # -> sample_result_submission/predictions.jsonl

# Scoring uses only the Python standard library:
python3 scoring_program/run_scoring.py          # -> scoring_output/scores.json
```

To test a *different* backend locally, copy that starter kit's two files into a
folder and point ingestion at it (or drop them into `sample_code_submission/`).

## Make it YOUR challenge

The defaults are a trivial trivia task (capital of France, 2+2, …) so the
plumbing is easy to verify. To turn it into a real competition:

1. **Swap the data.** Replace
   [`input_data/prompts.jsonl`](LLM-Inference-Competition/input_data/prompts.jsonl)
   (`{"id": ..., "prompt": ...}` per line) and
   [`reference_data/reference.jsonl`](LLM-Inference-Competition/reference_data/reference.jsonl)
   (`{"id": ..., "answer": ...}` per line). Keep the `id`s aligned.
   **In a real competition the reference data stays hidden on the server** — it
   is public here only because this is a template.
2. **Adjust the metrics** in
   [`scoring_program/score.py`](LLM-Inference-Competition/scoring_program/score.py).
   It currently does SQuAD-style `accuracy` (reference contained in answer),
   token `f1`, and `avg_latency`. Change these to fit your task (exact match,
   BLEU, an LLM judge, cost, …) and update the leaderboard `columns` in
   `competition.yaml` to match the keys you emit in `scores.json`.
3. **Edit the pages** in
   [`pages/`](LLM-Inference-Competition/pages/) (overview / evaluation /
   terms) and `competition.yaml` (title, dates, phases). Replace `logo.png`.
4. **Pick the worker image** — see below.

## Backends and Docker images

The ingestion program installs each submission's `requirements.txt` at run
time, so **any** backend works on a GPU worker with internet access. Some
backends run much faster (or, for llama.cpp, only reach the GPU) when their
runtime is **baked into the worker image** — one image per backend lives in
[`docker/`](docker/):

| Backend | Starter kit | Worker image needed? |
| --- | --- | --- |
| **Transformers** | `starting_kit/transformers/` | None — stock `codalab/codalab-legacy:gpu310` already has torch + CUDA. This is the default sample submission. |
| **vLLM** | `starting_kit/vllm/` | *Recommended:* [`docker/vllm/`](docker/vllm/) → [moujar/codabench-gpu-vllm:0.6.3](https://hub.docker.com/repository/docker/moujar/codabench-gpu-vllm). Without it, vLLM re-downloads its own multi-GB torch build every run. |
| **llama.cpp** | `starting_kit/llama_cpp/` | *Recommended:* [`docker/llama_cpp/`](docker/llama_cpp/) → [moujar/codabench-llama_cpp:0.1](https://hub.docker.com/repository/docker/moujar/codabench-llama_cpp). GPU offload needs `llama-cpp-python` compiled with CUDA — the worker can't compile it, so without the image it runs CPU-only. |
| **Ollama** | `starting_kit/ollama/` | **Required:** [`docker/ollama/`](docker/ollama/) → [moujar/codabench-gpu-ollama:0.32.0](https://hub.docker.com/repository/docker/moujar/codabench-gpu-ollama). The worker can't apt-install; otherwise every run downloads a 1.4 GB tarball. |

Every starter kit is written to work **with or without** its image (it detects
the baked-in runtime and skips the install), so participants submit the same
two files regardless. Use the published images above or **build and push your
own** images; the `moujar/*` tags are the author's personal builds, not a
hard dependency. Full build/push/wire-in instructions and the important
caveats are in **[`docker/README.md`](docker/README.md)**.

> ⚠️ The vLLM and llama.cpp Dockerfiles are provided but **have not been
> build-tested** by the maintainer (llama.cpp compiles CUDA and is sensitive to
> glibc + GPU compute capability). Build and run a real submission on your own
> GPU queue before trusting them.

## Deploy to Codabench

1. Choose and set `docker_image` in
   [`competition.yaml`](LLM-Inference-Competition/competition.yaml)
   (stock image for Transformers; otherwise the matching custom image you built
   and pushed).
2. Zip the **contents** of `LLM-Inference-Competition/` (so
   `competition.yaml` sits at the zip root).
3. On [Codabench](https://www.codabench.org/): **Benchmarks → Management →
   Upload**, then attach the competition to a **queue with GPU workers**.
   Workers need **internet access** to `pip install` requirements and download
   models from the Hugging Face Hub.

## Contributing

Issues and PRs welcome — new backend starter kits, metric options, and
build-tested Dockerfiles especially. By contributing you agree your changes are
released under the repo's [MIT license](LICENSE).
