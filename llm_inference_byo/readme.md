# Example LLM Inference — Bring Your Own Backend

An LLM inference competition where participants **choose their own inference
library**. A submission is just two files — `requirements.txt` and
`submission.py` — where `submission.py` implements a tiny `Submission` contract
(`setup` / `generate` / `teardown`). The ingestion program installs the
requirements, runs the model on a hidden set of prompts, and the scoring program
evaluates the answers.

Starter kits are provided for **Transformers**, **vLLM**, **Ollama**, and
**llama.cpp**.

* [View the bundle](LLM-Inference-BYO-Competition/)
* [Download the bundle](llm_inference_byo_bundle.zip)
* [Download the test submission](example_submission.zip)
* [Starter kits](LLM-Inference-BYO-Competition/starting_kit/)

#### What the participant submits

* `requirements.txt` — extra pip packages for the chosen backend
* `submission.py` — a `Submission` class holding the HF repo id (and a token if
  private) plus the inference code

#### Flow

1. **Ingestion** (`ingestion_program/`) installs `requirements.txt`, imports
   `Submission`, calls `setup()`, then `generate(prompt)` for each prompt in
   `input_data/prompts.jsonl`, and writes `predictions.jsonl`.
2. **Scoring** (`scoring_program/`) compares predictions to
   `reference_data/reference.jsonl` and reports **accuracy**, **token F1**, and
   **average latency**.

#### Queue and docker

* Uses the Docker image `codalab/codalab-legacy:gpu310`.
* Set the queue to one with **GPU workers**.
* Workers need **internet access** to `pip install` requirements and download
  models from the Hugging Face Hub.

#### Try it locally

```bash
cd LLM-Inference-BYO-Competition
# runs the sample (transformers) submission — needs a GPU + the deps:
python3 ingestion_program/run_ingestion.py
# scoring only needs the standard library:
python3 scoring_program/run_scoring.py
```

> Related: see [`../llm_inference/`](../llm_inference/) for a simpler variant
> where participants submit only a `config.yaml` and the organizer fixes the
> backend (Transformers).
