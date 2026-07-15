# Example LLM Inference

The goal of this bundle is to run **LLM inference on a GPU worker**. Participants
submit a small config file pointing to a model on the Hugging Face Hub; the
ingestion program clones the model onto the worker, runs inference on a hidden
set of prompts, and the scoring program evaluates the generated answers.

* [View the bundle](LLM-Inference-Competition/)
* [Download the bundle](llm_inference_bundle.zip)
* [Download the test submission](example_submission.zip)

#### What the participant submits

A single `config.yaml` (see [sample](LLM-Inference-Competition/sample_code_submission/config.yaml)):

* `model_name` — display name for the leaderboard
* `hf_repo_id` — the Hugging Face repo to clone (`namespace/name`)
* `hf_token` — optional, only for **private / gated** repos
* `system_prompt`, `temperature`, `top_p`, `do_sample`, `max_new_tokens`,
  `max_context_length` — generation settings

#### Flow

1. **Ingestion** (`ingestion_program/`) reads the config, logs in to Hugging
   Face if a token is given, downloads the model, runs inference on
   `input_data/prompts.jsonl`, and writes `predictions.jsonl`.
2. **Scoring** (`scoring_program/`) compares the predictions to
   `reference_data/reference.jsonl` and reports **accuracy**, **token F1**, and
   **average latency**.

#### Queue and docker

* This bundle uses the Docker image `codalab/codalab-legacy:gpu310`.
* You must manually set the queue to one that contains **GPU workers**.
* `transformers` / `huggingface_hub` are installed at runtime by the ingestion
  program if they are not already in the image.

#### Try it locally

```bash
cd LLM-Inference-Competition
# needs a GPU + torch/transformers to actually run inference:
python3 ingestion_program/run_ingestion.py
# scoring only needs the standard library:
python3 scoring_program/run_scoring.py
```
