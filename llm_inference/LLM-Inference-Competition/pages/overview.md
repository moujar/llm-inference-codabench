# Overview
***

This is an example **LLM inference** competition.

Instead of submitting code or predictions, you submit a small **config file**
that points to a model on the [Hugging Face Hub](https://huggingface.co/). The
GPU worker then:

1. **Clones** your model from the Hub (using your token if the repo is private).
2. Runs **inference** on a hidden set of prompts.
3. **Scores** the generated answers against the reference answers.

## What you submit

A single `config.yaml` file (zipped) describing your model and generation
settings:

```yaml
model_name: "SmolLM2-135M-Instruct"          # leaderboard display name
hf_repo_id: "HuggingFaceTB/SmolLM2-135M-Instruct"  # namespace/name on HF Hub
hf_token: ""                                 # only for PRIVATE / gated repos
system_prompt: "You are a helpful assistant. Answer concisely."
temperature: 0.7
top_p: 0.9
do_sample: true
max_new_tokens: 128
max_context_length: 2048
```

> **Private models:** set `hf_token` to a read-only access token. Submissions
> are visible to the organizers, so use a disposable token.

You can start from the sample submission in `sample_code_submission/`.

## Goal

Pick / configure a model that answers the hidden questions accurately while
keeping latency low. See the **Evaluation** page for the metrics.
