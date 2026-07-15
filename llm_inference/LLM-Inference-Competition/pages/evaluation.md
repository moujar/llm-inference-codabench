# Evaluation
***

The ingestion program runs your model on the hidden prompts and produces one
answer per prompt. The scoring program compares each answer to the reference
answer and reports three metrics:

| Metric | Description | Better |
| --- | --- | --- |
| **Accuracy** | Fraction of answers that *contain* the (normalized) reference answer. | higher |
| **Token F1** | Mean token-level F1 overlap between answer and reference (SQuAD-style normalization: lower-cased, punctuation and articles removed). | higher |
| **Avg latency (s)** | Mean generation time per prompt on the GPU worker. | lower |

The leaderboard is ranked primarily by **Accuracy**.

## Normalization

Before comparison, both prediction and reference are lower-cased, stripped of
punctuation and the articles `a`/`an`/`the`, and whitespace-collapsed. So
`"The capital is Paris."` matches the reference `"Paris"`.

## Notes for organizers

- This is a **template**. Swap `input_data/prompts.jsonl` and
  `reference_data/reference.jsonl` for your own task, and adjust the metrics in
  `scoring_program/score.py` (e.g. exact match, ROUGE, BLEU, a judge model).
- Inference needs a GPU worker; set the queue to a GPU queue and keep the
  `codalab/codalab-legacy:gpu310` Docker image (or one with `torch` + CUDA).
