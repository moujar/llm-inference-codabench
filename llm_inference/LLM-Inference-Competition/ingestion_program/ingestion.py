# ------------------------------------------
# Ingestion program for the LLM Inference competition
#
# It:
#   1. Reads the participant's config (config.yaml) from the submission dir.
#   2. Downloads / clones the requested Hugging Face model onto the worker
#      (logging in with the HF token first if the repo is private).
#   3. Runs inference on the hidden prompts (input_data/prompts.jsonl).
#   4. Saves the generated answers to output/predictions.jsonl for scoring.
# ------------------------------------------
import os
import sys
import json
import time
import subprocess


def _lazy_import():
    """Import heavy deps, installing them at runtime if the image lacks them."""
    global torch, yaml, AutoModelForCausalLM, AutoTokenizer, hf_login
    try:
        import torch  # noqa: F401
        import yaml  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        from huggingface_hub import login as hf_login  # noqa: F401
    except ImportError:
        print("[ingestion] Installing missing dependencies ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "transformers>=4.44", "huggingface_hub>=0.24", "accelerate", "pyyaml"]
        )
        import torch  # noqa: F401
        import yaml  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        from huggingface_hub import login as hf_login  # noqa: F401


class Ingestion:

    DEFAULT_CONFIG = {
        "model_name": "unknown-model",
        "hf_repo_id": None,
        "hf_token": "",
        "system_prompt": "",
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "max_new_tokens": 128,
        "max_context_length": 2048,
    }

    def load_config(self, submission_dir):
        """Read the participant's config.yaml (or config.json)."""
        cfg = dict(self.DEFAULT_CONFIG)
        path_yaml = os.path.join(submission_dir, "config.yaml")
        path_yml = os.path.join(submission_dir, "config.yml")
        path_json = os.path.join(submission_dir, "config.json")

        if os.path.exists(path_yaml) or os.path.exists(path_yml):
            with open(path_yaml if os.path.exists(path_yaml) else path_yml) as f:
                user_cfg = yaml.safe_load(f) or {}
        elif os.path.exists(path_json):
            with open(path_json) as f:
                user_cfg = json.load(f)
        else:
            raise FileNotFoundError(
                f"No config.yaml / config.json found in submission dir: {submission_dir}"
            )

        cfg.update({k: v for k, v in user_cfg.items() if v is not None})
        if not cfg["hf_repo_id"]:
            raise ValueError("config must define a non-empty 'hf_repo_id'.")

        self.config = cfg
        print(f"[ingestion] Loaded config for model '{cfg['model_name']}' "
              f"(repo: {cfg['hf_repo_id']})")
        return cfg

    def load_prompts(self, input_dir):
        """Read the hidden prompts (one JSON object per line)."""
        prompts_file = os.path.join(input_dir, "prompts.jsonl")
        prompts = []
        with open(prompts_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(json.loads(line))
        print(f"[ingestion] Loaded {len(prompts)} prompts.")
        self.prompts = prompts
        return prompts

    def load_model(self):
        """Log in to HF (if a token is given) and clone/load the model."""
        cfg = self.config
        token = (cfg.get("hf_token") or "").strip()
        if token:
            print("[ingestion] Logging in to Hugging Face Hub with provided token.")
            hf_login(token=token)

        auth = token if token else None
        print(f"[ingestion] Downloading model '{cfg['hf_repo_id']}' ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ingestion] Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg["hf_repo_id"], token=auth)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["hf_repo_id"],
            token=auth,
            torch_dtype=(torch.float16 if self.device == "cuda" else torch.float32),
        ).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("[ingestion] Model loaded.")

    def _build_input(self, question):
        """Build model input, using a chat template when the tokenizer has one."""
        cfg = self.config
        system_prompt = cfg.get("system_prompt", "")
        if getattr(self.tokenizer, "chat_template", None):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": question})
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Plain completion fallback.
        prefix = (system_prompt + "\n\n") if system_prompt else ""
        return f"{prefix}Question: {question}\nAnswer:"

    def run_inference(self):
        """Generate an answer for every prompt and record latency."""
        cfg = self.config
        predictions = []
        do_sample = bool(cfg["do_sample"]) and float(cfg["temperature"]) > 0

        for item in self.prompts:
            text = self._build_input(item["prompt"])
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=int(cfg["max_context_length"]),
            ).to(self.device)

            start = time.time()
            with torch.no_grad():
                gen_kwargs = dict(
                    max_new_tokens=int(cfg["max_new_tokens"]),
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                if do_sample:
                    gen_kwargs["temperature"] = float(cfg["temperature"])
                    gen_kwargs["top_p"] = float(cfg["top_p"])
                output_ids = self.model.generate(**inputs, **gen_kwargs)
            latency = time.time() - start

            # Only decode the newly generated tokens.
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            predictions.append({
                "id": item["id"],
                "prediction": answer,
                "latency": round(latency, 4),
            })
            print(f"[ingestion] id={item['id']} ({latency:.2f}s): {answer[:80]!r}")

        self.predictions = predictions
        return predictions

    def save_predictions(self, output_dir):
        """Write predictions.jsonl + a small metadata file for scoring."""
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, "predictions.jsonl")
        with open(out_file, "w") as f:
            for p in self.predictions:
                f.write(json.dumps(p) + "\n")

        meta_file = os.path.join(output_dir, "metadata.json")
        with open(meta_file, "w") as f:
            json.dump({
                "model_name": self.config["model_name"],
                "hf_repo_id": self.config["hf_repo_id"],
                "num_predictions": len(self.predictions),
            }, f, indent=4)
        print(f"[ingestion] Saved {len(self.predictions)} predictions to {out_file}")


_lazy_import()
