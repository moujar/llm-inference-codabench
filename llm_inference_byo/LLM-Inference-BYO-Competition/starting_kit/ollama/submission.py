# =============================================================================
# Starter kit: Ollama backend
# =============================================================================
# Runs a local Ollama server and pulls a GGUF model straight from the HF Hub
# (ollama supports `hf.co/<user>/<repo>` model ids for GGUF repos).
# Edit the CONFIG block, then submit this file together with requirements.txt.
# =============================================================================
import os
import time
import shutil
import subprocess
import urllib.request

import ollama

# ---- CONFIG -----------------------------------------------------------------
# For Ollama, point at a GGUF repo. `hf.co/<user>/<repo>` pulls from HF Hub.
# You can also append a quant tag, e.g. "hf.co/user/repo:Q4_K_M".
OLLAMA_MODEL = "hf.co/bartowski/SmolLM2-135M-Instruct-GGUF:Q4_K_M"
HF_TOKEN = ""                 # only for PRIVATE / gated GGUF repos
SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 128
# -----------------------------------------------------------------------------


class Submission:

    backend = "ollama"
    model_name = OLLAMA_MODEL.split("/")[-1]

    def _install_ollama(self):
        """Install the ollama server binary if it is not already present."""
        if shutil.which("ollama"):
            return
        print("[ollama] installing ollama server ...")
        script = "/tmp/install_ollama.sh"
        urllib.request.urlretrieve("https://ollama.com/install.sh", script)
        subprocess.check_call(["sh", script])

    def setup(self):
        if HF_TOKEN:
            os.environ["HF_TOKEN"] = HF_TOKEN
        self._install_ollama()

        print("[ollama] starting server ...")
        self.server = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # Wait for the server to accept connections.
        for _ in range(30):
            try:
                ollama.list()
                break
            except Exception:
                time.sleep(1)

        print(f"[ollama] pulling {OLLAMA_MODEL} ...")
        ollama.pull(OLLAMA_MODEL)

    def generate(self, prompt):
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=(
                ([{"role": "system", "content": SYSTEM_PROMPT}] if SYSTEM_PROMPT else [])
                + [{"role": "user", "content": prompt}]
            ),
            options={
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "num_predict": MAX_NEW_TOKENS,
            },
        )
        return response["message"]["content"].strip()

    def teardown(self):
        if getattr(self, "server", None):
            self.server.terminate()
