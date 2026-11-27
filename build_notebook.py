"""Generate qwen_livekit_phone_agent.ipynb from the tested .py files in this folder.

Run:  python build_notebook.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
FILES = ["model_server.py", "qwen_plugins.py", "agent.py", "smoke_test.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


cells = [
    md("""
# 📞 Qwen voice agent on the phone — LiveKit + Google Colab

A caller dials a phone number → **LiveKit Cloud** answers and puts the call in a room →
this notebook's agent joins and talks:

```
caller ──phone──▶ LiveKit Cloud (free number + SIP) ──WebRTC──▶ agent (this Colab GPU)
                                                                  │  Silero VAD (turn taking)
                                                                  │  Qwen3-ASR-0.6B   (speech → text)
                                                                  │  Qwen2.5-7B via Ollama (thinking + tools)
                                                                  └  Qwen3-TTS-0.6B   (text → speech, voice "Ryan")
```

The agent only makes **outbound** connections to LiveKit, so Colab needs no open ports.

**Before you start** (one-time, ~5 minutes — details in the README next to this notebook):
1. Create a free account at **https://cloud.livekit.io** → new project → *Settings → API keys*: note the
   **URL** (`wss://…livekit.cloud`), **API key**, **API secret**.
2. In Colab, open the 🔑 **Secrets** panel (left sidebar) and add `LIVEKIT_URL`, `LIVEKIT_API_KEY`,
   `LIVEKIT_API_SECRET`, and toggle *Notebook access* on for each.
3. *Runtime → Change runtime type → **T4 GPU***.
4. In the LiveKit dashboard: *Telephony → Phone numbers* → get the free US number, and create a
   **dispatch rule** for it (*Individual* rooms). You can also test without a phone using the
   *Agents Playground* (step 8).

Run the cells top to bottom. First run downloads ~10 GB of models (≈10 min).
"""),
    md("## 1. Check the GPU"),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    md("""
## 2. Install packages
`qwen-asr` and `qwen-tts` pin different `transformers` patch versions (4.57.6 vs 4.57.3), so `qwen-tts`
is installed with `--no-deps` on top of `qwen-asr` and its few extra dependencies are added by hand.
"""),
    code("""
!apt-get -qq install -y sox libsox-fmt-all zstd > /dev/null
!pip install -q -U qwen-asr
!pip install -q -U --no-deps qwen-tts
!pip install -q onnxruntime einops torchaudio soundfile librosa
!pip install -q "livekit-agents[openai,silero]~=1.9"
!pip list 2>/dev/null | grep -iE "^(qwen-asr|qwen-tts|transformers|livekit-agents|torch) "
"""),
    md("## 3. Install Ollama and pull the Qwen LLM (~4.7 GB)"),
    code("""
import subprocess, time, urllib.request
!curl -fsSL https://ollama.com/install.sh | sh > /dev/null
ollama = subprocess.Popen(["ollama", "serve"], stdout=open("ollama.log", "w"), stderr=subprocess.STDOUT)
for _ in range(30):
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2); break
    except Exception:
        time.sleep(1)
LLM_MODEL = "qwen2.5:7b"   # smaller/faster: "qwen2.5:3b"
!ollama pull {LLM_MODEL}
"""),
    md("## 4. Write the agent files\nThese are the same files as in the `voice_agent_colab/` folder."),
]
for name in FILES:
    cells.append(code(f"%%writefile {name}\n" + (HERE / name).read_text(encoding="utf-8")))

cells += [
    md("""
## 5. Start the Qwen model server
Loads Qwen3-ASR and Qwen3-TTS once on the GPU (first run downloads them). Wait for **ready**.
Change the voice with `TTS_SPEAKER` (English: `Ryan`, `Aiden`; others: `Vivian`, `Serena`, `Uncle_Fu`,
`Dylan`, `Eric`, `Ono_Anna`, `Sohee`).
"""),
    code("""
import os, subprocess, time, json, urllib.request
os.environ.update({"TTS_SPEAKER": "Ryan", "ASR_LANGUAGE": "English", "TTS_LANGUAGE": "English"})
model_server = subprocess.Popen(["python", "model_server.py"], stdout=open("model_server.log", "w"),
                                stderr=subprocess.STDOUT)
for i in range(180):
    try:
        print("ready:", json.load(urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=2)))
        break
    except Exception:
        if model_server.poll() is not None:
            print("model server exited; log:"); print(open("model_server.log").read()[-3000:]); break
        time.sleep(5)
!tail -n 6 model_server.log
"""),
    md("## 6. Listen to the voice and test the whole pipeline"),
    code("""
import numpy as np, urllib.request, json
from IPython.display import Audio, display
req = urllib.request.Request("http://127.0.0.1:8765/tts", method="POST",
      data=json.dumps({"text": "Hi! I'm your Qwen voice assistant. How can I help you today?"}).encode())
with urllib.request.urlopen(req) as r:
    pcm = np.frombuffer(r.read(), dtype=np.int16); sr = int(r.headers["X-Sample-Rate"])
display(Audio(pcm.astype(np.float32) / 32768, rate=sr))
"""),
    code("""
import os
os.environ.update({"LLM_MODEL": LLM_MODEL, "OLLAMA_URL": "http://127.0.0.1:11434/v1"})
!python smoke_test.py 2>&1 | grep -vE "^(DEBUG|INFO)" | tail -n 15
!tail -n 4 model_server.log
"""),
    md("""
## 7. Start the phone agent
Reads your LiveKit credentials from Colab Secrets and connects to LiveKit Cloud.
Look for `registered worker` in the log.
"""),
    code("""
import os, subprocess, time
from google.colab import userdata
for k in ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]:
    os.environ[k] = userdata.get(k)
agent_proc = subprocess.Popen(["python", "agent.py", "start"], stdout=open("agent.log", "w"),
                              stderr=subprocess.STDOUT)
time.sleep(20)
!grep -iE "registered|error|warn" agent.log | tail -n 5
"""),
    md("""
## 8. Talk to it
* **Phone:** call the LiveKit number from your mobile. The first reply can take a few seconds while
  everything warms up.
* **Browser (no phone needed):** open https://agents-playground.livekit.io, sign in with the same
  LiveKit project, *Connect*, and allow the microphone.

Run the next cell any time to see what the caller said and what the models did.
"""),
    code("""
!grep -E "caller:|error|Error" agent.log | tail -n 20
!grep -E "^\\[(asr|tts)" model_server.log | tail -n 12
"""),
    md("## 9. Stop everything\n(Colab also stops it all when the runtime disconnects.)"),
    code("""
for p in ["agent_proc", "model_server", "ollama"]:
    if p in globals():
        globals()[p].terminate()
print("stopped")
"""),
]

nb = {"cells": cells, "metadata": {
    "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
out = HERE / "qwen_livekit_phone_agent.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} ({len(cells)} cells)")
