# Qwen phone agent on LiveKit, run from Google Colab

Call a phone number and talk to an AI agent whose **ears, brain and voice are all Qwen models**,
running on a free Colab GPU:

```
caller ──phone──▶ LiveKit Cloud (free US number + SIP) ──WebRTC──▶ agent.py  (Colab T4 GPU)
                                                                     │ Silero VAD        turn-taking
                                                                     │ Qwen3-ASR-0.6B    speech → text
                                                                     │ Qwen2.5-7B        reasoning + tools (via Ollama)
                                                                     └ Qwen3-TTS-0.6B    text → speech
```

| File | What it does |
|---|---|
| `qwen_livekit_phone_agent.ipynb` | **Open this in Colab.** Installs everything, starts the servers, runs the agent. |
| `model_server.py` | Loads Qwen3-ASR + Qwen3-TTS **once** on the GPU and serves `/asr` and `/tts` on localhost, so every call shares one copy of the models. `--fake` mode needs no GPU (for testing). |
| `qwen_plugins.py` | `QwenSTT` / `QwenTTS`: LiveKit Agents 1.9 plugins that call the model server. |
| `agent.py` | The LiveKit agent: instructions, tools (`get_current_time`, `web_search`), greeting. |
| `smoke_test.py` | Tests STT, TTS, ASR↔TTS round trip and the LLM + tools without a phone. |
| `build_notebook.py` | Regenerates the notebook from the `.py` files (run after editing them). |
| `.env.example` | The environment variables for local runs. Copy it to `.env`, which git ignores. In Colab, use Secrets instead. |

> **Clock tip:** LiveKit rejects tokens from a machine whose clock is more than about a minute off
> (you'll see HTTP 401). If that happens on your own PC, sync its clock. Colab's clock is already accurate.

## One-time setup (≈5 minutes, free)

1. **LiveKit Cloud account.** Sign up at https://cloud.livekit.io (the *Build* plan is $0, no card).
   Create a project, then *Settings → API keys* and copy the **WebSocket URL** (`wss://…livekit.cloud`),
   the **API key** and the **API secret**. Keep the secret private.
2. **Phone number.** In the dashboard, open *Telephony → Phone numbers*, get the **free US local number**,
   and add a **dispatch rule** for it (type *Individual*, so each caller gets a room). Leave the agent
   name empty: `agent.py` joins every new room automatically. The free plan includes **inbound** calls
   only (about 50 US minutes a month at the time of writing; check the pricing page).
3. **Colab.** Open the notebook in https://colab.research.google.com (*File → Upload notebook*),
   choose *Runtime → Change runtime type → T4 GPU*, then open the 🔑 **Secrets** panel and add
   `LIVEKIT_URL`, `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`, each with *Notebook access* turned on.

## Run it

Run the cells top to bottom. The first run downloads about 10 GB: the Qwen ASR and TTS models plus
the 4.7 GB LLM, which takes around 10 minutes. Then:
- **Step 6** plays the TTS voice in the notebook and runs `smoke_test.py` with the real models.
- **Step 7** connects the agent to LiveKit. Wait for `registered worker` in the log.
- **Step 8:** call your LiveKit number. If you don't have a phone handy, use the browser at
  https://agents-playground.livekit.io instead.

Tuning:
- **Voice:** set `TTS_SPEAKER` in step 5. English voices are `Ryan` and `Aiden`; the others are
  `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`, `Ono_Anna` and `Sohee`.
- **Faster replies:** set `LLM_MODEL = "qwen2.5:3b"` in step 3. Replies get quicker but less accurate.
- **Other languages:** set `ASR_LANGUAGE` and `TTS_LANGUAGE`. For example, `"Chinese"` uses one of
  the Chinese voices. `"auto"` / `"Auto"` detects the language.
- **Bigger models** (an A100/L4 Colab GPU is recommended):
  `ASR_MODEL=Qwen/Qwen3-ASR-1.7B` and `TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`. The 1.7B TTS
  also accepts style instructions.

## How it fits LiveKit

- Both Qwen models are *non-streaming* in LiveKit terms, and `AgentSession` wraps them automatically.
- **Speech-to-text:** Silero VAD cuts the caller's audio into turns, and each finished turn goes to
  Qwen3-ASR.
- **Text-to-speech:** the LLM's reply is split into sentences, and each sentence goes to Qwen3-TTS.
  The caller hears sentence 1 while sentence 2 is still being generated.
- LiveKit Agents' LLM step uses Ollama's OpenAI-compatible API (`openai.LLM.with_ollama`), so tool
  calling works through standard function tools.
- Colab can't accept incoming connections. That's fine here, because the agent dials *out* to
  LiveKit Cloud, and LiveKit sends it the calls.

## What was tested, and what wasn't

Tested locally on Windows with LiveKit Agents 1.9.0 and the model server in `--fake` mode:
- The STT and TTS plugins talk to the server correctly, and audio frames come back.
- The agent's LLM and tools ran through `AgentSession` on Ollama `qwen2.5:3b`.
  `get_current_time` and `web_search` were both called and answered correctly.

**Not tested yet:**
- The real Qwen3-ASR and Qwen3-TTS models on a Colab GPU. They need CUDA, and this PC has none.
- A live phone call through LiveKit Cloud. That needs your account.

Expect the first Colab run to surface package-version issues. The two Qwen packages pin different
`transformers` versions, which the notebook handles with `--no-deps`. If a model fails to load, the
model server log (`model_server.log`) shows why.

**Latency:** a T4 is a modest GPU. Expect roughly 1–3 seconds from when the caller stops talking to
the first spoken word. Run step 6 to see the ASR and TTS timings for your session.

**Colab limits:**
- Free runtimes disconnect when idle and after several hours, which takes the agent offline.
  Keep the tab open while testing.
- Colab is for experiments, not a 24/7 phone line. For that, run the same files on a GPU server.
