"""Plumbing test without LiveKit Cloud or a phone:
  1. STT plugin -> model server /asr
  2. TTS plugin -> model server /tts (audio frames come back)
  3. The agent's LLM + tools via AgentSession in text-only mode (needs Ollama)

Usage:  python smoke_test.py            (model server must be running, real or --fake)
        LLM_MODEL=qwen2.5:3b OLLAMA_URL=http://127.0.0.1:11434/v1 python smoke_test.py
"""

import asyncio
import os
import time

import numpy as np
from livekit import rtc
from livekit.agents import AgentSession
from livekit.plugins import openai

from qwen_plugins import QwenSTT, QwenTTS


async def main() -> None:
    sr = 48_000
    tone = (np.sin(2 * np.pi * 220 * np.arange(sr) / sr) * 6000).astype(np.int16)
    frame = rtc.AudioFrame(tone.tobytes(), sample_rate=sr, num_channels=1, samples_per_channel=sr)

    stt_ = QwenSTT()
    t = time.time()
    ev = await stt_.recognize([frame])
    print(f"STT ({time.time() - t:.2f}s): {ev.type.value} -> {ev.alternatives[0].text!r}")

    tts_ = QwenTTS()
    t = time.time()
    frames = [f async for f in tts_.synthesize("Hello! This is a smoke test of the voice pipeline.")]
    secs = sum(f.frame.samples_per_channel for f in frames) / tts_.sample_rate
    print(f"TTS ({time.time() - t:.2f}s): {len(frames)} frames, {secs:.2f}s of audio")

    # Optional: round-trip real TTS audio back through ASR (meaningful with real models)
    audio = rtc.combine_audio_frames([f.frame for f in frames])
    ev = await stt_.recognize([audio])
    print(f"TTS->ASR round trip: {ev.alternatives[0].text!r}")
    await stt_.aclose()
    await tts_.aclose()

    if os.environ.get("SKIP_LLM"):
        return
    from agent import LLM_MODEL, OLLAMA_URL, VoiceAssistant
    async with openai.LLM.with_ollama(model=LLM_MODEL, base_url=OLLAMA_URL) as llm, \
            AgentSession(llm=llm) as session:
        await session.start(VoiceAssistant())
        for q in ["What's the date today?", "Search the web: what is the latest Python release?"]:
            t = time.time()
            result = await session.run(user_input=q)
            calls = [e.item.name for e in result.events if e.type == "function_call"]
            reply = [e.item.text_content for e in result.events if e.type == "message"]
            print(f"LLM ({time.time() - t:.1f}s) {q!r}\n  tools={calls}\n  reply={reply[-1] if reply else None!r}")


if __name__ == "__main__":
    asyncio.run(main())
