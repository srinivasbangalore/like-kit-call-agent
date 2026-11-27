"""LiveKit phone/voice agent: Qwen3-ASR -> Qwen LLM (Ollama) -> Qwen3-TTS.

Run (after model_server.py and Ollama are up, and LIVEKIT_* env vars are set):
    python agent.py start        # production mode, waits for calls
    python agent.py dev          # verbose logs
Env: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LLM_MODEL (default qwen2.5:7b),
     OLLAMA_URL (default http://127.0.0.1:11434/v1), AGENT_NAME (optional).
"""

from __future__ import annotations

import html
import logging
import os
import re
import urllib.parse
from datetime import datetime, timezone

import aiohttp
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, JobProcess, RunContext, cli, function_tool
from livekit.plugins import openai, silero

from qwen_plugins import QwenSTT, QwenTTS

log = logging.getLogger("qwen-voice-agent")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/v1")
AGENT_NAME = os.environ.get("AGENT_NAME", "")  # empty = join every new room automatically

INSTRUCTIONS = """You are a friendly, concise voice assistant answering a phone call.
Speak naturally in short sentences; one to three sentences per reply unless asked for more.
Never use markdown, lists, emoji, or read out URLs. Spell numbers the way people say them.
If you need current information (news, prices, weather, recent facts), call web_search first
and answer only from what it returns. If you are unsure, say so. If the caller says goodbye,
say a short goodbye."""


class VoiceAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)

    @function_tool
    async def get_current_time(self, context: RunContext) -> str:
        """Get the current date and time (UTC)."""
        return datetime.now(timezone.utc).strftime("%A %d %B %Y, %H:%M UTC")

    @function_tool
    async def web_search(self, context: RunContext, query: str) -> str:
        """Search the web for current information and return the top results.

        Args:
            query: what to search for
        """
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
                async with s.post("https://lite.duckduckgo.com/lite/", data={"q": query},
                                  headers={"User-Agent": "Mozilla/5.0 voice-agent/0.1"}) as r:
                    page = await r.text()
        except Exception as e:  # noqa: BLE001
            return f"Search failed: {e}"
        titles = re.findall(r"class=['\"]result-link['\"][^>]*>(.*?)</a>", page, re.S)
        snippets = re.findall(r"class=['\"]result-snippet['\"][^>]*>(.*?)</td>", page, re.S)
        clean = lambda s: html.unescape(re.sub(r"<[^>]+>", "", s)).strip()  # noqa: E731
        results = [f"{clean(t)}: {clean(sn)}" for t, sn in zip(titles, snippets)][:4]
        return "\n".join(results) or "No results found."


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


# Colab has only 2 CPUs and the GPU models keep them busy; LiveKit's default
# (stop accepting calls above 70% CPU) would make the agent look "unavailable".
server = AgentServer(setup_fnc=prewarm, num_idle_processes=1,
                     load_threshold=float(os.environ.get("LOAD_THRESHOLD", "inf")))


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=QwenSTT(),
        llm=openai.LLM.with_ollama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0.4),
        tts=QwenTTS(),
        min_endpointing_delay=0.4,
    )

    @session.on("user_input_transcribed")
    def _heard(ev):
        if getattr(ev, "is_final", True):
            log.info("caller: %s", ev.transcript)

    await session.start(agent=VoiceAssistant(), room=ctx.room)
    await session.generate_reply(instructions="Greet the caller briefly and ask how you can help.")


if __name__ == "__main__":
    cli.run_app(server)
