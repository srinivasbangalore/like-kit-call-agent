"""LiveKit Agents STT/TTS plugins backed by the local Qwen model server.

Both are "non-streaming" in LiveKit terms:
  * QwenSTT gets one finished utterance at a time. AgentSession wraps it with the
    Silero VAD (stt.StreamAdapter) automatically, so speech is cut into turns.
  * QwenTTS synthesises one sentence at a time. AgentSession wraps it with a
    sentence tokenizer (tts.StreamAdapter), so the caller hears the first sentence
    while the LLM is still writing the next.
"""

from __future__ import annotations

import os
import uuid

import aiohttp
import numpy as np
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    stt,
    tts,
)
from livekit.agents.utils import AudioBuffer

SERVER = os.environ.get("MODEL_SERVER_URL", "http://127.0.0.1:8765")
TTS_SAMPLE_RATE = 24_000  # model_server.OUT_SR


class _HttpMixin:
    _session: aiohttp.ClientSession | None = None

    def _http(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60))
        return self._session

    async def aclose(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


class QwenSTT(_HttpMixin, stt.STT):
    def __init__(self, server_url: str = SERVER, language: str = "en"):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self.server_url = server_url.rstrip("/")
        self.language = language

    @property
    def model(self) -> str:
        return "qwen3-asr"

    @property
    def provider(self) -> str:
        return "local"

    async def _recognize_impl(self, buffer: AudioBuffer, *, language=None,
                              conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> stt.SpeechEvent:
        frame = rtc.combine_audio_frames(buffer)
        pcm = np.frombuffer(frame.data, dtype=np.int16)
        if frame.num_channels > 1:
            pcm = pcm.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
        try:
            async with self._http().post(f"{self.server_url}/asr?sr={frame.sample_rate}",
                                         data=pcm.tobytes()) as r:
                if r.status != 200:
                    raise APIStatusError(f"ASR server error: {await r.text()}", status_code=r.status)
                out = await r.json()
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"ASR server unreachable: {e}") from e
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=uuid.uuid4().hex,
            alternatives=[stt.SpeechData(language=self.language, text=out.get("text", ""))],
        )


class QwenTTS(_HttpMixin, tts.TTS):
    def __init__(self, server_url: str = SERVER, speaker: str | None = None, language: str | None = None):
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False),
                         sample_rate=TTS_SAMPLE_RATE, num_channels=1)
        self.server_url = server_url.rstrip("/")
        self.speaker = speaker
        self.language = language

    @property
    def model(self) -> str:
        return "qwen3-tts"

    @property
    def provider(self) -> str:
        return "local"

    def synthesize(self, text: str, *,
                   conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> tts.ChunkedStream:
        return _QwenChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _QwenChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        t: QwenTTS = self._tts  # type: ignore[assignment]
        body = {"text": self.input_text}
        if t.speaker:
            body["speaker"] = t.speaker
        if t.language:
            body["language"] = t.language
        try:
            async with t._http().post(f"{t.server_url}/tts", json=body) as r:
                if r.status != 200:
                    raise APIStatusError(f"TTS server error: {await r.text()}", status_code=r.status)
                sr = int(r.headers.get("X-Sample-Rate", TTS_SAMPLE_RATE))
                output_emitter.initialize(request_id=uuid.uuid4().hex, sample_rate=sr,
                                          num_channels=1, mime_type="audio/pcm")
                async for chunk in r.content.iter_chunked(8192):
                    output_emitter.push(chunk)
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"TTS server unreachable: {e}") from e
        output_emitter.flush()
