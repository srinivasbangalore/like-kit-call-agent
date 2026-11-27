"""Qwen speech model server: loads Qwen3-ASR and Qwen3-TTS ONCE on the GPU and
serves them over localhost, so every LiveKit call shares the same models.

  GET  /health                      -> {"ok": true, "asr": ..., "tts": ..., "speaker": ...}
  POST /asr?sr=16000                body: raw little-endian int16 mono PCM
                                    -> {"text": "...", "language": "English"}
  POST /tts                         body: {"text": "...", "speaker"?: "Ryan", "language"?: "English"}
                                    -> raw int16 mono PCM at 24 kHz (header X-Sample-Rate)

Run:  python model_server.py            (real models, needs a CUDA GPU)
      python model_server.py --fake     (no models: canned text + a tone, for plumbing tests)

Settings (env vars): ASR_MODEL, TTS_MODEL, TTS_SPEAKER, ASR_LANGUAGE, TTS_LANGUAGE,
MODEL_SERVER_PORT (default 8765).
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import numpy as np

OUT_SR = 24_000  # the TTS plugin declares this rate; we resample to it
ASR_MODEL = os.environ.get("ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")
TTS_MODEL = os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
TTS_SPEAKER = os.environ.get("TTS_SPEAKER", "Ryan")
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "English")  # "auto" = detect
TTS_LANGUAGE = os.environ.get("TTS_LANGUAGE", "English")  # "Auto" = detect
MAX_BODY = 20 * 1024 * 1024


class Models:
    def __init__(self, fake: bool):
        self.fake = fake
        self.asr = self.tts = None
        self.asr_lock = threading.Lock()  # one GPU job per model at a time
        self.tts_lock = threading.Lock()
        if fake:
            return
        import torch
        from qwen_asr import Qwen3ASRModel
        from qwen_tts import Qwen3TTSModel

        if not torch.cuda.is_available():
            raise SystemExit("No CUDA GPU. In Colab: Runtime > Change runtime type > T4 GPU.")
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16  # T4 (sm75) has no bf16
        attn = "sdpa"
        try:
            import flash_attn  # noqa: F401
            if major >= 8:
                attn = "flash_attention_2"
        except ImportError:
            pass
        print(f"[models] GPU {torch.cuda.get_device_name(0)}  dtype={dtype}  attn={attn}", flush=True)

        t = time.time()
        self.asr = Qwen3ASRModel.from_pretrained(
            ASR_MODEL, dtype=dtype, device_map="cuda:0", attn_implementation=attn,
            max_inference_batch_size=4, max_new_tokens=256)
        print(f"[models] ASR {ASR_MODEL} loaded in {time.time() - t:.0f}s", flush=True)

        t = time.time()
        self.tts = Qwen3TTSModel.from_pretrained(
            TTS_MODEL, dtype=dtype, device_map="cuda:0", attn_implementation=attn)
        speakers = self.tts.get_supported_speakers() or []
        print(f"[models] TTS {TTS_MODEL} loaded in {time.time() - t:.0f}s; speakers={speakers}", flush=True)
        self.warmup()

    def warmup(self):
        t = time.time()
        pcm, _ = self.synthesize("Hello, I'm ready.")
        self.transcribe(pcm, OUT_SR)
        print(f"[models] warm-up round trip {time.time() - t:.1f}s", flush=True)

    # -- inference -------------------------------------------------------------
    def transcribe(self, pcm: np.ndarray, sr: int) -> dict:
        if self.fake:
            return {"text": "what time is it" if len(pcm) > sr // 4 else "", "language": "English"}
        audio = pcm.astype(np.float32) / 32768.0
        lang = None if ASR_LANGUAGE.lower() == "auto" else ASR_LANGUAGE
        with self.asr_lock:
            res = self.asr.transcribe(audio=(audio, sr), language=lang)[0]
        return {"text": (res.text or "").strip(), "language": res.language}

    def synthesize(self, text: str, speaker: str | None = None,
                   language: str | None = None) -> tuple[np.ndarray, int]:
        if self.fake:
            t = np.arange(int(OUT_SR * min(3.0, 0.06 * len(text) + 0.3))) / OUT_SR
            return (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16), OUT_SR
        with self.tts_lock:
            wavs, sr = self.tts.generate_custom_voice(
                text=text, speaker=speaker or TTS_SPEAKER, language=language or TTS_LANGUAGE)
        wav = np.asarray(wavs[0], dtype=np.float32)
        if sr != OUT_SR:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=OUT_SR)
        return (np.clip(wav, -1, 1) * 32767).astype(np.int16), OUT_SR


def make_handler(models: Models):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json", headers=None):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ValueError("body too large")
            return self.rfile.read(n) if n else b""

        def do_GET(self):
            if urlsplit(self.path).path == "/health":
                return self._send(200, {"ok": True, "fake": models.fake, "asr": ASR_MODEL,
                                        "tts": TTS_MODEL, "speaker": TTS_SPEAKER})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            parts = urlsplit(self.path)
            try:
                body = self._body()
                if parts.path == "/asr":
                    sr = int(parse_qs(parts.query).get("sr", ["16000"])[0])
                    pcm = np.frombuffer(body, dtype=np.int16)
                    t = time.time()
                    out = models.transcribe(pcm, sr)
                    print(f"[asr {time.time() - t:.2f}s, {len(pcm) / sr:.1f}s audio] {out['text']!r}", flush=True)
                    return self._send(200, out)
                if parts.path == "/tts":
                    req = json.loads(body or b"{}")
                    text = str(req.get("text", "")).strip()
                    if not text:
                        return self._send(400, {"error": "empty text"})
                    t = time.time()
                    pcm, sr = models.synthesize(text, req.get("speaker"), req.get("language"))
                    print(f"[tts {time.time() - t:.2f}s -> {len(pcm) / sr:.1f}s audio] {text!r}", flush=True)
                    return self._send(200, pcm.tobytes(), "application/octet-stream",
                                      {"X-Sample-Rate": str(sr)})
            except Exception as e:  # noqa: BLE001 - report to the caller
                print(f"[error] {parts.path}: {e}", flush=True)
                return self._send(500, {"error": str(e)})
            self._send(404, {"error": "not found"})

        def log_message(self, *a):
            pass

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="no models; for testing the plumbing")
    ap.add_argument("--port", type=int, default=int(os.environ.get("MODEL_SERVER_PORT", "8765")))
    args = ap.parse_args()
    models = Models(fake=args.fake)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(models))
    print(f"[models] serving on http://127.0.0.1:{args.port}  fake={args.fake}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
