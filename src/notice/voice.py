"""Notice's own in-ear voice (ElevenLabs streaming TTS) for proactive cues: coaching, curiosity, fatigue.

Apprentice speaks its normal replies itself; this is for short cues Notice decides to say.
Key: env ELEVENLABS_API_KEY, else macOS Keychain service "ElevenLabs" (account "api_key").
No key -> falls back to macOS `say`, so the pipeline always works.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess

import httpx

log = logging.getLogger("notice.voice")

API = "https://api.elevenlabs.io/v1"
RATE = 22050
MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_flash_v2_5")  # lowest-latency model
VOICE = os.environ.get("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # premade "George"; override freely


def api_key() -> str | None:
    if k := os.environ.get("ELEVENLABS_API_KEY"):
        return k
    try:
        r = subprocess.run(["security", "find-generic-password", "-s", "ElevenLabs", "-a", "api_key", "-w"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


class Voice:
    def __init__(self, backend: str = "auto"):
        self.key = api_key() if backend in ("auto", "eleven") else None
        self.backend = "eleven" if self.key else ("say" if backend != "none" else "none")
        self._lock = asyncio.Lock()
        log.info("voice backend: %s", self.backend)

    async def say(self, text: str):
        """Speak one short cue. Serialised so cues never talk over each other."""
        if not text or self.backend == "none":
            return
        async with self._lock:
            if self.backend == "eleven":
                try:
                    return await self._eleven(text)
                except Exception as e:
                    log.warning("ElevenLabs failed (%s), using macOS say", e)
            await asyncio.to_thread(subprocess.run, ["say", text])

    async def _eleven(self, text: str):
        import sounddevice as sd

        url = f"{API}/text-to-speech/{VOICE}/stream?output_format=pcm_{RATE}&optimize_streaming_latency=3"
        body = {"text": text, "model_id": MODEL}
        stream = sd.RawOutputStream(samplerate=RATE, channels=1, dtype="int16")
        stream.start()
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                async with c.stream("POST", url, json=body, headers={"xi-api-key": self.key}) as r:
                    if r.status_code != 200:
                        raise RuntimeError(f"HTTP {r.status_code}: {(await r.aread())[:200]!r}")
                    buf = b""
                    async for chunk in r.aiter_bytes():
                        buf += chunk
                        n = len(buf) - (len(buf) % 2)  # whole int16 samples only
                        if n:
                            await asyncio.to_thread(stream.write, buf[:n])
                            buf = buf[n:]
        finally:
            await asyncio.to_thread(stream.stop)
            stream.close()


async def check() -> str:
    """Validate the key without spending characters."""
    k = api_key()
    if not k:
        return "no ElevenLabs key (set ELEVENLABS_API_KEY or Keychain service 'ElevenLabs')"
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{API}/user/subscription", headers={"xi-api-key": k})
    if r.status_code != 200:
        return f"key rejected: HTTP {r.status_code}"
    s = r.json()
    return f"ok: tier {s.get('tier')}, {s.get('character_count')}/{s.get('character_limit')} characters used"


def main():  # uv run notice-voice "text"   |   uv run notice-voice --check
    import sys
    logging.basicConfig(level=logging.INFO)
    if sys.argv[1:] == ["--check"]:
        print(asyncio.run(check()))
    else:
        asyncio.run(Voice().say(" ".join(sys.argv[1:]) or "Notice voice is working."))
