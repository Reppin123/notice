"""AssemblyAI Universal-Streaming (v3) with word timings anchored to the Mac clock."""
from __future__ import annotations

import asyncio
import json
import os
import time
from urllib.parse import urlencode

import numpy as np
import sounddevice as sd
import websockets

from .fusion import Word

SR = 16000
CHUNK = 800  # 50 ms


class Transcriber:
    """Calls on_partial(words) while speaking and on_turn(words, text) at end of turn.

    Word times from AssemblyAI are ms since the first audio byte we sent; t0 is the Mac
    time that byte was captured, so word.start = t0 + ms/1000.
    """

    def __init__(self, on_turn, on_partial=None, api_key: str | None = None, device=None):
        self.api_key = api_key or os.environ["ASSEMBLYAI_API_KEY"]
        self.on_turn, self.on_partial, self.device = on_turn, on_partial, device
        self.t0: float | None = None
        self._q: asyncio.Queue[bytes] = asyncio.Queue()
        self._done_turns: set[int] = set()

    def _words(self, raw) -> list[Word]:
        return [Word(w["text"], self.t0 + w["start"] / 1000, self.t0 + w["end"] / 1000) for w in raw]

    async def run(self):
        loop = asyncio.get_running_loop()

        def cb(indata, frames, tinfo, status):
            if self.t0 is None:
                # capture time of this buffer's first sample on the Mac wall clock
                self.t0 = time.time() - (tinfo.currentTime - tinfo.inputBufferAdcTime) - frames / SR
            loop.call_soon_threadsafe(self._q.put_nowait, bytes(indata))

        params = urlencode({"sample_rate": SR, "encoding": "pcm_s16le", "format_turns": "true",
                            "end_of_turn_confidence_threshold": 0.6, "min_end_of_turn_silence_when_confident": 300})
        url = f"wss://streaming.assemblyai.com/v3/ws?{params}"
        async with websockets.connect(url, additional_headers={"Authorization": self.api_key}) as ws:
            with sd.RawInputStream(samplerate=SR, channels=1, dtype="int16", blocksize=CHUNK,
                                   device=self.device, callback=cb):
                sender = asyncio.create_task(self._send(ws))
                try:
                    async for raw in ws:
                        self._handle(json.loads(raw))
                finally:
                    sender.cancel()

    async def _send(self, ws):
        while True:
            await ws.send(await self._q.get())

    def _handle(self, m: dict):
        if m.get("type") != "Turn" or self.t0 is None:
            return
        words = self._words(m.get("words") or [])
        order = m.get("turn_order")
        if m.get("end_of_turn"):
            # act on the first end-of-turn (unformatted) for latency; skip the formatted repeat
            if order in self._done_turns:
                return
            self._done_turns.add(order)
            text = m.get("transcript") or " ".join(w.text for w in words)
            self.on_turn(words, text)
        elif self.on_partial:
            self.on_partial(words)
