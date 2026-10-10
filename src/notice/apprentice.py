"""Apprentice = the hands. Talks to the local agent's WebSocket (127.0.0.1:8765) as a client."""
from __future__ import annotations

import asyncio
import glob
import json
import logging
import os

import websockets

log = logging.getLogger("notice.apprentice")


def _tokens() -> list[str]:
    if t := os.environ.get("APPRENTICE_BRIDGE_TOKEN"):
        return [t]
    paths = sorted(glob.glob(os.path.expanduser("~/Library/Application Support/*/bridge_token")),
                   key=os.path.getmtime, reverse=True)
    return [open(p).read().strip() for p in paths]


class Apprentice:
    def __init__(self, url: str = "ws://127.0.0.1:8765", on_event=None):
        self.url, self.on_event = url, on_event or (lambda m: None)
        self.ws = None
        self._lock = asyncio.Lock()

    async def connect(self):
        last = None
        for tok in _tokens():
            try:
                ws = await websockets.connect(self.url, origin=None, max_size=None)
                await ws.send(json.dumps({"type": "hello", "protocol_version": "0", "client_kind": "notice", "auth_token": tok}))
                first = json.loads(await asyncio.wait_for(ws.recv(), 5))
                if first.get("type") == "hello":
                    self.ws = ws
                    asyncio.create_task(self._pump())
                    log.info("connected to Apprentice %s", first.get("app_version"))
                    return self
                await ws.close()
            except Exception as e:  # wrong token closes the socket
                last = e
        raise RuntimeError(f"could not connect to Apprentice at {self.url}: {last}")

    async def _pump(self):
        try:
            async for raw in self.ws:
                try:
                    self.on_event(json.loads(raw))
                except Exception:
                    log.exception("event handler")
        finally:
            self.ws = None

    async def ask(self, text: str, context: str = "", app_name: str = ""):
        """Send resolved context, then the utterance as a normal user turn."""
        async with self._lock:
            if self.ws is None:
                await self.connect()
            if context:
                await self.ws.send(json.dumps({"type": "current_context", "app_name": app_name, "bundle_id": "", "details": context}))
            await self.ws.send(json.dumps({"type": "transcript_final", "text": text, "source": "notice"}))
