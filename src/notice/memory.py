"""Attention memory: only what the wearer FIXATED on (>= 300 ms) is kept, as a crop + caption.

The caption cache doubles as the fast path for voice turns: by the time a turn arrives, the thing
the user was looking at is usually already captioned, so the agent gets text, not an image.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import httpx

from .scene import crop_around, decode, dhash, hamming

log = logging.getLogger("notice.memory")
DATA = Path(os.environ.get("NOTICE_DATA", Path.home() / "interface-hack" / "notice-data"))
MIN_FIX_S = 0.3
DEDUPE_S = 30
CAPTION_MODEL = os.environ.get("NOTICE_CAPTION_MODEL", "claude-haiku-5-5")
PROMPT = ("This is a crop centred on what a person wearing eye-tracking glasses was looking at. "
          "In at most 30 words say what the object at the centre is. Copy any readable text there "
          "verbatim (names, dates, times, places, prices). No preamble.")


class AttentionMemory:
    def __init__(self, scene, enabled: bool = True):
        self.scene = scene
        self.enabled = enabled
        (DATA / "crops").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(DATA / "memory.db", check_same_thread=False)
        self.db.execute("""create table if not exists seen(
            id integer primary key, t real, dwell_ms int, scene_x real, scene_y real,
            crop text, hash text, caption text)""")
        self.db.execute("create virtual table if not exists seen_fts using fts5(caption, content='seen', content_rowid='id')")
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(2, thread_name_prefix="caption")
        self.pending: dict[int, object] = {}
        self.key = os.environ.get("ANTHROPIC_API_KEY")
        self.recent: list[tuple[float, int, int, float]] = []  # (t, hash, row id, fixation start)

    # ---- write -------------------------------------------------------------------------
    def on_fixation(self, f) -> int | None:
        """Called when a world fixation reaches 300 ms. Returns the memory row id."""
        if not self.enabled or f.scene_x is None or f.duration < MIN_FIX_S:
            return None
        dwell = round(f.duration * 1000)
        for _, _, rid, fs in self.recent:
            if abs(fs - f.start) < 0.05:  # same fixation seen again (it grew): keep the longest dwell
                with self.lock:
                    self.db.execute("update seen set dwell_ms = max(dwell_ms, ?) where id = ?", (dwell, rid))
                    self.db.commit()
                return rid
        fr = self.scene.nearest((f.start + f.end) / 2)
        if fr is None:
            return None
        crop = crop_around(decode(fr[1]), f.scene_x, f.scene_y)
        h = dhash(crop)
        now = time.time()
        self.recent = [r for r in self.recent if now - r[0] < DEDUPE_S]
        for _, oh, rid, _ in self.recent:
            if hamming(h, oh) <= 6:
                with self.lock:  # same thing again: extend dwell instead of a new row
                    self.db.execute("update seen set dwell_ms = dwell_ms + ? where id = ?", (round(f.duration * 1000), rid))
                    self.db.commit()
                return rid
        path = DATA / "crops" / f"{int(now * 1000)}.jpg"
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with self.lock:
            cur = self.db.execute("insert into seen(t, dwell_ms, scene_x, scene_y, crop, hash, caption) values(?,?,?,?,?,?,'')",
                                  (f.start, round(f.duration * 1000), f.scene_x, f.scene_y, str(path), str(h)))
            self.db.commit()
            rid = cur.lastrowid
        self.recent.append((now, h, rid, f.start))
        if self.key:
            self.pending[rid] = self.pool.submit(self._caption, rid, path)
        return rid

    def _caption(self, rid: int, path: Path) -> str:
        t0 = time.time()
        b64 = base64.b64encode(path.read_bytes()).decode()
        try:
            r = httpx.post("https://api.anthropic.com/v1/messages", timeout=20, headers={
                "x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": CAPTION_MODEL, "max_tokens": 120, "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": PROMPT}]}]})
            r.raise_for_status()
            cap = "".join(b.get("text", "") for b in r.json()["content"]).strip()
        except Exception as e:
            log.warning("caption failed: %s", e)
            return ""
        with self.lock:
            self.db.execute("update seen set caption = ? where id = ?", (cap, rid))
            self.db.execute("insert into seen_fts(rowid, caption) values(?, ?)", (rid, cap))
            self.db.commit()
        log.info("caption %d in %d ms: %s", rid, (time.time() - t0) * 1000, cap[:80])
        self.pending.pop(rid, None)
        return cap

    # ---- read --------------------------------------------------------------------------
    def get(self, rid: int, wait_s: float = 0.0) -> dict | None:
        fut = self.pending.get(rid)
        if fut is not None and wait_s > 0:
            try:
                fut.result(timeout=wait_s)
            except Exception:
                pass
        with self.lock:
            row = self.db.execute("select id, t, dwell_ms, crop, caption from seen where id = ?", (rid,)).fetchone()
        return dict(zip(("id", "t", "dwell_ms", "crop", "caption"), row)) if row else None

    def recall(self, query: str = "", since: float | None = None, limit: int = 5) -> list[dict]:
        since = since or 0
        with self.lock:
            if query.strip():
                q = " OR ".join(json.dumps(w) for w in query.split() if len(w) > 2) or json.dumps(query)
                rows = self.db.execute(
                    "select s.id, s.t, s.dwell_ms, s.crop, s.caption, bm25(seen_fts) from seen_fts "
                    "join seen s on s.id = seen_fts.rowid where seen_fts match ? and s.t >= ? limit 50", (q, since)).fetchall()
            else:
                rows = self.db.execute("select id, t, dwell_ms, crop, caption, 0 from seen where t >= ? "
                                       "order by t desc limit 50", (since,)).fetchall()
        now = time.time()
        # bm25 is negative-better; add dwell and recency so "the one I looked at longest" ranks up
        scored = sorted(rows, key=lambda r: r[5] - r[2] / 2000 + (now - r[1]) / 3600)
        return [dict(zip(("id", "t", "dwell_ms", "crop", "caption"), r[:5])) for r in scored[:limit]]
