"""Attention memory: only what the wearer FIXATED on is kept, as a small crop + its text.

Privacy rules (2026-10-10):
- Capture only while armed (Apprentice arms it while the user is on a call; auto-off after 10 min
  without renewal). Not armed = nothing written; the scene stays in a 10 s in-memory buffer.
- Looks of >= 500 ms only, near-duplicates skipped, at most 10 crops a minute.
- Crops that are mostly a face are never saved (Apple Vision face check, on device).
- Crops older than 24 h are deleted; forget() wipes everything.
- Text is read on the Mac (Apple Vision OCR), so the agent normally gets text, not images.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path

import cv2

from . import ocr
from .scene import crop_around, decode, dhash, hamming

log = logging.getLogger("notice.memory")
DATA = Path(os.environ.get("NOTICE_DATA", Path.home() / "interface-hack" / "notice-data"))
MIN_FIX_S = 0.5
DEDUPE_S = 30
MAX_PER_MIN = 10
FACE_MAX = 0.12  # skip if a face covers more than 12% of the crop
KEEP_S = 24 * 3600
ARM_TTL_S = 600


class AttentionMemory:
    def __init__(self, scene, enabled: bool = True, always_armed: bool = False):
        self.scene = scene
        self.enabled = enabled
        self.always_armed = always_armed
        self.armed_until = 0.0
        (DATA / "crops").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(DATA / "memory.db", check_same_thread=False)
        self.db.execute("""create table if not exists seen(
            id integer primary key, t real, dwell_ms int, scene_x real, scene_y real,
            crop text, hash text, caption text, codes text default '', labels text default '')""")
        if "conf" not in {r[1] for r in self.db.execute("pragma table_info(seen)")}:
            self.db.execute("alter table seen add column conf real default 0")
        self.db.execute("create virtual table if not exists seen_fts using fts5(caption, content='seen', content_rowid='id')")
        self.lock = threading.Lock()
        self.recent: list[tuple[float, int, int, float]] = []  # (t, hash, row id, fixation start)
        self.saved_at: list[float] = []
        self.stats = {"saved": 0, "dupe": 0, "face": 0, "rate": 0, "unarmed": 0}
        self.prune()
        threading.Thread(target=self._warm, daemon=True).start()

    def _warm(self):  # Vision loads its models on first use (~4 s)
        import numpy as np
        try:
            ocr.analyze(np.full((64, 64, 3), 255, np.uint8))
        except Exception as e:
            log.warning("OCR warm-up failed: %s", e)

    # ---- arming ------------------------------------------------------------------------
    @property
    def armed(self) -> bool:
        return self.enabled and (self.always_armed or time.time() < self.armed_until)

    def arm(self, on: bool):
        self.armed_until = time.time() + ARM_TTL_S if on else 0.0

    # ---- write -------------------------------------------------------------------------
    def on_fixation(self, f, force: bool = False) -> int | None:
        """Store a world fixation (>= 500 ms, or any length when `force` = the user just spoke about it)."""
        if not self.armed:
            self.stats["unarmed"] += 1
            return None
        if f.scene_x is None or (f.duration < MIN_FIX_S and not force):
            return None
        dwell = round(f.duration * 1000)
        for _, _, rid, fs in self.recent:
            if abs(fs - f.start) < 0.05:  # same fixation again (it grew): keep the longest dwell
                with self.lock:
                    self.db.execute("update seen set dwell_ms = max(dwell_ms, ?) where id = ?", (dwell, rid))
                    self.db.commit()
                return rid
        now = time.time()
        self.saved_at = [t for t in self.saved_at if now - t < 60]
        if len(self.saved_at) >= MAX_PER_MIN and not force:
            self.stats["rate"] += 1
            return None
        fr = self.scene.nearest((f.start + f.end) / 2)
        if fr is None:
            return None
        full = decode(fr[1])
        crop = crop_around(full, f.scene_x, f.scene_y)
        h = dhash(crop)
        self.recent = [r for r in self.recent if now - r[0] < DEDUPE_S]
        for _, oh, rid, _ in self.recent:
            if hamming(h, oh) <= 6:  # same thing again: extend dwell instead of a new row
                with self.lock:
                    self.db.execute("update seen set dwell_ms = dwell_ms + ? where id = ?", (dwell, rid))
                    self.db.commit()
                self.stats["dupe"] += 1
                return rid
        t0 = time.time()
        a = ocr.analyze(crop)
        text, face = a["text"], a["face"]
        if face > FACE_MAX:
            self.stats["face"] += 1
            log.info("skipped a crop with a face (%.0f%%)", face * 100)
            return None
        found = a["codes"] or ocr.codes(crop_around(full, f.scene_x, f.scene_y, size=800))
        codes, labels = " ".join(found), ", ".join(a["labels"])
        path = DATA / "crops" / f"{int(now * 1000)}.jpg"
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with self.lock:
            cur = self.db.execute("insert into seen(t, dwell_ms, scene_x, scene_y, crop, hash, caption, codes, labels, conf) "
                                  "values(?,?,?,?,?,?,?,?,?,?)",
                                  (f.start, dwell, f.scene_x, f.scene_y, str(path), str(h), text, codes, labels, a["conf"]))
            rid = cur.lastrowid
            searchable = " ".join(x for x in (text, labels, codes) if x)
            if searchable:
                self.db.execute("insert into seen_fts(rowid, caption) values(?, ?)", (rid, searchable))
            self.db.commit()
        self.recent.append((now, h, rid, f.start))
        self.saved_at.append(now)
        self.stats["saved"] += 1
        log.info("saved %d in %d ms: %s", rid, (time.time() - t0) * 1000, text[:80] or "(no text)")
        return rid

    # ---- housekeeping ------------------------------------------------------------------
    def prune(self):
        cutoff = time.time() - KEEP_S
        with self.lock:
            old = self.db.execute("select id, crop from seen where t < ?", (cutoff,)).fetchall()
            for rid, crop in old:
                Path(crop).unlink(missing_ok=True)
                self.db.execute("delete from seen where id = ?", (rid,))
            self.db.execute("insert into seen_fts(seen_fts) values('rebuild')")
            self.db.commit()
        if old:
            log.info("pruned %d crops older than 24 h", len(old))

    def forget(self) -> int:
        with self.lock:
            n = self.db.execute("select count(*) from seen").fetchone()[0]
            self.db.execute("delete from seen")
            self.db.execute("insert into seen_fts(seen_fts) values('rebuild')")
            self.db.commit()
        shutil.rmtree(DATA / "crops", ignore_errors=True)
        (DATA / "crops").mkdir(parents=True, exist_ok=True)
        self.recent.clear()
        return n

    # ---- read --------------------------------------------------------------------------
    def get(self, rid: int, wait_s: float = 0.0) -> dict | None:
        with self.lock:
            row = self.db.execute("select id, t, dwell_ms, crop, caption, codes, labels, conf from seen where id = ?", (rid,)).fetchone()
        return dict(zip(("id", "t", "dwell_ms", "crop", "caption", "codes", "labels", "conf"), row)) if row else None

    def recall(self, query: str = "", since: float | None = None, limit: int = 5) -> list[dict]:
        since = since or 0
        with self.lock:
            if query.strip():
                q = " OR ".join(json.dumps(w) for w in query.split() if len(w) > 2) or json.dumps(query)
                rows = self.db.execute(
                    "select s.id, s.t, s.dwell_ms, s.crop, s.caption, bm25(seen_fts), s.codes, s.labels from seen_fts "
                    "join seen s on s.id = seen_fts.rowid where seen_fts match ? and s.t >= ? limit 50", (q, since)).fetchall()
                if not rows:  # text-less things (a bottle): fall back to recent, the model can look
                    rows = self.db.execute("select id, t, dwell_ms, crop, caption, 0, codes, labels from seen where t >= ? "
                                           "order by t desc limit 20", (since,)).fetchall()
            else:
                rows = self.db.execute("select id, t, dwell_ms, crop, caption, 0, codes, labels from seen where t >= ? "
                                       "order by t desc limit 50", (since,)).fetchall()
        now = time.time()
        # bm25 is negative-better; dwell and recency push "the one I looked at longest" up
        scored = sorted(rows, key=lambda r: r[5] - r[2] / 2000 + (now - r[1]) / 3600)
        return [dict(zip(("id", "t", "dwell_ms", "crop", "caption", "codes", "labels"), r[:5] + r[6:8])) for r in scored[:limit]]
