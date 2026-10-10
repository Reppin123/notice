"""Notice daemon: gaze source + voice + fusion + Attention API (HTTP) + Apprentice hands.

    uv run notice --gaze mouse            # mouse as fake gaze
    uv run notice --gaze replay:file.jsonl
    uv run notice --gaze neon [--neon-ip 192.168.1.20]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections import deque
from pathlib import Path

from aiohttp import web

from . import fusion, screen
from .core import AttentionState

log = logging.getLogger("notice")
PORT = 8790
ADDRESS_MIN = 0.5  # word counts as said to the agent if >= half the gaze around it is on screen


def fmt_target(r: dict) -> str:
    f = r["fixation"]
    if not f:
        return f"'{r['word']}': no fixation found around the word (ask the user what they meant)."
    if f["x"] is None:
        return f"'{r['word']}': user was looking away from the screen (world object; see Notice world memory)."
    lines = [f"'{r['word']}': user looked at screen point ({round(f['x'])},{round(f['y'])}) for {f['duration_ms']} ms. "
             "Nearby elements, nearest first (gaze is accurate to ~60 px, use the words to choose):"]
    for i, c in enumerate(r["candidates"], 1):
        lines.append(f"  {i}. {c['role']} \"{c['label'][:120]}\" in {c['app']} at center {tuple(c['center'])}, {c['distance_px']} px away")
    if not r["candidates"]:
        lines.append("  (no accessibility elements there; use read_screen near that point)")
    return "\n".join(lines)


class Notice:
    def __init__(self, state: AttentionState, apprentice=None, dry_run=False, scene=None, memory=None):
        self.state = state
        self.scene, self.memory = scene, memory
        self.apprentice = apprentice
        self.dry_run = dry_run
        self.events: deque = deque(maxlen=200)
        self.listeners: set[asyncio.Queue] = set()
        self.last_resolution: list[dict] = []
        self.memory_on = False
        self.neon = None
        self.source = ""

    def emit(self, kind: str, **data):
        ev = {"type": kind, "t": time.time(), **data}
        self.events.append(ev)
        for q in list(self.listeners):
            q.put_nowait(ev)
        log.info("%s %s", kind, json.dumps(data, default=str)[:300])

    # ---- voice -> fusion -> hands -------------------------------------------------
    def addressed(self, w: fusion.Word) -> bool:
        frac = self.state.on_screen_between(w.start - 0.3, w.end)
        return frac is None or frac >= ADDRESS_MIN

    def on_partial(self, words):
        # instant feedback: the bubble can start flying as soon as "this" appears
        for r in fusion.resolve(self.state, words, snap=None):
            if r["fixation"] and r["fixation"]["x"] is not None:
                self.emit("pointing", word=r["word"], x=r["fixation"]["x"], y=r["fixation"]["y"])
        self.emit("partial", text=" ".join(w.text for w in words))

    async def handle_turn(self, words: list[fusion.Word], text: str):
        kept = [w for w in words if self.addressed(w)]
        if not kept:
            self.emit("ignored", text=text, reason="eyes off screen (talking to someone else)")
            return
        if len(kept) < len(words):
            text = " ".join(w.text for w in kept)
            self.emit("trimmed", text=text, dropped=len(words) - len(kept))
        res = await asyncio.to_thread(fusion.resolve, self.state, kept, screen.candidates_near)
        self.last_resolution = res
        for r in res:
            c = r["candidates"][0] if r["candidates"] else None
            self.emit("resolved", word=r["word"], fixation=r["fixation"], top=c)
        context = ""
        if res:
            context = ("Notice (eye tracking): the user referred to things by looking at them.\n"
                       + "\n".join(fmt_target(r) for r in res))
        app = next((r["candidates"][0]["app"] for r in res if r["candidates"]), "")
        self.emit("ask", text=text, context=context)
        if self.apprentice and not self.dry_run:
            await self.apprentice.ask(text, context, app)

    def on_apprentice(self, m: dict):
        t = m.get("type")
        if t in ("say", "agent_text"):
            self.emit("agent_say", text=m.get("text") or m.get("content"))
        elif t == "point_at_element":
            self.emit("agent_point", **{k: m.get(k) for k in ("x", "y", "space", "width", "height", "instruction")})
        elif t in ("tool_started", "agent_turn_complete", "set_state"):
            self.emit("agent_" + t, name=m.get("name") or m.get("tool") or m.get("state"))

    # ---- Attention API -----------------------------------------------------------
    def focus(self) -> dict:
        f = self.state.latest_fixation()
        g = self.state.last_gaze()
        out = {"gaze": g.__dict__ if g else None, "fixation": f.to_dict() if f else None, "candidates": []}
        if f and f.on_screen:
            out["candidates"] = screen.candidates_near(f.x, f.y)
        return out

    def addressing(self) -> dict:
        frac = self.state.on_screen_fraction()
        return {"addressing": frac >= ADDRESS_MIN, "on_screen_fraction": round(frac, 2),
                "reason": "eyes on screen" if frac >= ADDRESS_MIN else "eyes off screen"}

    def resolve_at(self, t: float, word: str) -> dict:
        r = fusion.resolve(self.state, [fusion.Word(word, t, t + 0.2)] if word.lower() in fusion.DEICTICS
                           else [fusion.Word("this", t, t + 0.2)], screen.candidates_near)
        return r[0] if r else {}


def make_app(n: Notice) -> web.Application:
    app = web.Application()
    j = lambda d: web.json_response(d, dumps=lambda o: json.dumps(o, default=str))

    async def focus(_):
        return j(await asyncio.to_thread(n.focus))

    async def resolve(req):
        t = float(req.query.get("t", time.time()))
        return j(await asyncio.to_thread(n.resolve_at, t, req.query.get("word", "this")))

    async def addressing(_):
        return j(n.addressing())

    async def events(_):
        return j(list(n.events))

    async def say(req):  # inject a typed utterance with fake word times (testing without a mic)
        body = await req.json()
        now = time.time()
        toks = body["text"].split()
        span = body.get("duration", 0.3 * len(toks))
        words = [fusion.Word(tk, now - span + i * span / len(toks), now - span + (i + 1) * span / len(toks)) for i, tk in enumerate(toks)]
        await n.handle_turn(words, body["text"])
        return j({"ok": True, "resolution": n.last_resolution})

    async def stream(req):
        ws = web.WebSocketResponse()
        await ws.prepare(req)
        q: asyncio.Queue = asyncio.Queue()
        n.listeners.add(q)
        try:
            while True:
                await ws.send_str(json.dumps(await q.get(), default=str))
        finally:
            n.listeners.discard(q)
        return ws

    async def overlay(_):
        return web.FileResponse(Path(__file__).parent / "overlay.html")

    async def gaze(_):
        g = n.state.last_gaze()
        a = n.addressing()
        return j({"x": g.x if g else None, "y": g.y if g else None, "addressing": a["addressing"],
                  "memory": bool(n.memory and n.memory.armed)})

    async def layout(_):
        from .sources import markers
        return j({"origins": markers.tag_origins(), "size": markers.SIZE})

    async def tag(req):
        from .sources import markers
        i = int(req.match_info["i"])
        p = Path(f"/tmp/notice-tag{i}-{markers.SIZE}.png")
        if not p.exists():
            markers.marker_png(i, str(p), side=markers.SIZE * 2)
        return web.FileResponse(p)

    async def window_(req):
        """Glasses context for one utterance. start/end = Mac epoch seconds."""
        from . import window
        end = float(req.query.get("end", time.time()))
        start = float(req.query.get("start", end - 3.0))
        if req.query.get("arm") == "1":
            n.memory.arm(True)
        out = await asyncio.to_thread(window.build, n.state, n.memory, start, end, req.query.get("words", ""))
        n.emit("window", note=out["note"], items=out["items"], build_ms=out["build_ms"])
        return j(out)

    async def frame(req):
        """Latest scene frame with the gaze circle (what the wearer sees right now), JPEG."""
        from .scene import decode, mark_gaze
        import cv2
        fr = n.scene.latest() if n.scene else None
        if fr is None or time.time() - fr[0] > 1.5:
            return web.Response(status=404, text="no fresh scene frame (glasses not streaming)")
        g = next((g for g in reversed(n.state.gaze) if g.sx is not None), None)
        img = decode(fr[1])
        if g:
            img = mark_gaze(img, g.sx, g.sy, max_w=int(req.query.get("max_w", 1024)))
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return web.Response(body=jpg.tobytes(), content_type="image/jpeg",
                            headers={"X-Frame-Age-Ms": str(round((time.time() - fr[0]) * 1000))})

    async def health(_):
        """Everything the preflight checks: is each link alive and fresh?"""
        now = time.time()
        with n.state.lock:
            g2 = [g for g in n.state.gaze if g.t >= now - 2]
            fx10 = [f for f in n.state.fixations if f.end >= now - 10]
        fr = n.scene.latest() if n.scene else None
        last = n.state.last_gaze()
        out = {"source": n.source,
               "gaze_hz": round(len(g2) / 2, 1),
               "gaze_age_s": round(now - last.t, 2) if last else None,
               "on_screen_fraction": round(n.state.on_screen_fraction(1.0), 2),
               "fixations_10s": len(fx10),
               "frame_age_s": round(now - fr[0], 2) if fr else None,
               "frame_size": list(fr[2]) if fr else None,
               "memory": {"armed": n.memory.armed, **n.memory.stats} if n.memory else None,
               "recording": bool(getattr(n, "recorder", None) and n.recorder.active)}
        if n.neon:
            out["neon"] = {**n.neon.status, **n.neon.stats,
                           "screen_tags_age_s": round(now - n.neon.screen_seen_at, 1) if n.neon.screen_seen_at else None}
        return j(out)

    async def capture(req):
        """Apprentice arms capture while the user is on a call (on=1, renewed each turn) and disarms at hang-up."""
        n.memory.arm(req.query.get("on") == "1")
        n.memory_on = n.memory.armed
        n.emit("capture", on=n.memory.armed)
        return j({"armed": n.memory.armed, "stats": n.memory.stats})

    async def forget(_):
        k = await asyncio.to_thread(n.memory.forget)
        n.emit("forget", removed=k)
        return j({"removed": k})

    async def recall(req):
        if not n.memory:
            return j([])
        since = req.query.get("since")
        return j(await asyncio.to_thread(n.memory.recall, req.query.get("q", ""), float(since) if since else None))

    async def crop(req):  # only files inside Notice's own crops folder
        from .memory import DATA
        p = Path(req.query.get("path", "")).resolve()
        if (DATA / "crops").resolve() not in p.parents or not p.exists():
            return web.Response(status=404)
        return web.FileResponse(p)

    app.router.add_get("/crop", crop)
    app.router.add_get("/health", health)
    app.router.add_post("/capture", capture)
    app.router.add_get("/capture", capture)
    app.router.add_post("/forget", forget)
    app.router.add_get("/window", window_)
    app.router.add_get("/frame", frame)
    app.router.add_get("/recall", recall)
    app.router.add_get("/", overlay)
    app.router.add_get("/gaze", gaze)
    app.router.add_get("/layout", layout)
    app.router.add_get("/tag/{i}.png", tag)
    app.router.add_get("/focus", focus)
    app.router.add_get("/resolve", resolve)
    app.router.add_get("/addressing", addressing)
    app.router.add_get("/events", events)
    app.router.add_post("/say", say)

    async def speak(req):  # proactive in-ear cue via Notice's own voice (ElevenLabs)
        text = (await req.json()).get("text", "")
        n.emit("notice_say", text=text)
        asyncio.create_task(n.voice.say(text))
        return web.json_response({"ok": True, "backend": n.voice.backend})

    app.router.add_post("/speak", speak)
    app.router.add_get("/stream", stream)
    return app


async def memory_writer(state: AttentionState, memory):
    """While armed, store world fixations that reach 500 ms (crop + on-device OCR). Prune hourly."""
    done: set[int] = set()
    last_prune = time.time()
    while True:
        await asyncio.sleep(0.2)
        if time.time() - last_prune > 3600:
            last_prune = time.time()
            await asyncio.to_thread(memory.prune)
        if not memory.armed:
            continue
        with state.lock:
            recent = [f for f in list(state.fixations)[-10:] if id(f) not in done and f.scene_x is not None
                      and f.x is None and f.duration >= 0.5]
        for f in recent:
            done.add(id(f))
            await asyncio.to_thread(memory.on_fixation, f)


async def amain(args):
    from .apprentice import Apprentice

    from .memory import AttentionMemory
    from .scene import SceneBuffer, WebcamScene

    state = AttentionState()
    scene = SceneBuffer()
    memory = AttentionMemory(scene, enabled=not args.no_memory, always_armed=args.capture_always)
    n = Notice(state, dry_run=args.dry_run, scene=scene, memory=memory)
    n.memory_on = memory.enabled
    src = args.gaze
    n.source = src
    if src in ("mouse", "mouse-world"):
        from .sources.mouse import MouseGaze
        from .sources.replay import GazeRecorder
        world = None
        if src == "mouse-world":  # webcam (or --scene file) = scene camera, mouse = gaze into it
            from .scene import FileScene
            cam = FileScene(scene, args.scene).start() if args.scene else WebcamScene(scene).start()
            while cam.size is None:
                await asyncio.sleep(0.1)
            world = cam.size
        MouseGaze(state, recorder=GazeRecorder(args.record) if args.record else None, world_size=world).start()
    elif src.startswith("neon-replay:"):  # a real Neon recording (Native Recording Data), time-shifted to now
        from .sources.neon_replay import NeonReplay
        n.neon_replay = NeonReplay(state, scene, os.path.expanduser(src.split(":", 1)[1])).start()
    elif src.startswith("session:"):
        from .recorder import SessionReplay
        SessionReplay(state, scene, src.split(":", 1)[1]).start()
    elif src.startswith("replay:"):
        from .sources.replay import ReplayGaze
        ReplayGaze(state, src.split(":", 1)[1], loop=True).start()
    elif src == "neon":
        from .sources.neon import NeonGaze
        n.neon = NeonGaze(state, scene, ip=args.neon_ip, emit=n.emit)
        asyncio.create_task(n.neon.run())
    asyncio.create_task(memory_writer(state, memory))
    if args.record_sessions:
        from .recorder import Recorder
        n.recorder = Recorder(state, scene, memory)
        asyncio.create_task(n.recorder.run())
    from .voice import Voice
    n.voice = Voice(args.voice)

    if not args.dry_run:
        n.apprentice = Apprentice(on_event=n.on_apprentice)
        try:
            await n.apprentice.connect()
        except Exception as e:
            log.warning("Apprentice not reachable (%s); will retry on first turn", e)

    runner = web.AppRunner(make_app(n))
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", PORT).start()
    log.info("Attention API on http://127.0.0.1:%d", PORT)

    if not args.no_mic:
        from .stt import Transcriber
        loop = asyncio.get_running_loop()
        tr = Transcriber(on_turn=lambda w, t: loop.create_task(n.handle_turn(w, t)), on_partial=n.on_partial)
        while True:
            try:
                await tr.run()
            except Exception as e:
                log.warning("STT dropped (%s), reconnecting", e)
                tr.t0 = None
                await asyncio.sleep(1)
    else:
        await asyncio.Event().wait()


def main():
    ap = argparse.ArgumentParser(prog="notice")
    ap.add_argument("--gaze", default="mouse", help="mouse | mouse-world | neon | session:<dir> | replay:<file.jsonl>")
    ap.add_argument("--neon-ip")
    ap.add_argument("--record", help="append raw gaze to this JSONL (mouse mode)")
    ap.add_argument("--no-mic", action="store_true", help="no STT; drive with POST /say")
    ap.add_argument("--dry-run", action="store_true", help="resolve but don't send to Apprentice")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--scene", help="mouse-world: image/video file as the scene camera instead of the webcam")
    ap.add_argument("--record-sessions", action="store_true",
                    help="while armed (on a call) also save raw gaze + 2 fps frames for replay (may include people)")
    ap.add_argument("--capture-always", action="store_true", help="dev: save crops without a call arming it")
    ap.add_argument("--no-memory", action="store_true", help="don't store fixated crops")
    ap.add_argument("--tags", action="store_true", help="show AprilTags even without Neon")
    ap.add_argument("--debug", action="store_true", help="show the gaze dot")
    ap.add_argument("--voice", default="auto", help="Notice's in-ear voice: auto (ElevenLabs if a key exists, else macOS say) | eleven | say | none")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    if args.no_overlay:
        asyncio.run(amain(args))
        return
    import threading
    from .overlay import run_overlay
    threading.Thread(target=lambda: asyncio.run(amain(args)), daemon=True, name="notice-async").start()
    import socket
    for _ in range(100):  # wait for the API so the page loads first time
        try:
            socket.create_connection(("127.0.0.1", PORT), 0.1).close()
            break
        except OSError:
            time.sleep(0.05)
    q = "?tags=" + ("1" if args.gaze == "neon" or args.tags else "0") + ("&debug=1" if args.debug else "")
    run_overlay(f"http://127.0.0.1:{PORT}/{q}")
