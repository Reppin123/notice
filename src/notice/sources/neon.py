"""Pupil Labs Neon: gaze + on-device fixations/blinks, mapped to screen pixels via AprilTags.

Clock: Neon timestamps are phone unix time; mac = neon + offset (re-estimated every 60 s).
Screen: real-time-screen-gaze locates the 4 on-screen AprilTags in each scene frame and maps
scene-camera gaze to the screen surface (head movement is fine).
"""
from __future__ import annotations

import asyncio
import logging
import time

from pupil_labs.realtime_api import Device
from pupil_labs.realtime_api.discovery import discover_devices
from pupil_labs.realtime_api.time_echo import TimeOffsetEstimator
from pupil_labs.realtime_api.streaming.eye_events import BlinkEventData, FixationEventData, receive_eye_events_data
from pupil_labs.realtime_api.streaming.gaze import receive_gaze_data
from pupil_labs.realtime_api.streaming.video import receive_video_frames
from pupil_labs.real_time_screen_gaze.gaze_mapper import GazeMapper

from ..core import AttentionState, Fixation, GazePoint
from . import markers

log = logging.getLogger("notice.neon")


class NeonGaze:
    def __init__(self, state: AttentionState, scene, ip: str | None = None, emit=None, scene_hz: float = 10,
                 map_screen: bool = True):
        self.state, self.scene, self.ip, self.emit = state, scene, ip, emit or (lambda *a, **k: None)
        self.offset_s = 0.0
        self.scene_period = 1.0 / scene_hz
        self.map_screen = map_screen  # AprilTag search on every frame: only when the screen tags are shown
        self._newest = None  # newest undecoded scene frame; the worker only ever processes this one
        self._new_frame = asyncio.Event()
        self.mapper: GazeMapper | None = None
        self.surface = None
        self.screen_seen_at = 0.0  # last time the tags were located
        self.stats = {"gaze": 0, "fixations": 0, "blinks": 0, "frames": 0, "reconnects": 0}
        self.status = {"connected": False, "device": None, "offset_ms": None, "roundtrip_ms": None, "error": None}
        self.last_gaze_at = 0.0

    async def _device(self) -> Device:
        if self.ip:
            return Device(self.ip, 8080)
        async with asyncio.timeout(10):
            async for info in discover_devices():
                return Device.from_discovered_device(info)
        raise RuntimeError("no Neon found (mDNS); pass --neon-ip")

    async def _offset_loop(self, device: Device):
        status = await device.get_status()
        port = status.phone.time_echo_port
        if port is None:
            self.emit("neon_clock", error="Companion app too old for Time Echo; offset stays 0")
            return
        estimator = TimeOffsetEstimator(status.phone.ip, port)
        while True:
            est = await estimator.estimate(number_of_measurements=50)
            if est:
                self.offset_s = est.time_offset_ms.mean / 1000
                self.status.update(offset_ms=round(est.time_offset_ms.mean, 1),
                                   roundtrip_ms=round(est.roundtrip_duration_ms.mean, 1))
                self.emit("neon_clock", offset_ms=round(est.time_offset_ms.mean, 1),
                          roundtrip_ms=round(est.roundtrip_duration_ms.mean, 1))
            await asyncio.sleep(60)

    def to_screen(self, sx: float, sy: float) -> tuple[float | None, float | None]:
        """Scene-camera px -> screen points, or (None, None) if the screen isn't in view / gaze off it."""
        if self.mapper is None or time.time() - self.screen_seen_at > 0.5:
            return None, None
        res = self.mapper.process_gaze((sx, sy))
        if not res:
            return None, None
        hits = res.mapped_gaze.get(self.surface.uid) or []
        if not hits or not hits[0].is_on_aoi:
            return None, None
        w, h = markers.screen_size()
        return hits[0].x * w, (1 - hits[0].y) * h

    async def run(self):
        """Watchdog: (re)connect forever; a stream that errors or goes quiet for 3 s restarts everything."""
        backoff = 1.0
        while True:
            try:
                await self._run_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.status.update(connected=False, error=f"{type(e).__name__}: {e}"[:200])
                self.emit("neon_down", error=self.status["error"], retry_s=backoff)
                log.warning("Neon down (%s); retry in %.0f s", e, backoff)
            self.stats["reconnects"] += 1
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 8.0)

    async def _stall_watch(self):
        await asyncio.sleep(5)  # streams take a moment to start
        while True:
            await asyncio.sleep(1)
            if time.time() - self.last_gaze_at > 3:
                raise RuntimeError("no gaze for 3 s")

    async def _run_once(self):
        device = await self._device()
        async with device:
            status = await device.get_status()
            calib = await device.get_calibration()
            self.mapper = GazeMapper(calib)
            verts, size = markers.marker_verts()
            self.surface = self.mapper.add_surface(verts, size)
            self.status.update(connected=True, device=status.phone.device_name, error=None)
            self.emit("neon_connected", name=status.phone.device_name, ip=status.phone.ip,
                      battery=status.phone.battery_level)
            tasks = [asyncio.create_task(c) for c in (
                self._offset_loop(device),
                self._scene(status.direct_world_sensor().url),
                self._scene_worker(),
                self._gaze(status.direct_gaze_sensor().url),
                self._events(status.direct_eye_events_sensor().url),
                self._stall_watch())]
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    t.result()  # re-raise the first failure
                raise RuntimeError("a Neon stream ended")
            finally:
                for t in tasks:
                    t.cancel()

    async def _scene(self, url):
        """Reader: keep only the newest frame. Never await processing here, or unread video piles up
        and every crop shows the past (on real hardware frames fell 3.7 s behind)."""
        async for frame in receive_video_frames(url, run_loop=True):
            self._newest = frame
            self._new_frame.set()

    async def _scene_worker(self):
        """Process the newest frame at most scene_hz times a second; older frames are dropped."""
        while True:
            await self._new_frame.wait()
            self._new_frame.clear()
            frame, started = self._newest, time.time()
            t = frame.timestamp_unix_seconds + self.offset_s
            await asyncio.to_thread(self._process, t, frame)
            if self.map_screen and self.mapper._surface_locations.get(self.surface.uid) is not None:
                self.screen_seen_at = time.time()
            self.stats["frames"] += 1
            await asyncio.sleep(max(0.0, self.scene_period - (time.time() - started)))

    def _process(self, t, frame):
        bgr = frame.bgr_buffer()
        self.scene.add(t, bgr)
        if self.map_screen:
            self.mapper.process_scene(bgr)

    async def _gaze(self, url):
        async for g in receive_gaze_data(url, run_loop=True):
            t = g.timestamp_unix_seconds + self.offset_s
            self.last_gaze_at = time.time()
            x, y = self.to_screen(g.x, g.y)
            self.state.add_gaze(GazePoint(t, x, y, g.worn, g.x, g.y))
            self.stats["gaze"] += 1

    async def _events(self, url):
        async for e in receive_eye_events_data(url, run_loop=True):
            if isinstance(e, FixationEventData) and e.event_type == 1:
                x, y = self.to_screen(e.mean_gaze_x, e.mean_gaze_y)
                self.state.add_fixation(Fixation(
                    e.start_time_ns / 1e9 + self.offset_s, e.end_time_ns / 1e9 + self.offset_s,
                    x, y, e.mean_gaze_x, e.mean_gaze_y, source="neon"))
                self.stats["fixations"] += 1
            elif isinstance(e, BlinkEventData):
                self.stats["blinks"] += 1
