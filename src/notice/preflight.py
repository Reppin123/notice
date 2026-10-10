"""notice-preflight: one command that checks every link before a run. Green = go."""
from __future__ import annotations

import json
import re
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

OK, WARN, BAD = "\033[32m✔\033[0m", "\033[33m!\033[0m", "\033[31m✘\033[0m"
AGENT_LOG_DIR = Path.home() / "Library/Logs/Apprentice"
results: list[str] = []


def check(cond: bool | None, label: str, fix: str = "", warn: bool = False):
    mark = OK if cond else (WARN if warn else BAD)
    results.append(mark)
    print(f" {mark} {label}" + (f"\n     → {fix}" if not cond and fix else ""))


def port_open(host, port) -> bool:
    try:
        socket.create_connection((host, port), 0.3).close()
        return True
    except OSError:
        return False


def main():
    want_screen = "--screen" in sys.argv
    print("Network")
    ssid = subprocess.run(["networksetup", "-getairportnetwork", "en0"], capture_output=True, text=True).stdout.strip()
    print(f"   {ssid}")

    print("Notice")
    try:
        h = json.load(urllib.request.urlopen("http://127.0.0.1:8790/health", timeout=2))
    except Exception:
        check(False, "Notice running on :8790", "uv run notice --gaze neon [--neon-ip <phone ip>]")
        h = None
    if h:
        check(True, f"Notice running (source: {h['source']})")
        neon = h.get("neon")
        if h["source"] == "neon":
            check(neon and neon.get("connected"), f"Neon connected ({(neon or {}).get('device')})",
                  f"same network as the Neon phone? last error: {(neon or {}).get('error')}")
            rt = (neon or {}).get("roundtrip_ms")
            check(rt is not None and rt < 60, f"clock sync: offset {(neon or {}).get('offset_ms')} ms, roundtrip {rt} ms",
                  "roundtrip high: network is slow; use the hotspot", warn=True)
            check((neon or {}).get("reconnects", 0) == 0, f"stream reconnects: {(neon or {}).get('reconnects')}",
                  "the stream dropped at least once; watch the network", warn=True)
        age = h.get("gaze_age_s")
        check(age is not None and age < 1, f"gaze fresh ({age} s old, {h['gaze_hz']} Hz)", "glasses not streaming / not worn")
        if h["source"] == "neon":
            check(h["gaze_hz"] >= 150, "gaze at 200 Hz", "Companion app → settings → gaze data rate 200 Hz", warn=True)
        check(h["fixations_10s"] > 0, f"fixations arriving ({h['fixations_10s']} in 10 s)",
              "Companion app: enable fixation detection (needs 200 Hz)")
        fa = h.get("frame_age_s")
        check(fa is not None and fa < 1, f"scene frames fresh ({fa} s old, {h.get('frame_size')})", "scene video not streaming")
        if want_screen and neon:
            ta = neon.get("screen_tags_age_s")
            check(ta is not None and ta < 2, f"screen AprilTags seen ({ta} s ago)",
                  "tags visible on screen? sit closer / raise brightness", warn=True)
        m = h.get("memory") or {}
        print(f"   memory armed: {m.get('armed')} (arms on a call) · saved {m.get('saved')} · faces skipped {m.get('face')}"
              f" · recording sessions: {h.get('recording')}")

    print("Apprentice (Mac agent)")
    check(port_open("127.0.0.1", 8765), "agent listening on :8765",
          "launchctl kickstart -k gui/$(id -u)/com.akshitbansal.apprentice-dev-agent")
    logs = sorted(AGENT_LOG_DIR.glob("agent-20*.log"))
    if logs:
        lines = logs[-1].read_text(errors="ignore").splitlines()
        start = max((i for i, l in enumerate(lines) if "Apprentice agent listening" in l), default=0)
        tail = lines[start:]
        check(any("Swift client connected" in l for l in tail), "Mac app connected since the last agent start",
              "open the Apprentice Mac app (calls go to the cloud box when it's closed)")
        check(any("glasses: tools registered" in l for l in tail), "glasses tools loaded in the agent",
              "start Notice BEFORE the agent, then restart the agent", warn=True)
        lat = [l for l in tail if "glasses: window" in l][-1:]
        if lat:
            print("   last glasses turn:", re.sub(r"^.*glasses: ", "", lat[0]))
    bad = results.count(BAD)
    print(f"\n{'READY' if not bad else f'{bad} problem(s) to fix'} · {results.count(WARN)} warning(s)")
    sys.exit(1 if bad else 0)
