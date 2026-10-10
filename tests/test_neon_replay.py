import os
import time
from pathlib import Path

import pytest

SAMPLE = Path(os.environ.get("NEON_SAMPLE", Path.home() / "interface-hack/neon-sample/player-sample"))
pytestmark = pytest.mark.skipif(not (SAMPLE / "info.json").exists(), reason=f"no Neon sample at {SAMPLE}")


def test_replay_feeds_gaze_fixations_and_frames():
    from notice.core import AttentionState
    from notice.scene import SceneBuffer
    from notice.sources.neon_replay import NeonReplay

    state, scene = AttentionState(), SceneBuffer()
    r = NeonReplay(state, scene, str(SAMPLE), speed=4.0, loop=False).start()
    t0 = time.time()
    while time.time() - t0 < 30 and (r.stats["fixations"] < 3 or r.stats["frames"] < 3):
        time.sleep(0.2)
    r.stop()

    g = state.last_gaze()
    assert g is not None and g.x is None and g.sx is not None
    assert abs(g.t - time.time()) < 2  # re-timed to now
    f = state.latest_fixation()
    assert f is not None and f.source == "neon" and f.end > f.start and f.scene_x is not None
    assert len(scene.frames) >= 3
