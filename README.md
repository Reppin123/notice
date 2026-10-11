# Notice: what your eyes are on, for your AI

Built at INTERFACE (Oct 10-11, 2026), Vision track, on the Pupil Labs Neon eye tracker.

You are on a phone call with your AI assistant. You look at something in the real world
(a poster, a snack label, two products on a shelf) and ask "what does that say?" or "which
one is cheaper?". Notice knows what your eyes were on *while you spoke*, freezes those
frames, and hands them to the assistant, which answers by voice.

Notice runs on a Mac next to the Neon companion phone. The assistant is Apprentice, my
existing Mac agent with an iPhone call app (not in this repo; see Credits).

## How it works

1. **Gaze + scene in sync.** Neon's gaze and scene camera stream over the Pupil Labs realtime
   API. Clocks are aligned with the Time Echo protocol, so a spoken word can be matched to
   where the eyes were at that moment.
2. **What you looked at while speaking.** For each utterance, fixations are grouped into the
   things you looked at, ranked by dwell. One object seen from several angles is merged into
   one thing (colour histogram + OCR text agreement).
3. **A short note first.** On-device Apple Vision OCR reads each thing. If the text already
   answers the question, the assistant gets just the note (no image, fast). Shaky OCR is
   marked unsure so the model trusts the picture over it.
4. **One picture when needed.** The assistant's `look_through_glasses` tool calls
   `/snapshot`, which returns exactly one image, frozen at the moment you spoke:
   - `whole`: the full scene at 1280 px with a red circle where your eyes were (default)
   - `sheet`: a numbered 2x2 sheet when you looked at several things ("which one")
   - `motion`: 4 frames across your sentence, for "what just happened"
   - `zoom`: native resolution. The model can pick the area itself; otherwise Notice snaps
     to the text block or object nearest your gaze, so gaze that lands a little off still
     zooms to the right thing.

## Privacy

- Capture only happens while a call is active.
- Faces are blurred in every image that leaves Notice and in every stored crop.
- Stored crops are deleted after 24 hours; `/forget` clears them now.
- OCR, face detection and zoom snapping run on the Mac (Apple Vision).

## Run

```
uv sync
uv run notice-preflight                             # finds Neon, checks gaze, scene, clock sync
uv run notice --gaze neon --neon-ip <phone ip> --no-overlay
uv run notice --gaze mouse-world --scene testdata/poster_wall.jpg --no-mic   # no glasses
uv run pytest
```

Speech to text uses AssemblyAI (`ASSEMBLYAI_API_KEY`); `--no-mic` drives it with `POST /say`.

API on http://127.0.0.1:8790: `/window` (what the eyes were on during an utterance),
`/snapshot` (the one image), `/frame`, `/recall`, `/capture`, `/forget`, `/health`.
There is also a screen mode (AprilTags + macOS accessibility, `/focus`, `/resolve`) from the
first hours of the event; the demo is the real-world mode.

## What is real and what is not

- Tested live on Neon over real iPhone calls with Apprentice: first spoken words in
  about 1.6-3.3 s, full answer in about 6-7 s.
- At reading distance Neon's gaze tends to sit slightly above held objects; the whole view
  and snapped zoom are there to absorb that. No per-user offset correction yet.
- Body text on paper is often too small for the scene camera; headlines and labels read well.
- Smart zoom was tuned on a test poster wall and real Neon crops, not on a large dataset.

## Credits

- Pupil Labs `pupil-labs-realtime-api` (gaze, scene video, Time Echo) and
  `real-time-screen-gaze` (AprilTag screen mapping).
- Apple Vision via PyObjC (OCR, faces, saliency), OpenCV, AssemblyAI Universal-Streaming,
  ElevenLabs (optional in-ear voice).
- Apprentice, my Mac agent and iPhone app, existed before the event. Notice talks to it over
  its local WebSocket; the call, the voice and the model loop are Apprentice. The glasses
  tool on the Apprentice side (`look_through_glasses`) was added during the event.
- All code in this repo was written during INTERFACE (Oct 10-11, 2026).
