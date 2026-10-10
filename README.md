# Notice: attention as an API

Voice tells an agent what to do; your eyes tell it what you mean. Notice fuses gaze
(Pupil Labs Neon), voice (AssemblyAI word timings) and the screen (macOS accessibility)
so "reply to **this**" or "move **this** **there**" resolves to real targets, then hands
the command to Apprentice to act.

```
uv sync
uv run notice --gaze mouse --debug          # mouse as fake gaze (needs ASSEMBLYAI_API_KEY)
uv run notice --gaze mouse --no-mic --dry-run --tags --debug
curl -XPOST localhost:8790/say -d '{"text":"reply to this"}'   # typed utterance, no mic
uv run notice --gaze neon --neon-ip <phone ip>
uv run notice-mcp                           # Attention MCP server (stdio), proxies the daemon
```

Attention API (http://127.0.0.1:8790): `/focus`, `/resolve?t=&word=`, `/addressing`, `/events`, `/stream` (ws).

## Credits
- Pupil Labs `pupil-labs-realtime-api` and `real-time-screen-gaze` (AprilTag screen mapping).
- AssemblyAI Universal-Streaming (speech to text with word timestamps).
- Apprentice (Mac agent) executes the actions over its local WebSocket.
- All code in this repo was written during INTERFACE (Oct 10-11, 2026).
