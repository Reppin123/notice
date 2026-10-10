"""Attention MCP server: any agent can ask what the user is paying attention to.

Thin stdio proxy over the Notice daemon's local API (http://127.0.0.1:8790).
"""
from __future__ import annotations

import time

import httpx
from mcp.server.mcpserver import MCPServer

API = "http://127.0.0.1:8790"
mcp = MCPServer("notice-attention")


def _get(path: str, **params):
    return httpx.get(API + path, params=params, timeout=5).json()


@mcp.tool()
def get_focus() -> dict:
    """What the user is looking at right now: latest fixation plus nearby on-screen elements."""
    return _get("/focus")


@mcp.tool()
def resolve_reference(word: str = "this", seconds_ago: float = 0.0) -> dict:
    """Which thing the user meant by a deictic word ('this', 'that', 'there') said `seconds_ago` seconds ago."""
    return _get("/resolve", word=word, t=time.time() - seconds_ago)


@mcp.tool()
def is_addressing_agent() -> dict:
    """True when the user's eyes are on the screen (talking to the agent), false when looking away (talking to a person)."""
    return _get("/addressing")


@mcp.tool()
def recent_attention_events() -> list:
    """The last Notice events: resolved references, ignored speech, struggle offers."""
    return _get("/events")


def main():
    mcp.run()
