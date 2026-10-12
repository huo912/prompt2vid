"""配音师 agent。

Agent 实例由 crew.py 从 config/agents/voiceover_artist.yaml 加载；
工具为 prompt2vid.tools.tts_tool（edge-tts）。
"""

from __future__ import annotations

AGENT_NAME = "voiceover_artist"
