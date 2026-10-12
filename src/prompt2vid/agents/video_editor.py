"""剪辑师 agent。

Agent 实例由 crew.py 从 config/agents/video_editor.yaml 加载；
工具为 prompt2vid.tools.ffmpeg_tool（拼接/混音/字幕烧录）。
"""

from __future__ import annotations

AGENT_NAME = "video_editor"
