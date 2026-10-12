"""视频生成师 agent。

Agent 实例由 crew.py 从 config/agents/video_generator.yaml 加载；
工具为 prompt2vid.tools.comfyui_tool（LTX-Video 图生视频）。
"""

from __future__ import annotations

AGENT_NAME = "video_generator"
