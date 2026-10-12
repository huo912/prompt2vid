"""图像生成师 agent。

Agent 实例由 crew.py 从 config/agents/image_generator.yaml 加载；
工具为 prompt2vid.tools.comfyui_tool（文生图）。
"""

from __future__ import annotations

AGENT_NAME = "image_generator"
