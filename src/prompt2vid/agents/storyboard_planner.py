"""分镜师 agent。

Agent 实例由 crew.py 从 config/agents/storyboard_planner.yaml 加载；
本模块仅承载 agent 专属逻辑（无工具，纯 LLM 推理）。
"""

from __future__ import annotations

AGENT_NAME = "storyboard_planner"
