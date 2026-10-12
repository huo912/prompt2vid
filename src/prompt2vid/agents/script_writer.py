"""编剧 agent。

Agent 实例由 crew.py 从 config/agents/script_writer.yaml 加载；
本模块仅承载 agent 专属逻辑（无工具，纯 LLM 推理）。
"""

from __future__ import annotations

AGENT_NAME = "script_writer"
