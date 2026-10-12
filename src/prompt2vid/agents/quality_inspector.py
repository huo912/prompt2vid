"""质检师 agent。

Agent 实例由 crew.py 从 config/agents/quality_inspector.yaml 加载；
工具为 prompt2vid.tools.file_tool（文件与探测）。
"""

from __future__ import annotations

AGENT_NAME = "quality_inspector"
