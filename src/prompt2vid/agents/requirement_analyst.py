"""需求分析师 agent。

Agent 实例（role/goal/backstory）由 crew.py 从 config/agents/requirement_analyst.yaml
加载；本模块仅承载 agent 专属逻辑（无工具，纯 LLM 推理）。
"""

from __future__ import annotations

AGENT_NAME = "requirement_analyst"
