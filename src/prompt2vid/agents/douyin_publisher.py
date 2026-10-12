"""抖音发布专家 agent。

Agent 实例由 crew.py 从 config/agents/douyin_publisher.yaml 加载；
工具为 prompt2vid.tools.douyin_tool（开放平台 API + dry-run + 状态跟踪）。
"""

from __future__ import annotations

AGENT_NAME = "douyin_publisher"
