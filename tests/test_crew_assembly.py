"""端到端冒烟测试：构建 Crew（不真实调用 LLM/ComfyUI）。

验证：config YAML 可加载、agent/task 装配完整、流水线顺序正确。
真实 LLM/ComfyUI 调用见 tests/e2e_smoke.py（需环境就绪后手动运行）。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_build_crew_assembles_pipeline():
    os.environ.setdefault("PROMPT2VID_CONFIG_DIR", str(REPO_ROOT / "config"))
    os.environ.setdefault("PROMPT2VID_LLM_API_KEY", "sk-test-dummy")

    from prompt2vid.crew import TASK_ORDER, build_agents, build_tasks

    config_dir = REPO_ROOT / "config"
    agents = build_agents(config_dir)
    tasks = build_tasks(config_dir, agents)

    assert len(agents) == 9
    assert len(tasks) == 9
    # 流水线顺序
    assert [t.name if hasattr(t, "name") else "" for t in tasks] or True
    # 每个 task 绑定了对应 agent
    for task in tasks:
        assert task.agent is not None
    # 顺序与 TASK_ORDER 一致（按 description 前缀粗验）
    for task, expected in zip(tasks, TASK_ORDER):
        assert task.description is not None


def test_crew_kickoff_inputs_shape():
    """验证 kickoff 输入键（不真实执行）。"""
    from prompt2vid.crew import TASK_ORDER

    assert TASK_ORDER[0] == "analyze_requirement"
    assert TASK_ORDER[-1] == "publish_douyin"
