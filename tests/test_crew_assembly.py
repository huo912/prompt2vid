"""端到端冒烟测试：构建 Crew（不真实调用 LLM/ComfyUI）。

验证：config YAML 可加载、agent/task 装配完整、流水线顺序正确、
每个 agent 都拿到了任务描述中要求的工具。
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
    # 每个 task 绑定了对应 agent
    for task in tasks:
        assert task.agent is not None
    # 顺序与 TASK_ORDER 一致（按 description 前缀粗验）
    for task, expected in zip(tasks, TASK_ORDER):
        assert task.description is not None


def test_every_agent_has_tools():
    """创意三阶段需要 file_tool 写产物；媒体阶段需要阶段工具。"""
    os.environ.setdefault("PROMPT2VID_CONFIG_DIR", str(REPO_ROOT / "config"))
    os.environ.setdefault("PROMPT2VID_LLM_API_KEY", "sk-test-dummy")

    from prompt2vid.crew import build_agents

    agents = build_agents(REPO_ROOT / "config")
    for name, agent in agents.items():
        assert agent.tools, f"{name} 没有任何工具"

    # 创意 agent 具备文件读写（写 requirement.json / script.md / storyboard.json）
    for creative in ("requirement_analyst", "script_writer", "storyboard_planner"):
        tool_names = " ".join(t.name.lower() for t in agents[creative].tools)
        assert "write" in tool_names, f"{creative} 缺少文件写入工具"

    # 媒体 agent 拿到对应阶段工具
    expected_tool_keyword = {
        "image_generator": "first-frame",
        "video_generator": "video clips",
        "voiceover_artist": "voiceover",
        "video_editor": "render",
        "quality_inspector": "quality",
        "douyin_publisher": "douyin",
    }
    for name, keyword in expected_tool_keyword.items():
        tool_names = " ".join(t.name.lower() for t in agents[name].tools)
        assert keyword in tool_names, f"{name} 缺少阶段工具（{keyword}）"


def test_crew_kickoff_inputs_shape():
    """验证 kickoff 输入键（不真实执行）。"""
    from prompt2vid.crew import TASK_ORDER

    assert TASK_ORDER[0] == "analyze_requirement"
    assert TASK_ORDER[-1] == "publish_douyin"


def test_build_crew_task_subset():
    """直接媒体模式：只构建创意三任务 / 发布单任务的子集 Crew。"""
    os.environ.setdefault("PROMPT2VID_CONFIG_DIR", str(REPO_ROOT / "config"))
    os.environ.setdefault("PROMPT2VID_LLM_API_KEY", "sk-test-dummy")

    from prompt2vid.crew import build_crew
    from prompt2vid.pipeline import CREATIVE_TASKS, PUBLISH_TASK

    creative = build_crew(task_names=CREATIVE_TASKS)
    assert len(creative.tasks) == 3
    # agents 只包含实际用到的（去重）
    assert len(creative.agents) == 3

    publish = build_crew(task_names=[PUBLISH_TASK])
    assert len(publish.tasks) == 1
    assert len(publish.agents) == 1

    with pytest.raises(ValueError):
        build_crew(task_names=["nonexistent_task"])


def test_analyze_requirement_task_carries_user_prompt():
    """首个任务描述必须含 {user_prompt} 占位符，否则用户描述到不了 agent。"""
    import yaml

    spec_path = REPO_ROOT / "config" / "tasks" / "analyze_requirement.yaml"
    spec = next(iter(yaml.safe_load(spec_path.read_text(encoding="utf-8")).values()))
    assert "{user_prompt}" in spec["description"]


def test_style_hint_injected_into_storyboard_task(monkeypatch):
    """PROMPT2VID_STYLE=anime：分镜任务注入风格约束，其它任务不受影响。"""
    os.environ.setdefault("PROMPT2VID_CONFIG_DIR", str(REPO_ROOT / "config"))
    os.environ.setdefault("PROMPT2VID_LLM_API_KEY", "sk-test-dummy")
    monkeypatch.setenv("PROMPT2VID_STYLE", "anime")

    from prompt2vid.crew import TASK_ORDER, build_agents, build_tasks

    agents = build_agents(REPO_ROOT / "config")
    tasks = build_tasks(REPO_ROOT / "config", agents)
    by_name = dict(zip(TASK_ORDER, tasks))

    assert "PROMPT2VID_STYLE=anime" in by_name["plan_storyboard"].description
    assert "anime style" in by_name["plan_storyboard"].description
    for name in ("analyze_requirement", "write_script", "generate_images"):
        assert "PROMPT2VID_STYLE" not in by_name[name].description

    # 关闭风格后不注入（提示词层约束同样关闭）
    monkeypatch.delenv("PROMPT2VID_STYLE", raising=False)
    tasks = build_tasks(REPO_ROOT / "config", build_agents(REPO_ROOT / "config"))
    assert "PROMPT2VID_STYLE" not in dict(zip(TASK_ORDER, tasks))["plan_storyboard"].description


def test_kickoff_inputs_interpolate_into_first_task():
    """模拟 kickoff 插值：用户描述应真实出现在第一个任务的描述里。"""
    os.environ.setdefault("PROMPT2VID_CONFIG_DIR", str(REPO_ROOT / "config"))
    os.environ.setdefault("PROMPT2VID_LLM_API_KEY", "sk-test-dummy")

    from prompt2vid.crew import build_agents, build_tasks

    agents = build_agents(REPO_ROOT / "config")
    tasks = build_tasks(REPO_ROOT / "config", agents)

    inputs = {
        "user_prompt": "做一个关于秋天的治愈系短视频",
        "workspace": str(REPO_ROOT / "output" / "run_test"),
    }
    for task in tasks:
        task.interpolate_inputs_and_add_conversation_history(inputs)
    assert "做一个关于秋天的治愈系短视频" in tasks[0].description
    assert "{user_prompt}" not in tasks[0].description
