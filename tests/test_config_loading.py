"""配置文件加载测试：YAML 结构、agent/task 对应关系。"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = REPO_ROOT / "config" / "agents"
TASKS_DIR = REPO_ROOT / "config" / "tasks"

EXPECTED_AGENTS = [
    "requirement_analyst",
    "script_writer",
    "storyboard_planner",
    "image_generator",
    "video_generator",
    "voiceover_artist",
    "video_editor",
    "quality_inspector",
    "douyin_publisher",
]

EXPECTED_TASKS = [
    "analyze_requirement",
    "write_script",
    "plan_storyboard",
    "generate_images",
    "generate_videos",
    "generate_voiceover",
    "edit_video",
    "inspect_quality",
    "publish_douyin",
]


def _load_one(path: Path) -> tuple[str, dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and len(data) == 1
    return next(iter(data.items()))


def test_agent_yaml_files():
    names = sorted(p.stem for p in AGENTS_DIR.glob("*.yaml"))
    assert names == sorted(EXPECTED_AGENTS)
    for path in AGENTS_DIR.glob("*.yaml"):
        name, spec = _load_one(path)
        assert name == path.stem
        for key in ("role", "goal", "backstory"):
            assert key in spec, f"{path} 缺少 {key}"
            assert isinstance(spec[key], str) and spec[key].strip()


def test_task_yaml_files():
    names = sorted(p.stem for p in TASKS_DIR.glob("*.yaml"))
    assert names == sorted(EXPECTED_TASKS)
    for path in TASKS_DIR.glob("*.yaml"):
        name, spec = _load_one(path)
        assert name == path.stem
        for key in ("description", "expected_output", "agent"):
            assert key in spec, f"{path} 缺少 {key}"
        assert spec["agent"] in EXPECTED_AGENTS


def test_task_agent_mapping():
    mapping = {
        "analyze_requirement": "requirement_analyst",
        "write_script": "script_writer",
        "plan_storyboard": "storyboard_planner",
        "generate_images": "image_generator",
        "generate_videos": "video_generator",
        "generate_voiceover": "voiceover_artist",
        "edit_video": "video_editor",
        "inspect_quality": "quality_inspector",
        "publish_douyin": "douyin_publisher",
    }
    for task_name, agent_name in mapping.items():
        _, spec = _load_one(TASKS_DIR / f"{task_name}.yaml")
        assert spec["agent"] == agent_name
