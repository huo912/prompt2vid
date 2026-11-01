"""Agent Skills 集成测试（agentskills.io 标准 + 外部 skill 下载）。

覆盖：目录发现（多根 / 环境变量）、SKILL.md 解析校验、agent YAML 的
``skills:`` 解析（含 always_on 提升）、crew 装配接线、预检项、以及
:mod:`prompt2vid.skills_sync` 的离线安装路径（用临时假仓库，不联网）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"

GOOD_SKILL = """---
name: {name}
description: 测试用 skill，说明它做什么以及在什么情况下使用（至少二十个字符）。
license: MIT
---

# 测试

正文。
"""


def _write_skill(root: Path, name: str, body: str | None = None, dir_name: str | None = None) -> Path:
    skill_dir = root / (dir_name or name)
    skill_dir.mkdir(parents=True, exist_ok=True)
    text = body if body is not None else GOOD_SKILL.format(name=name)
    (skill_dir / "SKILL.md").write_text(text, encoding="utf-8")
    return skill_dir


# ---------------------------------------------------------------- 目录发现


def test_find_skills_dirs_includes_project_skills(monkeypatch):
    from prompt2vid.skills import find_skills_dirs

    monkeypatch.delenv("PROMPT2VID_SKILLS_DIR", raising=False)
    dirs = find_skills_dirs(CONFIG_DIR)
    assert CONFIG_DIR / "skills" in dirs


def test_find_skills_dirs_env_override(tmp_path, monkeypatch):
    from prompt2vid.skills import find_skills_dirs

    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv("PROMPT2VID_SKILLS_DIR", f"{first}:{second}")
    dirs = find_skills_dirs(CONFIG_DIR)
    assert dirs == [first, second]


def test_find_skills_dirs_env_missing_dir_returns_empty(tmp_path, monkeypatch):
    from prompt2vid.skills import find_skills_dirs

    monkeypatch.setenv("PROMPT2VID_SKILLS_DIR", str(tmp_path / "nope"))
    assert find_skills_dirs(CONFIG_DIR) == []


# ---------------------------------------------------------------- 解析校验


def test_project_skills_all_parse():
    from prompt2vid.skills import catalog_summary, find_skills_dirs, load_catalog, validate_skills_dirs

    dirs = find_skills_dirs(CONFIG_DIR)
    names, errors = validate_skills_dirs(dirs)
    assert errors == []
    assert set(names) >= {
        "sdxl-prompt-engineering",
        "ltx-motion-prompting",
        "character-consistency",
        "storyboard-schema-guide",
        "chinese-short-video-script",
        "douyin-metadata-and-compliance",
        "delivery-qa-spec",
        "visual-style-presets",
    }
    catalog = load_catalog(dirs)
    assert set(names) <= set(catalog)
    assert "skill（" in catalog_summary(dirs, catalog)


def test_validate_reports_bad_frontmatter(tmp_path):
    from prompt2vid.skills import validate_skills_dir

    root = tmp_path / "skills"
    _write_skill(root, "good")
    _write_skill(root, "broken", body="---\nname: broken\n---\n\n缺 description\n")
    names, errors = validate_skills_dir(root)
    assert names == ["good"]
    assert errors and "broken" in errors[0]


def test_validate_reports_dir_name_mismatch(tmp_path):
    from prompt2vid.skills import validate_skills_dir

    root = tmp_path / "skills"
    _write_skill(root, "actual-name", dir_name="wrong-dir")
    names, errors = validate_skills_dir(root)
    assert names == []
    assert errors and "wrong-dir" in errors[0]


def test_load_catalog_skips_broken_and_dedupes(tmp_path):
    from prompt2vid.skills import load_catalog

    first = tmp_path / "a"
    second = tmp_path / "b"
    _write_skill(first, "dup")
    _write_skill(second, "dup")
    _write_skill(second, "only-in-b")
    catalog = load_catalog([first, second])
    # 同名 skill 只保留先出现的那个
    assert catalog["dup"].path.parent == first
    assert "only-in-b" in catalog


# ---------------------------------------------------------------- 按 agent 解析


def test_resolve_skills_supports_plain_and_always_on(tmp_path):
    from prompt2vid.skills import load_catalog, resolve_skills

    root = tmp_path / "skills"
    _write_skill(root, "alpha")
    _write_skill(root, "beta")
    catalog = load_catalog(root)

    resolved = resolve_skills(["alpha", {"name": "beta", "always_on": True}], catalog)
    assert resolved is not None
    by_name = {s.name: s for s in resolved}
    assert set(by_name) == {"alpha", "beta"}
    # always_on -> 全文常驻注入（INSTRUCTIONS 级）
    assert by_name["beta"].disclosure_level > by_name["alpha"].disclosure_level
    assert by_name["beta"].instructions


def test_resolve_skills_handles_single_string_and_dedupes(tmp_path):
    from prompt2vid.skills import load_catalog, resolve_skills

    root = tmp_path / "skills"
    _write_skill(root, "alpha")
    catalog = load_catalog(root)
    assert len(resolve_skills("alpha", catalog) or []) == 1
    assert len(resolve_skills(["alpha", "alpha"], catalog) or []) == 1


def test_resolve_skills_none_when_unconfigured():
    from prompt2vid.skills import resolve_skills

    assert resolve_skills(None, {}) is None
    assert resolve_skills([], {}) is None


def test_resolve_skills_unknown_name_raises():
    from prompt2vid.skills import resolve_skills

    with pytest.raises(KeyError) as excinfo:
        resolve_skills(["does-not-exist"], {})
    assert "does-not-exist" in str(excinfo.value)


def test_env_always_on_forces_activation(tmp_path, monkeypatch):
    from prompt2vid.skills import load_catalog, resolve_skills

    root = tmp_path / "skills"
    _write_skill(root, "alpha")
    _write_skill(root, "beta")
    monkeypatch.setenv("PROMPT2VID_SKILLS_ALWAYS_ON", "alpha")
    resolved = {s.name: s for s in resolve_skills(["alpha", "beta"], load_catalog(root)) or []}
    assert resolved["alpha"].disclosure_level > resolved["beta"].disclosure_level


# ---------------------------------------------------------------- crew 接线


def test_agents_receive_skills():
    import os

    os.environ.setdefault("PROMPT2VID_CONFIG_DIR", str(CONFIG_DIR))
    os.environ.setdefault("PROMPT2VID_LLM_API_KEY", "sk-test-dummy")

    from prompt2vid.crew import build_agents

    agents = build_agents(CONFIG_DIR)
    planner = {s.name: s for s in agents["storyboard_planner"].skills or []}
    assert set(planner) == {
        "sdxl-prompt-engineering",
        "ltx-motion-prompting",
        "character-consistency",
        "storyboard-schema-guide",
    }
    # 关键 skill 常驻（小模型不一定主动调用 load_skill）
    assert planner["ltx-motion-prompting"].instructions
    assert planner["storyboard-schema-guide"].instructions

    assert [s.name for s in agents["script_writer"].skills or []] == ["chinese-short-video-script"]
    assert [s.name for s in agents["douyin_publisher"].skills or []] == [
        "douyin-metadata-and-compliance"
    ]
    assert [s.name for s in agents["quality_inspector"].skills or []] == ["delivery-qa-spec"]
    # 纯代码驱动的媒体 agent 不挂 skill
    assert not agents["image_generator"].skills


def test_skills_can_be_disabled_by_env(monkeypatch):
    import os

    os.environ.setdefault("PROMPT2VID_CONFIG_DIR", str(CONFIG_DIR))
    os.environ.setdefault("PROMPT2VID_LLM_API_KEY", "sk-test-dummy")
    monkeypatch.setenv("PROMPT2VID_SKILLS", "off")

    from prompt2vid.crew import build_agents

    agents = build_agents(CONFIG_DIR)
    assert all(not agent.skills for agent in agents.values())


# ---------------------------------------------------------------- 预检


def test_preflight_reports_skills_non_blocking():
    from prompt2vid.preflight import Check, _append_skills_check

    checks: list[Check] = []
    _append_skills_check(checks)
    skill_checks = [c for c in checks if c.name == "Agent Skills"]
    assert len(skill_checks) == 1
    check = skill_checks[0]
    assert check.required is False
    assert check.ok is True
    assert "sdxl-prompt-engineering" in check.detail


# ---------------------------------------------------------------- 外部 skill 下载


def _fake_repo(root: Path, names: tuple[str, ...] = ("skill-creator", "webapp-testing")) -> Path:
    (root / "skills").mkdir(parents=True, exist_ok=True)
    for name in names:
        _write_skill(root / "skills", name)
    return root


def test_sync_skills_installs_from_local_source(tmp_path, monkeypatch):
    from prompt2vid.skills_sync import MANIFEST_NAME, sync_skills

    repo = _fake_repo(tmp_path / "repo")
    target = tmp_path / "external"
    monkeypatch.setenv("PROMPT2VID_SKILLS_SOURCE", str(repo))

    report = sync_skills(["skill-creator"], target=target)

    assert report.ok
    assert report.installed == ["skill-creator"]
    assert (target / "skill-creator" / "SKILL.md").is_file()
    manifest = json.loads((target / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["standard"] == "agentskills.io"
    assert "skill-creator" in manifest["skills"]


def test_sync_skills_skips_existing_unless_forced(tmp_path, monkeypatch):
    from prompt2vid.skills_sync import sync_skills

    repo = _fake_repo(tmp_path / "repo")
    target = tmp_path / "external"
    monkeypatch.setenv("PROMPT2VID_SKILLS_SOURCE", str(repo))

    sync_skills(["skill-creator"], target=target)
    again = sync_skills(["skill-creator"], target=target)
    assert again.installed == []
    assert again.skipped == ["skill-creator"]

    forced = sync_skills(["skill-creator"], target=target, force=True)
    assert forced.installed == ["skill-creator"]


def test_sync_skills_rejects_unknown_name(tmp_path):
    from prompt2vid.skills_sync import SkillSyncError, sync_skills

    with pytest.raises(SkillSyncError):
        sync_skills(["pdf-nope"], target=tmp_path / "external")


def test_sync_skills_reports_missing_upstream_skill(tmp_path, monkeypatch):
    from prompt2vid.skills_sync import sync_skills

    repo = _fake_repo(tmp_path / "repo", names=())  # 空仓库
    monkeypatch.setenv("PROMPT2VID_SKILLS_SOURCE", str(repo))

    report = sync_skills(["skill-creator"], target=tmp_path / "external")
    assert not report.ok
    assert "skill-creator" in report.failed


def test_repo_urls_prefers_mirror_then_direct(monkeypatch):
    from prompt2vid.skills_sync import DEFAULT_REPO, repo_urls

    monkeypatch.delenv("PROMPT2VID_SKILLS_REPO", raising=False)
    monkeypatch.delenv("PROMPT2VID_SKILLS_MIRROR", raising=False)
    urls = repo_urls()
    assert urls[0].startswith("https://ghfast.top/")
    assert urls[-1] == DEFAULT_REPO

    monkeypatch.setenv("PROMPT2VID_SKILLS_MIRROR", "off")
    assert repo_urls() == [DEFAULT_REPO]
