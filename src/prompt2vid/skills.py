"""Agent Skills 集成（agentskills.io 标准，CrewAI 原生实现）。

Skills 与 Tools 的分工（最容易混淆的一点）：
- **Tools** 给 agent 可调用的函数（写文件、提交 ComfyUI、发布抖音）；
- **Skills** 只往 prompt 里注入"怎么做"的程序性知识（提示词规范、
  质检口径、合规清单），不提供任何可调用能力。

本模块只负责"发现 → 解析 → 按 agent 挂载"，不改动任何工具装配逻辑：
- 发现：扫描 ``config/skills/`` 下每个含 ``SKILL.md`` 的子目录
  （``crewai.skills.discover_skills``，只读 frontmatter，成本极低）；
- 解析：按 agent YAML 里的 ``skills:`` 列表挑出子集（支持
  ``always_on`` —— 常驻注入全文，不依赖小模型主动调用 ``load_skill``）；
- 挂载：``crew.build_agents`` 把 ``Skill`` 对象传给 ``Agent(skills=...)``。

环境变量：
- ``PROMPT2VID_SKILLS_DIR``：自定义 skill 根目录（默认 ``config/skills``）；
- ``PROMPT2VID_SKILLS=off``：整体关闭 skills（等价于回到引入前的行为）；
- ``PROMPT2VID_SKILLS_ALWAYS_ON=name1,name2``：临时把某些 skill 设为常驻。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from crewai.skills import Skill, activate_skill, discover_skills
from crewai.skills.parser import load_skill_metadata

logger = logging.getLogger("prompt2vid.skills")

SKILLS_DIR_VAR = "PROMPT2VID_SKILLS_DIR"
SKILLS_DISABLE_VAR = "PROMPT2VID_SKILLS"
SKILLS_ALWAYS_ON_VAR = "PROMPT2VID_SKILLS_ALWAYS_ON"
SKILL_FILENAME = "SKILL.md"

_DISABLED_VALUES = {"off", "0", "false", "no", "none", "disable", "disabled"}


def skills_enabled() -> bool:
    """skills 是否启用（``PROMPT2VID_SKILLS=off`` 可整体关闭）。"""
    raw = os.getenv(SKILLS_DISABLE_VAR, "").strip().lower()
    return raw not in _DISABLED_VALUES


def default_config_dir() -> Path | None:
    """定位 config/ 目录（与 crew._find_config_dir 同序，但不抛异常、不导 crewai）。

    预检要在不构造任何 Agent 的前提下检查 skills，因此这里独立实现。
    """
    env_dir = os.getenv("PROMPT2VID_CONFIG_DIR", "").strip()
    if env_dir:
        path = Path(env_dir)
        if path.is_dir():
            return path
    pkg_dir = Path(__file__).resolve().parent.parent / "config"
    if pkg_dir.is_dir():
        return pkg_dir
    repo_dir = Path(__file__).resolve().parents[2] / "config"
    return repo_dir if repo_dir.is_dir() else None


def find_skills_dirs(config_dir: Path) -> list[Path]:
    """定位所有 skill 根目录（按优先级去重）。

    两个来源：项目自带 ``config/skills``（随仓库/镜像分发）与
    ``config/skills-external``（:mod:`prompt2vid.skills_sync` 下载的第三方
    skill，可在 .gitignore 里排除）。环境变量 ``PROMPT2VID_SKILLS_DIR``
    可用 ``os.pathsep`` 分隔多个目录，此时只使用显式指定的目录。
    """
    raw = os.getenv(SKILLS_DIR_VAR, "").strip()
    candidates: list[Path] = []
    if raw:
        candidates.extend(
            Path(item).expanduser() for item in raw.split(os.pathsep) if item.strip()
        )
        existing = [p for p in candidates if p.is_dir()]
        if not existing:
            logger.warning("PROMPT2VID_SKILLS_DIR 指向的目录不存在: %s", raw)
        return existing

    for base in (config_dir, config_dir.parent):
        candidates.append(base / "skills")
        candidates.append(base / "skills-external")

    seen: set[Path] = set()
    found: list[Path] = []
    for cand in candidates:
        resolved = cand.resolve()
        if resolved in seen or not cand.is_dir():
            continue
        seen.add(resolved)
        found.append(cand)
    return found


def find_skills_dir(config_dir: Path) -> Path | None:
    """返回主 skill 根目录（:func:`find_skills_dirs` 的第一项）。"""
    dirs = find_skills_dirs(config_dir)
    return dirs[0] if dirs else None


def load_catalog(skills_dir: Path | Iterable[Path] | None) -> dict[str, Skill]:
    """扫描一个或多个目录，返回 ``{skill 名: Skill(METADATA 级)}``。

    解析失败的 skill 会被 ``discover_skills`` 跳过（只记 warning），
    需要逐项错误详情请用 :func:`validate_skills_dir`。先出现的目录优先，
    重名 skill 不会被后者覆盖。
    """
    if skills_dir is None:
        return {}
    dirs = [skills_dir] if isinstance(skills_dir, Path) else list(skills_dir)
    catalog: dict[str, Skill] = {}
    for directory in dirs:
        if not directory.is_dir():
            logger.warning("skill 目录不可用: %s", directory)
            continue
        try:
            discovered = discover_skills(directory)
        except FileNotFoundError:
            logger.warning("skill 目录不可用: %s", directory)
            continue
        for skill in discovered:
            if skill.name in catalog:
                logger.warning(
                    "skill 重名，忽略 %s（已存在同名 skill，来自 %s）",
                    skill.path,
                    catalog[skill.name].path,
                )
                continue
            catalog[skill.name] = skill
    return catalog


def validate_skills_dir(skills_dir: Path) -> tuple[list[str], list[str]]:
    """逐项校验 skill 目录，返回 ``(合法 skill 名, 错误信息)``。

    与 ``load_catalog`` 的区别：不吞异常。``SKILL.md`` frontmatter 非法、
    ``name`` 与目录名不一致等问题都会被显式报出（预检与 ``--list-skills`` 用）。
    """
    names: list[str] = []
    errors: list[str] = []
    if not skills_dir.is_dir():
        return names, [f"目录不存在: {skills_dir}"]

    for child in sorted(skills_dir.iterdir()):
        if not child.is_dir() or not (child / SKILL_FILENAME).is_file():
            continue
        try:
            skill = load_skill_metadata(child)
        except Exception as exc:  # noqa: BLE001 - 预检要把所有问题汇总报出
            errors.append(f"{child.name}: {exc}")
            continue
        names.append(skill.name)
    return names, errors


def validate_skills_dirs(
    skills_dirs: Iterable[Path],
) -> tuple[list[str], list[str]]:
    """逐项校验多个 skill 根目录，返回 ``(合法 skill 名, 错误信息)``。"""
    names: list[str] = []
    errors: list[str] = []
    for directory in skills_dirs:
        dir_names, dir_errors = validate_skills_dir(directory)
        names.extend(dir_names)
        errors.extend(dir_errors)
    return names, errors


def _always_on_names() -> set[str]:
    raw = os.getenv(SKILLS_ALWAYS_ON_VAR, "").strip()
    return {item.strip() for item in raw.split(",") if item.strip()}


def _normalize_entries(entries: Any) -> list[tuple[str, bool]]:
    """把 YAML 里的 skills 列表归一化为 ``[(name, always_on), ...]``。

    支持三种写法：
    - ``skills: [sdxl-prompt-engineering]``
    - ``skills: [{name: ltx-motion-prompting, always_on: true}]``
    - ``skills: sdxl-prompt-engineering``（单个字符串）
    """
    if not entries:
        return []
    items = entries if isinstance(entries, (list, tuple)) else [entries]
    parsed: list[tuple[str, bool]] = []
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            always_on = bool(item.get("always_on", False))
        else:
            name, always_on = str(item).strip(), False
        if name:
            parsed.append((name, always_on))
    return parsed


def resolve_skills(entries: Any, catalog: dict[str, Skill]) -> list[Skill] | None:
    """按 agent 的 ``skills:`` 配置从 catalog 挑出 Skill 对象。

    Args:
        entries: agent YAML 里的 ``skills`` 值（None / str / list）。
        catalog: :func:`load_catalog` 的结果。

    Returns:
        Skill 列表（未配置时返回 None，直接透传给 ``Agent(skills=...)``）。
        ``always_on: true`` 或 ``PROMPT2VID_SKILLS_ALWAYS_ON`` 命中的 skill
        会被提升到 INSTRUCTIONS 级（全文常驻注入）。

    Raises:
        KeyError: 引用了 catalog 中不存在的 skill（拼写错误必须显式暴露，
            否则会静默少挂一个 skill）。
    """
    specs = _normalize_entries(entries)
    if not specs:
        return None

    forced = _always_on_names()
    available = ", ".join(sorted(catalog)) or "（未发现任何 skill）"
    resolved: dict[str, Skill] = {}
    for name, always_on in specs:
        skill = catalog.get(name)
        if skill is None:
            raise KeyError(f"未知 skill: {name}（可用: {available}）")
        if name in resolved:
            continue
        if always_on or name in forced:
            resolved[name] = activate_skill(skill)
        else:
            resolved[name] = skill
    return list(resolved.values())


def catalog_summary(
    skills_dirs: Path | Iterable[Path] | None, catalog: dict[str, Skill]
) -> str:
    """生成一行人类可读的 catalog 摘要（预检 / CLI 用）。"""
    if skills_dirs is None:
        return "未发现 skill 目录（skills 为可选增强，不影响运行）"
    dirs = (
        [skills_dirs] if isinstance(skills_dirs, Path) else list(skills_dirs)
    )
    if not dirs:
        return "未发现 skill 目录（skills 为可选增强，不影响运行）"
    where = " + ".join(str(d) for d in dirs)
    if not catalog:
        return f"{where} 下没有可用 skill"
    names = ", ".join(sorted(catalog))
    return f"{len(catalog)} 个 skill（{where}）: {names}"


def always_on_skills(entries: Any) -> list[str]:
    """列出某 agent 配置里会常驻注入的 skill 名（排查 prompt 膨胀用）。"""
    forced = _always_on_names()
    return [name for name, flag in _normalize_entries(entries) if flag or name in forced]
