"""Crew 构建：从 config/ 目录加载 agent 与 task YAML 并组装流水线。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from crewai import Agent, Crew, Process, Task

from .dynamic_llm import build_llm
from .env_config import load_env
from .style import storyboard_style_hint

# agent 名 → 工具模块列表（prompt2vid.tools 下的模块名）
# 默认取模块的 TOOLS 属性；也可用 "module:ATTR" 语法指定其他导出
# （如 comfyui_tool:IMAGE_TOOLS 只给图像生成阶段工具）。
# 注意：创意三阶段（需求/剧本/分镜）需要 file_tool 写 requirement.json /
# script.md / storyboard.json，此前漏配导致产物从未落盘。
AGENT_TOOL_ATTRS: dict[str, list[str]] = {
    "requirement_analyst": ["file_tool"],
    "script_writer": ["file_tool"],
    "storyboard_planner": ["file_tool"],
    "image_generator": ["comfyui_tool:IMAGE_TOOLS"],
    "video_generator": ["comfyui_tool:VIDEO_TOOLS"],
    "voiceover_artist": ["tts_tool"],
    "video_editor": ["ffmpeg_tool"],
    "quality_inspector": ["quality_tool"],
    "douyin_publisher": ["douyin_tool", "file_tool"],
}

# 流水线顺序（task 名与 config/tasks/*.yaml 顶层键一致）
TASK_ORDER: list[str] = [
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

# 需要注入风格约束的任务：分镜表是 image_prompt 的唯一来源，
# 风格（PROMPT2VID_STYLE）从这里开始约束 LLM 的画面组织方式
STYLE_CONSTRAINED_TASKS = ("plan_storyboard",)


def _bypass_proxy_for_internal_hosts() -> None:
    """把内网服务地址加入 NO_PROXY，避免公司代理劫持内网请求。

    httpx/aiohttp 等库默认读取 HTTP_PROXY/HTTPS_PROXY 环境变量；公司
    代理（SWG）会拦截发往内网 LLM 网关的请求并返回 403 拦截页。
    把 PROMPT2VID_LLM_BASE_URL / COMFYUI_URL 的主机名追加到
    NO_PROXY / no_proxy 即可让这些请求直连。
    """
    hosts: set[str] = set()
    for var in ("PROMPT2VID_LLM_BASE_URL", "COMFYUI_URL"):
        raw = os.getenv(var, "").strip()
        if not raw:
            continue
        parsed = urlparse(raw if "://" in raw else f"http://{raw}")
        host = parsed.hostname
        if host:
            hosts.add(host)

    if not hosts:
        return

    for var in ("NO_PROXY", "no_proxy"):
        current = os.getenv(var, "")
        parts = [p.strip() for p in current.split(",") if p.strip()]
        merged = ",".join(dict.fromkeys(parts + sorted(hosts)))
        os.environ[var] = merged


def _find_config_dir() -> Path:
    """定位 config/ 目录：环境变量优先，其次包内、仓库根。"""
    env_dir = os.getenv("PROMPT2VID_CONFIG_DIR", "").strip()
    if env_dir:
        path = Path(env_dir)
        if path.is_dir():
            return path
    pkg_dir = Path(__file__).resolve().parent.parent / "config"
    if pkg_dir.is_dir():
        return pkg_dir
    repo_dir = Path(__file__).resolve().parents[2] / "config"
    if repo_dir.is_dir():
        return repo_dir
    raise FileNotFoundError(
        "找不到 config/ 目录，请设置 PROMPT2VID_CONFIG_DIR 环境变量"
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"{path} 应只包含一个顶层键")
    return data


def _collect_tools(agent_name: str) -> list:
    """从 tools/<module>.py 收集工具实例（支持 module / module:ATTR 语法）。"""
    if agent_name not in AGENT_TOOL_ATTRS:
        raise KeyError(f"未知 agent: {agent_name}")
    tools: list = []
    if not AGENT_TOOL_ATTRS[agent_name]:
        return tools
    from importlib import import_module

    for spec in AGENT_TOOL_ATTRS[agent_name]:
        module_name, _, attr = spec.partition(":")
        module = import_module(f"prompt2vid.tools.{module_name}")
        tools.extend(getattr(module, attr or "TOOLS"))
    return tools


def build_agents(config_dir: Path) -> dict[str, Agent]:
    """从 config/agents/*.yaml 构建 Agent 实例。"""
    agents: dict[str, Agent] = {}
    agents_dir = config_dir / "agents"
    for path in sorted(agents_dir.glob("*.yaml")):
        name, spec = next(iter(_load_yaml(path).items()))
        tools = _collect_tools(name)
        agents[name] = Agent(
            role=spec["role"],
            goal=spec["goal"],
            backstory=spec["backstory"],
            llm=build_llm(name),
            tools=tools,
            verbose=spec.get("verbose", False),
            allow_delegation=spec.get("allow_delegation", False),
            max_iter=spec.get("max_iter", 15),
        )
    return agents


def build_tasks(
    config_dir: Path,
    agents: dict[str, Agent],
    task_names: list[str] | None = None,
) -> list[Task]:
    """按流水线顺序从 config/tasks/*.yaml 构建 Task 实例。

    Args:
        task_names: 只构建指定任务（直接媒体模式拆分创意/发布两段用）；
            缺省构建全流程。
    """
    tasks_dir = config_dir / "tasks"
    specs: dict[str, dict[str, Any]] = {}
    for path in sorted(tasks_dir.glob("*.yaml")):
        name, spec = next(iter(_load_yaml(path).items()))
        specs[name] = spec

    order = task_names if task_names is not None else TASK_ORDER
    unknown = [name for name in order if name not in specs]
    if unknown:
        raise ValueError(f"未知任务: {unknown}（可选: {', '.join(TASK_ORDER)}）")

    tasks: list[Task] = []
    style_hint = storyboard_style_hint()
    for name in order:
        spec = specs[name]
        agent_name = spec["agent"]
        if agent_name not in agents:
            raise ValueError(f"任务 {name} 引用了未知 agent: {agent_name}")
        description = spec["description"]
        if style_hint and name in STYLE_CONSTRAINED_TASKS:
            # 全片风格固定（PROMPT2VID_STYLE）：生成阶段在提示词层还会
            # 硬性追加风格词，这里让分镜 LLM 写画面描述时就与之对齐
            description += style_hint
        tasks.append(
            Task(
                description=description,
                expected_output=spec["expected_output"],
                agent=agents[agent_name],
            )
        )
    return tasks


def build_crew(task_names: list[str] | None = None) -> Crew:
    """构建流水线 Crew。

    Args:
        task_names: 只包含指定任务（用于直接媒体模式分段执行）；
            缺省构建全流程（9 任务）。
    """
    # 幂等：CLI 已用 --env-file 指定档位时保持不变，否则按
    # PROMPT2VID_ENV_FILE / PROMPT2VID_PROFILE / 默认 .env 解析
    load_env()
    _bypass_proxy_for_internal_hosts()
    config_dir = _find_config_dir()
    agents = build_agents(config_dir)
    tasks = build_tasks(config_dir, agents, task_names)
    used_agents = list(dict.fromkeys(task.agent for task in tasks))
    return Crew(
        agents=used_agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )
