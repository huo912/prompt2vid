"""Crew 构建：从 config/ 目录加载 agent 与 task YAML 并组装流水线。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv

from .dynamic_llm import build_llm

# agent 名 → (agent 模块, tools 属性名列表)
# tools 属性名对应 agents/<name>.py 中定义的模块级变量
AGENT_TOOL_ATTRS: dict[str, list[str]] = {
    "requirement_analyst": [],
    "script_writer": [],
    "storyboard_planner": [],
    "image_generator": ["comfyui_tool"],
    "video_generator": ["comfyui_tool"],
    "voiceover_artist": ["tts_tool"],
    "video_editor": ["ffmpeg_tool"],
    "quality_inspector": ["file_tool"],
    "douyin_publisher": ["douyin_tool"],
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
    """从 agents/<name>.py 模块收集工具实例。"""
    if agent_name not in AGENT_TOOL_ATTRS:
        raise KeyError(f"未知 agent: {agent_name}")
    tools: list = []
    if not AGENT_TOOL_ATTRS[agent_name]:
        return tools
    from importlib import import_module

    for attr in AGENT_TOOL_ATTRS[agent_name]:
        module = import_module(f"prompt2vid.tools.{attr}")
        tools.extend(getattr(module, "TOOLS"))
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


def build_tasks(config_dir: Path, agents: dict[str, Agent]) -> list[Task]:
    """按流水线顺序从 config/tasks/*.yaml 构建 Task 实例。"""
    tasks_dir = config_dir / "tasks"
    specs: dict[str, dict[str, Any]] = {}
    for path in sorted(tasks_dir.glob("*.yaml")):
        name, spec = next(iter(_load_yaml(path).items()))
        specs[name] = spec

    tasks: list[Task] = []
    for name in TASK_ORDER:
        if name not in specs:
            raise ValueError(f"缺少任务配置: config/tasks/{name}.yaml")
        spec = specs[name]
        agent_name = spec["agent"]
        if agent_name not in agents:
            raise ValueError(f"任务 {name} 引用了未知 agent: {agent_name}")
        tasks.append(
            Task(
                description=spec["description"],
                expected_output=spec["expected_output"],
                agent=agents[agent_name],
            )
        )
    return tasks


def build_crew() -> Crew:
    """构建完整流水线 Crew。"""
    load_dotenv()
    _bypass_proxy_for_internal_hosts()
    config_dir = _find_config_dir()
    agents = build_agents(config_dir)
    tasks = build_tasks(config_dir, agents)
    return Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )
