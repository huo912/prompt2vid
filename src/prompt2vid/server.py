"""A2A 服务器：以 a2a-sdk 暴露 prompt2vid Crew，供 kagent 调用。

同时挂载 Web 调试控制台（GET / 页面 + /api/*，见 webui.py），与
A2A 路由共存于同一端口。

端口选择：KAGENT_PORT 环境变量（kagent harness 注入）优先，其次
PROMPT2VID_PORT，默认 8081。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
    create_rest_routes,
)
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Part,
    Task,
    TaskState,
    TaskStatus,
)
from fastapi import FastAPI

DEFAULT_PORT = 8081


def _resolve_port() -> int:
    for var in ("KAGENT_PORT", "PROMPT2VID_PORT"):
        value = os.getenv(var, "").strip()
        if value:
            try:
                return int(value)
            except ValueError:
                continue
    return DEFAULT_PORT


def _load_agent_card() -> AgentCard:
    card_path = Path(__file__).resolve().parent / "agent-card.json"
    with card_path.open("r", encoding="utf-8") as f:
        import json

        data = json.load(f)
    return AgentCard(
        name=data["name"],
        description=data["description"],
        version=data["version"],
        provider=AgentProvider(
            organization=data["provider"]["organization"],
            url=data["provider"].get("url"),
        ),
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=data.get("default_input_modes", ["text"]),
        default_output_modes=data.get("default_output_modes", ["text", "task-status"]),
        skills=[
            AgentSkill(
                id=skill["id"],
                name=skill["name"],
                description=skill["description"],
                tags=skill.get("tags", []),
                examples=skill.get("examples", []),
                input_modes=skill.get("input_modes", ["text"]),
                output_modes=skill.get("output_modes", ["text", "task-status"]),
            )
            for skill in data.get("skills", [])
        ],
        supported_interfaces=[
            AgentInterface(
                protocol_binding=iface["protocol_binding"],
                protocol_version=iface["protocol_version"],
                url=iface.get("url"),
            )
            for iface in data.get("supported_interfaces", [])
        ],
    )


class Prompt2VidExecutor(AgentExecutor):
    """把 A2A 请求转交给 Crew 流水线执行。"""

    def __init__(self) -> None:
        self.running_tasks: set[str] = set()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        context_id = context.context_id
        if (
            not task_id
            or not context_id
            or not context.message
        ):
            return
        self.running_tasks.add(task_id)

        user_message = context.message
        await event_queue.enqueue_event(
            Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                history=[user_message],
            )
        )
        updater = TaskUpdater(event_queue=event_queue, task_id=task_id, context_id=context_id)

        try:
            await updater.start_work(
                message=updater.new_agent_message(
                    parts=[Part(text="已收到视频生成请求，开始执行流水线…")]
                )
            )
            query = context.get_user_input()

            run_dir = Path("output") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_dir.mkdir(parents=True, exist_ok=True)

            from .pipeline import run_pipeline
            from .webui import EXECUTION_LOCK

            def _run() -> Any:
                # 与 Web 调试控制台共用进程级执行锁：工作目录经
                # PROMPT2VID_WORKSPACE 进程级环境变量传递，任何并发执行
                # 都会互相覆盖；GPU 生成本身也不适合并行
                with EXECUTION_LOCK:
                    return run_pipeline(query, run_dir)

            result = await asyncio.to_thread(_run)

            await updater.add_artifact(
                parts=[Part(text=str(result))],
                name="video_generation_result",
                last_chunk=True,
            )
            await updater.complete()
        except Exception as exc:  # noqa: BLE001 - A2A 需要把失败回传给调用方
            await updater.failed(
                message=updater.new_agent_message(
                    parts=[Part(text=f"流水线执行失败: {exc}")]
                )
            )
        finally:
            self.running_tasks.discard(task_id)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        if task_id:
            self.running_tasks.discard(task_id)
            updater = TaskUpdater(
                event_queue=event_queue,
                task_id=task_id,
                context_id=context.context_id,
            )
            await updater.cancel()


def serve(host: str = "0.0.0.0", port: int | None = None, env_file: str | None = None) -> None:
    """启动 A2A 服务器。"""
    if port is None:
        port = _resolve_port()

    # 环境配置：--env-file 显式指定，否则 PROMPT2VID_ENV_FILE /
    # PROMPT2VID_PROFILE（k8s/systemd 注入），最后默认 .env
    from .env_config import load_env

    load_env(env_file)

    agent_card = _load_agent_card()
    request_handler = DefaultRequestHandler(
        agent_executor=Prompt2VidExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    agent_card_routes = create_agent_card_routes(agent_card=agent_card)
    jsonrpc_routes = create_jsonrpc_routes(
        request_handler=request_handler,
        rpc_url="/a2a/jsonrpc",
        enable_v0_3_compat=True,
    )
    rest_routes = create_rest_routes(
        request_handler=request_handler,
        path_prefix="/a2a/rest",
        enable_v0_3_compat=True,
    )

    app = FastAPI(title="prompt2vid A2A server")

    # Web 调试控制台：GET / 页面 + /static/* 资源 + /api/*
    # 必须先于 A2A 路由注册：a2a-sdk 的 rest 路由含租户 Mount /{tenant}，
    # 会吞掉其后注册的单段前缀路径（/api/*、/static/* 全部 404）
    from . import webui

    webui.register_webui(app)
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=agent_card_routes,
        jsonrpc_routes=jsonrpc_routes,
        rest_routes=rest_routes,
    )

    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


def main() -> None:
    parser = argparse.ArgumentParser(prog="prompt2vid-server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--env-file",
        default=None,
        metavar="FILE",
        help="指定环境配置文件（路径或档位简名，如 .env.4090 或 4090；默认 .env）",
    )
    args = parser.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        serve(host=args.host, port=args.port, env_file=args.env_file)


if __name__ == "__main__":
    main()
