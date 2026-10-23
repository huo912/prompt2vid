"""Web 调试控制台：REST API + 后台执行管理 + 静态页面。

挂在 server.py 的同一个 FastAPI 应用上（默认端口 8081），与 A2A
路由共存互不影响：
- GET /        → 调试页面（prompt2vid/web/index.html）
- /static/*    → 页面静态资源
- /api/*       → 步骤目录 / 运行管理 / 单步执行 / 日志 / 产物

执行模型：POST /api/runs/{id}/steps 在后台线程执行 steps.run_steps，
前端 2s 轮询 run 详情与 run.log。EXECUTION_LOCK 是进程级执行锁，
与 A2A 执行共用——PROMPT2VID_WORKSPACE 是进程级环境变量，任何
并发执行都会串目录，因此全进程同时只允许一个执行流。
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import steps

# 运行根目录（与 main.py / server.py 的 output/ 一致，相对服务启动目录）
OUTPUT_BASE = Path("output")
WEB_DIR = Path(__file__).resolve().parent / "web"

# 进程级执行锁（server.py 的 A2A 执行同样持有此锁）
EXECUTION_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class RunManager:
    """后台线程执行 steps.run_steps，同一时刻仅一个执行流。"""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._guard = threading.Lock()
        self.current: dict[str, Any] | None = None

    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        run_dir: Path,
        run_id: str,
        step_ids: list[str],
        force: bool = False,
        prompt: str | None = None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """启动执行；busy 时返回 (None, 原因)。"""
        with self._guard:
            if self.active():
                cur = self.current or {}
                return None, (
                    f"已有任务在执行（{cur.get('run_id', '?')}），"
                    "请等待完成或先取消"
                )
            record: dict[str, Any] = {
                "run_id": run_id,
                "steps": list(step_ids),
                "force": force,
                "started_at": _now(),
                "cancel_requested": False,
                "cancel_event": threading.Event(),
                "finished_at": None,
                "error": None,
            }
            self.current = record
            self._thread = threading.Thread(
                target=self._worker,
                args=(run_dir, step_ids, force, prompt, record),
                name=f"p2v-run-{run_id}",
                daemon=True,
            )
            self._thread.start()
            return self.snapshot_record(record), None

    def _worker(
        self,
        run_dir: Path,
        step_ids: list[str],
        force: bool,
        prompt: str | None,
        record: dict[str, Any],
    ) -> None:
        # A2A 请求可能正占着锁（UI 与 A2A 共用进程）——立即让位
        if not EXECUTION_LOCK.acquire(blocking=False):
            record["error"] = "A2A 请求正在执行流水线，请稍后重试"
            record["finished_at"] = _now()
            return
        try:
            steps.run_steps(
                run_dir,
                step_ids,
                prompt=prompt,
                force=force,
                cancel_event=record["cancel_event"],
            )
        except Exception as exc:  # noqa: BLE001 - 失败原因回传给轮询方
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            EXECUTION_LOCK.release()
            record["finished_at"] = _now()

    def cancel(self, run_id: str) -> str | None:
        """请求取消（当前步骤完成后停止）；返回错误原因或 None。"""
        if not self.active() or not self.current or self.current["run_id"] != run_id:
            return "该运行当前没有正在执行的任务"
        self.current["cancel_requested"] = True
        self.current["cancel_event"].set()
        return None

    def snapshot_record(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v for k, v in record.items() if k != "cancel_event"
        }

    def snapshot(self) -> dict[str, Any] | None:
        """当前/最近一次执行的信息（可 JSON 序列化）。"""
        if self.current is None:
            return None
        snap = self.snapshot_record(self.current)
        snap["active"] = self.active()
        return snap

    def wait(self, timeout: float = 10.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)


manager = RunManager()

# ---------------------------------------------------------------- 路由

router = APIRouter(prefix="/api")


class CreateRunRequest(BaseModel):
    prompt: str = Field(min_length=1)
    auto_run: list[str] | None = None  # 步骤 id 列表，可含 "all"


class RunStepsRequest(BaseModel):
    steps: list[str] = Field(min_length=1)
    force: bool = False
    prompt: str | None = None


def _runs_base() -> Path:
    return OUTPUT_BASE


def _resolve_run(run_id: str) -> Path:
    """按目录名定位运行目录；拒绝任何带路径分隔符的输入。"""
    if not run_id or Path(run_id).name != run_id or run_id.startswith("."):
        raise HTTPException(status_code=404, detail=f"运行不存在: {run_id}")
    run_dir = _runs_base() / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"运行不存在: {run_id}")
    return run_dir


@router.get("/steps")
def get_steps() -> dict:
    return {"steps": steps.catalog()}


@router.get("/runs")
def list_runs() -> dict:
    base = _runs_base()
    snap = manager.snapshot()
    items: list[dict] = []
    if base.is_dir():
        run_dirs = sorted(
            (d for d in base.glob("run_*") if d.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for run_dir in run_dirs[:100]:
            meta = steps.read_meta(run_dir)
            statuses = steps.step_statuses(run_dir)
            values = [v.get("status") for v in statuses.values()]
            items.append(
                {
                    "id": run_dir.name,
                    "prompt": meta.get("prompt", ""),
                    "created_at": meta.get("created_at"),
                    "done": values.count("done"),
                    "failed": values.count("failed") + values.count("interrupted"),
                    "total": len(steps.STEPS),
                    "running": bool(
                        snap and snap["active"] and snap["run_id"] == run_dir.name
                    ),
                }
            )
    return {"runs": items}


@router.post("/runs", status_code=201)
def create_run(req: CreateRunRequest) -> dict:
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt 不能为空")
    base = _runs_base()
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{stamp}"
    n = 1
    while (base / run_id).exists():
        run_id = f"run_{stamp}_{n}"
        n += 1
    run_dir = base / run_id
    run_dir.mkdir(parents=True)
    steps.write_meta(run_dir, prompt)

    started = False
    warning: str | None = None
    if req.auto_run:
        ids = steps.STEP_IDS if "all" in req.auto_run else req.auto_run
        unknown = [i for i in ids if i != "all" and i not in steps.STEP_INDEX]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"未知步骤: {unknown}（可选: {', '.join(steps.STEP_IDS)}）",
            )
        _, err = manager.start(
            run_dir.resolve(), run_id, list(ids), prompt=prompt
        )
        if err:
            warning = err
        else:
            started = True
    return {"run_id": run_id, "started": started, "warning": warning}


@router.get("/runs/{run_id}")
def run_detail(run_id: str) -> dict:
    run_dir = _resolve_run(run_id)
    snap = manager.snapshot()
    executing_here = bool(snap and snap["active"] and snap["run_id"] == run_id)
    statuses = steps.step_statuses(run_dir, active=executing_here)

    current_step = None
    step_rows: list[dict] = []
    for entry in steps.catalog():
        state = statuses.get(entry["id"], {})
        if state.get("status") == "running":
            current_step = entry["id"]
        step_rows.append(
            {
                **entry,
                **state,
                "artifact_exists": bool(
                    entry["artifact"]
                    and (run_dir / entry["artifact"]).is_file()
                ),
            }
        )

    execution = None
    if snap and snap["active"]:
        execution = {**snap, "current_step": current_step if executing_here else None}

    final = run_dir / "final" / "final.mp4"
    return {
        "id": run_id,
        "dir": str(run_dir.resolve()),
        "meta": steps.read_meta(run_dir),
        "steps": step_rows,
        "execution": execution,
        "final_video": (
            f"/api/runs/{run_id}/files/final/final.mp4" if final.is_file() else None
        ),
    }


@router.post("/runs/{run_id}/steps", status_code=202)
def execute_steps(run_id: str, req: RunStepsRequest) -> dict:
    run_dir = _resolve_run(run_id)
    ids = [s.strip() for s in req.steps if s.strip()]
    unknown = [i for i in ids if i not in steps.STEP_INDEX]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"未知步骤: {unknown}（可选: {', '.join(steps.STEP_IDS)}）",
        )
    prompt = None
    if req.prompt is not None and req.prompt.strip():
        prompt = req.prompt.strip()
        # 持久化到 meta.json，后续单步补跑也能拿到
        steps.write_meta(run_dir, prompt)
    _, err = manager.start(run_dir.resolve(), run_id, ids, force=req.force, prompt=prompt)
    if err:
        raise HTTPException(status_code=409, detail=err)
    return {"started": True, "steps": ids, "force": req.force}


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict:
    _resolve_run(run_id)
    err = manager.cancel(run_id)
    if err:
        raise HTTPException(status_code=409, detail=err)
    return {"ok": True, "message": "已请求取消，当前步骤完成后停止"}


@router.get("/runs/{run_id}/log")
def run_log(run_id: str, tail: int = Query(default=300, ge=1, le=2000)) -> dict:
    run_dir = _resolve_run(run_id)
    path = run_dir / "run.log"
    if not path.is_file():
        return {"lines": [], "size": 0}
    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    return {"lines": text.splitlines()[-tail:], "size": len(data)}


@router.get("/runs/{run_id}/files/{file_path:path}")
def run_file(run_id: str, file_path: str) -> FileResponse:
    run_dir = _resolve_run(run_id).resolve()
    target = (run_dir / file_path).resolve()
    if not target.is_relative_to(run_dir):
        raise HTTPException(status_code=403, detail="路径越界")
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"文件不存在: {file_path}")
    return FileResponse(target)


def register_webui(app: FastAPI) -> None:
    """把调试控制台（页面 + /api）挂到 FastAPI 应用上。"""
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="webui")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")
