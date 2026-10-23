"""分步执行：9 步注册表 + 单步执行器 + steps.json 状态落盘。

调试用：不必整条流水线跑到底，可以执行/重跑任意单个 agent 步骤。
- 创意步骤（需求/剧本/分镜）与发布步骤：单任务 crew（LLM）执行；
- 媒体步骤（生图/生视频/配音/渲染/质检）：直接调用 stages.py 的
  确定性函数——与该步骤 agent 转调的工具是同一实现，省一次 LLM
  往返（与直接媒体模式同一思路）。

执行状态写入 <run_dir>/steps.json（status/起止时间/摘要/错误），
产物存在性作为补充信号（CLI 全流程跑出的历史 run 也能被 Web
控制台识别为"已完成"）。CLI --only 与 Web 控制台共用本模块。
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .pipeline import _STAGE_FUNCS, setup_logging
from .stages import (
    AUDIO_DIR,
    CLIPS_DIR,
    FINAL_DIR,
    FRAMES_DIR,
    REPORT_DIR,
)
from .workspace import set_workspace

logger = logging.getLogger("prompt2vid.steps")

STEPS_FILE = "steps.json"
META_FILE = "meta.json"

# 媒体阶段 → 产物目录（--force 强制重跑时整目录删除）
STAGE_OUTPUT_DIRS: dict[str, str] = {
    "frames": FRAMES_DIR,
    "clips": CLIPS_DIR,
    "voiceovers": AUDIO_DIR,
    "render": FINAL_DIR,
    "quality": REPORT_DIR,
}

# 9 步注册表（顺序 = crew.TASK_ORDER；id 即 task 名）
STEPS: list[dict[str, Any]] = [
    {
        "id": "analyze_requirement",
        "label": "需求分析",
        "agent": "requirement_analyst",
        "kind": "creative",
        "stage": None,
        "artifact": "requirement.json",
        "requires": [],
        "description": "解析一句话描述，生成 requirement.json（LLM）",
    },
    {
        "id": "write_script",
        "label": "剧本撰写",
        "agent": "script_writer",
        "kind": "creative",
        "stage": None,
        "artifact": "script.md",
        "requires": ["requirement.json"],
        "description": "基于需求撰写剧本 script.md（LLM）",
    },
    {
        "id": "plan_storyboard",
        "label": "分镜规划",
        "agent": "storyboard_planner",
        "kind": "creative",
        "stage": None,
        "artifact": "storyboard.json",
        "requires": ["script.md"],
        "description": "把剧本拆成分镜表 storyboard.json（LLM）",
    },
    {
        "id": "generate_images",
        "label": "生成分镜图",
        "agent": "image_generator",
        "kind": "media",
        "stage": "frames",
        "artifact": f"{FRAMES_DIR}/manifest.json",
        "requires": ["storyboard.json"],
        "description": "逐镜 SDXL 文生图 → frames/（直跑 stages，不耗 LLM）",
    },
    {
        "id": "generate_videos",
        "label": "生成视频片段",
        "agent": "video_generator",
        "kind": "media",
        "stage": "clips",
        "artifact": f"{CLIPS_DIR}/manifest.json",
        "requires": [f"{FRAMES_DIR}/manifest.json"],
        "description": "逐镜 LTX 图生视频 → clips/（直跑 stages，不耗 LLM）",
    },
    {
        "id": "generate_voiceover",
        "label": "生成配音",
        "agent": "voiceover_artist",
        "kind": "media",
        "stage": "voiceovers",
        "artifact": f"{AUDIO_DIR}/manifest.json",
        "requires": ["storyboard.json"],
        "description": "逐镜中文配音 → audio/（直跑 stages，不耗 LLM）",
    },
    {
        "id": "edit_video",
        "label": "剪辑合成",
        "agent": "video_editor",
        "kind": "media",
        "stage": "render",
        "artifact": f"{FINAL_DIR}/final.mp4",
        "requires": [f"{CLIPS_DIR}/manifest.json"],
        "description": "拼接/字幕/混流 → final/final.mp4（直跑 stages，不耗 LLM）",
    },
    {
        "id": "inspect_quality",
        "label": "质量检查",
        "agent": "quality_inspector",
        "kind": "media",
        "stage": "quality",
        "artifact": f"{REPORT_DIR}/quality_report.json",
        "requires": [f"{FINAL_DIR}/final.mp4"],
        "description": "成片与产物核验 → report/quality_report.json（直跑 stages）",
    },
    {
        "id": "publish_douyin",
        "label": "发布抖音",
        "agent": "douyin_publisher",
        "kind": "publish",
        "stage": None,
        "artifact": None,
        "requires": [f"{REPORT_DIR}/quality_report.json"],
        "description": "质检通过后发布到抖音（LLM）",
    },
]

STEP_IDS: list[str] = [s["id"] for s in STEPS]
STEP_INDEX: dict[str, int] = {s["id"]: i for i, s in enumerate(STEPS)}
STEP_BY_ID: dict[str, dict[str, Any]] = {s["id"]: s for s in STEPS}

KIND_LABELS: dict[str, str] = {
    "creative": "创意",
    "media": "媒体",
    "publish": "发布",
}


def catalog() -> list[dict[str, Any]]:
    """步骤目录（含序号与中文类型名），供 Web 控制台 / --list-steps 使用。"""
    return [
        {**step, "index": i + 1, "kind_label": KIND_LABELS[step["kind"]]}
        for i, step in enumerate(STEPS)
    ]


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- meta.json

def write_meta(run_dir: str | Path, prompt: str) -> None:
    """记录运行的一句话描述（Web 控制台展示与单步补跑 analyze 时使用）。"""
    run_dir = Path(run_dir)
    meta: dict[str, Any] = {}
    meta_path = run_dir / META_FILE
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = {}
    meta["prompt"] = prompt
    meta.setdefault("created_at", _now())
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_meta(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / META_FILE
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def read_prompt(run_dir: str | Path) -> str:
    return str(read_meta(run_dir).get("prompt") or "")


# -------------------------------------------------------------- steps.json

def _load_states(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / STEPS_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _update_state(run_dir: Path, step_id: str, **fields: Any) -> None:
    states = _load_states(run_dir)
    entry = states.get(step_id, {})
    entry.update(fields)
    states[step_id] = entry
    (run_dir / STEPS_FILE).write_text(
        json.dumps(states, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def step_statuses(
    run_dir: str | Path, active: bool = False
) -> dict[str, dict[str, Any]]:
    """读取每步状态。

    steps.json 的记录优先；没有记录但产物已存在的步骤视为 done
    （CLI 全流程跑出的历史 run 也能识别）；active=False 时残留的
    running 状态视为 interrupted（服务重启/进程退出导致的中断）。
    """
    run_dir = Path(run_dir)
    states = _load_states(run_dir)
    out: dict[str, dict[str, Any]] = {}
    for step in STEPS:
        sid = step["id"]
        entry = dict(states.get(sid, {}))
        status = entry.get("status")
        if status == "running" and not active:
            entry["status"] = "interrupted"
            entry.setdefault("error", "执行中断（服务重启或进程退出）")
        if status is None and step["artifact"]:
            if (run_dir / step["artifact"]).is_file():
                entry["status"] = "done"
                entry.setdefault("summary", "产物已存在（历史运行生成）")
        out[sid] = entry
    return out


# ---------------------------------------------------------------- 执行器

def _clean_stage_output(run_dir: Path, stage: str) -> None:
    """强制重跑：删除媒体阶段的产物目录，已生成的文件不再跳过。"""
    target = run_dir / STAGE_OUTPUT_DIRS[stage]
    if target.is_dir():
        shutil.rmtree(target)
        logger.info("强制重跑：已删除 %s", target)


def run_step(
    run_dir: str | Path,
    step_id: str,
    prompt: str = "",
    force: bool = False,
) -> dict[str, Any]:
    """执行单个步骤，返回 {step, label, status, duration_seconds, summary, error}。

    媒体步骤失败以返回值里的 error 字段呈现（阶段函数不抛异常），
    创意/发布步骤的 crew 异常与"未写出期望产物"同样归一为 failed，
    单步失败不影响调用方继续执行其他步骤。
    """
    step = STEP_BY_ID.get(step_id)
    if step is None:
        raise ValueError(f"未知步骤: {step_id}（可选: {', '.join(STEP_IDS)}）")
    run_dir = set_workspace(run_dir)
    setup_logging(run_dir)

    began = time.monotonic()
    logger.info("步骤 %s（%s）开始", step_id, step["label"])
    summary = ""
    error: str | None = None
    try:
        if step["kind"] == "media":
            if force:
                _clean_stage_output(run_dir, step["stage"])
            result = _STAGE_FUNCS[step["stage"]](str(run_dir))
            summary = json.dumps(result, ensure_ascii=False)[:400]
            if "error" in result:
                error = str(result["error"])
        else:
            if force and step["artifact"]:
                artifact = run_dir / step["artifact"]
                if artifact.is_file():
                    artifact.unlink()
                    logger.info("强制重跑：已删除 %s", artifact)
            from .crew import build_crew

            crew = build_crew(task_names=[step_id])
            output = crew.kickoff(
                inputs={"user_prompt": prompt, "workspace": str(run_dir)}
            )
            summary = str(output).strip()[:400]
            if step["artifact"]:
                artifact = run_dir / step["artifact"]
                if not artifact.is_file():
                    error = f"未生成期望产物 {step['artifact']}（LLM 可能没有调用写文件工具）"
    except Exception as exc:  # noqa: BLE001 - 单步失败以状态返回而非中断调用方
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("步骤 %s 异常", step_id)

    status = "done" if error is None else "failed"
    duration = round(time.monotonic() - began, 1)
    logger.info(
        "步骤 %s %s (%.1fs)%s", step_id, "完成" if status == "done" else "失败",
        duration, f": {error}" if error else "",
    )
    return {
        "step": step_id,
        "label": step["label"],
        "status": status,
        "duration_seconds": duration,
        "summary": summary or None,
        "error": error,
    }


def run_steps(
    run_dir: str | Path,
    step_ids: list[str] | None = None,
    prompt: str | None = None,
    force: bool = False,
    cancel_event: threading.Event | None = None,
    on_event: Callable[[str, str], None] | None = None,
) -> list[dict[str, Any]]:
    """按流水线顺序执行指定步骤（缺省全部），失败即停。

    每步执行前后更新 steps.json；cancel_event 在步骤间检查，
    当前步骤（如 ComfyUI 批量生成）跑完后停止。
    """
    run_dir = Path(run_dir).resolve()
    if step_ids is None:
        step_ids = list(STEP_IDS)
    unknown = [i for i in step_ids if i not in STEP_INDEX]
    if unknown:
        raise ValueError(f"未知步骤: {unknown}（可选: {', '.join(STEP_IDS)}）")
    if prompt is None:
        prompt = read_prompt(run_dir)

    ordered = sorted(set(step_ids), key=lambda i: STEP_INDEX[i])
    results: list[dict[str, Any]] = []
    for step_id in ordered:
        if cancel_event is not None and cancel_event.is_set():
            logger.warning("已请求取消，停止执行（已完成 %d 步）", len(results))
            break
        _update_state(run_dir, step_id, status="running", started_at=_now())
        if on_event:
            on_event(step_id, "start")
        result = run_step(run_dir, step_id, prompt=prompt, force=force)
        _update_state(
            run_dir,
            step_id,
            status=result["status"],
            finished_at=_now(),
            summary=result.get("summary"),
            error=result.get("error"),
        )
        if on_event:
            on_event(step_id, result["status"])
        results.append(result)
        if result["status"] != "done":
            logger.error(
                "步骤 %s 失败，停止后续步骤: %s", step_id, result.get("error")
            )
            break
    return results
