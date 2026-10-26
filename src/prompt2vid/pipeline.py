"""流水线编排：Crew 全流程模式 / 直接媒体模式 / 断点续跑 / 输出清理。

两种执行模式：
- crew 模式（默认）：9 个 LLM agent 顺序执行全流程（A2A/kagent 形态）；
- 直接媒体模式（PROMPT2VID_DIRECT_MEDIA=1 或 CLI --direct-media）：
  只用 LLM 做创意三阶段（需求→剧本→分镜），生图/生视频/配音/渲染/质检
  由 stages.py 确定性代码直接执行，发布再由一次 LLM 调用完成——省去 6 次
  LLM 往返（更快、更省 token、更少故障面）。

统一日志：每次运行的日志写入 <工作目录>/run.log（阶段开始/结束、
耗时、失败原因），排障不再只依赖 stdout。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from . import stages
from .llm_output import salvage_payload
from .workspace import set_workspace

# 直接媒体模式下的 LLM 任务拆分
CREATIVE_TASKS = ["analyze_requirement", "write_script", "plan_storyboard"]
PUBLISH_TASK = "publish_douyin"

# 媒体阶段（顺序）
MEDIA_STAGES = ["frames", "clips", "voiceovers", "render", "quality"]
_STAGE_FUNCS: dict[str, Any] = {
    "frames": stages.generate_frames,
    "clips": stages.generate_clips,
    "voiceovers": stages.generate_voiceovers,
    "render": stages.render_final,
    "quality": stages.run_quality_checks,
}

logger = logging.getLogger("prompt2vid.pipeline")

# 记录当前挂载的 run.log FileHandler，切换运行目录时替换（server 多请求）
_FILE_HANDLERS: list[logging.FileHandler] = []


def direct_media_enabled() -> bool:
    """是否启用直接媒体模式（PROMPT2VID_DIRECT_MEDIA 真值）。"""
    return os.getenv("PROMPT2VID_DIRECT_MEDIA", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def setup_logging(run_dir: str | Path) -> logging.Logger:
    """把本次运行的日志写入 run_dir/run.log（幂等，可跨请求切换目录）。"""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s", "%H:%M:%S"
    )

    # 替换上一次运行挂的文件 handler（server 场景）
    for handler in _FILE_HANDLERS:
        root.removeHandler(handler)
        handler.close()
    _FILE_HANDLERS.clear()

    file_handler = logging.FileHandler(
        Path(run_dir) / "run.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    _FILE_HANDLERS.append(file_handler)

    # 控制台 handler 只挂一次
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    )
    if not has_console:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

    return logger


def run_media_stages(start: str = "frames") -> dict:
    """按顺序执行媒体阶段，返回最后一个阶段的汇总。

    某阶段返回 error 即停止后续阶段（质检除外——渲染失败后仍跑质检，
    让质检报告如实反映失败原因）。
    """
    if start not in MEDIA_STAGES:
        raise ValueError(f"未知阶段: {start}（可选: {', '.join(MEDIA_STAGES)}）")
    last: dict = {}
    for name in MEDIA_STAGES[MEDIA_STAGES.index(start):]:
        began = time.monotonic()
        logger.info("阶段 %s 开始", name)
        last = _STAGE_FUNCS[name]()
        if "error" in last:
            logger.error("阶段 %s 失败 (%.1fs): %s",
                         name, time.monotonic() - began, last["error"])
            if name != "render":
                break
            # 渲染失败仍执行质检，报告如实记录
            quality = _STAGE_FUNCS["quality"]()
            if "error" not in quality:
                last = quality
            break
        logger.info("阶段 %s 完成 (%.1fs): %s", name, time.monotonic() - began,
                    json.dumps(last, ensure_ascii=False)[:400])
    return last


def _ensure_creative_artifact(run_dir: Path, artifact: str, output: str) -> None:
    """创意产物缺失时从 LLM 最终回答抢救写入（best-effort，不抛异常）。

    直连模式下三个创意任务在同一次 crew 里执行，kickoff 只返回最后一个
    （分镜）任务的回答，因此这里只能救 storyboard.json——它恰好是媒体
    阶段唯一的创意依赖，缺了它整条链路都无法启动。
    """
    path = run_dir / artifact
    if path.is_file() or not output:
        return
    payload = salvage_payload(artifact, output)
    if payload is None:
        return
    path.write_text(payload, encoding="utf-8")
    logger.warning("LLM 未写出 %s，已从最终回答中恢复（建议换用支持工具调用的模型）", artifact)


def run_pipeline(
    prompt: str,
    run_dir: str | Path,
    direct_media: bool | None = None,
) -> Any:
    """执行完整流水线，返回最终结果（crew 模式为 CrewOutput，直连模式为发布结果）。"""
    run_dir = set_workspace(run_dir)
    setup_logging(run_dir)
    from .crew import build_crew

    direct = direct_media_enabled() if direct_media is None else direct_media
    inputs = {"user_prompt": prompt, "workspace": str(run_dir)}

    if not direct:
        logger.info("以 crew 模式执行（9 agent 全流程）")
        crew = build_crew()
        return crew.kickoff(inputs=inputs)

    logger.info("以直接媒体模式执行（LLM 仅创意三阶段 + 发布）")
    creative = build_crew(task_names=CREATIVE_TASKS)
    output = creative.kickoff(inputs=inputs)
    _ensure_creative_artifact(run_dir, "storyboard.json", str(output))
    logger.info("创意阶段完成，storyboard: %s", run_dir / "storyboard.json")

    run_media_stages()

    publish = build_crew(task_names=[PUBLISH_TASK])
    return publish.kickoff(inputs=inputs)


def run_resume(
    run_dir: str | Path,
    from_stage: str = "frames",
    publish: bool = False,
) -> Any:
    """断点续跑：从指定媒体阶段继续（创意产物复用工作目录里已有的）。

    已生成的分镜帧/片段/音频会被自动跳过，只补缺失与后续阶段。
    """
    run_dir = set_workspace(run_dir)
    setup_logging(run_dir)
    logger.info("断点续跑 %s，从 %s 阶段开始", run_dir, from_stage)
    last = run_media_stages(from_stage)

    if not publish:
        return last
    if "error" in last:
        logger.warning("媒体阶段存在失败，跳过发布")
        return last
    from .crew import build_crew

    publish_crew = build_crew(task_names=[PUBLISH_TASK])
    return publish_crew.kickoff(inputs={"user_prompt": "", "workspace": str(run_dir)})


def clean_outputs(keep: int = 5) -> list[Path]:
    """清理 output/run_* 旧运行目录，保留最近 keep 个；返回已删除列表。"""
    base = Path("output")
    if not base.is_dir():
        return []
    runs = sorted(
        (d for d in base.glob("run_*") if d.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    removed = runs[max(0, keep):]
    for run_dir in removed:
        shutil.rmtree(run_dir, ignore_errors=True)
    return removed


def trim_publish_state(keep: int = 20) -> int:
    """截断 .douyin/state.json 的 published 历史，保留最近 keep 条。"""
    state_path = Path(os.getenv("DOUYIN_STATE_FILE", "").strip() or Path(".douyin") / "state.json")
    if not state_path.is_file():
        return 0
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    published = state.get("published")
    if not isinstance(published, list) or len(published) <= keep:
        return 0
    state["published"] = published[-keep:]
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(published) - keep
