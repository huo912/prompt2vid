"""CLI 入口：python -m prompt2vid.main "一句话描述"。

执行流程：
1. 加载 .env，解析参数；
2. 预检环境（ComfyUI 可达、模型就位、ffmpeg 可用）——把配置错误
   拦截在启动阶段，而不是流水线跑到一半才失败；
3. 创建 output/run_<时间戳> 工作目录并写入 PROMPT2VID_WORKSPACE；
4. 执行流水线（crew 全流程 / 直接媒体模式）；
5. 汇总产物路径（成片 / 质检报告 / 发布状态）。

常用命令：
    prompt2vid "做一个关于秋天的治愈系短视频"          # 全流程
    prompt2vid "..." --direct-media                    # 直连媒体模式（省 6 次 LLM 往返）
    prompt2vid --check-only                            # 只做环境预检
    prompt2vid --resume output/run_xxx --from render   # 断点续跑
    prompt2vid --clean --keep 5                        # 清理旧运行目录
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def _prepare_workspace() -> Path:
    """创建本次运行的工作目录 output/run_<时间戳>。"""
    base = Path("output").resolve()
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _print_run_summary(run_dir: Path) -> None:
    """打印本次运行的产物汇总。"""
    final = run_dir / "final" / "final.mp4"
    report = run_dir / "report" / "quality_report.json"
    state = Path(".douyin") / "state.json"

    print("\n===== 运行结果 =====")
    print(f"工作目录: {run_dir}")
    print(f"运行日志: {run_dir / 'run.log'}")
    print(f"成片: {final}{'（已生成）' if final.is_file() else '（未生成）'}")
    if report.is_file():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            print(f"质检: {'通过' if data.get('passed') else '未通过'}（{report}）")
        except (json.JSONDecodeError, OSError):
            pass
    if state.is_file():
        try:
            last = json.loads(state.read_text(encoding="utf-8")).get("last_publish")
            if last:
                print(f"发布: {last.get('status')} item_id={last.get('item_id')}")
        except (json.JSONDecodeError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prompt2vid",
        description="一句话描述 → 自动生成视频并发布到抖音",
    )
    parser.add_argument("prompt", nargs="?", help="视频内容的一句话描述")
    parser.add_argument(
        "--workspace",
        default=None,
        help="运行工作目录（默认 output/run_<时间戳>）",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只做环境预检（ComfyUI/模型/ffmpeg），不执行流水线",
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="跳过环境预检（调试用）",
    )
    direct_group = parser.add_mutually_exclusive_group()
    direct_group.add_argument(
        "--direct-media",
        dest="direct_media",
        action="store_true",
        default=None,
        help="直接媒体模式：LLM 只做创意三阶段+发布，媒体阶段由代码直跑（默认读 PROMPT2VID_DIRECT_MEDIA）",
    )
    direct_group.add_argument(
        "--no-direct-media",
        dest="direct_media",
        action="store_false",
        help="强制 crew 全流程模式（9 agent）",
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_DIR",
        default=None,
        help="断点续跑：复用已有工作目录，从 --from 指定阶段继续",
    )
    parser.add_argument(
        "--from",
        dest="from_stage",
        default="frames",
        choices=["frames", "clips", "voiceovers", "render", "quality"],
        help="续跑起始阶段（默认 frames，配合 --resume）",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="续跑完成后执行抖音发布（配合 --resume）",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="清理 output/ 下的旧运行目录（保留最近 --keep 个）",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=5,
        help="clean 时保留的运行目录数（默认 5）",
    )
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv()

    if args.clean:
        from .pipeline import clean_outputs, trim_publish_state

        removed = clean_outputs(max(0, args.keep))
        trimmed = trim_publish_state()
        print(f"已删除 {len(removed)} 个旧运行目录（保留最近 {args.keep} 个）")
        if trimmed:
            print(f"已截断抖音发布历史 {trimmed} 条")
        return 0

    if args.check_only:
        from .preflight import run_preflight

        return 0 if run_preflight() else 2

    if args.resume:
        run_dir = Path(args.resume).resolve()
        if not run_dir.is_dir():
            print(f"错误: 工作目录不存在: {run_dir}")
            return 2
    else:
        if not args.prompt:
            parser.error("请提供视频内容的一句话描述（或使用 --check-only / --resume / --clean）")
        if args.workspace:
            run_dir = Path(args.workspace).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = _prepare_workspace()

    if not args.no_preflight:
        from .preflight import run_preflight

        if not run_preflight():
            return 2

    from .pipeline import run_pipeline, run_resume
    from .workspace import set_workspace

    if args.resume:
        result = run_resume(run_dir, from_stage=args.from_stage, publish=args.publish)
    else:
        # 覆盖式设置：同一进程内多次运行 / server 复用时不串目录
        set_workspace(run_dir)
        result = run_pipeline(args.prompt, run_dir, direct_media=args.direct_media)

    print(f"最终输出:\n{result}")
    _print_run_summary(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
