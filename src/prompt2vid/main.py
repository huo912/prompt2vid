"""CLI 入口：python -m prompt2vid.main "一句话描述"。"""

from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prompt2vid",
        description="一句话描述 → 自动生成视频并发布到抖音",
    )
    parser.add_argument("prompt", help="视频内容的一句话描述")
    parser.add_argument(
        "--workspace",
        default=None,
        help="运行工作目录（默认 output/run_<时间戳>）",
    )
    args = parser.parse_args(argv)

    if args.workspace:
        run_dir = Path(args.workspace).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = _prepare_workspace()

    import os

    os.environ.setdefault("PROMPT2VID_WORKSPACE", str(run_dir))

    from .crew import build_crew

    crew = build_crew()
    result = crew.kickoff(inputs={"user_prompt": args.prompt, "workspace": str(run_dir)})

    print("\n===== 运行结果 =====")
    print(f"工作目录: {run_dir}")
    print(f"最终输出:\n{result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
