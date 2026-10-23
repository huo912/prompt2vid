"""端到端冒烟测试（需环境就绪后手动运行，不在 pytest 自动收集内）。

前置条件：
1. ComfyUI 已运行且模型就位（见 deploy/comfyui/MODELS.md）
2. .env 已配置 LLM API Key
3. ffmpeg / ffprobe 在 PATH

运行：
    python tests/e2e_smoke.py "做一个关于秋天的治愈系短视频"
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "做一个关于秋天的治愈系短视频"
    print(f"[e2e] 用户描述: {prompt}")

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")

    from prompt2vid.preflight import run_preflight

    if not run_preflight():
        print("[e2e] 预检未通过，终止")
        return 2

    run_dir = REPO_ROOT / "output" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[e2e] 工作目录: {run_dir}")

    from prompt2vid.workspace import set_workspace

    set_workspace(run_dir)

    from prompt2vid.pipeline import run_pipeline

    result = run_pipeline(prompt, run_dir)

    print("\n===== e2e 结果 =====")
    print(result)
    final = run_dir / "final" / "final.mp4"
    print(f"成片: {final}{'（已生成）' if final.is_file() else '（未生成）'}")
    print(f"日志: {run_dir / 'run.log'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
