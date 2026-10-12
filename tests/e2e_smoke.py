"""端到端冒烟测试（需环境就绪后手动运行，不在 pytest 自动收集内）。

前置条件：
1. ComfyUI 已运行且模型就位（见 deploy/comfyui/MODELS.md）
2. .env 已配置 LLM API Key
3. ffmpeg / ffprobe 在 PATH

运行：
    python tests/e2e_smoke.py "做一个关于秋天的治愈系短视频"
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> int:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "做一个关于秋天的治愈系短视频"
    print(f"[e2e] 用户描述: {prompt}")

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")

    from prompt2vid.crew import build_crew

    crew = build_crew()
    result = crew.kickoff(inputs={"user_prompt": prompt})
    print("\n===== e2e 结果 =====")
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
