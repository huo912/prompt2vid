"""质检工具（CrewAI 薄包装）。

阶段工具 run_quality_checks：确定性核验成片存在/时长/分辨率/音轨与
分镜产物完整性，产出 report/quality_report.json。
"""

from __future__ import annotations

import json

from crewai.tools import tool

from .. import stages


@tool("Run quality checks on the final video and assets")
def run_quality_checks() -> str:
    """对最终成片与全流程产物做质量检查。

    自动读取工作目录下的 final/final.mp4、requirement.json 与三个
    manifest 清单，逐项核验：
    1. 成片存在且可读；
    2. 时长与需求 duration_seconds 误差 ±2 秒内；
    3. 分辨率达标（宽 ≥ 512）；
    4. 包含音轨；
    5. 分镜产物齐全（frames/clips/audio 清单中每个文件真实存在）。
    结果写入 report/quality_report.json。无需任何参数。

    Returns:
        JSON 质检报告：passed、checks、summary、final_video。
    """
    return json.dumps(stages.run_quality_checks(), ensure_ascii=False)


TOOLS = [run_quality_checks]
