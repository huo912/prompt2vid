"""配音工具（CrewAI 薄包装）。

阶段工具 generate_voiceovers：读 storyboard.json 逐镜配音 + manifest；
实现（edge-tts → 网关 TTS → 静音占位回退链）在 prompt2vid.tts 纯函数模块。
"""

from __future__ import annotations

import json

from crewai.tools import tool

from .. import stages


@tool("Generate voiceover audio for all storyboard shots")
def generate_voiceovers() -> str:
    """为分镜表中每个分镜生成中文配音（edge-tts）。

    自动读取工作目录下的 storyboard.json，逐镜合成 audio/shot_<id>.mp3
    （音色 zh-CN-XiaoxiaoNeural，可用环境变量覆盖；edge-tts 不可用时自动
    回退），并写入 audio/manifest.json 清单。无需任何参数。

    Returns:
        JSON 汇总：stage/total/succeeded/failed/manifest。
    """
    return json.dumps(stages.generate_voiceovers(), ensure_ascii=False)


TOOLS = [generate_voiceovers]
