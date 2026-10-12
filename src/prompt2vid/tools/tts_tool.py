"""edge-tts 配音工具：中文旁白转语音。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import edge_tts
from crewai.tools import tool

DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
DEFAULT_RATE = "+0%"


def _workspace() -> Path:
    ws = os.getenv("PROMPT2VID_WORKSPACE", "").strip()
    path = Path(ws) if ws else Path("output") / "run_default"
    path.mkdir(parents=True, exist_ok=True)
    return path


@tool("Synthesize Chinese voiceover via edge-tts")
def synthesize(text: str, output_path: str, voice: str = "", rate: str = "") -> str:
    """将中文文本合成为语音音频。

    Args:
        text: 要合成的文本（中文旁白）。
        output_path: 音频保存路径（建议 .mp3）。
        voice: edge-tts 音色（留空使用 zh-CN-XiaoxiaoNeural）。
        rate: 语速调整（如 "+10%" 加快、"-10%" 放慢，留空不变）。

    Returns:
        保存后的音频绝对路径。
    """

    async def _run() -> None:
        voice_name = voice or os.getenv("EDGE_TTS_VOICE", "").strip() or DEFAULT_VOICE
        rate_value = rate or os.getenv("EDGE_TTS_RATE", "").strip() or DEFAULT_RATE
        communicate = edge_tts.Communicate(text=text, voice=voice_name, rate=rate_value)
        await communicate.save(str(dest))

    dest = Path(output_path)
    if not dest.is_absolute():
        dest = _workspace() / dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(_run())
    if not dest.is_file():
        raise RuntimeError(f"edge-tts 未产出音频: {dest}")
    return str(dest.resolve())


TOOLS = [synthesize]
