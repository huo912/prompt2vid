"""剪辑工具（CrewAI 薄包装）。

阶段工具 render_final：确定性完成拼接、逐镜音频对齐、SRT 字幕生成、
混流烧录，产出 final/final.mp4；ffmpeg 原语在 prompt2vid.ffmpeg_ops。
"""

from __future__ import annotations

import json

from crewai.tools import tool

from .. import stages
from ..ffmpeg_ops import probe_file
from ..workspace import resolve_in_workspace


@tool("Render the final video from clips and voiceovers")
def render_final() -> str:
    """把所有分镜片段与配音合成为最终成片。

    自动读取工作目录下的 clips/manifest.json、audio/manifest.json 与
    storyboard.json，依次完成：
    1. 逐镜探测真实时长并把配音对齐到该时长（不足静音填充）；
    2. 拼接视频片段；
    3. 按真实时间轴生成 SRT 中文字幕；
    4. 混流音轨并烧录字幕，产出 final/final.mp4 与 final/render_report.json。
    无需任何参数。

    Returns:
        JSON 汇总：final_video/duration_seconds/width/height/has_audio/shots。
    """
    return json.dumps(stages.render_final(), ensure_ascii=False)


@tool("Probe media file with ffprobe")
def probe(media_path: str) -> str:
    """探测媒体文件的时长、分辨率、帧率、音轨信息。

    Args:
        media_path: 媒体文件路径（相对工作目录或绝对路径）。

    Returns:
        JSON 字符串：duration_seconds、width、height、fps、has_audio。
    """
    media = resolve_in_workspace(media_path)
    return json.dumps(probe_file(media), ensure_ascii=False)


TOOLS = [render_final, probe]
