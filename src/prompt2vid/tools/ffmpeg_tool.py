"""剪辑工具（CrewAI 薄包装）。

阶段工具 render_final：确定性完成拼接、逐镜音频对齐、SRT 字幕生成、
混流烧录，产出 final/final.mp4；ffmpeg 原语在 prompt2vid.ffmpeg_ops。
阶段工具 sync_lips：配音驱动的唇形同步（LatentSync），产物在 lipsync/，
渲染阶段会优先使用；缺节点/失败时保留原片段，不阻塞出片。
"""

from __future__ import annotations

import json

from crewai.tools import tool

from .. import stages
from ..ffmpeg_ops import probe_file
from ..workspace import resolve_in_workspace


@tool("Sync lips of every clip to its voiceover")
def sync_lips() -> str:
    """用配音驱动每个分镜片段的唇形（LatentSync 口型同步）。

    自动读取工作目录下的 clips/manifest.json 与 audio/manifest.json，
    把片段与配音上传 ComfyUI 做音频驱动的嘴部重绘，产出
    lipsync/shot_<id>.mp4 与 lipsync/manifest.json。配音比片段长时以
    往返补帧延长（旁白零截断）。单镜失败（无正脸/动漫脸/OOM）保留原
    片段并在清单里记录；渲染阶段会优先使用口型同步产物。
    必须在 render_final 之前调用。无需任何参数。

    Returns:
        JSON 汇总：total/succeeded/failed/skipped/manifest。
    """
    return json.dumps(stages.sync_lips(), ensure_ascii=False)


@tool("Render the final video from clips and voiceovers")
def render_final() -> str:
    """把所有分镜片段与配音合成为最终成片。

    自动读取工作目录下的 clips/manifest.json、audio/manifest.json 与
    storyboard.json，依次完成：
    1. 逐镜探测真实时长并把配音对齐到该时长（不足静音填充）；
    2. 已有 lipsync/manifest.json 中状态 ok 的镜头优先使用口型同步片段；
    3. 拼接视频片段；
    4. 按真实时间轴生成 SRT 中文字幕；
    5. 混流音轨并烧录字幕，产出 final/final.mp4 与 final/render_report.json。
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


TOOLS = [sync_lips, render_final, probe]
