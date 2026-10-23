"""ComfyUI 工具（CrewAI 薄包装）。

阶段工具（推荐，供 agent 调用）：
- generate_frames：读 storyboard.json，逐镜生成首帧图 + manifest + 重试；
- generate_clips：读 frames/manifest.json，逐镜图生视频 + manifest + 重试。

单镜原语（text_to_image / image_to_video）保留用于调试与高级用法，
实现已迁移到 comfyui_client.py 纯函数。
"""

from __future__ import annotations

import json

from crewai.tools import tool

from .. import stages
from ..comfyui_client import run_image_to_video, run_text_to_image
from ..workspace import resolve_in_workspace


@tool("Generate first-frame images for all storyboard shots")
def generate_frames() -> str:
    """为分镜表中每个分镜生成首帧图（ComfyUI SDXL 文生图）。

    自动读取工作目录下的 storyboard.json，逐镜生成 frames/shot_<id>.png，
    失败自动重试，并写入 frames/manifest.json 清单。无需任何参数。

    Returns:
        JSON 汇总：stage/total/succeeded/failed/manifest 及失败明细。
    """
    return json.dumps(stages.generate_frames(), ensure_ascii=False)


@tool("Generate video clips for all storyboard shots")
def generate_clips() -> str:
    """将每个分镜的首帧图转为视频片段（ComfyUI LTX-Video 图生视频）。

    自动读取 frames/manifest.json 与 storyboard.json，逐镜生成
    clips/shot_<id>.webm，帧数按时长自动换算（8n+1），失败自动重试，
    并写入 clips/manifest.json 清单。无需任何参数。

    Returns:
        JSON 汇总：stage/total/succeeded/failed/manifest 及失败明细。
    """
    return json.dumps(stages.generate_clips(), ensure_ascii=False)


@tool("Text to image via ComfyUI SDXL")
def text_to_image(
    prompt: str,
    output_path: str,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 1344,
) -> str:
    """根据英文提示词生成一张图片（单镜原语，通常用 generate_frames 即可）。

    Args:
        prompt: 英文 SDXL 提示词（含风格/构图/光线）。
        output_path: 图片保存路径（相对工作目录或绝对路径，建议 .png）。
        negative_prompt: 负面提示词，留空使用环境默认。
        width: 图片宽度（默认 768）。
        height: 图片高度（默认 1344，竖屏 9:16）。

    Returns:
        保存后的图片绝对路径。
    """
    dest = resolve_in_workspace(output_path)
    path = run_text_to_image(
        prompt=prompt,
        dest=dest,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
    )
    return str(path.resolve())


@tool("Image to video via ComfyUI LTX-Video")
def image_to_video(
    image_path: str,
    prompt: str,
    output_path: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 768,
    length: int = 41,
) -> str:
    """将首帧图转为视频片段（单镜原语，通常用 generate_clips 即可）。

    Args:
        image_path: 首帧图路径（本地文件，将自动上传到 ComfyUI）。
        prompt: 英文提示词 = 画面描述 + 运动描述（如 'autumn park, slow zoom in'）。
        output_path: 视频保存路径（建议 .webm）。
        negative_prompt: 负面提示词，留空使用环境默认。
        width: 视频宽度（默认 512，8GB 显存预算）。
        height: 视频高度（默认 768，竖屏）。
        length: 视频帧数（必须 8n+1；25fps 下 41 帧 ≈ 1.6s）。

    Returns:
        保存后的视频绝对路径。
    """
    image = resolve_in_workspace(image_path)
    dest = resolve_in_workspace(output_path)
    path = run_image_to_video(
        image=image,
        dest=dest,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        length=length,
    )
    return str(path.resolve())


IMAGE_TOOLS = [generate_frames]
VIDEO_TOOLS = [generate_clips]
TOOLS = [generate_frames, generate_clips]
