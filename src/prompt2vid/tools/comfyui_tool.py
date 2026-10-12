"""ComfyUI 工具：文生图（SDXL）与图生视频（LTX-Video）。

通过 ComfyUI HTTP API（/prompt、/history、/view、/upload/image）提交
API 格式工作流并轮询结果。工作流模板基于 ComfyUI 官方
ltxv_image_to_video_simple.0.9.5.json / ltxv_text_to_video.json
（节点 schema 已对照 ComfyUI master nodes_lt.py / nodes_custom_sampler.py 验证）。
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from crewai.tools import tool

DEFAULT_COMFYUI_URL = "http://comfyui.comfyui.svc.cluster.local:8188"
DEFAULT_T2I_CHECKPOINT = "sd_xl_base_1.0.safetensors"
DEFAULT_LTX_CHECKPOINT = "ltx-video-2b-v0.9.5.safetensors"
DEFAULT_T5 = "t5xxl_fp8_e4m3fn.safetensors"
DEFAULT_T5_DEVICE = "cpu"
DEFAULT_TIMEOUT = 1800.0
DEFAULT_POLL_INTERVAL = 5.0

DEFAULT_T2I_NEGATIVE = (
    "low quality, worst quality, deformed, distorted, disfigured, "
    "bad anatomy, extra limbs, ugly, watermark, text"
)
DEFAULT_LTX_NEGATIVE = (
    "low quality, worst quality, deformed, distorted, disfigured, "
    "motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"
)


def _env(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


def _workspace() -> Path:
    ws = os.getenv("PROMPT2VID_WORKSPACE", "").strip()
    path = Path(ws) if ws else Path("output") / "run_default"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _timeout() -> float:
    try:
        return float(_env("COMFYUI_TIMEOUT", str(DEFAULT_TIMEOUT)))
    except ValueError:
        return DEFAULT_TIMEOUT


def _seed() -> int:
    return random.randint(0, 2**32 - 1)


class ComfyUIClient:
    """ComfyUI HTTP API 最小客户端。"""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or _env("COMFYUI_URL", DEFAULT_COMFYUI_URL)).rstrip("/")

    def submit(self, workflow: dict[str, Any]) -> str:
        """提交工作流，返回 prompt_id。"""
        prompt_id = str(uuid.uuid4())
        payload = {"prompt": workflow, "client_id": prompt_id}
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{self.base_url}/prompt", json=payload)
            resp.raise_for_status()
            data = resp.json()
        if "error" in data:
            raise RuntimeError(f"ComfyUI 拒绝工作流: {data['error']}")
        return data["prompt_id"]

    def wait(self, prompt_id: str, timeout: float | None = None) -> dict[str, Any]:
        """轮询 /history 直到任务完成，返回 history 条目。"""
        deadline = time.monotonic() + (timeout or _timeout())
        with httpx.Client(timeout=30.0) as client:
            while True:
                resp = client.get(f"{self.base_url}/history/{prompt_id}")
                resp.raise_for_status()
                history = resp.json()
                entry = history.get(prompt_id)
                if entry:
                    status = entry.get("status", {})
                    if status.get("completed"):
                        return entry
                    if status.get("status_str") == "error":
                        raise RuntimeError(f"ComfyUI 执行失败: {status}")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"等待 ComfyUI 任务 {prompt_id} 超时")
                time.sleep(DEFAULT_POLL_INTERVAL)

    def outputs(self, entry: dict[str, Any]) -> list[dict[str, str]]:
        """从 history 条目解析全部输出文件（兼容 images/gifs/videos 等键）。"""
        files: list[dict[str, str]] = []
        for node_outputs in entry.get("outputs", {}).values():
            ui = node_outputs.get("ui") or node_outputs.get("images") or {}
            for key in ("images", "gifs", "videos", "images_preview"):
                for item in ui.get(key, []) if isinstance(ui, dict) else []:
                    if isinstance(item, dict) and item.get("filename"):
                        files.append(
                            {
                                "filename": item["filename"],
                                "subfolder": item.get("subfolder", ""),
                                "type": item.get("type", "output"),
                            }
                        )
        return files

    def download(self, file_info: dict[str, str], dest: Path) -> Path:
        """通过 /view 下载输出文件到本地。"""
        params = {
            "filename": file_info["filename"],
            "subfolder": file_info.get("subfolder", ""),
            "type": file_info.get("type", "output"),
        }
        with httpx.Client(timeout=120.0) as client:
            resp = client.get(f"{self.base_url}/view", params=params)
            resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest

    def upload_image(self, image_path: Path, name: str | None = None) -> str:
        """上传图片到 ComfyUI input 目录，返回上传后的文件名。"""
        image_path = Path(image_path)
        name = name or image_path.name
        mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
        with httpx.Client(timeout=120.0) as client:
            with image_path.open("rb") as f:
                resp = client.post(
                    f"{self.base_url}/upload/image",
                    files={"image": (name, f, mime)},
                    data={"overwrite": "true"},
                )
            resp.raise_for_status()
        return resp.json()["name"]


def build_t2i_workflow(
    prompt: str,
    negative_prompt: str,
    width: int = 768,
    height: int = 1344,
    seed: int | None = None,
) -> dict[str, Any]:
    """SDXL 文生图工作流（API 格式）。

    节点链：CheckpointLoaderSimple → CLIPTextEncode(正/负) →
    EmptyLatentImage → KSampler → VAEDecode → SaveImage
    """
    ckpt = _env("COMFYUI_T2I_CHECKPOINT", DEFAULT_T2I_CHECKPOINT)
    negative = negative_prompt or _env("COMFYUI_NEGATIVE_PROMPT", DEFAULT_T2I_NEGATIVE)
    return {
        "44": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["44", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["44", 1]},
        },
        "70": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["44", 0],
                "seed": seed if seed is not None else _seed(),
                "steps": 30,
                "cfg": 7.0,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["70", 0],
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["44", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "prompt2vid"},
        },
    }


def build_i2v_workflow(
    image_name: str,
    prompt: str,
    negative_prompt: str,
    width: int = 512,
    height: int = 768,
    length: int = 41,
    seed: int | None = None,
) -> dict[str, Any]:
    """LTX-Video 图生视频工作流（API 格式）。

    节点链（对照官方 ltxv_image_to_video_simple.0.9.5.json）：
    CLIPLoader(T5) → CLIPTextEncode(正/负) → LTXVImgToVideo → LTXVConditioning →
    SamplerCustom(cfg=3.0) ← KSamplerSelect(euler) + LTXVScheduler；
    CheckpointLoaderSimple(LTX) 提供 MODEL/VAE；
    LoadImage → LTXVPreprocess；VAEDecode → SaveWEBM
    """
    ltx_ckpt = _env("COMFYUI_LTX_CHECKPOINT", DEFAULT_LTX_CHECKPOINT)
    t5 = _env("COMFYUI_T5", DEFAULT_T5)
    t5_device = _env("COMFYUI_T5_DEVICE", DEFAULT_T5_DEVICE)
    negative = negative_prompt or _env("COMFYUI_LTX_NEGATIVE_PROMPT", DEFAULT_LTX_NEGATIVE)
    return {
        # T5 文本编码器（fp8 放 text_encoders，device=cpu 省显存）
        "38": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": t5,
                "type": "ltxv",
                "device": t5_device,
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["38", 0]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["38", 0]},
        },
        # LTX checkpoint 提供 MODEL / VAE
        "44": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ltx_ckpt},
        },
        # 首帧图
        "78": {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        },
        "82": {
            "class_type": "LTXVPreprocess",
            "inputs": {"image": ["78", 0], "img_compression": 35},
        },
        "95": {
            "class_type": "LTXVImgToVideo",
            "inputs": {
                "positive": ["6", 0],
                "negative": ["7", 0],
                "vae": ["44", 2],
                "image": ["82", 0],
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1,
                "strength": 1.0,
            },
        },
        "69": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["95", 0],
                "negative": ["95", 1],
                "frame_rate": 25.0,
            },
        },
        "71": {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps": 30,
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stretch": True,
                "terminal": 0.1,
            },
        },
        "73": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "72": {
            "class_type": "SamplerCustom",
            "inputs": {
                "add_noise": True,
                "noise_seed": seed if seed is not None else _seed(),
                "cfg": 3.0,
                "model": ["44", 0],
                "positive": ["69", 0],
                "negative": ["69", 1],
                "sampler": ["73", 0],
                "sigmas": ["71", 0],
                "latent_image": ["69", 2],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["72", 0], "vae": ["44", 2]},
        },
        "86": {
            "class_type": "SaveWEBM",
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": "prompt2vid",
                "codec": "vp9",
                "fps": 24.0,
                "crf": 32.0,
            },
        },
    }


def _run_workflow(workflow: dict[str, Any], dest: Path) -> Path:
    client = ComfyUIClient()
    prompt_id = client.submit(workflow)
    entry = client.wait(prompt_id)
    files = client.outputs(entry)
    if not files:
        raise RuntimeError("ComfyUI 任务完成但没有输出文件")
    return client.download(files[0], dest)


@tool("Text to image via ComfyUI SDXL")
def text_to_image(
    prompt: str,
    output_path: str,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 1344,
) -> str:
    """根据英文提示词生成一张图片。

    Args:
        prompt: 英文 SDXL 提示词（含风格/构图/光线）。
        output_path: 图片保存路径（相对工作目录或绝对路径，建议 .png）。
        negative_prompt: 负面提示词，留空使用环境默认。
        width: 图片宽度（默认 768）。
        height: 图片高度（默认 1344，竖屏 9:16）。

    Returns:
        保存后的图片绝对路径。
    """
    dest = Path(output_path)
    if not dest.is_absolute():
        dest = _workspace() / dest
    workflow = build_t2i_workflow(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
    )
    path = _run_workflow(workflow, dest)
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
    """将首帧图转为视频片段（LTX-Video 图生视频）。

    Args:
        image_path: 首帧图路径（本地文件，将自动上传到 ComfyUI）。
        prompt: 英文提示词 = 画面描述 + 运动描述（如 'autumn park, slow zoom in'）。
        output_path: 视频保存路径（建议 .webm）。
        negative_prompt: 负面提示词，留空使用环境默认。
        width: 视频宽度（默认 512，8GB 显存预算）。
        height: 视频高度（默认 768，竖屏）。
        length: 视频帧数（25fps 下 41 帧 ≈ 1.6s；帧数 = 秒 × 8 + 1）。

    Returns:
        保存后的视频绝对路径。
    """
    image = Path(image_path)
    if not image.is_absolute():
        image = _workspace() / image
    if not image.is_file():
        raise FileNotFoundError(f"首帧图不存在: {image}")

    dest = Path(output_path)
    if not dest.is_absolute():
        dest = _workspace() / dest

    client = ComfyUIClient()
    uploaded = client.upload_image(image)
    workflow = build_i2v_workflow(
        image_name=uploaded,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        length=length,
    )
    path = _run_workflow(workflow, dest)
    return str(path.resolve())


TOOLS = [text_to_image, image_to_video]
