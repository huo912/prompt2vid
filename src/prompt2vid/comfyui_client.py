"""ComfyUI 客户端与工作流构建（纯函数，不依赖 CrewAI）。

通过 ComfyUI HTTP API（/prompt、/history、/view、/upload/image）提交
API 格式工作流并轮询结果。工作流模板基于 ComfyUI 官方
ltxv_image_to_video_simple.0.9.5.json / ltxv_text_to_video.json
（节点 schema 已对照 ComfyUI master nodes_lt.py / nodes_custom_sampler.py 验证）。

从 tools/comfyui_tool.py 迁移而来：工具层只留薄包装，循环/重试等编排
逻辑下沉到 stages.py，供确定性流水线直接调用。
"""

from __future__ import annotations

import itertools
import mimetypes
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"
DEFAULT_T2I_CHECKPOINT = "sd_xl_base_1.0.safetensors"
DEFAULT_LTX_CHECKPOINT = "ltx-video-2b-v0.9.5.safetensors"
DEFAULT_T5 = "t5xxl_fp8_e4m3fn.safetensors"
DEFAULT_T5_DEVICE = "cpu"
DEFAULT_TIMEOUT = 1800.0
# 轮询指数退避：1s 起步、1.5 倍增长、5s 封顶（短视频任务比固定 5s 省 10-20s）
POLL_START = 1.0
POLL_MAX = 5.0
# 采样步数：LTX 0.9.5 是 distilled 蒸馏版，官方建议 8-13 步；
# 30 步不会更好，只会慢 2-3 倍。SDXL 25 步 dpmpp_2m 足够。
DEFAULT_T2I_STEPS = 25
DEFAULT_LTX_STEPS = 10

DEFAULT_T2I_NEGATIVE = (
    "low quality, worst quality, deformed, distorted, disfigured, "
    "bad anatomy, extra limbs, ugly, watermark, text"
)
DEFAULT_LTX_NEGATIVE = (
    "low quality, worst quality, deformed, distorted, disfigured, "
    "motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"
)

# 角色一致性（IPAdapter 参考图注入）：需在 ComfyUI 安装
# ComfyUI_IPAdapter_plus 自定义节点 + 对应模型（见 docs/LOCAL_SETUP.md）。
# 预设决定 Unified Loader 加载哪个 adapter 模型：
# "PLUS FACE (portraits)" 只需 ip-adapter-plus-face_sdxl + CLIP-ViT-H；
# FACEID 系列更强但需要 insightface。
DEFAULT_IPADAPTER_PRESET = "PLUS FACE (portraits)"
DEFAULT_IPADAPTER_WEIGHT = 0.8

# LTX 输出帧率（与 LTXVConditioning.frame_rate 保持一致）
LTX_FPS = 25.0

# 种子计数器：PROMPT2VID_SEED 固定后，逐次 +1 保证分镜间种子不同但整跑可复现
_SEED_COUNTER = itertools.count()


def env_str(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, str(default)))
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(float(env_str(name, str(default))))
    except ValueError:
        return default


def next_seed() -> int:
    """取下一个种子：设了 PROMPT2VID_SEED 则 base+n（可复现），否则随机。"""
    base = os.getenv("PROMPT2VID_SEED", "").strip()
    if base:
        try:
            return int(base) + next(_SEED_COUNTER)
        except ValueError:
            pass
    return random.randint(0, 2**32 - 1)


def ipadapter_enabled() -> bool:
    """是否启用 IPAdapter 角色参考注入（COMFYUI_IPADAPTER 真值）。"""
    return os.getenv("COMFYUI_IPADAPTER", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


class ComfyUITaskError(RuntimeError):
    """单个 ComfyUI 任务执行失败（wait/wait_all 收集，不中断同批其他任务）。"""


def parse_output_files(entry: dict[str, Any]) -> list[dict[str, str]]:
    """解析 history 条目的全部输出文件（兼容新旧 ComfyUI 格式）。

    outputs[node_id] 的两种形态都支持：
    - images 直接挂在节点下（经典文档格式，也是 2025+ 新版 master
      的实际格式：ui_outputs[node_id] = ui_info["output"]）；
    - 旧变体：{"ui": {"images": [...]}} 包装。
    """
    files: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for node_outputs in entry.get("outputs", {}).values():
        if not isinstance(node_outputs, dict):
            continue
        containers: list[dict[str, Any]] = [node_outputs]
        ui = node_outputs.get("ui")
        if isinstance(ui, dict):
            containers.append(ui)
        for container in containers:
            for key in ("images", "gifs", "videos", "images_preview"):
                items = container.get(key)
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not (isinstance(item, dict) and item.get("filename")):
                        continue
                    record = {
                        "filename": str(item["filename"]),
                        "subfolder": str(item.get("subfolder") or ""),
                        "type": str(item.get("type") or "output"),
                    }
                    identity = (
                        record["filename"], record["subfolder"], record["type"]
                    )
                    if identity not in seen:
                        seen.add(identity)
                        files.append(record)
    return files


class ComfyUIClient:
    """ComfyUI HTTP API 最小客户端（单实例复用连接池，支持批量等待）。

    用法：
        with ComfyUIClient() as client:
            pid = client.submit(workflow)
            entry = client.wait(pid)
        或批量：results = client.wait_all([pid1, pid2, ...])
    """

    def __init__(self, base_url: str | None = None, http: httpx.Client | None = None) -> None:
        self.base_url = (base_url or env_str("COMFYUI_URL", DEFAULT_COMFYUI_URL)).rstrip("/")
        self._http = http or httpx.Client(timeout=30.0)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ComfyUIClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def submit(self, workflow: dict[str, Any]) -> str:
        """提交工作流，返回 prompt_id。"""
        prompt_id = str(uuid.uuid4())
        payload = {"prompt": workflow, "client_id": prompt_id}
        resp = self._http.post(f"{self.base_url}/prompt", json=payload)
        if resp.status_code != 200:
            # 把响应体带进错误信息：节点缺失/校验失败的原因在 body 里
            # （如 IPAdapter 未安装时 ComfyUI 返回 400 + node_errors）
            raise RuntimeError(
                f"ComfyUI 拒绝工作流（HTTP {resp.status_code}）: {resp.text[:500]}"
            )
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"ComfyUI 拒绝工作流: {data['error']}")
        return data["prompt_id"]

    def _history_entry(self, prompt_id: str) -> dict[str, Any] | None:
        """查询单个任务：未完成返回 None；执行出错抛 ComfyUITaskError。"""
        resp = self._http.get(f"{self.base_url}/history/{prompt_id}")
        resp.raise_for_status()
        entry = resp.json().get(prompt_id)
        if not entry:
            return None
        status = entry.get("status") or {}
        status_str = str(status.get("status_str") or "").lower()
        if status.get("completed") or status_str == "success":
            return entry
        if status_str == "error":
            raise ComfyUITaskError(f"ComfyUI 执行失败: {status}")
        return None

    def wait(self, prompt_id: str, timeout: float | None = None) -> dict[str, Any]:
        """轮询 /history 直到任务完成，返回 history 条目（指数退避）。"""
        deadline = time.monotonic() + (timeout or env_float("COMFYUI_TIMEOUT", DEFAULT_TIMEOUT))
        interval = POLL_START
        while True:
            entry = self._history_entry(prompt_id)
            if entry is not None:
                return entry
            if time.monotonic() > deadline:
                raise TimeoutError(f"等待 ComfyUI 任务 {prompt_id} 超时")
            time.sleep(interval)
            interval = min(interval * 1.5, POLL_MAX)

    def wait_all(
        self, prompt_ids: list[str], timeout: float | None = None
    ) -> dict[str, Any]:
        """批量等待多个任务完成。

        ComfyUI 按 GPU 队列串行执行，一次提交整批再统一轮询可消除逐任务
        的调度空隙。返回 prompt_id → history 条目；单个任务执行失败不抛
        异常，对应值为 ComfyUITaskError 实例（由调用方决定重试）。
        """
        deadline = time.monotonic() + (timeout or env_float("COMFYUI_TIMEOUT", DEFAULT_TIMEOUT))
        done: dict[str, Any] = {}
        pending: set[str] = set(prompt_ids)
        interval = POLL_START
        while pending:
            for pid in sorted(pending):
                try:
                    entry = self._history_entry(pid)
                except ComfyUITaskError as exc:
                    done[pid] = exc
                    pending.discard(pid)
                    continue
                if entry is not None:
                    done[pid] = entry
                    pending.discard(pid)
            if not pending:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"等待 ComfyUI 任务批量完成超时: {sorted(pending)}")
            time.sleep(interval)
            interval = min(interval * 1.5, POLL_MAX)
        return done

    def outputs(self, entry: dict[str, Any]) -> list[dict[str, str]]:
        """从 history 条目解析全部输出文件（见 parse_output_files）。"""
        return parse_output_files(entry)

    def download(self, file_info: dict[str, str], dest: Path) -> Path:
        """通过 /view 下载输出文件到本地。"""
        params = {
            "filename": file_info["filename"],
            "subfolder": file_info.get("subfolder", ""),
            "type": file_info.get("type", "output"),
        }
        resp = self._http.get(f"{self.base_url}/view", params=params, timeout=120.0)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest

    def upload_image(self, image_path: Path, name: str | None = None) -> str:
        """上传图片到 ComfyUI input 目录，返回上传后的文件名。"""
        image_path = Path(image_path)
        name = name or image_path.name
        mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
        with image_path.open("rb") as f:
            resp = self._http.post(
                f"{self.base_url}/upload/image",
                files={"image": (name, f, mime)},
                data={"overwrite": "true"},
                timeout=120.0,
            )
        resp.raise_for_status()
        return resp.json()["name"]


def build_t2i_workflow(
    prompt: str,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 1344,
    seed: int | None = None,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """SDXL 文生图工作流（API 格式）。

    节点链：CheckpointLoaderSimple → CLIPTextEncode(正/负) →
    EmptyLatentImage → KSampler → VAEDecode → SaveImage

    Args:
        reference: 角色参考图注入（IPAdapter）。传 {"image_name": 已上传的
            参考图名} 即可，weight/preset 缺省读 COMFYUI_IPADAPTER_WEIGHT /
            COMFYUI_IPADAPTER_PRESET。节点链变为：
            CheckpointLoader → IPAdapterUnifiedLoader → IPAdapter(参考图)
            → KSampler，需要 ComfyUI 安装 ComfyUI_IPAdapter_plus。
    """
    ckpt = env_str("COMFYUI_T2I_CHECKPOINT", DEFAULT_T2I_CHECKPOINT)
    negative = negative_prompt or env_str("COMFYUI_NEGATIVE_PROMPT", DEFAULT_T2I_NEGATIVE)
    workflow: dict[str, Any] = {
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
                "seed": seed if seed is not None else next_seed(),
                "steps": env_int("COMFYUI_T2I_STEPS", DEFAULT_T2I_STEPS),
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
    if reference and reference.get("image_name"):
        workflow["3"]["inputs"]["model"] = _apply_ipadapter(workflow, reference)
    return workflow


def _apply_ipadapter(
    workflow: dict[str, Any], reference: dict[str, Any]
) -> list[Any]:
    """注入 IPAdapter 参考图节点链，返回 KSampler 应使用的 MODEL 输出。

    节点签名对照 ComfyUI_IPAdapter_plus（cubiq）当前版本；未安装该
    自定义节点或模型时 ComfyUI 会拒绝工作流，由调用方（stages 的
    重试机制）降级为无参考生成。
    """
    weight = reference.get("weight")
    if weight is None:
        weight = env_float("COMFYUI_IPADAPTER_WEIGHT", DEFAULT_IPADAPTER_WEIGHT)
    preset = reference.get("preset") or env_str(
        "COMFYUI_IPADAPTER_PRESET", DEFAULT_IPADAPTER_PRESET
    )
    workflow["ref_image"] = {
        "class_type": "LoadImage",
        "inputs": {"image": reference["image_name"]},
    }
    workflow["ref_loader"] = {
        "class_type": "IPAdapterUnifiedLoader",
        "inputs": {"model": ["44", 0], "ipadapter": preset},
    }
    workflow["ref_apply"] = {
        "class_type": "IPAdapter",
        "inputs": {
            "model": ["ref_loader", 0],
            "ipadapter": ["ref_loader", 1],
            "image": ["ref_image", 0],
            "weight": weight,
            "weight_faceidv2": False,
            "weight_kolors": False,
            "combine_embeds": "concat",
        },
    }
    return ["ref_apply", 0]


def build_i2v_workflow(
    image_name: str,
    prompt: str,
    negative_prompt: str = "",
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
    ltx_ckpt = env_str("COMFYUI_LTX_CHECKPOINT", DEFAULT_LTX_CHECKPOINT)
    t5 = env_str("COMFYUI_T5", DEFAULT_T5)
    t5_device = env_str("COMFYUI_T5_DEVICE", DEFAULT_T5_DEVICE)
    negative = negative_prompt or env_str("COMFYUI_LTX_NEGATIVE_PROMPT", DEFAULT_LTX_NEGATIVE)
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
                "frame_rate": LTX_FPS,
            },
        },
        "71": {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps": env_int("COMFYUI_LTX_STEPS", DEFAULT_LTX_STEPS),
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
                "noise_seed": seed if seed is not None else next_seed(),
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
                "fps": LTX_FPS,
                "crf": 32.0,
            },
        },
    }


def run_workflow(workflow: dict[str, Any], dest: Path, client: ComfyUIClient | None = None) -> Path:
    """提交工作流并等待完成，下载第一个输出文件到 dest。"""
    client = client or ComfyUIClient()
    prompt_id = client.submit(workflow)
    entry = client.wait(prompt_id)
    files = client.outputs(entry)
    if not files:
        raise RuntimeError("ComfyUI 任务完成但没有输出文件")
    return client.download(files[0], dest)


def run_text_to_image(
    prompt: str,
    dest: str | Path,
    negative_prompt: str = "",
    width: int = 768,
    height: int = 1344,
    seed: int | None = None,
    client: ComfyUIClient | None = None,
) -> Path:
    """SDXL 文生图：生成一张图片并保存到 dest。"""
    dest = Path(dest)
    workflow = build_t2i_workflow(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        seed=seed,
    )
    return run_workflow(workflow, dest, client)


def run_image_to_video(
    image: str | Path,
    dest: str | Path,
    prompt: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 768,
    length: int = 41,
    seed: int | None = None,
    client: ComfyUIClient | None = None,
) -> Path:
    """LTX-Video 图生视频：把首帧图转为视频片段并保存到 dest。

    首帧图会自动上传到 ComfyUI。length 必须满足 8n+1（LTX 约束）。
    """
    image = Path(image)
    if not image.is_file():
        raise FileNotFoundError(f"首帧图不存在: {image}")
    client = client or ComfyUIClient()
    uploaded = client.upload_image(image)
    workflow = build_i2v_workflow(
        image_name=uploaded,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        length=length,
        seed=seed,
    )
    return run_workflow(workflow, Path(dest), client)
