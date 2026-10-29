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
import logging
import math
import mimetypes
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from .style import apply_style, apply_style_negative

logger = logging.getLogger("prompt2vid.comfyui")

DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"
DEFAULT_T2I_CHECKPOINT = "sd_xl_base_1.0.safetensors"
DEFAULT_LTX_CHECKPOINT = "ltx-video-2b-v0.9.5.safetensors"
DEFAULT_T5 = "t5xxl_fp8_e4m3fn.safetensors"
DEFAULT_T5_DEVICE = "cpu"
DEFAULT_TIMEOUT = 1800.0

# 文生图画幅（竖屏 4:7）。首帧与视频必须共用同一画幅，否则 LTX 会把
# 竖图裁/压进横向或更窄的画布，人物肢体就会被裁掉或拉变形。
DEFAULT_T2I_WIDTH = 768
DEFAULT_T2I_HEIGHT = 1344
# LTX 分辨率必须是 32 的倍数（官方模型卡约束）。
LTX_SIZE_MULTIPLE = 32
# LTX 单帧像素预算：显存越大可以给越高。
# 2B / 8-12GB 档：512x768；13B / 24-32GB 档：576x1024（约 1.5 倍像素）。
DEFAULT_LTX_MAX_PIXELS = 512 * 768
LTX_13B_MAX_PIXELS = 576 * 1024
# 轮询指数退避：1s 起步、1.5 倍增长、5s 封顶（短视频任务比固定 5s 省 10-20s）
POLL_START = 1.0
POLL_MAX = 5.0
# 采样步数：LTX 0.9.5 是 distilled 蒸馏版，官方建议 8-13 步；
# 30 步不会更好，只会慢 2-3 倍。SDXL 25 步 dpmpp_2m 足够。
DEFAULT_T2I_STEPS = 25
DEFAULT_LTX_STEPS = 10
# T2I 采样参数（SDXL 系）。CFG/sampler/scheduler 可按 checkpoint 微调：
# SDXL Base 7.0 / dpmpp_2m / karras；RealVisXL V5 等写实系建议 CFG 6 左右。
DEFAULT_T2I_CFG = 7.0
DEFAULT_T2I_SAMPLER = "dpmpp_2m"
DEFAULT_T2I_SCHEDULER = "karras"
# WEBM 中转片段质量：VP9 CRF。旧值 32 压缩过狠，渐变/暗部会有色带，
# 且这些片段之后还要再经历两次转码；20 足够干净且文件仍小。
DEFAULT_WEBM_CRF = 20.0

# 自动挑选替代 checkpoint 时的优先级（t2i）：写实/通用 SDXL 系模型。
# LTX 是图生视频模型，不能拿来文生图。
_T2I_CHECKPOINT_HINTS = (
    "realvis", "sd_xl", "sdxl", "juggernaut", "dreamshaper", "pony", "illustrious",
)
CHECKPOINT_LOADER = "CheckpointLoaderSimple"

# LoRA：文生图用 LoraLoader（model + clip 双输出），视频用
# LoraLoaderModelOnly——LTX checkpoint 不带 CLIP（T5 单独经 CLIPLoader
# 加载），接不了 LoraLoader 的 clip 输入。
LORA_LOADER = "LoraLoader"
LORA_MODEL_ONLY_LOADER = "LoraLoaderModelOnly"
DEFAULT_LORA_STRENGTH = 0.8

DEFAULT_T2I_NEGATIVE = (
    "low quality, worst quality, deformed, distorted, disfigured, "
    "bad anatomy, bad hands, extra fingers, fused fingers, extra limbs, "
    "malformed limbs, ugly, watermark, text"
)
DEFAULT_LTX_NEGATIVE = (
    "low quality, worst quality, deformed, distorted, disfigured, "
    "motion smear, motion artifacts, fused fingers, extra fingers, "
    "missing fingers, extra limbs, malformed limbs, bad anatomy, "
    "bad hands, weird hand, twisted body, body morphing, ugly"
)

# 角色一致性（IPAdapter 参考图注入）：需在 ComfyUI 安装
# ComfyUI_IPAdapter_plus 自定义节点 + 对应模型（见 docs/LOCAL_SETUP.md）。
# 预设决定 Unified Loader 加载哪个 adapter 模型：
# "PLUS FACE (portraits)" 只需 ip-adapter-plus-face_sdxl + CLIP-ViT-H；
# FACEID 系列更强但需要 insightface。
DEFAULT_IPADAPTER_PRESET = "PLUS FACE (portraits)"
DEFAULT_IPADAPTER_WEIGHT = 0.8

# 配音（IndexTTS-2.5，经 chenpipi0807/ComfyUI-Index-TTS 自定义节点，
# 见 docs/LOCAL_SETUP.md）。默认音色取插件 TimbreModel 目录自带的
# "甜美女声"，全部候选见该目录（59 个，含新闻女声/说书人/御姐等）；
# PROMPT2VID_TTS_REF_AUDIO 指向自己的录音时可克隆自定义音色。
DEFAULT_TTS_TIMBRE = "甜美女声.mp3"
DEFAULT_TTS_LANG = "ZH"
DEFAULT_TTS_SPEED = 1.0
# 句间停顿（毫秒）：IndexTTS 的 interval_silence_ms，200ms 让长句
# 旁白有自然换气感，而不是一口气念完
DEFAULT_TTS_SILENCE_MS = 200

# 口型同步（LatentSync 1.6，经 ShmuelRonen/ComfyUI-LatentSyncWrapper
# 自定义节点 + VideoHelperSuite，见 docs/LOCAL_SETUP.md）。
# lips_expression 即 guidance_scale：1.0-1.5 平静语速自然，2.0-3.0
# 演讲式夸张；推理步数 20 够用，30-50 更慢更稳。
DEFAULT_LIPSYNC_EXPRESSION = 1.5
DEFAULT_LIPSYNC_STEPS = 20
DEFAULT_LIPSYNC_CRF = 19

# LTX 输出帧率（与 LTXVConditioning.frame_rate 保持一致）。
# 经 COMFYUI_LTX_FPS 配置：stages 的时长→帧数换算、LTXVConditioning、
# SaveWEBM 与剪辑阶段的 fps 必须一致，否则配音/时长会对不上。
LTX_FPS = 25.0


def ltx_fps() -> float:
    """当前 LTX 帧率（COMFYUI_LTX_FPS，默认 25；非法值回退默认）。"""
    fps = env_float("COMFYUI_LTX_FPS", LTX_FPS)
    return fps if fps > 0 else LTX_FPS

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


def _snap_multiple(value: float, multiple: int = LTX_SIZE_MULTIPLE) -> int:
    """向下取整到 multiple 的倍数（至少一个 multiple）。"""
    return max(multiple, int(value) // multiple * multiple)


def image_dimensions(path: str | Path) -> tuple[int, int] | None:
    """读取 PNG 头里的宽高（首帧都是 ComfyUI 输出的 PNG）。

    只读 24 字节，不加载整图；非 PNG / 读取失败返回 None，由调用方回退到
    环境变量里的默认画幅（测试用的假文件就是这种情况）。
    """
    try:
        with Path(path).open("rb") as f:
            head = f.read(24)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    width = int.from_bytes(head[16:20], "big")
    height = int.from_bytes(head[20:24], "big")
    if width <= 0 or height <= 0:
        return None
    return width, height


def default_t2i_size() -> tuple[int, int]:
    """文生图画幅（COMFYUI_T2I_WIDTH/HEIGHT，默认 768x1344 竖屏）。"""
    return (
        env_int("COMFYUI_T2I_WIDTH", DEFAULT_T2I_WIDTH),
        env_int("COMFYUI_T2I_HEIGHT", DEFAULT_T2I_HEIGHT),
    )


def is_ltx_13b(checkpoint: str | None = None) -> bool:
    name = checkpoint or env_str("COMFYUI_LTX_CHECKPOINT", DEFAULT_LTX_CHECKPOINT)
    return "13b" in name.lower()


def ltx_max_pixels(checkpoint: str | None = None) -> int:
    """LTX 单帧像素预算：显式 COMFYUI_LTX_MAX_PIXELS 优先，否则按模型档位。"""
    override = env_int("COMFYUI_LTX_MAX_PIXELS", 0)
    if override > 0:
        return override
    return LTX_13B_MAX_PIXELS if is_ltx_13b(checkpoint) else DEFAULT_LTX_MAX_PIXELS


def fit_ltx_size(
    width: int, height: int, max_pixels: int | None = None
) -> tuple[int, int]:
    """把任意画幅换算成 LTX 可用的分辨率。

    约束：
    - 宽高均为 32 的倍数（LTX 官方硬约束）；
    - 像素数不超过预算（显存）；
    - 尽量保持首帧的宽高比，且绝不上采样。

    这一步是修复「人物变形」的关键：此前首帧是 768x1344，视频却固定按
    512x768 生成，宽高比 0.571 → 0.667，LTX 只能裁切/挤压首帧，人物
    头脚被裁、身体比例失真。
    """
    width = max(LTX_SIZE_MULTIPLE, int(width))
    height = max(LTX_SIZE_MULTIPLE, int(height))
    budget = max(LTX_SIZE_MULTIPLE * LTX_SIZE_MULTIPLE, int(max_pixels or DEFAULT_LTX_MAX_PIXELS))

    if (
        width % LTX_SIZE_MULTIPLE == 0
        and height % LTX_SIZE_MULTIPLE == 0
        and width * height <= budget
    ):
        return width, height

    ratio = width / height
    # 允许 2% 的宽高比误差，优先取预算内更大的分辨率
    tolerance = 0.02
    upper = int(budget * (1 + tolerance))
    max_w = min(width, int(math.sqrt(upper * ratio)) + LTX_SIZE_MULTIPLE)
    best: tuple[int, int, int] | None = None  # (面积, 宽, 高)
    for candidate_w in range(LTX_SIZE_MULTIPLE, max_w + 1, LTX_SIZE_MULTIPLE):
        ideal_h = candidate_w / ratio
        steps = {
            int(round(ideal_h / LTX_SIZE_MULTIPLE)) * LTX_SIZE_MULTIPLE,
            int(ideal_h // LTX_SIZE_MULTIPLE) * LTX_SIZE_MULTIPLE,
            int(math.ceil(ideal_h / LTX_SIZE_MULTIPLE)) * LTX_SIZE_MULTIPLE,
        }
        for candidate_h in steps:
            if candidate_h < LTX_SIZE_MULTIPLE or candidate_h > height:
                continue
            if candidate_w > width:
                continue
            area = candidate_w * candidate_h
            if area > upper:
                continue
            if abs((candidate_w / candidate_h) - ratio) / ratio > tolerance:
                continue
            if best is None or area > best[0]:
                best = (area, candidate_w, candidate_h)

    if best is not None:
        return best[1], best[2]

    # 极端画幅兜底：预算内的正方形，至少保证 32 倍数
    side = _snap_multiple(math.sqrt(budget))
    return side, side


def resolve_ltx_size(
    width: int, height: int, checkpoint: str | None = None
) -> tuple[int, int]:
    """把首帧尺寸换算为 LTX 画布；显式 COMFYUI_LTX_WIDTH/HEIGHT 时直接采用。"""
    env_w = env_int("COMFYUI_LTX_WIDTH", 0)
    env_h = env_int("COMFYUI_LTX_HEIGHT", 0)
    if env_w > 0 and env_h > 0:
        return (
            _snap_multiple(env_w),
            _snap_multiple(env_h),
        )
    return fit_ltx_size(width, height, ltx_max_pixels(checkpoint))


def default_ltx_size(checkpoint: str | None = None) -> tuple[int, int]:
    """按文生图画幅推导 LTX 画幅（不传首帧尺寸时使用）。"""
    width, height = default_t2i_size()
    return resolve_ltx_size(width, height, checkpoint)


def ltx_sampling_defaults(checkpoint: str | None = None) -> tuple[float, int]:
    """按 LTX 模型档位返回 (cfg, steps) 默认值。

    - 0.9.7/0.9.8 distilled：Lightricks 官方节点说明是 cfg=1.0、8 步左右；
      用 3.0 会把蒸馏模型「过采样」，肢体/手部更容易崩。
    - 13B dev：cfg=3.0、30 步。
    - 2B 0.9.5：cfg=3.0、10 步。
    COMFYUI_LTX_CFG / COMFYUI_LTX_STEPS 可显式覆盖。
    """
    name = (checkpoint or env_str("COMFYUI_LTX_CHECKPOINT", DEFAULT_LTX_CHECKPOINT)).lower()
    if "distilled" in name:
        return 1.0, 8
    if "13b" in name or "-dev" in name:
        return 3.0, 30
    return 3.0, DEFAULT_LTX_STEPS


def next_seed() -> int:
    """取下一个种子：设了 PROMPT2VID_SEED 则 base+n（可复现），否则随机。"""
    base = os.getenv("PROMPT2VID_SEED", "").strip()
    if base:
        try:
            return int(base) + next(_SEED_COUNTER)
        except ValueError:
            pass
    return random.randint(0, 2**32 - 1)


def combo_values(node_spec: dict[str, Any], field_name: str) -> list[str]:
    """从 object_info 的节点定义里取 combo 候选值。

    兼容三种真实格式：
    - 新版 ComfyUI：`{"input": {"required": {"ckpt_name": [[...候选...], {...}]}}}`
      （候选列表外面还包了一层 [值列表, 配置]）；
    - `{"input": {"ckpt_name": [...]}}`；
    - `{"input": {"ckpt_name": {"value": [...]}}}`。
    """
    inputs = node_spec.get("input") or {}
    raw: Any = inputs.get(field_name)
    if raw is None:
        for group in ("required", "optional"):
            group_inputs = inputs.get(group)
            if isinstance(group_inputs, dict) and field_name in group_inputs:
                raw = group_inputs[field_name]
                break
        else:
            return []

    # 新版 ComfyUI：候选包在列表里，可能带配置尾项
    # （checkpoint: [[候选...], {"tooltip": ...}]；CLIPLoader: [[候选...]]）
    while isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]

    if isinstance(raw, dict):
        for key in ("value", "values", "options"):
            values = raw.get(key)
            if isinstance(values, list):
                return [str(v) for v in values]
        return []
    return [str(v) for v in raw] if isinstance(raw, list) else []


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
            # "audio" 是 SaveAudio 的输出键（IndexTTS 配音经它落盘）；
            # "gifs" 是 VHS_VideoCombine 的历史遗留键（口型同步产物）
            for key in ("images", "gifs", "videos", "images_preview", "audio"):
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
        # 已解析的模型文件名缓存（role → 文件名；LoRA 的空串表示"已确认关闭"）
        self._checkpoint_cache: dict[str, str] = {}
        # object_info 的节点定义缓存（class_type → 定义；空 dict = 查询失败/不存在）
        self._node_spec_cache: dict[str, dict[str, Any]] = {}

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ComfyUIClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def object_info(self, node: str) -> dict[str, Any]:
        """GET /object_info/<node>；服务不支持或出错时返回空 dict。"""
        try:
            resp = self._http.get(f"{self.base_url}/object_info/{node}", timeout=10.0)
        except httpx.HTTPError:
            return {}
        if resp.status_code != 200:
            return {}
        try:
            data = resp.json()
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    def node_spec(self, node: str) -> dict[str, Any]:
        """节点定义（object_info 结果进程内缓存）；不存在/查询失败返回空 dict。"""
        if node not in self._node_spec_cache:
            self._node_spec_cache[node] = self.object_info(node).get(node) or {}
        return self._node_spec_cache[node]

    def node_exists(self, node: str) -> bool:
        """ComfyUI 是否注册了该节点（自定义节点是否安装的判定依据）。"""
        return bool(self.node_spec(node))

    def declared_inputs(self, node: str) -> set[str]:
        """节点声明的全部输入名（required + optional + hidden + 扁平旧格式）。

        空集合表示定义未知（节点不存在 / object_info 不可用）。
        """
        spec = self.node_spec(node)
        inputs = spec.get("input") or {}
        if not isinstance(inputs, dict):
            return set()
        # 扁平旧格式的字段名直接挂在 input 下；嵌套格式下顶层键是
        # 分组名（required/optional/hidden），不能算输入名
        names = {
            key for key in inputs
            if isinstance(key, str) and key not in ("required", "optional", "hidden")
        }
        for group in ("required", "optional", "hidden"):
            group_inputs = inputs.get(group)
            if isinstance(group_inputs, dict):
                names.update(group_inputs.keys())
        return names

    def filter_workflow_inputs(self, workflow: dict[str, Any]) -> dict[str, Any]:
        """按服务端节点定义过滤工作流里未声明的输入键（返回副本）。

        自定义节点（VideoHelperSuite 等）不同版本的输入集有差异：送出
        未声明的键会被 ComfyUI 静默忽略，但缺 required 键会被 400 拒绝。
        这里只保留服务端声明过的键，让同一份构建器适配新旧版本；定义
        查询失败时原样保留（由 submit 的 400 错误兜底定位）。
        """
        filtered: dict[str, Any] = {}
        for node_id, node in workflow.items():
            class_type = str(node.get("class_type") or "")
            declared = self.declared_inputs(class_type) if class_type else set()
            if not declared:
                filtered[node_id] = node
                continue
            inputs = {
                key: value
                for key, value in (node.get("inputs") or {}).items()
                if key in declared
            }
            filtered[node_id] = {**node, "inputs": inputs}
        return filtered

    def available_checkpoints(self) -> list[str]:
        """ComfyUI 当前可用的 checkpoint 文件名（查询失败时为空列表）。"""
        spec = self.object_info(CHECKPOINT_LOADER).get(CHECKPOINT_LOADER) or {}
        return combo_values(spec, "ckpt_name")

    def resolve_checkpoint(self, role: str = "t2i") -> str:
        """返回实际可用的 checkpoint 文件名（配置名缺失时自动降级并告警）。

        云机上模型文件名常与 `.env` 默认值不一致（例如只下到
        `RealVisXL_V5.0_fp16.safetensors`），旧行为是每个分镜都收到
        HTTP 400 `value_not_in_list`，整段流水线全失败。这里先查
        `/object_info/CheckpointLoaderSimple`：配置值在候选里就原样用，
        不在则按角色挑同类模型（t2i 排除 LTX 视频模型；ltx 只选文件名
        含 `ltx` 的）。查询失败（旧版 ComfyUI / 网络异常）时返回配置值，
        保持原行为。
        """
        if role in self._checkpoint_cache:
            return self._checkpoint_cache[role]
        if role == "ltx":
            requested = env_str("COMFYUI_LTX_CHECKPOINT", DEFAULT_LTX_CHECKPOINT)
        else:
            requested = env_str("COMFYUI_T2I_CHECKPOINT", DEFAULT_T2I_CHECKPOINT)
        candidates = self.available_checkpoints()
        chosen = requested
        if candidates and requested not in candidates:
            usable = self._rank_checkpoints(candidates, role)
            if usable:
                chosen = usable[0]
                logger.warning(
                    "ComfyUI: %s checkpoint %s 不可用，自动改用 %s（服务端可用: %s）",
                    role, requested, chosen, ", ".join(candidates),
                )
        self._checkpoint_cache[role] = chosen
        return chosen

    @staticmethod
    def _rank_checkpoints(candidates: list[str], role: str) -> list[str]:
        """按角色对候选 checkpoint 排序（相似度优先，其余按名字稳定排序）。"""
        if role == "ltx":
            usable = [c for c in candidates if "ltx" in c.lower()]

            def ltx_score(name: str) -> tuple[int, str]:
                low = name.lower()
                if "13b" in low and "distilled" in low:
                    return 0, low  # 5090/24G+ 首选：质量最好且采样便宜
                if "13b" in low:
                    return 1, low  # 13B dev：质量高但需要更多步数
                if "distilled" in low:
                    return 2, low  # 2B distilled
                return 3, low  # 2B 0.9.5 / 其它

            usable.sort(key=ltx_score)
            return usable

        def score(name: str) -> int:
            low = name.lower()
            for index, hint in enumerate(_T2I_CHECKPOINT_HINTS):
                if hint in low:
                    return index
            return len(_T2I_CHECKPOINT_HINTS)

        usable = [c for c in candidates if "ltx" not in c.lower()]
        usable.sort(key=lambda c: (score(c), c))
        return usable

    def available_loras(self, model_only: bool = False) -> list[str]:
        """ComfyUI 当前可用的 LoRA 文件名（查询失败时为空列表）。"""
        node = LORA_MODEL_ONLY_LOADER if model_only else LORA_LOADER
        spec = self.object_info(node).get(node) or {}
        return combo_values(spec, "lora_name")

    def resolve_lora(self, role: str = "t2i") -> str:
        """返回该档位应使用的 LoRA 文件名；不可用时告警并返回空串（关闭）。

        与 checkpoint 不同，LoRA 是风格增强件：文件缺失 / 名字与服务端
        不一致时降级为无 LoRA 生成，不让整条流水线失败。也不会自动挑
        别的 LoRA 顶替——风格是刻意配置，静默换一个等于换风格。

        Args:
            role: "t2i"（读 COMFYUI_T2I_LORA）或 "ltx"（读 COMFYUI_LTX_LORA）。
        """
        cache_key = f"{role}_lora"
        if cache_key in self._checkpoint_cache:
            return self._checkpoint_cache[cache_key]
        if role == "ltx":
            requested = env_str("COMFYUI_LTX_LORA", "")
            model_only = True
        else:
            requested = env_str("COMFYUI_T2I_LORA", "")
            model_only = False
        chosen = requested
        if requested:
            candidates = self.available_loras(model_only)
            if candidates and requested not in candidates:
                logger.warning(
                    "ComfyUI: %s LoRA %s 不可用，本次不挂 LoRA（服务端可用: %s）",
                    role, requested, ", ".join(candidates[:8]),
                )
                chosen = ""
        self._checkpoint_cache[cache_key] = chosen
        return chosen

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

    def upload_file(self, file_path: Path, name: str | None = None) -> str:
        """上传任意文件（图片/音频/视频）到 ComfyUI input 目录，返回文件名。

        ComfyUI 的 /upload/image 不校验扩展名，VHS/LatentSync 等自定义
        节点的音频、视频同样经它上传，之后用文件名在节点里引用。
        """
        file_path = Path(file_path)
        name = name or file_path.name
        mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        with file_path.open("rb") as f:
            resp = self._http.post(
                f"{self.base_url}/upload/image",
                files={"image": (name, f, mime)},
                data={"overwrite": "true"},
                timeout=120.0,
            )
        resp.raise_for_status()
        return resp.json()["name"]

    def upload_image(self, image_path: Path, name: str | None = None) -> str:
        """上传图片到 ComfyUI input 目录，返回上传后的文件名。"""
        image_path = Path(image_path)
        return self.upload_file(image_path, name or image_path.name)


def build_t2i_workflow(
    prompt: str,
    negative_prompt: str = "",
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    reference: dict[str, Any] | None = None,
    checkpoint: str | None = None,
    lora: str | None = None,
) -> dict[str, Any]:
    """SDXL 文生图工作流（API 格式）。

    节点链：CheckpointLoaderSimple → [LoraLoader] → CLIPTextEncode(正/负) →
    EmptyLatentImage → KSampler → VAEDecode → SaveImage

    Args:
        checkpoint: 覆盖 `COMFYUI_T2I_CHECKPOINT` / 默认值；调用方通常传
            `ComfyUIClient.resolve_checkpoint("t2i")` 的返回值，以适配云机上
            实际存在的模型文件名。
        lora: 覆盖 `COMFYUI_T2I_LORA`（空串 = 不挂 LoRA）；通常传
            `ComfyUIClient.resolve_lora("t2i")` 的返回值。权重读
            COMFYUI_T2I_LORA_STRENGTH（默认 0.8）。
        reference: 角色参考图注入（IPAdapter）。传 {"image_name": 已上传的
            参考图名} 即可，weight/preset 缺省读 COMFYUI_IPADAPTER_WEIGHT /
            COMFYUI_IPADAPTER_PRESET。节点链变为：
            CheckpointLoader → LoraLoader → IPAdapterUnifiedLoader →
            IPAdapter(参考图) → KSampler，需要 ComfyUI 安装
            ComfyUI_IPAdapter_plus。

    风格约束（PROMPT2VID_STYLE，见 style.py）：正向提示词自动追加风格
    正向词、负向提示词自动追加风格排斥词，调用方无需自行拼接。
    """
    ckpt = checkpoint or env_str("COMFYUI_T2I_CHECKPOINT", DEFAULT_T2I_CHECKPOINT)
    styled_prompt = apply_style(prompt)
    negative = negative_prompt or env_str("COMFYUI_NEGATIVE_PROMPT", DEFAULT_T2I_NEGATIVE)
    negative = apply_style_negative(negative)
    if width is None or height is None:
        default_width, default_height = default_t2i_size()
        width = default_width if width is None else width
        height = default_height if height is None else height
    workflow: dict[str, Any] = {
        "44": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": styled_prompt, "clip": ["44", 1]},
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
                "cfg": env_float("COMFYUI_T2I_CFG", DEFAULT_T2I_CFG),
                "sampler_name": env_str("COMFYUI_T2I_SAMPLER", DEFAULT_T2I_SAMPLER),
                "scheduler": env_str("COMFYUI_T2I_SCHEDULER", DEFAULT_T2I_SCHEDULER),
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
    # LoRA 链在 checkpoint 之后、文本编码与采样之前（IPAdapter 之前）：
    # 文本编码器用 LoRA 之后的 CLIP，采样器用 LoRA 之后的 MODEL
    model_src: list[Any] = ["44", 0]
    lora_name = lora if lora is not None else env_str("COMFYUI_T2I_LORA", "")
    if lora_name:
        strength = env_float("COMFYUI_T2I_LORA_STRENGTH", DEFAULT_LORA_STRENGTH)
        workflow["lora"] = {
            "class_type": LORA_LOADER,
            "inputs": {
                "lora_name": lora_name,
                "strength_model": strength,
                "strength_clip": strength,
                "model": ["44", 0],
                "clip": ["44", 1],
            },
        }
        model_src = ["lora", 0]
        for text_node in ("6", "7"):
            workflow[text_node]["inputs"]["clip"] = ["lora", 1]
    if reference and reference.get("image_name"):
        workflow["3"]["inputs"]["model"] = _apply_ipadapter(workflow, reference, model_src)
    elif lora_name:
        workflow["3"]["inputs"]["model"] = model_src
    return workflow


def _apply_ipadapter(
    workflow: dict[str, Any], reference: dict[str, Any], model_src: list[Any]
) -> list[Any]:
    """注入 IPAdapter 参考图节点链，返回 KSampler 应使用的 MODEL 输出。

    model_src 是 IPAdapter 之前的 MODEL 来源（checkpoint 直出，或经过
    LoRA 之后的输出），保证 LoRA 与 IPAdapter 可以叠加。

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
        "inputs": {"model": model_src, "ipadapter": preset},
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
    width: int | None = None,
    height: int | None = None,
    length: int = 41,
    seed: int | None = None,
    checkpoint: str | None = None,
    lora: str | None = None,
) -> dict[str, Any]:
    """LTX-Video 图生视频工作流（API 格式）。

    节点链（对照官方 ltxv_image_to_video_simple.0.9.5.json）：
    CLIPLoader(T5) → CLIPTextEncode(正/负) → LTXVImgToVideo → LTXVConditioning →
    SamplerCustom ← KSamplerSelect(euler) + LTXVScheduler；
    CheckpointLoaderSimple(LTX) → [LoraLoaderModelOnly] 提供 MODEL；VAE；
    LoadImage → LTXVPreprocess；VAEDecode → SaveWEBM

    Args:
        width/height: 不传时按首帧默认画幅（COMFYUI_T2I_WIDTH/HEIGHT）
            推导出 32 倍数且符合显存预算的 LTX 画幅。调用方应尽量传首帧
            的真实宽高，保证画幅一致，避免人物被裁切/拉伸。
        checkpoint: 覆盖 `COMFYUI_LTX_CHECKPOINT` / 默认值；通常传
            `ComfyUIClient.resolve_checkpoint("ltx")` 的返回值。cfg/步数
            会按模型档位自动选择（distilled：cfg 1.0 / 8 步；13B dev：
            cfg 3.0 / 30 步；2B：cfg 3.0 / 10 步）。
        lora: 覆盖 `COMFYUI_LTX_LORA`（空串 = 不挂 LoRA）；通常传
            `ComfyUIClient.resolve_lora("ltx")` 的返回值。LTX checkpoint
            不带 CLIP，所以用只接 MODEL 的 LoraLoaderModelOnly。权重读
            COMFYUI_LTX_LORA_STRENGTH（默认 0.8）。

    风格约束（PROMPT2VID_STYLE，见 style.py）：正向提示词自动追加风格
    正向词、负向提示词自动追加风格排斥词——视频风格跟随首帧之余，
    提示词层再钉一次风格。
    """
    ltx_ckpt = checkpoint or env_str("COMFYUI_LTX_CHECKPOINT", DEFAULT_LTX_CHECKPOINT)
    if width is None or height is None:
        default_width, default_height = default_ltx_size(ltx_ckpt)
        width = default_width if width is None else width
        height = default_height if height is None else height
    cfg_default, steps_default = ltx_sampling_defaults(ltx_ckpt)
    cfg = env_float("COMFYUI_LTX_CFG", cfg_default)
    steps = env_int("COMFYUI_LTX_STEPS", steps_default)
    img_compression = env_int("COMFYUI_LTX_IMG_COMPRESSION", 35)
    t5 = env_str("COMFYUI_T5", DEFAULT_T5)
    t5_device = env_str("COMFYUI_T5_DEVICE", DEFAULT_T5_DEVICE)
    styled_prompt = apply_style(prompt)
    negative = negative_prompt or env_str("COMFYUI_LTX_NEGATIVE_PROMPT", DEFAULT_LTX_NEGATIVE)
    negative = apply_style_negative(negative)
    # LoRA 只接 MODEL（LTX checkpoint 的 CLIP 输出为空，接 LoraLoader 会
    # 在预校验阶段被拒）
    model_src: list[Any] = ["44", 0]
    lora_name = lora if lora is not None else env_str("COMFYUI_LTX_LORA", "")
    if lora_name:
        model_src = ["lora", 0]
    workflow = {
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
            "inputs": {"text": styled_prompt, "clip": ["38", 0]},
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
            "inputs": {"image": ["78", 0], "img_compression": img_compression},
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
                "frame_rate": ltx_fps(),
            },
        },
        # max_shift/base_shift 是官方 simple 工作流的默认值（按 512-768p
        # 调校）；分辨率/帧数上去后可经 COMFYUI_LTX_MAX_SHIFT /
        # COMFYUI_LTX_BASE_SHIFT 微调（详见 docs/QUALITY.md）。
        "71": {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps": steps,
                "max_shift": env_float("COMFYUI_LTX_MAX_SHIFT", 2.05),
                "base_shift": env_float("COMFYUI_LTX_BASE_SHIFT", 0.95),
                "stretch": True,
                "terminal": 0.1,
            },
        },
        "73": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": env_str("COMFYUI_LTX_SAMPLER", "euler"),
            },
        },
        "72": {
            "class_type": "SamplerCustom",
            "inputs": {
                "add_noise": True,
                "noise_seed": seed if seed is not None else next_seed(),
                "cfg": cfg,
                "model": model_src,
                "positive": ["69", 0],
                "negative": ["69", 1],
                "sampler": ["73", 0],
                "sigmas": ["71", 0],
                # latent 直接取 LTXVImgToVideo 的第 3 个输出（index 2）。
                # 不能接 LTXVConditioning 的 index 2：该节点只有 2 个输出
                # （positive/negative），引用越界会让 ComfyUI 在预校验阶段
                # 报 "list index out of range"（HTTP 400）。
                "latent_image": ["95", 2],
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
                "fps": ltx_fps(),
                "crf": env_float("COMFYUI_WEBM_CRF", DEFAULT_WEBM_CRF),
            },
        },
    }
    if lora_name:
        workflow["lora"] = {
            "class_type": LORA_MODEL_ONLY_LOADER,
            "inputs": {
                "lora_name": lora_name,
                "strength_model": env_float(
                    "COMFYUI_LTX_LORA_STRENGTH", DEFAULT_LORA_STRENGTH
                ),
                "model": ["44", 0],
            },
        }
    return workflow


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
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    client: ComfyUIClient | None = None,
    checkpoint: str | None = None,
    lora: str | None = None,
) -> Path:
    """SDXL 文生图：生成一张图片并保存到 dest（尺寸缺省读环境变量）。"""
    dest = Path(dest)
    workflow = build_t2i_workflow(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        seed=seed,
        checkpoint=checkpoint,
        lora=lora,
    )
    return run_workflow(workflow, dest, client)


def run_image_to_video(
    image: str | Path,
    dest: str | Path,
    prompt: str,
    negative_prompt: str = "",
    width: int | None = None,
    height: int | None = None,
    length: int = 41,
    seed: int | None = None,
    client: ComfyUIClient | None = None,
    checkpoint: str | None = None,
    lora: str | None = None,
) -> Path:
    """LTX-Video 图生视频：把首帧图转为视频片段并保存到 dest。

    首帧图会自动上传到 ComfyUI。length 必须满足 8n+1（LTX 约束）；
    width/height 不传时会读取首帧真实宽高，换算成 32 倍数且符合显存
    预算的画幅，避免画幅不一致导致人物变形。
    """
    image = Path(image)
    if not image.is_file():
        raise FileNotFoundError(f"首帧图不存在: {image}")
    if width is None or height is None:
        frame_size = image_dimensions(image)
        frame_w, frame_h = frame_size or default_t2i_size()
        resolved_w, resolved_h = resolve_ltx_size(frame_w, frame_h, checkpoint)
        width = resolved_w if width is None else width
        height = resolved_h if height is None else height
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
        checkpoint=checkpoint,
        lora=lora,
    )
    return run_workflow(workflow, Path(dest), client)


def build_tts_workflow(
    text: str,
    filename_prefix: str = "prompt2vid_tts",
    reference_audio: str | None = None,
    timbre: str | None = None,
    lang: str | None = None,
    speed: float | None = None,
    silence_ms: int | None = None,
    keep_cached: bool | None = None,
) -> dict[str, Any]:
    """IndexTTS-2.5 配音工作流（API 格式）。

    节点链（需在 ComfyUI 安装 chenpipi0807/ComfyUI-Index-TTS）：
    TimbreAudioLoader（自带音色）或 LoadAudio（自定义参考音频，声音克隆）
    → [IndexTTS25CacheControlNode] → IndexTTS25BaseNode → SaveAudio

    Args:
        text: 待合成文本（中文）。
        reference_audio: 已上传到 ComfyUI input 目录的参考音频文件名
            （3-15 秒干净人声，优先于 timbre——克隆自定义音色用）。
        timbre: 插件 TimbreModel 目录里的自带音色文件名；缺省读
            PROMPT2VID_TTS_TIMBRE（默认 "甜美女声.mp3"）。
        lang: 语种（ZH/EN/JA/ES/AR/ZH/EN），缺省读 PROMPT2VID_TTS_LANG。
        speed: duration_factor（>1 变慢、<1 变快），缺省读 PROMPT2VID_TTS_SPEED。
        silence_ms: 句间停顿毫秒数（interval_silence_ms），缺省读
            PROMPT2VID_TTS_SILENCE_MS（默认 200，长句旁白有换气感）。
        keep_cached: 推理后模型是否留在显存（整批逐镜合成时避免反复
            加载 ~8GB 模型）；缺省读 PROMPT2VID_TTS_KEEP_CACHED（默认开）。
    """
    timbre = timbre or env_str("PROMPT2VID_TTS_TIMBRE", DEFAULT_TTS_TIMBRE)
    lang_value = lang or env_str("PROMPT2VID_TTS_LANG", DEFAULT_TTS_LANG)
    speed_value = env_float("PROMPT2VID_TTS_SPEED", DEFAULT_TTS_SPEED) if speed is None else speed
    silence = (
        env_int("PROMPT2VID_TTS_SILENCE_MS", DEFAULT_TTS_SILENCE_MS)
        if silence_ms is None else silence_ms
    )
    if keep_cached is None:
        keep_cached = os.getenv("PROMPT2VID_TTS_KEEP_CACHED", "1").strip().lower() not in (
            "0", "false", "no", "off", "",
        )

    reference_link: list[Any]
    if reference_audio:
        # 声音克隆：上传的参考音频经核心 LoadAudio 节点加载
        reference_node = {
            "class_type": "LoadAudio",
            "inputs": {"audio": reference_audio},
        }
    else:
        reference_node = {
            "class_type": "TimbreAudioLoader",
            "inputs": {"audio_file": timbre, "refresh": False},
        }

    tts_inputs: dict[str, Any] = {
        "text": text,
        "reference_audio": ["timbre", 0],
        "lang": lang_value,
        "duration_factor": speed_value,
        "interval_silence_ms": silence,
    }
    workflow: dict[str, Any] = {"timbre": reference_node}
    if keep_cached:
        workflow["cache"] = {
            "class_type": "IndexTTS25CacheControlNode",
            "inputs": {"keep_models_cached": True},
        }
        tts_inputs["cache_control"] = ["cache", 0]
    workflow["tts"] = {
        "class_type": "IndexTTS25BaseNode",
        "inputs": tts_inputs,
    }
    workflow["save"] = {
        "class_type": "SaveAudio",
        "inputs": {"audio": ["tts", 0], "filename_prefix": filename_prefix},
    }
    return workflow


def build_lipsync_workflow(
    clip_name: str,
    audio_name: str,
    mode: str = "pingpong",
    fps: float | None = None,
    filename_prefix: str = "prompt2vid_lipsync",
    seed: int | None = None,
    expression: float | None = None,
    steps: int | None = None,
) -> dict[str, Any]:
    """LatentSync 口型同步工作流（API 格式）。

    节点链（需在 ComfyUI 安装 ShmuelRonen/ComfyUI-LatentSyncWrapper
    与 Kosinkadink/ComfyUI-VideoHelperSuite）：
    VHS_LoadVideo(25fps) + LoadAudio(配音) → VideoLengthAdjuster(长度对齐)
    → LatentSyncNode(唇形重绘) → VHS_VideoCombine(h264 mp4)

    LatentSync 只认 25fps 且需要与音频等长的帧序列：force_rate 强制
    25fps，VideoLengthAdjuster 负责长度对齐——mode=pingpong 在配音更长
    时把片段往返补帧到配音长度（旁白零截断），mode=normal 在片段更长
    时裁掉多余帧并给音频尾部补静音。

    Args:
        clip_name / audio_name: 已上传到 ComfyUI input 目录的片段与配音
            文件名（调用方先 upload_file）。
        mode: "pingpong"（配音更长，补帧）或 "normal"（片段更长，裁帧）。
        fps: 帧率，缺省与 LTX 一致（COMFYUI_LTX_FPS，默认 25）。
        expression: lips_expression（即 guidance_scale），缺省读
            PROMPT2VID_LIPSYNC_EXPRESSION（默认 1.5，平静语速自然）。
        steps: 推理步数，缺省读 PROMPT2VID_LIPSYNC_STEPS（默认 20）。
    """
    fps_value = ltx_fps() if fps is None else fps
    expression_value = (
        env_float("PROMPT2VID_LIPSYNC_EXPRESSION", DEFAULT_LIPSYNC_EXPRESSION)
        if expression is None else expression
    )
    steps_value = (
        env_int("PROMPT2VID_LIPSYNC_STEPS", DEFAULT_LIPSYNC_STEPS)
        if steps is None else steps
    )
    return {
        "load_video": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": clip_name,
                "force_rate": fps_value,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
                "format": "AnimateDiff",
            },
        },
        "load_audio": {
            "class_type": "LoadAudio",
            "inputs": {"audio": audio_name},
        },
        "adjust": {
            "class_type": "VideoLengthAdjuster",
            "inputs": {
                "images": ["load_video", 0],
                "audio": ["load_audio", 0],
                "mode": mode,
                "fps": fps_value,
                "silent_padding_sec": 0.5,
            },
        },
        "latentsync": {
            "class_type": "LatentSyncNode",
            "inputs": {
                "images": ["adjust", 0],
                "audio": ["adjust", 1],
                "seed": seed if seed is not None else next_seed(),
                "lips_expression": expression_value,
                "inference_steps": steps_value,
            },
        },
        "save": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["latentsync", 0],
                "frame_rate": fps_value,
                "loop_count": 0,
                "filename_prefix": filename_prefix,
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
                "codec": "libx264",
                "crf": env_int("PROMPT2VID_LIPSYNC_CRF", DEFAULT_LIPSYNC_CRF),
                "pix_fmt": "yuv420p",
            },
        },
    }
