"""运行前环境预检：ComfyUI 可达性 / 模型就位 / LTX 节点 / ffmpeg。

目标：把"配置好 ComfyUI 后直接运行"变成可验证的事实——配置缺失时
在启动几秒内给出可操作的错误清单，而不是跑了几分钟、生成了半个
分镜表之后才在某个工具里炸掉。

CLI：prompt2vid "描述" 默认自动预检；也可 `prompt2vid --check-only`
单独体检。检查项：
1. ffmpeg 可用（ffprobe 可选：缺失时探测回退解析 ffmpeg 输出）；
2. ComfyUI 可达（GET /system_stats）；
3. T2I checkpoint（CheckpointLoaderSimple 的 ckpt_name 候选中）；
4. LTX checkpoint（同上）；
5. T5 文本编码器（CLIPLoader 的 clip_name 候选中）；
6. LTXV 节点存在（ComfyUI 版本不能太旧）；
7. GPU 显存与画质档位校对（system_stats 的 CUDA 设备；13B/720p
   预算超出显存、IPAdapter 在 8GB 卡上开启时警告——全部非阻断）；
8. 配音可用性（ComfyUI IndexTTS-2.5 / 网关 / piper 离线 / edge-tts
   至少一条路，非阻断）；
9. 口型同步节点（LatentSync + VideoHelperSuite；auto 模式缺节点只
   提示跳过，on 模式缺节点为阻断性配置错误）。
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .comfyui_client import (
    DEFAULT_COMFYUI_URL,
    DEFAULT_LTX_CHECKPOINT,
    DEFAULT_T2I_CHECKPOINT,
    DEFAULT_T5,
    combo_values,
    env_str,
    ipadapter_enabled,
    ltx_max_pixels,
)
from .ffmpeg_ops import resolve_tool
from .style import style_enabled, style_name

POLL_TIMEOUT = 8.0

# edge-tts 连通性探测结果缓存（None = 未探测；进程内只探一次）
_EDGE_TTS_OK: bool | None = None


@dataclass
class Check:
    """单项检查结果。"""

    name: str
    ok: bool
    detail: str = ""
    hint: str = ""
    required: bool = True


@dataclass
class PreflightResult:
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks if c.required)

    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.required and not c.ok]

    def format_report(self) -> str:
        lines = ["===== 环境预检 ====="]
        for c in self.checks:
            mark = "OK " if c.ok else "NG "
            lines.append(f"{mark}{c.name}: {c.detail}")
            if not c.ok and c.hint:
                lines.append(f"    -> {c.hint}")
        if self.passed:
            lines.append("预检通过，可以开始运行。")
        else:
            lines.append("预检未通过，请按上面的提示修复后重试。")
        return "\n".join(lines)


def _combo_values(node_spec: dict[str, Any], field_name: str) -> list[str]:
    """从 object_info 的节点定义里取 combo 候选值（兼容各版 schema）。"""
    return combo_values(node_spec, field_name)


def _fetch_node_info(client: httpx.Client, node: str, base_url: str) -> dict[str, Any]:
    """GET /object_info/<node>，返回该节点定义；404 时返回空。"""
    resp = client.get(f"{base_url}/object_info/{node}")
    if resp.status_code != 200:
        return {}
    return resp.json()


def _append_gpu_checks(
    checks: list[Check], system_stats: dict[str, Any], ltx_ckpt: str
) -> None:
    """GPU 显存与画质档位校对（全部非阻断：只提示，不拦截）。

    显存决定 LTX 模型与分辨率预算：配置超出显卡能力会在生成阶段 OOM，
    配置低于显卡能力则浪费画质。显存阈值按本项目三档（8/24/32GB）划分。
    """
    devices = system_stats.get("devices") or [] if isinstance(system_stats, dict) else []
    gpu = next((d for d in devices if isinstance(d, dict) and d.get("type") == "cuda"), None)
    if gpu is None:
        return  # 旧版 ComfyUI / 无 CUDA：不臆测
    vram_gb = float(gpu.get("vram_total") or 0) / 2**30
    name = str(gpu.get("name") or "CUDA 设备")
    checks.append(
        Check(
            name=f"GPU 显存（{name}）",
            ok=True,
            detail=f"{vram_gb:.0f}GB",
            required=False,
        )
    )
    pixels = ltx_max_pixels(ltx_ckpt)
    ltx_name = ltx_ckpt.lower()
    if "13b" in ltx_name and vram_gb < 16:
        checks.append(
            Check(
                name="档位校对：LTX 13B",
                ok=False,
                detail=f"{vram_gb:.0f}GB 显存放不下 13B fp8（权重约 15.7GB），生成阶段会 OOM",
                hint="换回 2B：COMFYUI_LTX_CHECKPOINT=ltx-video-2b-v0.9.5.safetensors，"
                     "并清空 COMFYUI_LTX_MAX_PIXELS（2B 默认 393216）",
                required=False,
            )
        )
    if pixels > 700_000 and vram_gb < 26:
        checks.append(
            Check(
                name="档位校对：分辨率预算",
                ok=False,
                detail=f"COMFYUI_LTX_MAX_PIXELS={pixels}（720p 档）在 {vram_gb:.0f}GB 显存上偏激进，可能 OOM",
                hint="24GB 档建议 589824（约 576x1024）；OOM 后回调即可",
                required=False,
            )
        )
    elif vram_gb >= 26 and pixels <= 600_000:
        checks.append(
            Check(
                name="档位校对：分辨率预算",
                ok=True,
                detail=f"显存富余（{vram_gb:.0f}GB），可把 COMFYUI_LTX_MAX_PIXELS 提到 921600 上 720p（约 704x1248）",
                required=False,
            )
        )
    if ipadapter_enabled() and vram_gb < 10:
        checks.append(
            Check(
                name="档位校对：IPAdapter",
                ok=False,
                detail=f"COMFYUI_IPADAPTER=true 但只有 {vram_gb:.0f}GB 显存（T2I 阶段需多加载约 3.4GB）",
                hint="8GB 档建议关闭 IPAdapter，角色一致性靠分镜角色卡的文字契约",
                required=False,
            )
        )


def _append_tts_check(
    checks: list[Check],
    comfyui_url: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> None:
    """配音可用性预检（非阻断）：IndexTTS / 网关 / piper / edge-tts。

    优先报告 ComfyUI IndexTTS-2.5（主引擎，需自定义节点），再探测回退链。
    全部不可用时只警告不拦截——静音占位保证流水线能跑完，但成片没旁白，
    提前让用户知道比跑完 20 分钟才发现要好。
    """
    from . import tts as tts_mod
    from .comfyui_client import DEFAULT_TTS_TIMBRE

    engine = os.getenv("PROMPT2VID_TTS_ENGINE", "auto").strip().lower() or "auto"
    ref_audio = tts_mod._ref_audio_path()
    # ComfyUI IndexTTS-2.5（auto/comfyui 引擎时优先探测）
    if comfyui_url and engine in ("auto", "comfyui"):
        try:
            with httpx.Client(timeout=POLL_TIMEOUT, transport=transport) as client:
                tts_spec = _fetch_node_info(client, "IndexTTS25BaseNode", comfyui_url)
                timbre_spec = _fetch_node_info(client, "TimbreAudioLoader", comfyui_url)
        except Exception:  # noqa: BLE001 - object_info 异常按不可用处理
            tts_spec = timbre_spec = {}
        if tts_spec:
            if ref_audio:
                checks.append(Check(
                    name="配音（IndexTTS-2.5）", ok=True,
                    detail=f"ComfyUI 本地推理，克隆参考音频: {Path(ref_audio).name}",
                    required=False,
                ))
                return
            timbre = (
                os.getenv("PROMPT2VID_TTS_TIMBRE", "").strip()
                or DEFAULT_TTS_TIMBRE
            )
            available = _combo_values(timbre_spec.get("TimbreAudioLoader", {}), "audio_file")
            if timbre in available:
                checks.append(Check(
                    name="配音（IndexTTS-2.5）", ok=True,
                    detail=f"ComfyUI 本地推理，音色: {timbre}",
                    required=False,
                ))
                return
            if available:
                checks.append(Check(
                    name="配音（IndexTTS-2.5）", ok=False,
                    detail=f"音色 {timbre} 不在插件自带音色列表里"
                           f"（可选: {', '.join(available[:5])}… 共 {len(available)} 个）",
                    hint="修改 PROMPT2VID_TTS_TIMBRE 为列表中的文件名，或设置 "
                         "PROMPT2VID_TTS_REF_AUDIO 指向 3-15 秒干净人声克隆自定义音色",
                    required=False,
                ))
                return
        elif engine == "comfyui":
            checks.append(Check(
                name="配音（IndexTTS-2.5）", ok=False,
                detail="显式指定 comfyui 引擎但 ComfyUI 缺少 IndexTTS 节点",
                hint="安装 chenpipi0807/ComfyUI-Index-TTS 并运行 TTS2_5_download.py"
                     " 下载模型（见 docs/LOCAL_SETUP.md）；本次运行将走回退链",
                required=False,
            ))

    gateway = os.getenv("PROMPT2VID_TTS_BASE_URL", "").strip()
    if gateway:
        checks.append(Check(
            name="配音（TTS 网关）", ok=True, detail=gateway, required=False,
        ))
        return
    piper_model = tts_mod._piper_model()
    if piper_model and shutil.which("piper"):
        checks.append(Check(
            name="配音（piper 离线）", ok=True,
            detail=f"音色: {Path(piper_model).name}", required=False,
        ))
        return
    # edge-tts 现场试连（公司网络常拦 WSS，握手 403 立刻失败）；
    # 结果进程内缓存，避免预检被反复调用时重复等网络
    global _EDGE_TTS_OK
    if _EDGE_TTS_OK is None:
        _EDGE_TTS_OK = False
        try:
            import asyncio
            import tempfile

            import edge_tts

            with tempfile.TemporaryDirectory() as tmp:
                probe = Path(tmp) / "probe.mp3"
                asyncio.run(asyncio.wait_for(
                    edge_tts.Communicate(text="测试", voice="zh-CN-XiaoxiaoNeural")
                    .save(str(probe)),
                    timeout=10.0,
                ))
                _EDGE_TTS_OK = probe.is_file() and probe.stat().st_size > 0
        except Exception:  # noqa: BLE001 - 任何失败都归为不可用
            pass
    if _EDGE_TTS_OK:
        checks.append(Check(
            name="配音（edge-tts）", ok=True, detail="公网连通",
            required=False,
        ))
        return
    checks.append(Check(
        name="配音（TTS）", ok=False,
        detail="edge-tts / 网关 / piper 均不可用，本次运行的旁白将是静音占位",
        hint="推荐离线方案：pip install piper-tts 并下载中文音色到 models/tts/"
             "（命令见 docs/QUALITY.md §配音）；或配置 PROMPT2VID_TTS_BASE_URL；"
             "或设 EDGE_TTS_PROXY 走代理",
        required=False,
    ))


def check_environment(
    base_url: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> PreflightResult:
    """执行全部预检项，返回 PreflightResult。"""
    comfyui_url = (base_url or env_str("COMFYUI_URL", DEFAULT_COMFYUI_URL)).rstrip("/")
    result = PreflightResult()
    checks = result.checks

    # 1. ffmpeg / ffprobe（ffmpeg 必需；ffprobe 缺失时探测会回退到 ffmpeg 解析）
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool) or resolve_tool(tool)
        ok = path is not None
        detail = path or "未找到"
        if not ok and tool == "ffprobe" and resolve_tool("ffmpeg"):
            ok = True
            detail = "未找到独立 ffprobe，探测将回退用 ffmpeg 解析（功能等价）"
        checks.append(
            Check(
                name=tool,
                ok=ok,
                detail=detail,
                hint=f"请安装 {tool} 并加入 PATH（Docker 镜像已内置，本地需自行安装）",
            )
        )

    # 2. ComfyUI 可达性（顺便取 system_stats，供 GPU 档位校对）
    comfyui_ok = False
    system_stats: dict[str, Any] = {}
    try:
        with httpx.Client(timeout=POLL_TIMEOUT, transport=transport) as client:
            resp = client.get(f"{comfyui_url}/system_stats")
            comfyui_ok = resp.status_code == 200
            detail = resp.text[:80]
            if comfyui_ok:
                try:
                    stats = resp.json()
                except ValueError:
                    stats = {}
                if isinstance(stats, dict):
                    system_stats = stats
    except Exception as exc:  # noqa: BLE001 - 任何网络错误都归为不可达
        detail = str(exc)
    checks.append(
        Check(
            name="ComfyUI 可达",
            ok=comfyui_ok,
            detail=detail if comfyui_ok else f"{comfyui_url} 不可达（{detail}）",
            hint="请确认 ComfyUI 已启动，且 .env 中 COMFYUI_URL 正确（本地默认 http://127.0.0.1:8188）",
        )
    )

    # 3-6. 依赖 ComfyUI 的检查
    if comfyui_ok:
        t2i_ckpt = env_str("COMFYUI_T2I_CHECKPOINT", DEFAULT_T2I_CHECKPOINT)
        ltx_ckpt = env_str("COMFYUI_LTX_CHECKPOINT", DEFAULT_LTX_CHECKPOINT)
        t5 = env_str("COMFYUI_T5", DEFAULT_T5)
        # GPU 显存与画质档位校对（非阻断）
        _append_gpu_checks(checks, system_stats, ltx_ckpt)
        # 配音可用性校对（非阻断：IndexTTS / 网关 / piper / edge-tts 至少一条路）
        _append_tts_check(checks, comfyui_url, transport)
        # 口型同步节点校对（on 模式缺节点会阻断运行，属配置错误）
        lipsync_mode = (
            os.getenv("PROMPT2VID_LIPSYNC", "auto").strip().lower() or "auto"
        )
        if lipsync_mode != "off":
            try:
                with httpx.Client(timeout=POLL_TIMEOUT, transport=transport) as client:
                    lipsync_missing = [
                        node for node in (
                            "LatentSyncNode", "VideoLengthAdjuster",
                            "VHS_LoadVideo", "VHS_VideoCombine",
                        )
                        if not _fetch_node_info(client, node, comfyui_url)
                    ]
            except Exception as exc:  # noqa: BLE001 - object_info 异常按缺节点处理
                lipsync_missing = [f"object_info 查询失败: {exc}"]
            if not lipsync_missing:
                checks.append(Check(
                    name="口型同步（LatentSync）", ok=True,
                    detail="LatentSync + VideoHelperSuite 节点已注册",
                    required=False,
                ))
            else:
                checks.append(Check(
                    name="口型同步（LatentSync）", ok=False,
                    detail="缺少节点: " + ", ".join(lipsync_missing) + (
                        "（PROMPT2VID_LIPSYNC=on，运行将在口型同步阶段失败）"
                        if lipsync_mode == "on" else "（auto 模式将跳过口型同步，保留原片段）"
                    ),
                    hint="安装 ShmuelRonen/ComfyUI-LatentSyncWrapper 与 "
                         "Kosinkadink/ComfyUI-VideoHelperSuite（见 docs/LOCAL_SETUP.md），"
                         "或设 PROMPT2VID_LIPSYNC=off",
                    required=lipsync_mode == "on",
                ))
        try:
            with httpx.Client(timeout=POLL_TIMEOUT, transport=transport) as client:
                loader = _fetch_node_info(client, "CheckpointLoaderSimple", comfyui_url)
                clip_loader = _fetch_node_info(client, "CLIPLoader", comfyui_url)
                ltx_node = _fetch_node_info(client, "LTXVImgToVideo", comfyui_url)
                save_webm = _fetch_node_info(client, "SaveWEBM", comfyui_url)
                lora_loader = _fetch_node_info(client, "LoraLoader", comfyui_url)
                lora_model_only = _fetch_node_info(
                    client, "LoraLoaderModelOnly", comfyui_url
                )

            ckpts = _combo_values(loader.get("CheckpointLoaderSimple", {}), "ckpt_name")
            clips = _combo_values(clip_loader.get("CLIPLoader", {}), "clip_name")
            loras = _combo_values(lora_loader.get("LoraLoader", {}), "lora_name")
            loras_model_only = _combo_values(
                lora_model_only.get("LoraLoaderModelOnly", {}), "lora_name"
            )

            checks.append(
                Check(
                    name=f"T2I checkpoint（{t2i_ckpt}）",
                    ok=t2i_ckpt in ckpts,
                    detail=f"可用: {', '.join(ckpts[:5]) or '无'}",
                    hint="把 SDXL checkpoint 放入 ComfyUI models/checkpoints/，或修改 .env 的 COMFYUI_T2I_CHECKPOINT",
                )
            )
            checks.append(
                Check(
                    name=f"LTX checkpoint（{ltx_ckpt}）",
                    ok=ltx_ckpt in ckpts,
                    detail=f"可用: {', '.join(ckpts[:5]) or '无'}",
                    hint="把 LTX-Video checkpoint 放入 ComfyUI models/checkpoints/，或修改 .env 的 COMFYUI_LTX_CHECKPOINT",
                )
            )
            checks.append(
                Check(
                    name=f"T5 文本编码器（{t5}）",
                    ok=t5 in clips,
                    detail=f"可用: {', '.join(clips[:5]) or '无'}",
                    hint="把 t5xxl fp8 放入 ComfyUI models/text_encoders/，或修改 .env 的 COMFYUI_T5",
                )
            )
            checks.append(
                Check(
                    name="LTXV 节点",
                    ok=bool(ltx_node) and bool(save_webm),
                    detail="LTXVImgToVideo / SaveWEBM 已注册" if ltx_node and save_webm else "ComfyUI 缺少 LTXV 节点",
                    hint="请升级 ComfyUI 到支持 LTX-Video 0.9.5 的版本（master）",
                )
            )
            # LoRA 是风格增强件：缺失时运行期自动降级为无 LoRA 生成，
            # 只提示不拦截（风格词约束仍在，画面不会跑偏回默认风格）
            t2i_lora = env_str("COMFYUI_T2I_LORA", "")
            if t2i_lora:
                checks.append(
                    Check(
                        name=f"T2I LoRA（{t2i_lora}）",
                        ok=t2i_lora in loras,
                        detail=f"可用: {', '.join(loras[:5]) or '无'}",
                        hint="把 LoRA 放入 ComfyUI models/loras/，或清空 .env 的 COMFYUI_T2I_LORA（将退回纯提示词风格约束）",
                        required=False,
                    )
                )
            ltx_lora = env_str("COMFYUI_LTX_LORA", "")
            if ltx_lora:
                checks.append(
                    Check(
                        name=f"LTX LoRA（{ltx_lora}）",
                        ok=ltx_lora in loras_model_only,
                        detail=f"可用: {', '.join(loras_model_only[:5]) or '无'}",
                        hint="把 LTX LoRA 放入 ComfyUI models/loras/，或清空 .env 的 COMFYUI_LTX_LORA",
                        required=False,
                    )
                )
            if style_enabled():
                checks.append(
                    Check(
                        name=f"风格约束（{style_name()}）",
                        ok=True,
                        detail="每镜提示词将自动追加风格词（见 style.py 预设）",
                        required=False,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - object_info 异常
            checks.append(Check(name="ComfyUI object_info", ok=False, detail=str(exc),
                                hint="检查 ComfyUI 版本是否过旧"))

    return result


def run_preflight(base_url: str | None = None) -> bool:
    """执行预检并打印报告；返回是否通过。"""
    from .env_config import load_env, loaded_env_file

    load_env()  # 幂等：确保独立调用时（不经 CLI）环境配置也已就位
    result = check_environment(base_url)
    env_file = loaded_env_file()
    if env_file is not None:
        print(f"配置文件: {env_file}")
    print(result.format_report())
    return result.passed
