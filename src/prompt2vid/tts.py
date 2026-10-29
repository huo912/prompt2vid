"""配音纯函数：IndexTTS-2.5（ComfyUI 本地推理）优先，失败时逐级回退。

回退顺序由 PROMPT2VID_TTS_ENGINE 控制（auto / comfyui / edge / gateway /
piper，默认 auto=按下列顺序）：
1. ComfyUI IndexTTS-2.5（本地显卡推理，中文自然度最佳；支持插件自带
   59 个音色与自定义音色克隆。需在 ComfyUI 安装
   chenpipi0807/ComfyUI-Index-TTS 节点并下载模型，见 docs/LOCAL_SETUP.md；
   PROMPT2VID_TTS_REF_AUDIO 指向 3-15 秒干净人声时克隆该音色，
   否则用 PROMPT2VID_TTS_TIMBRE 指定的自带音色，默认"甜美女声"）
2. edge-tts（微软公网语音；公司网络拦 WSS 时可用 EDGE_TTS_PROXY 走
   SSH 动态转发 `ssh -D 1080 <云机>` 后的 socks5://127.0.0.1:1080）
3. PROMPT2VID_TTS_BASE_URL 指定的 OpenAI 兼容 /v1/audio/speech 接口
4. piper 本地离线小模型（机械感明显，需 `pip install piper-tts` 且音色
   onnx 就位；PROMPT2VID_TTS_PIPER_MODEL 指定，缺省探测 ./models/tts/）
5. 静音占位音频（流水线不中断，质检阶段会明确报出"旁白无声"）

每一级降级都会 logger.warning——静默降级曾导致整条片子全程用上了
机械音色而不被发现。
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from pathlib import Path

import edge_tts
import httpx

from .ffmpeg_ops import resolve_tool, run_ffmpeg

logger = logging.getLogger("prompt2vid.tts")

DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
DEFAULT_RATE = "+0%"
# 静音占位：8kHz 单声道 1 秒的极小 MP3 帧
_SILENT_MP3 = (
    b"\xff\xfb\x90\x00" + b"\x00" * 413
) * 1

# 引擎偏好顺序（auto 模式；显式指定某引擎时它排第一，其余仍作降级）
ENGINE_ORDER = ("comfyui", "edge", "gateway", "piper")

# ComfyUI IndexTTS-2.5 可用性探测缓存（None = 未探测；进程内只探一次，
# 避免逐镜配音都等服务端 object_info）
_COMFYUI_TTS_OK: bool | None = None


def reset_engine_cache() -> None:
    """清空引擎可用性缓存（测试切换环境变量后调用）。"""
    global _COMFYUI_TTS_OK
    _COMFYUI_TTS_OK = None


def _engine_list() -> list[str]:
    """解析 PROMPT2VID_TTS_ENGINE 为尝试顺序（显式引擎优先，其余兜底）。"""
    engine = os.getenv("PROMPT2VID_TTS_ENGINE", "auto").strip().lower() or "auto"
    if engine in ENGINE_ORDER:
        return [engine] + [e for e in ENGINE_ORDER if e != engine]
    return list(ENGINE_ORDER)


def _ref_audio_path() -> str:
    """自定义参考音频路径（PROMPT2VID_TTS_REF_AUDIO，存在才有效）。"""
    raw = os.getenv("PROMPT2VID_TTS_REF_AUDIO", "").strip()
    return raw if raw and Path(raw).is_file() else ""


def _comfyui_tts_available() -> bool:
    """探测 ComfyUI 侧 IndexTTS-2.5 节点（结果进程内缓存，失败只报一次）。"""
    global _COMFYUI_TTS_OK
    if _COMFYUI_TTS_OK is not None:
        return _COMFYUI_TTS_OK
    _COMFYUI_TTS_OK = False
    try:
        from .comfyui_client import ComfyUIClient

        with ComfyUIClient() as client:
            ok = client.node_exists("IndexTTS25BaseNode")
            # 用自带音色时还需要音色加载节点；自定义参考音频走核心 LoadAudio
            if ok and not _ref_audio_path():
                ok = client.node_exists("TimbreAudioLoader")
    except Exception as exc:  # noqa: BLE001 - ComfyUI 不可达按不可用处理
        logger.warning("探测 ComfyUI IndexTTS-2.5 失败: %s", exc)
        ok = False
    _COMFYUI_TTS_OK = ok
    if not ok:
        logger.warning(
            "ComfyUI IndexTTS-2.5 不可用（未装 chenpipi0807/ComfyUI-Index-TTS "
            "节点或服务未启动），配音降级到回退链（安装见 docs/LOCAL_SETUP.md）"
        )
    return _COMFYUI_TTS_OK


def _finalize_audio(wav: Path, dest: Path, bitrate: str = "192k") -> bool:
    """把 wav 转码为 mp3 落到 dest（manifest 约定 mp3）。

    无 ffmpeg 时直接写 wav 内容（探测/拼接按文件内容识别格式，扩展名
    不一致也能读）；ffmpeg 在但转码失败返回 False，走下一级回退。
    """
    try:
        run_ffmpeg(["-i", str(wav), "-c:a", "libmp3lame", "-b:a", bitrate, str(dest)])
    except RuntimeError:
        if resolve_tool("ffmpeg") is not None:
            return False  # ffmpeg 在但转码失败：走下一级回退
        shutil.copyfile(wav, dest)  # 无 ffmpeg：直接落 wav 内容（按内容识别）
    finally:
        wav.unlink(missing_ok=True)
    return dest.is_file() and dest.stat().st_size > 0


def _tts_via_comfyui(text: str, dest: Path) -> bool:
    """ComfyUI IndexTTS-2.5 配音（本地显卡推理，中文自然度最佳）。"""
    if not _comfyui_tts_available():
        return False
    from .comfyui_client import (
        ComfyUIClient,
        build_tts_workflow,
        run_workflow,
    )

    wav = dest.with_suffix(".comfyui.wav")
    try:
        with ComfyUIClient() as client:
            reference = None
            ref_path = _ref_audio_path()
            if ref_path:
                reference = client.upload_file(Path(ref_path))
            workflow = client.filter_workflow_inputs(
                build_tts_workflow(text=text, reference_audio=reference)
            )
            run_workflow(workflow, wav, client=client)
    except Exception as exc:  # noqa: BLE001 - 失败降级到下一引擎
        logger.warning("ComfyUI IndexTTS-2.5 配音失败: %s", exc)
        wav.unlink(missing_ok=True)
        return False
    return _finalize_audio(wav, dest)


def _tts_via_edge(text: str, dest: Path, voice: str = "", rate: str = "") -> bool:
    """edge-tts 配音（微软公网语音，网络可达时音质尚可）。"""
    voice_name = voice or os.getenv("EDGE_TTS_VOICE", "").strip() or DEFAULT_VOICE
    rate_value = rate or os.getenv("EDGE_TTS_RATE", "").strip() or DEFAULT_RATE
    # 公司网络拦 WSS 时可走代理（如云机 SSH 动态转发 / HTTP 代理端口转发）
    proxy = os.getenv("EDGE_TTS_PROXY", "").strip() or None
    try:
        communicate = edge_tts.Communicate(
            text=text, voice=voice_name, rate=rate_value, proxy=proxy
        )
        asyncio.run(communicate.save(str(dest)))
    except Exception as exc:  # noqa: BLE001 - edge-tts 不可用时走回退链
        logger.warning("edge-tts 配音失败（%s），尝试下一级回退", exc)
        return False
    return dest.is_file() and dest.stat().st_size > 0


def _tts_via_gateway(text: str, dest: Path) -> bool:
    """尝试 OpenAI 兼容 TTS 网关（PROMPT2VID_TTS_BASE_URL）。"""
    base = os.getenv("PROMPT2VID_TTS_BASE_URL", "").strip()
    if not base:
        return False
    model = os.getenv("PROMPT2VID_TTS_MODEL", "tts-1").strip()
    voice = os.getenv("PROMPT2VID_TTS_VOICE", "alloy").strip()
    api_key = os.getenv("PROMPT2VID_TTS_API_KEY", "").strip()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            resp = client.post(
                f"{base.rstrip('/')}/audio/speech",
                headers=headers,
                json={"model": model, "input": text, "voice": voice},
            )
            if resp.status_code == 200 and resp.content:
                dest.write_bytes(resp.content)
                return True
            logger.warning(
                "TTS 网关返回 HTTP %s（%s），尝试下一级回退",
                resp.status_code, resp.text[:120],
            )
    except Exception as exc:  # noqa: BLE001 - 回退链中任何失败都继续降级
        logger.warning("TTS 网关请求失败（%s），尝试下一级回退", exc)
    return False


def _piper_model() -> str:
    """解析 piper 音色模型路径：env 优先，缺省探测 ./models/tts/ 下唯一 onnx。"""
    model = os.getenv("PROMPT2VID_TTS_PIPER_MODEL", "").strip()
    if model:
        return model if Path(model).is_file() else ""
    tts_dir = Path("models/tts")
    if tts_dir.is_dir():
        candidates = sorted(tts_dir.glob("*.onnx"))
        if len(candidates) == 1:
            return str(candidates[0])
    return ""


def _tts_via_piper(text: str, dest: Path) -> bool:
    """本地 piper 神经网络配音（离线，不吃网络；中文推荐 huayan 音色）。"""
    exe = shutil.which("piper")
    if not exe:
        return False
    model = _piper_model()
    if not model:
        return False
    wav = dest.with_suffix(".piper.wav")
    try:
        result = subprocess.run(
            [exe, "--model", model, "--output_file", str(wav)],
            input=text, capture_output=True, text=True,
            encoding="utf-8", timeout=600.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("piper 配音失败（%s），尝试下一级回退", exc)
        return False
    if result.returncode != 0 or not (wav.is_file() and wav.stat().st_size > 0):
        wav.unlink(missing_ok=True)
        logger.warning("piper 配音失败（exit %s），尝试下一级回退", result.returncode)
        return False
    return _finalize_audio(wav, dest, bitrate="128k")


def _write_silent_placeholder(dest: Path, text: str) -> None:
    """写静音占位音频并记录旁白文本，保证流水线不中断。"""
    seconds = max(1, min(60, int(len(text) / 4)))
    dest.write_bytes(_SILENT_MP3 * seconds)
    sidecar = dest.with_suffix(".txt")
    sidecar.write_text(text, encoding="utf-8")


def synthesize_audio(text: str, dest: str | Path, voice: str = "", rate: str = "") -> Path:
    """把文本合成为音频文件（IndexTTS-2.5 → edge-tts → 网关 → piper → 静音）。

    引擎选择见模块 docstring；每次降级都会写 warning 日志。
    voice/rate 仅对 edge-tts 生效（IndexTTS 的音色由
    PROMPT2VID_TTS_TIMBRE / PROMPT2VID_TTS_REF_AUDIO 控制）。

    Returns:
        保存后的音频绝对路径。
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    for engine in _engine_list():
        if engine == "comfyui" and _tts_via_comfyui(text, dest):
            return dest.resolve()
        if engine == "edge" and _tts_via_edge(text, dest, voice, rate):
            return dest.resolve()
        if engine == "gateway" and _tts_via_gateway(text, dest):
            return dest.resolve()
        if engine == "piper" and _tts_via_piper(text, dest):
            return dest.resolve()

    logger.warning(
        "全部配音引擎失败，写入静音占位（旁白文本已存 %s.txt）；"
        "修复方式见 docs/QUALITY.md §配音", dest,
    )
    _write_silent_placeholder(dest, text)
    return dest.resolve()
