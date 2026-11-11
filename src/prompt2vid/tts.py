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

# 角色声线表：分镜表里的角色没写 voice 时，按角色 id 轮转取一个，
# 保证同片不同角色不是同一个嗓子（这是「人物说话」与「旁白铺底」的
# 核心区别）。PROMPT2VID_TTS_VOICES 可用逗号列表覆盖。
DEFAULT_CHARACTER_VOICES = (
    "zh-CN-YunxiNeural",        # 男声（年轻、偏轻快）
    "zh-CN-XiaoyiNeural",       # 女声（年轻、偏甜）
    "zh-CN-YunjianNeural",      # 男声（中年、解说腔）
    "zh-CN-XiaochenNeural",     # 女声（温柔）
    "zh-CN-YunyangNeural",      # 男声（新闻播报）
    "zh-CN-XiaohanNeural",      # 女声（沉稳）
    "zh-CN-YunxiaNeural",       # 男童
    "zh-CN-XiaomengNeural",     # 女童
)

# IndexTTS-2.5 角色音色轮转表（插件 TimbreModel 目录标准自带的音色文件，
# 男女交替；分镜角色卡没写 timbre 时按角色 id 轮转，保证不同角色不同嗓
# 子）。音色是内容盲的（不看性别），角色卡写 timbre 可按人设精挑。
# 个别安装缺某个文件时该句自动降级 edge-tts，不影响出片。
# PROMPT2VID_TTS_TIMBRES 可用逗号列表覆盖。
DEFAULT_CHARACTER_TIMBRES = (
    "钟离.wav",            # 男声（低沉威严、帝王气度）
    "御姐配音.wav",         # 女声（成熟御姐）
    "温润男声.mp3",         # 男声（温润）
    "甜美女声.mp3",         # 女声（甜美）
    "播音中年男.mp3",        # 男声（中年播报）
    "阅历姐姐.mp3",         # 女声（沉稳阅历）
    "随性男青.mp3",         # 男声（随性青年）
    "温柔学姐.mp3",         # 女声（温柔学姐）
)
# 音色清单里的「陷阱选项」：前者是静音占位（选它整片没声），后者是朗读小说
# 腔（正是「配音像读课文」的来源）。分镜师偶尔会按字面意思挑中它们。
BLOCKED_TIMBRES = frozenset({"无音频文件", "抖音-读小说.wav"})
# 静音占位：8kHz 单声道 1 秒的极小 MP3 帧
_SILENT_MP3 = (
    b"\xff\xfb\x90\x00" + b"\x00" * 413
) * 1

# 引擎偏好顺序（auto 模式；显式指定某引擎时它排第一，其余仍作降级）
ENGINE_ORDER = ("comfyui", "edge", "gateway", "piper")

# ComfyUI IndexTTS-2.5 可用性探测缓存（None = 未探测；进程内只探一次，
# 避免逐镜配音都等服务端 object_info）
_COMFYUI_TTS_OK: bool | None = None
# 情感节点（IndexTTS25EmotionTextNode）探测缓存，语义同上
_EMOTION_NODE_OK: bool | None = None
# ComfyUI 侧音色文件名清单缓存（None = 未探测）；用于校验分镜里写的 timbre
_TIMBRES: list[str] | None = None


def reset_engine_cache() -> None:
    """清空引擎可用性缓存（测试切换环境变量后调用）。"""
    global _COMFYUI_TTS_OK
    global _EMOTION_NODE_OK
    global _TIMBRES
    _COMFYUI_TTS_OK = None
    _EMOTION_NODE_OK = None
    _TIMBRES = None


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


def _env_list(name: str) -> list[str]:
    """逗号/顿号分隔的环境变量列表（空项丢弃）。"""
    raw = os.getenv(name, "").strip()
    if not raw:
        return []
    for sep in ("、", ";", "；"):
        raw = raw.replace(sep, ",")
    return [item.strip() for item in raw.split(",") if item.strip()]


def voice_for_speaker(speaker_id: int) -> str:
    """返回某发言人应使用的 edge-tts 音色。

    旁白（id=0）返回空字符串：由 EDGE_TTS_VOICE / DEFAULT_VOICE 决定，
    与旧行为一致；角色按 id 在表中轮转，不同角色自然分到不同声线。
    """
    if speaker_id <= 0:
        return ""
    palette = _env_list("PROMPT2VID_TTS_VOICES") or list(DEFAULT_CHARACTER_VOICES)
    return palette[(speaker_id - 1) % len(palette)]


def timbre_for_speaker(speaker_id: int) -> str:
    """返回某发言人应使用的 IndexTTS 音色文件名。

    角色卡写了 timbre 或显式配置 PROMPT2VID_TTS_TIMBRES（逗号列表，
    按角色 id 顺序）时优先；否则按角色 id 在 DEFAULT_CHARACTER_TIMBRES
    轮转，保证不同角色不同嗓子（与 edge-tts 的 voice 轮转同构）。
    旁白（id=0）返回空字符串：用全局 PROMPT2VID_TTS_TIMBRE，与旁白
    铺底的定位一致。
    """
    if speaker_id <= 0:
        return ""
    palette = _env_list("PROMPT2VID_TTS_TIMBRES") or list(DEFAULT_CHARACTER_TIMBRES)
    return palette[(speaker_id - 1) % len(palette)]


def available_timbres() -> list[str]:
    """ComfyUI 侧 TimbreModel 目录里的音色文件名（探测失败返回 []）。

    分镜师的模型并不知道服务器上到底有哪些音色，很容易编出一个不存在的
    文件名；拿这个清单校验一次，编错的名字就退回按角色轮转，而不是让
    整句配音因为节点报错而降级到别的引擎（那会导致同一角色声音突变）。
    """
    global _TIMBRES
    if _TIMBRES is not None:
        return list(_TIMBRES)
    _TIMBRES = []
    try:
        from .comfyui_client import ComfyUIClient, combo_values

        with ComfyUIClient() as client:
            _TIMBRES = combo_values(client.node_spec("TimbreAudioLoader"), "audio_file")
    except Exception as exc:  # noqa: BLE001 - ComfyUI 不可达按无清单处理
        logger.debug("探测 ComfyUI 音色清单失败: %s", exc)
    return list(_TIMBRES)


def resolve_timbre(name: str) -> str:
    """把分镜里写的音色名解析成服务器上真实存在的文件名。

    精确命中优先，其次忽略大小写与扩展名比较（音色包里 .WAV/.wav/.MP3
    混用，LLM 抄名字经常抄成小写）。清单拿不到或确实没有返回空串，
    由调用方退回轮转音色。
    """
    name = (name or "").strip()
    if not name or name in BLOCKED_TIMBRES:
        return ""
    catalogue = available_timbres()
    if not catalogue:
        return name  # 探测失败时不拦，交给 TTS 自己回退
    if name in catalogue:
        return name
    lowered = name.lower()
    for item in catalogue:
        if item.lower() == lowered:
            return item
    stem = Path(lowered).stem
    for item in catalogue:
        if Path(item.lower()).stem == stem:
            return item
    logger.warning("分镜指定的音色 %s 不在 ComfyUI 音色清单里，按角色轮转", name)
    return ""


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


def _emotion_node_available() -> bool:
    """探测情感节点（IndexTTS25EmotionTextNode，2.5 版插件才有）。"""
    global _EMOTION_NODE_OK
    if _EMOTION_NODE_OK is not None:
        return _EMOTION_NODE_OK
    _EMOTION_NODE_OK = False
    try:
        from .comfyui_client import ComfyUIClient

        with ComfyUIClient() as client:
            _EMOTION_NODE_OK = client.node_exists("IndexTTS25EmotionTextNode")
    except Exception:  # noqa: BLE001 - ComfyUI 不可达按无情感节点处理
        _EMOTION_NODE_OK = False
    return _EMOTION_NODE_OK


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


def _tts_via_comfyui(text: str, dest: Path, timbre: str = "", emotion: str = "") -> bool:
    """ComfyUI IndexTTS-2.5 配音（本地显卡推理，中文自然度最佳）。

    timbre 为该句指定的插件音色文件名（不同角色用不同嗓子）；
    空串时回退到 PROMPT2VID_TTS_TIMBRE 全局音色。
    emotion 为该句的情感描述（IndexTTS25EmotionTextNode）；
    服务器缺情感节点时自动去掉 emotion 走 BaseNode（日志告警一次）。
    """
    if not _comfyui_tts_available():
        return False
    from .comfyui_client import (
        ComfyUIClient,
        build_tts_workflow,
        run_workflow,
    )

    if emotion and not _emotion_node_available():
        logger.warning(
            "ComfyUI 缺 IndexTTS25EmotionTextNode 节点（插件版本较旧），"
            "情感控制不可用，按无情感合成"
        )
        emotion = ""

    wav = dest.with_suffix(".comfyui.wav")
    try:
        with ComfyUIClient() as client:
            reference = None
            ref_path = _ref_audio_path()
            if ref_path:
                reference = client.upload_file(Path(ref_path))
            workflow = client.filter_workflow_inputs(
                build_tts_workflow(
                    text=text,
                    reference_audio=reference,
                    timbre=timbre or None,
                    emotion=emotion or None,
                )
            )
            run_workflow(workflow, wav, client=client)
    except Exception as exc:  # noqa: BLE001 - 失败降级到下一引擎
        logger.warning("ComfyUI IndexTTS-2.5 配音失败: %s", exc)
        wav.unlink(missing_ok=True)
        return False
    return _finalize_audio(wav, dest)


def _edge_hint(exc: Exception) -> str:
    """把 edge-tts 的典型失败翻译成可执行提示（省得翻文档猜）。"""
    msg = str(exc)
    if "403" in msg:
        # 微软按客户端版本校验 Sec-MS-GEC 令牌：旧版 edge-tts 一律 403
        return "；403 多为 edge-tts 版本过期，`pip install -U edge-tts`（>=7.3.1）"
    if "Proxy" in msg or "proxy" in msg or "Connect" in msg:
        return "；网络不可达，可设 EDGE_TTS_PROXY 走代理"
    return ""


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
        logger.warning("edge-tts 配音失败（%s）%s，尝试下一级回退", exc, _edge_hint(exc))
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
    # Windows 上 piper 的入口脚本用 locale（cp936）解码 stdin，中文文本变
    # 乱码 → espeak 产出空音素 → 一个采样都不写就退出（wave 报
    # `# channels not specified`）。强制子进程按 UTF-8 读 stdin。
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        result = subprocess.run(
            [exe, "--model", model, "--output_file", str(wav)],
            input=text, capture_output=True, text=True,
            encoding="utf-8", timeout=600.0, env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("piper 配音失败（%s），尝试下一级回退", exc)
        return False
    if result.returncode != 0 or not (wav.is_file() and wav.stat().st_size > 0):
        wav.unlink(missing_ok=True)
        tail = (result.stderr or "").strip().splitlines()[-1:]
        logger.warning(
            "piper 配音失败（exit %s%s），尝试下一级回退",
            result.returncode, f"：{tail[0]}" if tail else "",
        )
        return False
    return _finalize_audio(wav, dest, bitrate="128k")


def _write_silent_placeholder(dest: Path, text: str) -> None:
    """写静音占位音频并记录旁白文本，保证流水线不中断。"""
    seconds = max(1, min(60, int(len(text) / 4)))
    dest.write_bytes(_SILENT_MP3 * seconds)
    sidecar = dest.with_suffix(".txt")
    sidecar.write_text(text, encoding="utf-8")


def synthesize_audio(
    text: str,
    dest: str | Path,
    voice: str = "",
    rate: str = "",
    timbre: str = "",
    emotion: str = "",
) -> Path:
    """把文本合成为音频文件（IndexTTS-2.5 → edge-tts → 网关 → piper → 静音）。

    引擎选择见模块 docstring；每次降级都会写 warning 日志。
    voice 仅对 edge-tts 生效（该句的音色）；timbre 仅对 IndexTTS 生效
    （插件音色文件名）；emotion 仅对 IndexTTS 生效（该句的情感描述，
    空串时读 PROMPT2VID_TTS_EMOTION 全局描述）。两者都为空时用环境
    变量的全局音色。

    Returns:
        保存后的音频绝对路径。
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    for engine in _engine_list():
        if engine == "comfyui" and _tts_via_comfyui(text, dest, timbre, emotion):
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
