"""配音纯函数：edge-tts 优先，失败时回退到网关 TTS / 静音占位。

edge-tts 需要访问微软 speech.platform.bing.com（公网 WebSocket），
在代理网络下可能被拦截（403）。回退顺序：
1. edge-tts（公网可达时音质最佳）
2. PROMPT2VID_TTS_BASE_URL 指定的 OpenAI 兼容 /v1/audio/speech 接口
3. 静音占位音频（流水线不中断，最终视频无旁白）
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import edge_tts
import httpx

DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"
DEFAULT_RATE = "+0%"
# 静音占位：8kHz 单声道 1 秒的极小 MP3 帧
_SILENT_MP3 = (
    b"\xff\xfb\x90\x00" + b"\x00" * 413
) * 1


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
    except Exception:  # noqa: BLE001 - 回退链中任何失败都继续降级
        pass
    return False


def _write_silent_placeholder(dest: Path, text: str) -> None:
    """写静音占位音频并记录旁白文本，保证流水线不中断。"""
    seconds = max(1, min(60, int(len(text) / 4)))
    dest.write_bytes(_SILENT_MP3 * seconds)
    sidecar = dest.with_suffix(".txt")
    sidecar.write_text(text, encoding="utf-8")


def synthesize_audio(text: str, dest: str | Path, voice: str = "", rate: str = "") -> Path:
    """把文本合成为音频文件（edge-tts → 网关 TTS → 静音占位）。

    Returns:
        保存后的音频绝对路径。
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    voice_name = voice or os.getenv("EDGE_TTS_VOICE", "").strip() or DEFAULT_VOICE
    rate_value = rate or os.getenv("EDGE_TTS_RATE", "").strip() or DEFAULT_RATE

    try:
        communicate = edge_tts.Communicate(text=text, voice=voice_name, rate=rate_value)
        asyncio.run(communicate.save(str(dest)))
        if dest.is_file() and dest.stat().st_size > 0:
            return dest.resolve()
    except Exception:  # noqa: BLE001 - edge-tts 不可用时走回退链
        pass

    if _tts_via_gateway(text, dest):
        return dest.resolve()

    _write_silent_placeholder(dest, text)
    return dest.resolve()
