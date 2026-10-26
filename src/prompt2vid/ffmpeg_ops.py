"""ffmpeg 剪辑纯函数：探测、拼接、音频对齐、SRT 生成、混流烧录。

供 stages.py 确定性调用；tools/ffmpeg_tool.py 只做 CrewAI 薄包装。
"""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

# 字幕烧录默认样式：竖屏 512 宽下默认字体偏小，显式加大并留出底部边距。
# 可用 PROMPT2VID_SUBTITLE_STYLE 覆盖（libass force_style 语法）。
DEFAULT_SUBTITLE_STYLE = "FontSize=18,Outline=1,MarginV=40"

# 可执行文件解析：环境变量优先（便于 Windows/容器里指向绝对路径），
# 其次 PATH，再退到常见安装目录与随包轮子（imageio-ffmpeg / static-ffmpeg）。
_TOOL_ENV = {
    "ffmpeg": ("PROMPT2VID_FFMPEG", "FFMPEG_BIN"),
    "ffprobe": ("PROMPT2VID_FFPROBE", "FFPROBE_BIN"),
}
_COMMON_DIRS = (
    Path.home() / ".local/bin",
    Path("/usr/local/bin"),
    Path("/opt/homebrew/bin"),
    Path("/opt/ffmpeg/bin"),
    Path("C:/ffmpeg/bin"),
)


def _as_executable(candidate: object) -> str | None:
    """把候选路径规范化为可执行文件路径；不可用返回 None。"""
    if not candidate:
        return None
    path = Path(str(candidate)).expanduser()
    if path.is_file() and os.access(path, os.X_OK):
        return str(path)
    return None


def _bundled_candidates(name: str):
    """随 pip 包提供的二进制（缺失对应包时静默跳过）。

    注意：只探测已下载到本地的二进制，绝不触发联网下载——static-ffmpeg
    的公开 helper 会在调用时从 GitHub 拉取，网络受限时会长时间阻塞。
    """
    try:  # static-ffmpeg：同时提供 ffmpeg 与 ffprobe（仅本地已下载时）
        from static_ffmpeg.run import get_platform_dir  # type: ignore

        yield _as_executable(Path(get_platform_dir()) / name)
    except Exception:  # noqa: BLE001 - 可选依赖，任何异常都视为不可用
        pass
    if name == "ffmpeg":
        try:  # imageio-ffmpeg 只带 ffmpeg（不含 ffprobe）
            import imageio_ffmpeg  # type: ignore

            yield _as_executable(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:  # noqa: BLE001
            pass


def _missing_tool_error(name: str) -> RuntimeError:
    return RuntimeError(
        f"未找到 {name} 可执行文件。请安装 ffmpeg（含 {name}）并加入 PATH，"
        f"或设置 {_TOOL_ENV[name][0]} 指向可执行文件路径；"
        "本地也可 `pip install imageio-ffmpeg`（自动回退用 ffmpeg 探测）。"
    )


def _resolve_tool(name: str, allow_sibling: bool = True) -> str | None:
    for var in _TOOL_ENV[name]:
        raw = os.getenv(var, "").strip()
        if not raw:
            continue
        found = _as_executable(raw) or shutil.which(raw)
        if found:
            return found
    found = shutil.which(name)
    if found:
        return found
    for directory in _COMMON_DIRS:
        found = _as_executable(directory / name)
        if found:
            return found
    for candidate in _bundled_candidates(name):
        if candidate:
            return candidate
    if allow_sibling:
        # 最后尝试：与另一个工具同目录（例如 ffmpeg/ffprobe 并列安装）
        other = "ffprobe" if name == "ffmpeg" else "ffmpeg"
        sibling = _resolve_tool(other, allow_sibling=False)
        if sibling:
            found = _as_executable(Path(sibling).with_name(name))
            if found:
                return found
    return None


@functools.lru_cache(maxsize=None)
def resolve_tool(name: str) -> str | None:
    """解析 ffmpeg / ffprobe 可执行文件路径，找不到返回 None（结果缓存）。"""
    if name not in _TOOL_ENV:
        raise ValueError(f"不支持的工具名: {name}")
    return _resolve_tool(name)


def require_tool(name: str) -> str:
    """解析工具路径，找不到时抛出带安装指引的 RuntimeError。"""
    path = resolve_tool(name)
    if path is None:
        raise _missing_tool_error(name)
    return path


def run_ffmpeg(args: list[str]) -> str:
    """执行 ffmpeg，失败抛 RuntimeError。"""
    cmd = [require_tool("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败: {result.stderr.strip()}")
    return result.stdout


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")
_VIDEO_SIZE_RE = re.compile(r":\s*Video:.*?(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_PROGRESS_TIME_RE = re.compile(r"time=(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")


def _hms_to_seconds(hours: str, minutes: str, seconds: str) -> float:
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _probe_with_ffmpeg(ffmpeg: str, media: Path) -> dict:
    """没有 ffprobe 时用 ffmpeg 的流信息输出解析（webm/mp3/mp4 通用）。"""
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-i", str(media)]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    text = result.stderr or ""
    if "Stream #" not in text:
        raise RuntimeError(f"ffmpeg 解析媒体信息失败: {text.strip()[:400]}")

    duration = 0.0
    match = _DURATION_RE.search(text)
    if match:
        duration = _hms_to_seconds(*match.groups())

    video_line = next(
        (ln for ln in text.splitlines() if ": Video: " in ln), ""
    )
    size = _VIDEO_SIZE_RE.search(video_line)
    width = int(size.group(1)) if size else 0
    height = int(size.group(2)) if size else 0
    fps_match = _FPS_RE.search(video_line)
    fps = float(fps_match.group(1)) if fps_match else 0.0

    if duration <= 0:
        # 头部无时长（流式容器）：解码一遍取最后时间戳
        decoded = subprocess.run(
            [ffmpeg, "-hide_banner", "-nostdin", "-i", str(media), "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8",
        )
        times = _PROGRESS_TIME_RE.findall(decoded.stderr or "")
        if times:
            duration = _hms_to_seconds(*times[-1])

    return {
        "path": str(media.resolve()),
        "duration_seconds": round(duration, 3),
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "has_audio": ": Audio: " in text,
    }


def probe_file(media: str | Path) -> dict:
    """探测媒体文件的时长、分辨率、帧率、音轨信息。

    优先用 ffprobe；未安装 ffprobe 但 ffmpeg 可用时，回退解析
    ffmpeg 的流信息（结果字段一致）。

    Returns:
        dict：path、duration_seconds、width、height、fps、has_audio。
    """
    media = Path(media)
    if not media.is_file():
        raise FileNotFoundError(f"文件不存在: {media}")

    ffprobe = resolve_tool("ffprobe")
    if ffprobe is None:
        ffmpeg = resolve_tool("ffmpeg")
        if ffmpeg is None:
            raise _missing_tool_error("ffprobe")
        return _probe_with_ffmpeg(ffmpeg, media)

    cmd = [
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(media),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {result.stderr.strip()}")
    info = json.loads(result.stdout)

    duration = float(info.get("format", {}).get("duration", 0.0))
    video_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "video"), None
    )
    audio_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None
    )
    fps = 0.0
    if video_stream:
        rate = video_stream.get("avg_frame_rate", "0/1")
        try:
            num, _, den = rate.partition("/")
            fps = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            fps = 0.0

    return {
        "path": str(media.resolve()),
        "duration_seconds": round(duration, 3),
        "width": video_stream.get("width", 0) if video_stream else 0,
        "height": video_stream.get("height", 0) if video_stream else 0,
        "fps": round(fps, 3),
        "has_audio": audio_stream is not None,
    }


def _srt_timestamp(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def build_srt(
    timings: list[tuple[float, float, str]],
    dest: str | Path,
) -> Path:
    """按 (start, end, text) 时间轴生成 SRT 字幕文件。

    时间轴由剪辑阶段根据每个片段的真实时长累计得到（不依赖 LLM 手写）。
    """
    blocks = []
    for idx, (start, end, text) in enumerate(timings, start=1):
        text = text.strip()
        if not text:
            continue
        blocks.append(
            f"{idx}\n{_srt_timestamp(start)} --> {_srt_timestamp(end)}\n{text}\n"
        )
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(blocks), encoding="utf-8")
    return dest


def normalize_clip(clip: str | Path, dest: str | Path, extend_seconds: float = 0.0) -> Path:
    """统一转码为 h264/yuv420p/25fps 无音轨 mp4（concat 前提）。

    extend_seconds > 0 时用定格末帧（tpad）延长片段——用于配音比片段
    长的场景，避免旁白被截断。
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = ["-i", str(clip), "-c:v", "libx264", "-preset", "fast",
            "-pix_fmt", "yuv420p", "-r", "25", "-an"]
    if extend_seconds > 0.01:
        args += ["-vf", f"tpad=stop_mode=clone:stop_duration={extend_seconds:.3f}"]
    args.append(str(dest))
    run_ffmpeg(args)
    return dest


def concat_media(
    items: list[Path],
    dest: str | Path,
    *,
    reencode_audio: bool = False,
) -> Path:
    """用 concat demuxer 按顺序拼接同编码参数的文件。"""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    list_file = dest.parent / f"concat_{dest.stem}.txt"
    list_file.write_text(
        "\n".join(f"file '{p.as_posix()}'" for p in items), encoding="utf-8"
    )
    args = ["-f", "concat", "-safe", "0", "-i", str(list_file)]
    args += (["-c:a", "pcm_s16le"] if reencode_audio else ["-c", "copy"])
    args.append(str(dest))
    run_ffmpeg(args)
    return dest


def fit_audio(audio: str | Path | None, duration: float, dest: str | Path) -> Path:
    """把音频对齐到指定时长：长则截断、短则静音填充，输出统一 wav。

    音频经响度归一（loudnorm，-16 LUFS，edge-tts 输出偏轻）；
    音频缺失时生成等时长静音。
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dur = f"{max(0.1, duration):.3f}"
    audio = Path(audio) if audio else None
    if audio is not None and audio.is_file():
        run_ffmpeg(["-i", str(audio),
                    "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,apad", "-t", dur,
                    "-ar", "44100", "-ac", "1", str(dest)])
    else:
        run_ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                    "-t", dur, str(dest)])
    return dest


def mux_and_burn(
    video: str | Path,
    audio: str | Path,
    srt: str | Path | None,
    dest: str | Path,
    bgm: str | Path | None = None,
    bgm_volume: float | None = None,
    subtitle_style: str | None = None,
) -> Path:
    """混流视频与音轨，并（可选）烧录 SRT 字幕，输出最终成片。

    Args:
        bgm: 可选背景音乐文件（循环播放、低音量与旁白混音）。
        bgm_volume: BGM 音量（0-1），默认读 PROMPT2VID_BGM_VOLUME（0.25）。
        subtitle_style: libass force_style；默认读
            PROMPT2VID_SUBTITLE_STYLE（FontSize=18,Outline=1,MarginV=40）。
    """
    video, audio, dest = Path(video), Path(audio), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    args = ["-i", str(video), "-i", str(audio)]
    if bgm is not None:
        args += ["-stream_loop", "-1", "-i", str(bgm)]

    if srt is not None and Path(srt).is_file():
        style = subtitle_style or os.getenv(
            "PROMPT2VID_SUBTITLE_STYLE", DEFAULT_SUBTITLE_STYLE
        ).strip()
        # subtitles 滤镜路径转义（Windows 盘符冒号）
        srt_arg = Path(srt).as_posix().replace(":", "\\:")
        vf = f"subtitles='{srt_arg}'"
        if style:
            vf += f":force_style='{style}'"
        args += ["-vf", vf]

    if bgm is not None:
        if bgm_volume is None:
            try:
                bgm_volume = float(os.getenv("PROMPT2VID_BGM_VOLUME", "0.25"))
            except ValueError:
                bgm_volume = 0.25
        args += [
            "-filter_complex",
            f"[1:a]volume=1.0[a1];[2:a]volume={bgm_volume:.2f}[a2];"
            "[a1][a2]amix=inputs=2:duration=first[aout]",
            "-map", "0:v:0", "-map", "[aout]",
        ]
    else:
        args += ["-map", "0:v:0", "-map", "1:a:0"]

    args += [
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(dest),
    ]
    run_ffmpeg(args)
    return dest
