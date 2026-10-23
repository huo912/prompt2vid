"""ffmpeg 剪辑纯函数：探测、拼接、音频对齐、SRT 生成、混流烧录。

供 stages.py 确定性调用；tools/ffmpeg_tool.py 只做 CrewAI 薄包装。
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

# 字幕烧录默认样式：竖屏 512 宽下默认字体偏小，显式加大并留出底部边距。
# 可用 PROMPT2VID_SUBTITLE_STYLE 覆盖（libass force_style 语法）。
DEFAULT_SUBTITLE_STYLE = "FontSize=18,Outline=1,MarginV=40"


def run_ffmpeg(args: list[str]) -> str:
    """执行 ffmpeg，失败抛 RuntimeError。"""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败: {result.stderr.strip()}")
    return result.stdout


def probe_file(media: str | Path) -> dict:
    """探测媒体文件的时长、分辨率、帧率、音轨信息。

    Returns:
        dict：path、duration_seconds、width、height、fps、has_audio。
    """
    media = Path(media)
    if not media.is_file():
        raise FileNotFoundError(f"文件不存在: {media}")
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
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
