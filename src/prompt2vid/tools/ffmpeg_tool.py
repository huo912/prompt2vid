"""ffmpeg 剪辑工具：拼接、混音、字幕烧录、探测。"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from crewai.tools import tool


def _workspace() -> Path:
    ws = os.getenv("PROMPT2VID_WORKSPACE", "").strip()
    path = Path(ws) if ws else Path("output") / "run_default"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = _workspace() / path
    return path


def _run_ffmpeg(args: list[str]) -> str:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败: {result.stderr.strip()}")
    return result.stdout


@tool("Assemble video clips into one video")
def assemble(clip_manifest: str, output_path: str) -> str:
    """按清单顺序拼接视频片段为一个视频。

    Args:
        clip_manifest: clips/manifest.json 路径，结构
            {"shots": [{"id": 1, "video": "clips/shot_1.webm"}, ...]}。
        output_path: 输出视频路径（建议 .mp4）。

    Returns:
        拼接后视频的绝对路径。
    """
    manifest_path = _resolve(clip_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips = [_resolve(shot["video"]) for shot in manifest["shots"]]
    for clip in clips:
        if not clip.is_file():
            raise FileNotFoundError(f"缺少视频片段: {clip}")

    dest = _resolve(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # concat demuxer 要求同编码参数；先统一转码为 mp4(h264) 再拼接
    normalized_dir = dest.parent / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized: list[Path] = []
    for idx, clip in enumerate(clips):
        norm = normalized_dir / f"clip_{idx:03d}.mp4"
        _run_ffmpeg(["-i", str(clip), "-c:v", "libx264", "-preset", "fast",
                     "-pix_fmt", "yuv420p", "-an", str(norm)])
        normalized.append(norm)

    concat_file = dest.parent / "concat.txt"
    concat_file.write_text(
        "\n".join(f"file '{p.as_posix()}'" for p in normalized), encoding="utf-8"
    )
    _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(concat_file),
                 "-c", "copy", str(dest)])
    return str(dest.resolve())


@tool("Mix voiceover audio onto video")
def mix_audio(video_path: str, audio_manifest: str, output_path: str) -> str:
    """把配音音轨混入视频（配音替换/覆盖原音轨）。

    Args:
        video_path: 已拼接的视频路径。
        audio_manifest: audio/manifest.json 路径，结构
            {"shots": [{"id": 1, "audio": "audio/shot_1.mp3"}, ...]}。
        output_path: 输出视频路径（建议 .mp4）。

    Returns:
        混音后视频的绝对路径。
    """
    video = _resolve(video_path)
    manifest = json.loads(_resolve(audio_manifest).read_text(encoding="utf-8"))
    audios = [_resolve(shot["audio"]) for shot in manifest["shots"]]
    for audio in audios:
        if not audio.is_file():
            raise FileNotFoundError(f"缺少配音音频: {audio}")

    dest = _resolve(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 拼接全部配音为一条音轨
    voiceover = dest.parent / "voiceover_all.mp3"
    concat_file = dest.parent / "audio_concat.txt"
    concat_file.write_text(
        "\n".join(f"file '{p.as_posix()}'" for p in audios), encoding="utf-8"
    )
    _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(concat_file),
                 "-c:a", "libmp3lame", str(voiceover)])

    # 混流：视频时长为准，音轨不足处静音
    _run_ffmpeg(["-i", str(video), "-i", str(voiceover),
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                 "-map", "0:v:0", "-map", "1:a:0",
                 "-shortest", str(dest)])
    return str(dest.resolve())


@tool("Burn SRT subtitles into video")
def burn_subtitles(video_path: str, srt_path: str, output_path: str) -> str:
    """将 SRT 字幕烧录进视频画面。

    Args:
        video_path: 输入视频路径。
        srt_path: SRT 字幕文件路径。
        output_path: 输出视频路径（建议 .mp4）。

    Returns:
        烧录字幕后视频的绝对路径。
    """
    video = _resolve(video_path)
    srt = _resolve(srt_path)
    if not srt.is_file():
        raise FileNotFoundError(f"缺少字幕文件: {srt}")
    dest = _resolve(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # subtitles 滤镜路径转义（Windows 盘符冒号）
    srt_arg = srt.as_posix().replace(":", "\\:")
    vf = f"subtitles='{srt_arg}'"
    _run_ffmpeg(["-i", str(video), "-vf", vf,
                 "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                 "-c:a", "copy", str(dest)])
    return str(dest.resolve())


@tool("Probe media file with ffprobe")
def probe(media_path: str) -> str:
    """探测媒体文件的时长、分辨率、帧率、音轨信息。

    Args:
        media_path: 媒体文件路径。

    Returns:
        JSON 字符串：duration_seconds、width、height、fps、has_audio。
    """
    media = _resolve(media_path)
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

    return json.dumps(
        {
            "path": str(media.resolve()),
            "duration_seconds": round(duration, 3),
            "width": video_stream.get("width", 0) if video_stream else 0,
            "height": video_stream.get("height", 0) if video_stream else 0,
            "fps": round(fps, 3),
            "has_audio": audio_stream is not None,
        },
        ensure_ascii=False,
    )


TOOLS = [assemble, mix_audio, burn_subtitles, probe]
