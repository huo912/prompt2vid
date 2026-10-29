"""ffmpeg 工具定位与探测回退测试。

覆盖：环境变量覆盖 → PATH → 随包二进制的解析顺序；无 ffprobe 时
用 ffmpeg 流信息回退探测；两者都缺失时给出可操作的报错。
"""

from __future__ import annotations

import subprocess

import pytest

from prompt2vid import ffmpeg_ops


@pytest.fixture(autouse=True)
def _clear_tool_cache():
    ffmpeg_ops.resolve_tool.cache_clear()
    yield
    ffmpeg_ops.resolve_tool.cache_clear()


def _fake_bin(tmp_path, name: str):
    path = tmp_path / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_resolve_tool_prefers_env_override(monkeypatch, tmp_path):
    fake = _fake_bin(tmp_path, "my-ffprobe")
    monkeypatch.setenv("PROMPT2VID_FFPROBE", str(fake))
    monkeypatch.setattr("prompt2vid.ffmpeg_ops.shutil.which", lambda name: None)
    assert ffmpeg_ops.resolve_tool("ffprobe") == str(fake)


def test_resolve_tool_falls_back_to_path(monkeypatch):
    monkeypatch.setattr(
        "prompt2vid.ffmpeg_ops.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )
    monkeypatch.setattr("prompt2vid.ffmpeg_ops._as_executable", lambda p: str(p))
    assert ffmpeg_ops.resolve_tool("ffmpeg") == "/usr/bin/ffmpeg"


def test_resolve_tool_returns_none_when_absent(monkeypatch):
    monkeypatch.delenv("PROMPT2VID_FFMPEG", raising=False)
    monkeypatch.delenv("FFMPEG_BIN", raising=False)
    monkeypatch.setattr("prompt2vid.ffmpeg_ops.shutil.which", lambda name: None)
    monkeypatch.setattr("prompt2vid.ffmpeg_ops._COMMON_DIRS", ())
    monkeypatch.setattr("prompt2vid.ffmpeg_ops._bundled_candidates", lambda name: iter(()))
    assert ffmpeg_ops.resolve_tool("ffmpeg") is None


def test_require_tool_reports_install_hint(monkeypatch):
    monkeypatch.setattr("prompt2vid.ffmpeg_ops.resolve_tool", lambda name: None)
    with pytest.raises(RuntimeError) as excinfo:
        ffmpeg_ops.require_tool("ffprobe")
    message = str(excinfo.value)
    assert "PROMPT2VID_FFPROBE" in message
    assert "ffprobe" in message


FFMPEG_STDERR = """
Input #0, matroska,webm, from '/tmp/shot_1.webm':
  Duration: 00:00:05.12, start: 0.000000, bitrate: 1024 kb/s
  Stream #0:0: Video: vp9 (Profile 0), yuv420p(tv), 512x512, 24 fps, 24 tbr, 1k tbn
"""


def test_probe_file_falls_back_to_ffmpeg(monkeypatch, tmp_path):
    media = tmp_path / "shot_1.webm"
    media.write_bytes(b"fake")
    monkeypatch.setattr(
        ffmpeg_ops, "resolve_tool",
        lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
    )
    captured: list[str] = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd[0])
        return subprocess.CompletedProcess(cmd, 1, "", FFMPEG_STDERR)

    monkeypatch.setattr("prompt2vid.ffmpeg_ops.subprocess.run", fake_run)
    info = ffmpeg_ops.probe_file(media)
    assert captured == ["/usr/bin/ffmpeg"]
    assert info["duration_seconds"] == pytest.approx(5.12)
    assert (info["width"], info["height"]) == (512, 512)
    assert info["fps"] == pytest.approx(24.0)
    assert info["has_audio"] is False


def test_probe_file_prefers_ffprobe(monkeypatch, tmp_path):
    media = tmp_path / "a.mp4"
    media.write_bytes(b"fake")
    monkeypatch.setattr(ffmpeg_ops, "resolve_tool", lambda name: f"/usr/bin/{name}")
    payload = (
        '{"format": {"duration": "3.5"}, "streams": ['
        '{"codec_type": "video", "width": 720, "height": 1280,'
        ' "avg_frame_rate": "30/1"},'
        '{"codec_type": "audio"}]}'
    )

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, payload, "")

    monkeypatch.setattr("prompt2vid.ffmpeg_ops.subprocess.run", fake_run)
    info = ffmpeg_ops.probe_file(media)
    assert info["duration_seconds"] == pytest.approx(3.5)
    assert (info["width"], info["height"], info["fps"]) == (720, 1280, 30.0)
    assert info["has_audio"] is True


def test_probe_file_without_any_tool_raises(monkeypatch, tmp_path):
    media = tmp_path / "a.mp4"
    media.write_bytes(b"fake")
    monkeypatch.setattr("prompt2vid.ffmpeg_ops.resolve_tool", lambda name: None)
    with pytest.raises(RuntimeError, match="ffprobe"):
        ffmpeg_ops.probe_file(media)


def test_run_ffmpeg_uses_resolved_binary(monkeypatch):
    monkeypatch.setattr(
        "prompt2vid.ffmpeg_ops.resolve_tool", lambda name: "/opt/ffmpeg/bin/ffmpeg"
    )
    seen: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("prompt2vid.ffmpeg_ops.subprocess.run", fake_run)
    ffmpeg_ops.run_ffmpeg(["-i", "in.webm", "out.mp4"])
    assert seen["cmd"][0] == "/opt/ffmpeg/bin/ffmpeg"


# ------------------------------------------------------------ 画质参数

def _capture_ffmpeg_args(monkeypatch):
    """mock 掉 ffmpeg 执行，捕获完整参数列表。"""
    seen: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("prompt2vid.ffmpeg_ops.subprocess.run", fake_run)
    monkeypatch.setattr(
        "prompt2vid.ffmpeg_ops.resolve_tool", lambda name: "/usr/bin/ffmpeg"
    )
    return seen


def test_upscale_target_spec_forms():
    # 宽度形式：保持宽高比、偶数
    assert ffmpeg_ops.upscale_target(480, 832, "720") == (720, 1248)
    assert ffmpeg_ops.upscale_target(576, 1024, "720") == (720, 1280)
    # WxH 形式：作为上限，仍保持宽高比
    assert ffmpeg_ops.upscale_target(480, 832, "720x1280") == (720, 1248)
    # 源已达标（<=1.05）→ None；放大超过 2.5 倍 → None
    assert ffmpeg_ops.upscale_target(704, 1248, "720") is None
    assert ffmpeg_ops.upscale_target(480, 832, "1080") is None
    # 非法输入 → None
    for bad in ("", "abc", "0", "x10", "-5"):
        assert ffmpeg_ops.upscale_target(480, 832, bad) is None


def test_scaled_default_style_scales_font_and_margin():
    style = "FontSize=18,Outline=1,MarginV=40"
    scaled = ffmpeg_ops._scaled_default_style(style, 1.5)
    assert "FontSize=27" in scaled
    assert "MarginV=60" in scaled
    # 放大不明显时原样返回
    assert ffmpeg_ops._scaled_default_style(style, 1.0) == style


def test_normalize_clip_quality_defaults(monkeypatch, tmp_path):
    seen = _capture_ffmpeg_args(monkeypatch)
    ffmpeg_ops.normalize_clip(tmp_path / "in.webm", tmp_path / "out.mp4")
    args = seen[-1]
    joined = " ".join(args)
    assert "-crf 16" in joined          # 中转高质量（默认）
    assert "-r 25" in joined            # 帧率跟 LTX_FPS（默认 25）
    assert "minterpolate" not in joined  # 未配置补帧


def test_normalize_clip_smooth_fps_and_extend(monkeypatch, tmp_path):
    seen = _capture_ffmpeg_args(monkeypatch)
    monkeypatch.setenv("PROMPT2VID_SMOOTH_FPS", "50")
    ffmpeg_ops.normalize_clip(
        tmp_path / "in.webm", tmp_path / "out.mp4", extend_seconds=1.5
    )
    joined = " ".join(seen[-1])
    assert "minterpolate=fps=50" in joined
    assert "tpad=stop_mode=clone:stop_duration=1.500" in joined
    # 补帧时输出帧率由 minterpolate 决定，不再叠加 -r 做二次重采样
    assert " -r " not in f" {joined} "


def test_mux_and_burn_default_encode_quality(monkeypatch, tmp_path):
    seen = _capture_ffmpeg_args(monkeypatch)
    video, audio = tmp_path / "v.mp4", tmp_path / "a.wav"
    video.write_bytes(b"v")
    audio.write_bytes(b"a")
    ffmpeg_ops.mux_and_burn(video, audio, None, tmp_path / "final.mp4")
    joined = " ".join(seen[-1])
    assert "-crf 18" in joined            # 成片 CRF 18（默认）
    assert "-preset medium" in joined     # 默认 medium
    assert "+faststart" in joined         # moov 前置
    assert "-pix_fmt yuv420p" in joined


def test_mux_and_burn_upscale_filter_before_subtitles(monkeypatch, tmp_path):
    seen = _capture_ffmpeg_args(monkeypatch)
    video, audio, srt = tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "s.srt"
    video.write_bytes(b"v")
    audio.write_bytes(b"a")
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n测试\n", encoding="utf-8")

    def fake_probe(media):
        return {"path": str(media), "duration_seconds": 1.0, "width": 480,
                "height": 832, "fps": 25.0, "has_audio": True}

    monkeypatch.setattr(ffmpeg_ops, "probe_file", fake_probe)
    ffmpeg_ops.mux_and_burn(
        video, audio, srt, tmp_path / "final.mp4", upscale_to=(720, 1248)
    )
    joined = " ".join(seen[-1])
    # 放大在字幕之前（同一 -vf 链），锐化默认 0.3，默认字幕字号按 1.5x 放大
    assert "scale=720:1248:flags=lanczos" in joined
    assert "unsharp=5:5:0.3" in joined
    assert "subtitles=" in joined
    assert "FontSize=27" in joined  # 18 x (720/480) = 27
    scale_pos = joined.index("scale=")
    sub_pos = joined.index("subtitles=")
    assert scale_pos < sub_pos


def test_fit_audio_silent_input_skips_loudnorm(monkeypatch, tmp_path):
    """无声输入（TTS 被拦截的静音占位）不能过 loudnorm——会把量化
    底噪满增益放大成恒定噪声，应直接输出纯静音。"""
    seen = _capture_ffmpeg_args(monkeypatch)
    silent = tmp_path / "silent.mp3"
    silent.write_bytes(b"fake")

    monkeypatch.setattr(ffmpeg_ops, "audio_max_volume", lambda p: -91.0)
    ffmpeg_ops.fit_audio(silent, 3.0, tmp_path / "out.wav")
    joined = " ".join(seen[-1])
    assert "anullsrc" in joined and "loudnorm" not in joined

    # 有声输入走正常 loudnorm 路径
    monkeypatch.setattr(ffmpeg_ops, "audio_max_volume", lambda p: -12.0)
    ffmpeg_ops.fit_audio(silent, 3.0, tmp_path / "out2.wav")
    assert "loudnorm" in " ".join(seen[-1])

    # 探测失败（None）时保持旧行为：按有声处理
    monkeypatch.setattr(ffmpeg_ops, "audio_max_volume", lambda p: None)
    ffmpeg_ops.fit_audio(silent, 3.0, tmp_path / "out3.wav")
    assert "loudnorm" in " ".join(seen[-1])


def test_merge_with_transitions_builds_xfade_chain(monkeypatch, tmp_path):
    seen = _capture_ffmpeg_args(monkeypatch)
    clips = [tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "c.mp4"]
    for c in clips:
        c.write_bytes(b"x")
    ffmpeg_ops.merge_with_transitions(
        clips, offsets=[1.5, 3.2], duration=0.5, transition="fade",
        dest=tmp_path / "merged.mp4",
    )
    args = seen[-1]
    joined = " ".join(args)
    # 三个输入 + 两段 xfade 链
    assert joined.count("-i") == 3
    assert "xfade=transition=fade:duration=0.500:offset=1.495" in joined
    assert "xfade=transition=fade:duration=0.500:offset=3.195" in joined  # 5ms 安全余量
    assert "[vout]" in joined and "-map [vout]" in joined
    assert "-crf 16" in joined and "-an" in joined
    # 不支持的过渡名回退 fade
    ffmpeg_ops.merge_with_transitions(
        clips, offsets=[1.5, 3.2], duration=0.5, transition="nonexistent",
        dest=tmp_path / "merged2.mp4",
    )
    assert "transition=fade" in " ".join(seen[-1])
    # 参数校验
    import pytest as _pytest
    with _pytest.raises(ValueError):
        ffmpeg_ops.merge_with_transitions(clips[:1], [], 0.5, "fade", tmp_path / "m.mp4")
    with _pytest.raises(ValueError):
        ffmpeg_ops.merge_with_transitions(clips, [1.0], 0.5, "fade", tmp_path / "m.mp4")
