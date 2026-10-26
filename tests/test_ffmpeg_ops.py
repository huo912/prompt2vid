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
