"""配音引擎链测试：IndexTTS-2.5（ComfyUI）优先、逐级回退、降级告警。

IndexTTS 引擎用 fake 的 ComfyUIClient 模拟（提交→完成→下载），
不依赖真实 ComfyUI / ffmpeg / 网络。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from prompt2vid import tts


@pytest.fixture(autouse=True)
def _reset_engine_cache():
    """每个用例前后清空引擎探测缓存（环境变量可能不同）。"""
    tts.reset_engine_cache()
    yield
    tts.reset_engine_cache()


class FakeComfyTTSClient:
    """IndexTTS 工作流的 ComfyUI 客户端 mock（探测→上传→提交→下载）。"""

    def __init__(self, has_nodes: bool = True):
        self.has_nodes = has_nodes
        self.submitted: list[dict] = []
        self.uploaded: list[Path] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def node_exists(self, node: str) -> bool:
        return self.has_nodes and node in (
            "IndexTTS25BaseNode", "TimbreAudioLoader", "SaveAudio",
        )

    def upload_file(self, path, name=None) -> str:
        self.uploaded.append(Path(path))
        return Path(path).name

    def filter_workflow_inputs(self, workflow):
        return workflow

    def submit(self, workflow) -> str:
        self.submitted.append(workflow)
        return "pid-1"

    def wait(self, prompt_id, timeout=None) -> dict:
        return {"outputs": {"save": {"audio": [{"filename": "tts_00001.wav"}]}}}

    def outputs(self, entry):
        from prompt2vid.comfyui_client import parse_output_files

        return parse_output_files(entry)

    def download(self, file_info, dest) -> Path:
        dest = Path(dest)
        dest.write_bytes(b"fake-wav")
        return dest


def _fake_transcode(monkeypatch):
    """模拟 ffmpeg 转码：把源文件字节拷到目标（不依赖真实 ffmpeg）。"""

    def fake_run_ffmpeg(args):
        # args = ["-i", src, "-c:a", "libmp3lame", "-b:a", bitrate, dest]
        src, dest = Path(args[1]), Path(args[-1])
        dest.write_bytes(src.read_bytes())

    monkeypatch.setattr(tts, "run_ffmpeg", fake_run_ffmpeg)


def test_engine_list_default_and_explicit(monkeypatch):
    monkeypatch.delenv("PROMPT2VID_TTS_ENGINE", raising=False)
    assert tts._engine_list() == ["comfyui", "edge", "gateway", "piper"]
    monkeypatch.setenv("PROMPT2VID_TTS_ENGINE", "piper")
    # 显式引擎排第一，其余仍作降级兜底
    assert tts._engine_list() == ["piper", "comfyui", "edge", "gateway"]


def test_comfyui_engine_success(monkeypatch, tmp_path):
    fake = FakeComfyTTSClient()
    monkeypatch.setattr(
        "prompt2vid.comfyui_client.ComfyUIClient", lambda *a, **kw: fake
    )
    _fake_transcode(monkeypatch)
    monkeypatch.delenv("PROMPT2VID_TTS_REF_AUDIO", raising=False)

    dest = tmp_path / "shot_1.mp3"
    result = tts.synthesize_audio("秋天的公园", dest)

    assert result == dest.resolve()
    assert dest.is_file() and dest.stat().st_size > 0
    # 提交的工作流是 IndexTTS-2.5 节点链
    assert len(fake.submitted) == 1
    workflow = fake.submitted[0]
    assert workflow["tts"]["class_type"] == "IndexTTS25BaseNode"
    assert workflow["tts"]["inputs"]["text"] == "秋天的公园"
    assert workflow["timbre"]["class_type"] == "TimbreAudioLoader"
    assert workflow["save"]["class_type"] == "SaveAudio"
    # 中间 wav 已清理
    assert not (tmp_path / "shot_1.comfyui.wav").exists()


def test_comfyui_engine_clones_reference_audio(monkeypatch, tmp_path):
    """PROMPT2VID_TTS_REF_AUDIO 指向录音时上传并克隆该音色。"""
    fake = FakeComfyTTSClient()
    monkeypatch.setattr(
        "prompt2vid.comfyui_client.ComfyUIClient", lambda *a, **kw: fake
    )
    _fake_transcode(monkeypatch)
    ref = tmp_path / "my_voice.wav"
    ref.write_bytes(b"ref-audio")
    monkeypatch.setenv("PROMPT2VID_TTS_REF_AUDIO", str(ref))

    dest = tmp_path / "shot_1.mp3"
    tts.synthesize_audio("你好", dest)

    assert fake.uploaded == [ref]
    workflow = fake.submitted[0]
    assert workflow["timbre"]["class_type"] == "LoadAudio"
    assert workflow["timbre"]["inputs"]["audio"] == "my_voice.wav"


def test_comfyui_unavailable_falls_back_to_edge(monkeypatch, tmp_path, caplog):
    """节点未安装：告警一次后降级到下一引擎（不再是静默降级）。"""
    fake = FakeComfyTTSClient(has_nodes=False)
    monkeypatch.setattr(
        "prompt2vid.comfyui_client.ComfyUIClient", lambda *a, **kw: fake
    )

    def fake_edge(text, dest, voice="", rate=""):
        dest.write_bytes(b"edge-audio")
        return True

    monkeypatch.setattr(tts, "_tts_via_edge", fake_edge)

    with caplog.at_level(logging.WARNING):
        dest = tmp_path / "shot_1.mp3"
        tts.synthesize_audio("秋天的公园", dest)

    assert dest.read_bytes() == b"edge-audio"
    assert any("IndexTTS-2.5 不可用" in r.getMessage() for r in caplog.records)


def test_all_engines_fail_writes_silent_placeholder(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(tts, "_comfyui_tts_available", lambda: False)
    monkeypatch.setattr(tts, "_tts_via_edge", lambda *a, **kw: False)
    monkeypatch.setattr(tts, "_tts_via_gateway", lambda text, dest: False)
    monkeypatch.setattr(tts, "_tts_via_piper", lambda text, dest: False)

    with caplog.at_level(logging.WARNING):
        dest = tmp_path / "shot_1.mp3"
        tts.synthesize_audio("四个字的测试", dest)

    assert dest.is_file()
    # 旁白文本存 sidecar，质检阶段会报"旁白无声"
    assert (tmp_path / "shot_1.txt").read_text(encoding="utf-8") == "四个字的测试"
    assert any("全部配音引擎失败" in r.getMessage() for r in caplog.records)
