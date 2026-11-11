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
    # 默认带全局情感描述 → 情感节点；采样关束搜索
    assert workflow["tts"]["class_type"] == "IndexTTS25EmotionTextNode"
    assert workflow["tts"]["inputs"]["text"] == "秋天的公园"
    assert workflow["tts"]["inputs"]["num_beams"] == 1
    assert workflow["tts"]["inputs"]["emotion_description"]
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


def test_piper_forces_utf8_stdin(monkeypatch, tmp_path):
    """piper 子进程必须按 UTF-8 读 stdin：Windows 默认 cp936 会把中文变乱码，
    espeak 产出空音素 → piper 一个采样都不写就 exit 1（静音占位）。"""
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        captured["cmd"] = cmd
        Path(cmd[cmd.index("--output_file") + 1]).write_bytes(b"fake-wav")

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr(tts.shutil, "which", lambda name: "/fake/piper")
    monkeypatch.setattr(tts, "_piper_model", lambda: "/fake/voice.onnx")
    monkeypatch.setattr(tts.subprocess, "run", fake_run)
    monkeypatch.setattr(
        tts, "_finalize_audio",
        lambda wav, dest, bitrate="192k": bool(dest.write_bytes(b"mp3")),
    )

    dest = tmp_path / "shot_1.mp3"
    assert tts._tts_via_piper("秋天的公园", dest)

    assert captured["input"] == "秋天的公园"
    assert captured["env"]["PYTHONUTF8"] == "1"
    assert captured["env"]["PYTHONIOENCODING"].lower().startswith("utf-8")


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


# ------------------------------------------------------------ 角色声线

def test_voice_for_speaker_rotates_palette(monkeypatch):
    """旁白不指定音色；不同角色 id 拿到不同嗓子（避免全片一个声音）。"""
    monkeypatch.delenv("PROMPT2VID_TTS_VOICES", raising=False)
    assert tts.voice_for_speaker(0) == ""  # 旁白 → 用 EDGE_TTS_VOICE
    voices = [tts.voice_for_speaker(i) for i in range(1, 5)]
    assert all(voices)
    assert len(set(voices)) == len(voices)
    # 超出内置表长度后循环复用
    assert tts.voice_for_speaker(len(tts.DEFAULT_CHARACTER_VOICES) + 1) == voices[0]


def test_voice_for_speaker_env_override(monkeypatch):
    monkeypatch.setenv("PROMPT2VID_TTS_VOICES", "zh-CN-A,zh-CN-B")
    assert tts.voice_for_speaker(1) == "zh-CN-A"
    assert tts.voice_for_speaker(2) == "zh-CN-B"
    assert tts.voice_for_speaker(3) == "zh-CN-A"
    # 也支持中文顿号/分号分隔
    monkeypatch.setenv("PROMPT2VID_TTS_VOICES", "zh-CN-A、zh-CN-B；zh-CN-C")
    assert [tts.voice_for_speaker(i) for i in (1, 2, 3)] == [
        "zh-CN-A", "zh-CN-B", "zh-CN-C",
    ]


def test_timbre_for_speaker_defaults_empty(monkeypatch):
    """未配置音色库时按角色 id 轮转默认音色表；旁白恒为空（走全局值）。"""
    monkeypatch.delenv("PROMPT2VID_TTS_TIMBRES", raising=False)
    assert tts.timbre_for_speaker(0) == ""  # 旁白：全局音色
    first = tts.timbre_for_speaker(1)
    second = tts.timbre_for_speaker(2)
    assert first and second and first != second  # 不同角色不同嗓子
    assert tts.timbre_for_speaker(1) == first  # 同角色稳定
    monkeypatch.setenv("PROMPT2VID_TTS_TIMBRES", "甜美女声.mp3,播音中年男.mp3")
    assert tts.timbre_for_speaker(1) == "甜美女声.mp3"
    assert tts.timbre_for_speaker(2) == "播音中年男.mp3"
    assert tts.timbre_for_speaker(0) == ""


def test_resolve_timbre_matches_catalogue(monkeypatch):
    """分镜写的音色名按精确/忽略大小写/忽略扩展名三级匹配到真实文件名。"""
    catalogue = ["北京大爷.mp3", "说书人.WAV", "甜美女声.mp3"]
    monkeypatch.setattr(tts, "available_timbres", lambda: list(catalogue))
    assert tts.resolve_timbre("北京大爷.mp3") == "北京大爷.mp3"
    # 音色包里 .WAV/.wav/.MP3 混用，LLM 抄成小写也要认
    assert tts.resolve_timbre("说书人.wav") == "说书人.WAV"
    # 扩展名写错但主干对得上
    assert tts.resolve_timbre("甜美女声.wav") == "甜美女声.mp3"
    # 服务器上确实没有 → 空串（调用方退回轮转）
    assert tts.resolve_timbre("不存在的音色.mp3") == ""
    assert tts.resolve_timbre("") == ""
    # 清单里的坑：静音占位与朗读小说腔一律拒绝
    assert tts.resolve_timbre("无音频文件") == ""
    assert tts.resolve_timbre("抖音-读小说.wav") == ""


def test_resolve_timbre_without_catalogue_keeps_name(monkeypatch):
    """探测不到音色清单时不拦路，交给 TTS 自己回退。"""
    monkeypatch.setattr(tts, "available_timbres", lambda: [])
    assert tts.resolve_timbre("帝王音.mp3") == "帝王音.mp3"


def test_synthesize_audio_passes_timbre_to_comfyui(monkeypatch, tmp_path):
    """台词行的角色 timbre 会写进 IndexTTS 的 TimbreAudioLoader。"""
    fake = FakeComfyTTSClient()
    monkeypatch.setattr(
        "prompt2vid.comfyui_client.ComfyUIClient", lambda *a, **kw: fake
    )
    _fake_transcode(monkeypatch)
    monkeypatch.delenv("PROMPT2VID_TTS_REF_AUDIO", raising=False)

    tts.synthesize_audio("陛下", tmp_path / "shot_1_line_1.mp3", timbre="帝王音.mp3")

    workflow = fake.submitted[0]
    assert workflow["timbre"]["class_type"] == "TimbreAudioLoader"
    assert workflow["timbre"]["inputs"]["audio_file"] == "帝王音.mp3"


def test_synthesize_audio_passes_emotion_to_comfyui(monkeypatch, tmp_path):
    """台词行的 emotion 覆盖全局情感描述（逐句语气控制）。"""
    fake = FakeComfyTTSClient()
    monkeypatch.setattr(
        "prompt2vid.comfyui_client.ComfyUIClient", lambda *a, **kw: fake
    )
    _fake_transcode(monkeypatch)
    monkeypatch.delenv("PROMPT2VID_TTS_REF_AUDIO", raising=False)

    # 逐句情感：emotion 非空时不需要探测情感节点（测试环境 ComfyUI 不可达
    # 也不会被触发——探测只在 _tts_via_comfyui 且 emotion 非空时发生，
    # FakeComfyTTSClient 的 node_exists 不含情感节点会清空 emotion，
    # 因此这里直接 mock 探测结果为 True）
    monkeypatch.setattr(tts, "_emotion_node_available", lambda: True)
    tts.synthesize_audio(
        "这活，我们真不会。", tmp_path / "shot_5_line_2.mp3", emotion="哽咽，懊悔自责"
    )

    workflow = fake.submitted[0]
    assert workflow["tts"]["class_type"] == "IndexTTS25EmotionTextNode"
    assert workflow["tts"]["inputs"]["emotion_description"] == "哽咽，懊悔自责"
