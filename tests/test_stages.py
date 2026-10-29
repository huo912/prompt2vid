"""确定性阶段引擎测试。

generate_frames / generate_clips 用 mock 的 ComfyUIClient 验证批量提交
编排与清单；render / quality 依赖真实 ffmpeg（本机没有则跳过，
CI 或装有 ffmpeg 的环境会覆盖）。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from prompt2vid import stages
from prompt2vid.comfyui_client import ComfyUITaskError, fit_ltx_size, parse_output_files
from prompt2vid.ffmpeg_ops import resolve_tool

# 通过统一解析器定位 ffmpeg（PATH / 环境变量 / 随包二进制）；
# ffprobe 缺失时探测会回退解析 ffmpeg 输出，因此只要求 ffmpeg。
FFMPEG = resolve_tool("ffmpeg")
HAS_FFMPEG = FFMPEG is not None

STORYBOARD = {
    "shots": [
        {
            "id": 1,
            "narration": "秋天的公园",
            "visual": "金色树叶",
            "image_prompt": "autumn park, golden leaves",
            "motion_prompt": "slow zoom in",
            "duration_seconds": 4,
        },
        {
            "id": 2,
            "narration": "风吹落叶",
            "visual": "落叶飘落",
            "image_prompt": "falling leaves",
            "motion_prompt": "gentle wind",
            "duration_seconds": 3,
        },
    ],
    "total_duration_seconds": 7,
}

CHARACTER_DESCRIPTION = "a young woman with long black hair, wearing a red knit sweater"

STORYBOARD_WITH_CHARACTERS = {
    "characters": [
        {"id": 1, "name": "女主", "description": CHARACTER_DESCRIPTION},
    ],
    "shots": [
        {
            "id": 1,
            "characters": [1],
            "narration": "少女入园",
            "visual": "红衣少女站在公园",
            "image_prompt": "autumn park, golden leaves",
            "motion_prompt": "slow zoom in",
            "duration_seconds": 4,
        },
        {
            "id": 2,
            "narration": "空镜落叶",
            "visual": "落叶飘落",
            "image_prompt": "falling leaves",
            "motion_prompt": "gentle wind",
            "duration_seconds": 3,
        },
    ],
    "total_duration_seconds": 7,
}


class FakeComfyUIClient:
    """批量客户端 mock：按提交序号决定成败，可模拟一次性失败。

    fail_first_n：前 n 个提交失败（含 wait/wait_all）。
    fail_indices：指定序号集合失败（用于"定妆照成功、分镜失败"等场景）。
    missing_nodes：node_exists 返回 False 的节点集合（口型同步 gating 用）。
    """

    def __init__(
        self,
        fail_first_n: int = 0,
        fail_all: bool = False,
        fail_indices: set[int] | None = None,
        missing_nodes: set[str] | None = None,
    ):
        self.submitted: list[dict] = []
        self.uploads: list[Path] = []
        self._fail_first_n = fail_first_n
        self._fail_all = fail_all
        self._fail_indices = fail_indices or set()
        self._missing_nodes = missing_nodes or set()

    def _fails(self, idx: int) -> bool:
        return self._fail_all or idx in self._fail_indices or idx < self._fail_first_n

    def submit(self, workflow) -> str:
        pid = f"pid{len(self.submitted)}"
        self.submitted.append(workflow)
        return pid

    def wait(self, prompt_id) -> dict:
        if self._fails(int(prompt_id[3:])):
            raise ComfyUITaskError("ComfyUI exploded")
        return {"outputs": {"9": {"images": [{"filename": "out.png"}]}}}

    def wait_all(self, prompt_ids) -> dict:
        results = {}
        for pid in prompt_ids:
            if self._fails(int(pid[3:])):
                results[pid] = ComfyUITaskError("ComfyUI exploded")
            else:
                results[pid] = {"outputs": {"9": {"images": [{"filename": "out.png"}]}}}
        return results

    def outputs(self, entry) -> list[dict]:
        # 走真实解析函数：mock 的 history 结构必须与真实 ComfyUI 一致
        # （曾因 mock 绕过解析、格式与新版 ComfyUI 脱节导致线上全分镜失败）
        return parse_output_files(entry)

    def download(self, file_info, dest) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake-output")
        return dest

    def upload_image(self, image_path, name=None) -> str:
        self.uploads.append(Path(image_path))
        return Path(image_path).name

    def upload_file(self, file_path, name=None) -> str:
        self.uploads.append(Path(file_path))
        return Path(file_path).name

    def node_exists(self, node: str) -> bool:
        return node not in self._missing_nodes

    def filter_workflow_inputs(self, workflow):
        # 真实客户端会按服务端 object_info 过滤未声明输入；mock 原样返回
        return workflow

    def resolve_checkpoint(self, role: str = "t2i") -> str:
        # 真实客户端会查 /object_info 校准文件名；mock 固定返回可预测值
        return f"mock-{role}.safetensors"

    def resolve_lora(self, role: str = "t2i") -> str:
        # 真实客户端会查 /object_info；mock 默认不挂 LoRA（空串 = 关闭）
        return ""

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _setup_workspace(tmp_path: Path, monkeypatch, storyboard: dict | None = None) -> Path:
    monkeypatch.setenv("PROMPT2VID_WORKSPACE", str(tmp_path))
    # 渲染类测试固定基线：仓库 .env（含 PROMPT2VID_UPSCALE 等）可能被
    # 其它测试经 load_env 写进 os.environ，这里统一清掉保证断言稳定
    for name in ("PROMPT2VID_UPSCALE", "PROMPT2VID_UPSCALE_SHARPEN",
                 "PROMPT2VID_SMOOTH_FPS", "PROMPT2VID_VIDEO_CRF",
                 "PROMPT2VID_CLIP_CRF"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "storyboard.json").write_text(
        json.dumps(storyboard or STORYBOARD, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


def _use_fake_client(monkeypatch, **kwargs) -> FakeComfyUIClient:
    fake = FakeComfyUIClient(**kwargs)
    monkeypatch.setattr(stages, "ComfyUIClient", lambda *a, **kw: fake)
    return fake


def test_frames_for_duration_respects_ltx_constraints():
    # 8n+1 约束
    for seconds in (1, 2, 3, 4, 5, 8, 10):
        length = stages.frames_for_duration(seconds)
        assert (length - 1) % 8 == 0, f"{seconds}s -> {length}"
        assert length >= stages.MIN_FRAMES
    # 上限
    assert stages.frames_for_duration(60) <= 121


def test_frames_for_duration_follows_ltx_fps_env(monkeypatch):
    """COMFYUI_LTX_FPS 参与帧数换算（工作流的 frame_rate 同源）。"""
    monkeypatch.delenv("COMFYUI_LTX_FPS", raising=False)
    assert stages.frames_for_duration(2) == 57  # 2s x 25fps → 8n+1
    monkeypatch.setenv("COMFYUI_LTX_FPS", "50")
    assert stages.frames_for_duration(2) == 105  # 2s x 50fps → 8n+1


def test_generate_frames_batch_writes_manifest(tmp_path, monkeypatch):
    ws = _setup_workspace(tmp_path, monkeypatch)
    fake = _use_fake_client(monkeypatch)

    summary = stages.generate_frames()

    assert summary["stage"] == "frames"
    assert summary["total"] == 2
    assert summary["succeeded"] == 2
    assert summary["failed"] == 0
    # 整批一次性提交（两个工作流在同一次 wait_all 前提交完）
    assert len(fake.submitted) == 2
    manifest = json.loads((ws / "frames" / "manifest.json").read_text(encoding="utf-8"))
    assert [e["id"] for e in manifest["shots"]] == [1, 2]
    assert all(e["status"] == "ok" for e in manifest["shots"])
    # 清单里的 path 必须真实存在（绝不编造）
    for entry in manifest["shots"]:
        assert Path(entry["path"]).is_file()


def test_generate_frames_retries_failed_batch(tmp_path, monkeypatch):
    ws = _setup_workspace(tmp_path, monkeypatch)
    monkeypatch.setenv("COMFYUI_RETRIES", "1")
    # 首批两个提交（序号 0、1）失败，重试批（序号 2、3）成功
    fake = _use_fake_client(monkeypatch, fail_first_n=2)

    summary = stages.generate_frames()

    assert len(fake.submitted) == 4
    assert summary["succeeded"] == 2
    assert summary["failed"] == 0
    manifest = json.loads((ws / "frames" / "manifest.json").read_text(encoding="utf-8"))
    assert all(e["status"] == "ok" for e in manifest["shots"])


def test_generate_frames_reports_permanent_failure(tmp_path, monkeypatch):
    ws = _setup_workspace(tmp_path, monkeypatch)
    monkeypatch.setenv("COMFYUI_RETRIES", "1")
    fake = _use_fake_client(monkeypatch, fail_all=True)

    summary = stages.generate_frames()

    assert len(fake.submitted) == 4  # 2 镜 × (1 + 1 重试)
    assert summary["succeeded"] == 0
    assert summary["failed"] == 2
    assert "ComfyUI exploded" in summary["failed_shots"][0]["error"]
    manifest = json.loads((ws / "frames" / "manifest.json").read_text(encoding="utf-8"))
    assert all(e["status"] == "failed" for e in manifest["shots"])


def test_generate_frames_missing_storyboard(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPT2VID_WORKSPACE", str(tmp_path))
    summary = stages.generate_frames()
    assert "error" in summary
    assert "storyboard.json" in summary["error"]


def test_generate_frames_skips_existing(tmp_path, monkeypatch):
    """断点续跑：已存在的帧不再提交。"""
    ws = _setup_workspace(tmp_path, monkeypatch)
    frame = ws / "frames" / "shot_1.png"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"existing")
    fake = _use_fake_client(monkeypatch)

    summary = stages.generate_frames()

    assert len(fake.submitted) == 1  # 只有镜 2 需要生成
    assert summary["succeeded"] == 2


# ------------------------------------------------------------ 角色一致性


def test_generate_frames_with_character_reference(tmp_path, monkeypatch):
    """IPAdapter 开启：先出角色定妆照，出镜分镜带参考，空镜不带。"""
    ws = _setup_workspace(tmp_path, monkeypatch, STORYBOARD_WITH_CHARACTERS)
    monkeypatch.setenv("COMFYUI_IPADAPTER", "true")
    fake = _use_fake_client(monkeypatch)

    summary = stages.generate_frames()

    assert summary["succeeded"] == 2
    # 提交序：0=定妆照，1=镜1（带参考），2=镜2（空镜）
    assert len(fake.submitted) == 3
    ref_wf, shot1_wf, shot2_wf = fake.submitted
    # 定妆照：1024 方图，prompt 含角色外观契约
    assert ref_wf["70"]["inputs"]["width"] == ref_wf["70"]["inputs"]["height"] == 1024
    assert CHARACTER_DESCRIPTION in ref_wf["6"]["inputs"]["text"]
    assert (ws / "frames" / "character_ref_1.png").is_file()
    # 定妆照已上传，出镜分镜携带 IPAdapter 节点
    assert [p.name for p in fake.uploads] == ["character_ref_1.png"]
    assert shot1_wf["ref_apply"]["class_type"] == "IPAdapter"
    assert shot1_wf["3"]["inputs"]["model"] == ["ref_apply", 0]
    assert shot1_wf["ref_image"]["inputs"]["image"] == "character_ref_1.png"
    # 一期契约：出镜分镜的 prompt 含角色外观描述
    assert CHARACTER_DESCRIPTION in shot1_wf["6"]["inputs"]["text"]
    # 空镜分镜：纯 t2i，不注入参考
    assert not any(key.startswith("ref_") for key in shot2_wf)


def test_generate_frames_reference_disabled_by_default(tmp_path, monkeypatch):
    """默认关闭：只走一期 prompt 契约，无定妆照、无 IPAdapter 节点。"""
    _setup_workspace(tmp_path, monkeypatch, STORYBOARD_WITH_CHARACTERS)
    # 本地 .env 可能开了 IPAdapter，测试必须显式关闭
    monkeypatch.delenv("COMFYUI_IPADAPTER", raising=False)
    fake = _use_fake_client(monkeypatch)

    summary = stages.generate_frames()

    assert summary["succeeded"] == 2
    assert len(fake.submitted) == 2
    assert fake.uploads == []
    for wf in fake.submitted:
        assert not any(key.startswith("ref_") for key in wf)
    # 一期契约仍生效：出镜分镜 prompt 已含角色外观描述
    assert CHARACTER_DESCRIPTION in fake.submitted[0]["6"]["inputs"]["text"]


def test_generate_frames_reference_failure_falls_back(tmp_path, monkeypatch):
    """定妆照生成失败：警告并全部回退无参考，流水线不中断。"""
    ws = _setup_workspace(tmp_path, monkeypatch, STORYBOARD_WITH_CHARACTERS)
    monkeypatch.setenv("COMFYUI_IPADAPTER", "true")
    # 序号 0 = 定妆照 → 生成失败
    fake = _use_fake_client(monkeypatch, fail_indices={0})

    summary = stages.generate_frames()

    assert summary["succeeded"] == 2
    for wf in fake.submitted:
        assert not any(key.startswith("ref_") for key in wf)
    assert not (ws / "frames" / "character_ref_1.png").is_file()


def test_generate_frames_degrades_reference_on_retry(tmp_path, monkeypatch):
    """带参考的分镜失败后，重试轮自动降级为无参考。"""
    _setup_workspace(tmp_path, monkeypatch, STORYBOARD_WITH_CHARACTERS)
    monkeypatch.setenv("COMFYUI_IPADAPTER", "true")
    monkeypatch.setenv("COMFYUI_RETRIES", "1")
    # 0=定妆照(成功)，1=镜1带参考(失败)，2=镜2空镜(成功)
    fake = _use_fake_client(monkeypatch, fail_indices={1})

    summary = stages.generate_frames()

    assert summary["succeeded"] == 2
    assert len(fake.submitted) == 4  # 定妆照 + 首批 2 镜 + 重试 1 镜
    _, shot1_first, _, shot1_retry = fake.submitted
    assert shot1_first["ref_apply"]["class_type"] == "IPAdapter"
    # 重试轮降级：镜 1 不再携带参考节点
    assert not any(key.startswith("ref_") for key in shot1_retry)


def test_generate_frames_applies_style_and_lora(tmp_path, monkeypatch):
    """风格约束 + LoRA：每个提交的工作流都带风格词与 LoRA 节点。"""
    _setup_workspace(tmp_path, monkeypatch)
    monkeypatch.setenv("PROMPT2VID_STYLE", "anime")
    monkeypatch.setenv("COMFYUI_T2I_LORA", "anime.safetensors")
    fake = _use_fake_client(monkeypatch)
    fake.resolve_lora = lambda role="t2i": "anime.safetensors"

    summary = stages.generate_frames()

    assert summary["succeeded"] == 2
    assert len(fake.submitted) == 2
    for wf, original in zip(fake.submitted, ("autumn park, golden leaves", "falling leaves")):
        text = wf["6"]["inputs"]["text"]
        assert original in text  # 原始画面描述保留
        assert "anime style" in text  # 风格正向词已追加
        assert "photorealistic" in wf["7"]["inputs"]["text"]  # 负向排斥词
        assert wf["lora"]["class_type"] == "LoraLoader"
        assert wf["lora"]["inputs"]["lora_name"] == "anime.safetensors"
        # 文本编码与采样器都接在 LoRA 之后
        assert wf["6"]["inputs"]["clip"] == ["lora", 1]
        assert wf["3"]["inputs"]["model"] == ["lora", 0]


def test_generate_clips_applies_lora_model_only(tmp_path, monkeypatch):
    """视频侧 LoRA 用 LoraLoaderModelOnly（LTX checkpoint 无 CLIP 可接）。"""
    _setup_workspace(tmp_path, monkeypatch)
    _use_fake_client(monkeypatch)
    stages.generate_frames()  # 先产出首帧清单

    fake = _use_fake_client(monkeypatch)
    fake.resolve_lora = lambda role="t2i": (
        "shinkai-anime-style-lora-v1_58000.safetensors"
    )
    monkeypatch.delenv("PROMPT2VID_STYLE", raising=False)

    summary = stages.generate_clips()

    assert summary["succeeded"] == 2
    for wf in fake.submitted:
        assert wf["lora"]["class_type"] == "LoraLoaderModelOnly"
        assert wf["72"]["inputs"]["model"] == ["lora", 0]
        assert wf["lora"]["inputs"]["model"] == ["44", 0]


def test_generate_clips_batch_from_frames_manifest(tmp_path, monkeypatch):
    ws = _setup_workspace(tmp_path, monkeypatch)

    # 先生成首帧（复用 mock）
    _use_fake_client(monkeypatch)
    stages.generate_frames()

    # clips 阶段用新 fake（其 submitted 只包含视频工作流，便于断言）
    fake = _use_fake_client(monkeypatch)
    summary = stages.generate_clips()

    assert summary["succeeded"] == 2
    manifest = json.loads((ws / "clips" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["shots"][0]["status"] == "ok"
    # 整批一次性提交
    assert len(fake.submitted) == 2
    first_wf = fake.submitted[0]
    # prompt = image_prompt + motion_prompt + 肢体稳定提示
    text = first_wf["6"]["inputs"]["text"]
    assert "autumn park, golden leaves" in text
    assert "slow zoom in" in text
    assert "consistent anatomy" in text
    # 视频画幅跟首帧画幅一致（回归：固定 512x768 会把 768x1344 首帧裁变形）
    width = first_wf["95"]["inputs"]["width"]
    height = first_wf["95"]["inputs"]["height"]
    expected_w, expected_h = fit_ltx_size(768, 1344, 512 * 768)
    assert (width, height) == (expected_w, expected_h)
    assert width % 32 == 0 and height % 32 == 0
    assert manifest["shots"][0]["width"] == width
    assert manifest["shots"][0]["height"] == height
    # 4s → 105 帧（25fps、8n+1 约束）
    assert first_wf["95"]["inputs"]["length"] == 105
    assert manifest["shots"][0]["length"] == 105


def test_generate_clips_reads_first_frame_size(tmp_path, monkeypatch):
    """首帧 PNG 头里的 768x1344 必须成为视频画幅依据，而不是默认 512x768。"""
    ws = _setup_workspace(tmp_path, monkeypatch)
    frames_dir = ws / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for shot_id in (1, 2):
        frame = frames_dir / f"shot_{shot_id}.png"
        frame.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            + (768).to_bytes(4, "big")
            + (1344).to_bytes(4, "big")
        )
        entries.append(
            {"id": shot_id, "image": frame.name, "status": "ok", "path": str(frame)}
        )
    (frames_dir / "manifest.json").write_text(
        json.dumps({"shots": entries}), encoding="utf-8"
    )

    fake = _use_fake_client(monkeypatch)
    summary = stages.generate_clips()

    assert summary["succeeded"] == 2
    width = fake.submitted[0]["95"]["inputs"]["width"]
    height = fake.submitted[0]["95"]["inputs"]["height"]
    assert (width, height) == fit_ltx_size(768, 1344, 512 * 768)


def test_generate_voiceovers_writes_manifest(tmp_path, monkeypatch):
    ws = _setup_workspace(tmp_path, monkeypatch)

    def fake_tts(text, dest, **kwargs):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"mp3-fake")
        return dest

    monkeypatch.setattr(stages.tts, "synthesize_audio", fake_tts)
    summary = stages.generate_voiceovers()

    assert summary["succeeded"] == 2
    manifest = json.loads((ws / "audio" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["shots"][1]["audio"] == "audio/shot_2.mp3"
    assert Path(manifest["shots"][1]["path"]).is_file()


# ------------------------------------------------------------ 口型同步


def _lipsync_workspace(tmp_path: Path, monkeypatch, with_audio: bool = True) -> Path:
    """clips/ + audio/（可选）齐备的工作目录，供 sync_lips 使用。"""
    ws = _setup_workspace(tmp_path, monkeypatch)
    monkeypatch.delenv("PROMPT2VID_LIPSYNC", raising=False)
    clip_entries = []
    for shot_id in (1, 2):
        clip = ws / "clips" / f"shot_{shot_id}.webm"
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(b"fake-clip")
        clip_entries.append({
            "id": shot_id, "video": f"clips/shot_{shot_id}.webm",
            "status": "ok", "path": str(clip),
        })
    (ws / "clips" / "manifest.json").write_text(
        json.dumps({"shots": clip_entries}, ensure_ascii=False), encoding="utf-8"
    )
    if with_audio:
        audio_entries = []
        for shot_id in (1, 2):
            audio = ws / "audio" / f"shot_{shot_id}.mp3"
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(b"fake-audio")
            audio_entries.append({
                "id": shot_id, "audio": f"audio/shot_{shot_id}.mp3",
                "status": "ok", "path": str(audio),
            })
        (ws / "audio" / "manifest.json").write_text(
            json.dumps({"shots": audio_entries}, ensure_ascii=False),
            encoding="utf-8",
        )
    return ws


def _fake_durations(monkeypatch, clip_seconds: float, audio_seconds: float):
    """控制 _audio_drives 的时长探测（音视频按扩展名区分）。"""
    def fake_probe(path):
        path = Path(path)
        seconds = audio_seconds if path.suffix == ".mp3" else clip_seconds
        return {"path": str(path), "duration_seconds": seconds,
                "width": 512, "height": 768, "fps": 25.0, "has_audio": True}

    monkeypatch.setattr(stages.ffmpeg_ops, "probe_file", fake_probe)


def test_sync_lips_submits_latentsync_workflows(tmp_path, monkeypatch):
    ws = _lipsync_workspace(tmp_path, monkeypatch)
    fake = _use_fake_client(monkeypatch)
    # 配音 5s > 片段 2s → pingpong（补帧到配音长度，旁白零截断）
    _fake_durations(monkeypatch, clip_seconds=2.0, audio_seconds=5.0)

    result = stages.sync_lips()

    assert result["stage"] == "lipsync"
    assert result["succeeded"] == 2
    assert result["failed"] == 0
    assert "error" not in result
    assert len(fake.submitted) == 2
    for workflow in fake.submitted:
        assert workflow["latentsync"]["class_type"] == "LatentSyncNode"
        assert workflow["adjust"]["inputs"]["mode"] == "pingpong"
        assert workflow["save"]["class_type"] == "VHS_VideoCombine"
    # 片段与配音都上传；产物落盘并写清单
    assert {p.name for p in fake.uploads} == {
        "shot_1.webm", "shot_2.webm", "shot_1.mp3", "shot_2.mp3",
    }
    for shot_id in (1, 2):
        assert (ws / "lipsync" / f"shot_{shot_id}.mp4").is_file()
    manifest = json.loads((ws / "lipsync" / "manifest.json").read_text(encoding="utf-8"))
    assert [s["id"] for s in manifest["shots"]] == [1, 2]
    assert all(s["status"] == "ok" for s in manifest["shots"])


def test_sync_lips_normal_mode_when_clip_longer(tmp_path, monkeypatch):
    """片段比配音长 → normal（裁帧 + 音频尾部补静音）。"""
    _lipsync_workspace(tmp_path, monkeypatch)
    fake = _use_fake_client(monkeypatch)
    _fake_durations(monkeypatch, clip_seconds=5.0, audio_seconds=2.0)

    stages.sync_lips()

    for workflow in fake.submitted:
        assert workflow["adjust"]["inputs"]["mode"] == "normal"


def test_sync_lips_probe_failure_defaults_to_pingpong(tmp_path, monkeypatch):
    """探测失败按 pingpong 处理：音频只会被静音补齐、绝不被裁剪。"""
    _lipsync_workspace(tmp_path, monkeypatch)
    fake = _use_fake_client(monkeypatch)

    def failing_probe(path):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(stages.ffmpeg_ops, "probe_file", failing_probe)
    stages.sync_lips()

    for workflow in fake.submitted:
        assert workflow["adjust"]["inputs"]["mode"] == "pingpong"


def test_sync_lips_failed_shot_keeps_original(tmp_path, monkeypatch):
    """单镜失败（无人脸/OOM）保留原片段，不阻塞其余镜头。"""
    ws = _lipsync_workspace(tmp_path, monkeypatch)
    _fake_durations(monkeypatch, clip_seconds=2.0, audio_seconds=5.0)
    _use_fake_client(monkeypatch, fail_indices={0})  # 镜 1 失败

    result = stages.sync_lips()

    assert result["succeeded"] == 1
    assert result["failed"] == 1
    assert result["failed_shots"][0]["id"] == 1
    assert result["failed_shots"][0]["fallback"] == "clips/shot_1.webm"
    # 阶段不因单镜失败而报错（渲染会回退原片段）
    assert "error" not in result
    assert not (ws / "lipsync" / "shot_1.mp4").exists()
    assert (ws / "lipsync" / "shot_2.mp4").is_file()


def test_sync_lips_missing_nodes_auto_skips(tmp_path, monkeypatch):
    """auto 模式缺节点：告警跳过，清单记录原因，不提交任何工作流。"""
    _lipsync_workspace(tmp_path, monkeypatch)
    fake = _use_fake_client(monkeypatch, missing_nodes={"LatentSyncNode"})

    result = stages.sync_lips()

    assert "error" not in result
    assert result["skipped"] == 2
    assert fake.submitted == []
    manifest = json.loads(
        (tmp_path / "lipsync" / "manifest.json").read_text(encoding="utf-8")
    )
    assert all(s["status"] == "skipped" for s in manifest["shots"])
    assert any("节点未安装" in s["reason"] for s in manifest["shots"])


def test_sync_lips_on_mode_missing_nodes_errors(tmp_path, monkeypatch):
    """on 模式缺节点是配置错误：阶段报错（预检也会提前拦截）。"""
    _lipsync_workspace(tmp_path, monkeypatch)
    monkeypatch.setenv("PROMPT2VID_LIPSYNC", "on")
    _use_fake_client(monkeypatch, missing_nodes={"LatentSyncNode", "VHS_LoadVideo"})

    result = stages.sync_lips()

    assert "error" in result
    assert "LatentSyncNode" in result["error"]


def test_sync_lips_off_mode_skips(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPT2VID_LIPSYNC", "off")
    result = stages.sync_lips()
    assert result.get("skipped") == "PROMPT2VID_LIPSYNC=off"


def test_sync_lips_without_audio_marks_skipped(tmp_path, monkeypatch):
    """没有配音的镜头无从驱动口型：标记 skipped，保留原片段。"""
    _lipsync_workspace(tmp_path, monkeypatch, with_audio=False)
    fake = _use_fake_client(monkeypatch)

    result = stages.sync_lips()

    assert result["skipped"] == 2
    assert result["succeeded"] == 0
    assert fake.submitted == []
    manifest = json.loads(
        (tmp_path / "lipsync" / "manifest.json").read_text(encoding="utf-8")
    )
    assert all(s["status"] == "skipped" for s in manifest["shots"])
    assert any("无配音" in s["reason"] for s in manifest["shots"])


def test_sync_lips_resume_skips_existing(tmp_path, monkeypatch):
    """断点续跑：已有产物的镜头不再提交，仍计入成功。"""
    ws = _lipsync_workspace(tmp_path, monkeypatch)
    _fake_durations(monkeypatch, clip_seconds=2.0, audio_seconds=5.0)
    first = _use_fake_client(monkeypatch)
    stages.sync_lips()
    assert len(first.submitted) == 2

    second = _use_fake_client(monkeypatch)
    result = stages.sync_lips()

    assert second.submitted == []
    assert result["succeeded"] == 2
    manifest = json.loads((ws / "lipsync" / "manifest.json").read_text(encoding="utf-8"))
    assert all(s["status"] == "ok" for s in manifest["shots"])


@pytest.mark.skipif(not HAS_FFMPEG, reason="本机未安装 ffmpeg/ffprobe")
def test_render_final_and_quality_checks(tmp_path, monkeypatch):
    """用 lavfi 合成假片段与假配音，验证确定性剪辑与质检全链路。"""
    ws = _setup_workspace(tmp_path, monkeypatch)

    # 假首帧 + 假片段（testsrc 1s 视频）+ 假配音（sine 1s 音频）
    for shot_id in (1, 2):
        frame = ws / "frames" / f"shot_{shot_id}.png"
        frame.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "color=c=blue:s=512x768:d=0.04",
             "-frames:v", "1", str(frame)],
            check=True,
        )
        clip = ws / "clips" / f"shot_{shot_id}.webm"
        clip.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc=size=512x768:rate=25",
             "-t", "1", "-c:v", "libvpx-vp9", str(clip)],
            check=True,
        )
        audio = ws / "audio" / f"shot_{shot_id}.mp3"
        audio.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "sine=frequency=440",
             "-t", "0.8", str(audio)],
            check=True,
        )

    for rel, key, ext in (("frames", "image", "png"), ("clips", "video", "webm"), ("audio", "audio", "mp3")):
        folder = ws / rel
        entries = [
            {
                "id": i,
                key: f"{rel}/shot_{i}.{ext}",
                "status": "ok",
                "path": str(folder / f"shot_{i}.{ext}"),
            }
            for i in (1, 2)
        ]
        (folder / "manifest.json").write_text(
            json.dumps({"shots": entries}, ensure_ascii=False), encoding="utf-8"
        )

    render = stages.render_final()
    assert "error" not in render, render
    final = Path(render["final_video"])
    assert final.is_file()
    # 两个 1s 片段 → 成片约 2s，带音轨，字幕已烧录
    assert 1.0 <= render["duration_seconds"] <= 3.0
    assert render["width"] == 512
    assert render["has_audio"] is True
    assert Path(render["srt"]).is_file()
    assert (ws / "final" / "render_report.json").is_file()

    # 质检：需求时长写 3s（±2s 容差内）
    (ws / "requirement.json").write_text(
        json.dumps({"duration_seconds": 3}, ensure_ascii=False), encoding="utf-8"
    )
    report = stages.run_quality_checks()
    assert report["passed"], report
    names = {c["name"] for c in report["checks"]}
    assert {"final_video_exists", "duration_match", "resolution_ok",
            "audio_track_present", "assets_complete"} <= names
    assert (ws / "report" / "quality_report.json").is_file()


@pytest.mark.skipif(not HAS_FFMPEG, reason="本机未安装 ffmpeg/ffprobe")
def test_render_final_upscales_to_target_width(tmp_path, monkeypatch):
    """PROMPT2VID_UPSCALE=720：512x768 源 → 720x1080 成片，报告记录放大。"""
    ws = _setup_workspace(tmp_path, monkeypatch)
    monkeypatch.setenv("PROMPT2VID_UPSCALE", "720")

    for shot_id in (1, 2):
        clip = ws / "clips" / f"shot_{shot_id}.webm"
        clip.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc=size=512x768:rate=25",
             "-t", "1", "-c:v", "libvpx-vp9", str(clip)],
            check=True,
        )
        audio = ws / "audio" / f"shot_{shot_id}.mp3"
        audio.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "sine=frequency=440", "-t", "0.8", str(audio)],
            check=True,
        )
    entries = [
        {"id": i, "video": f"clips/shot_{i}.webm", "status": "ok",
         "path": str(ws / "clips" / f"shot_{i}.webm")}
        for i in (1, 2)
    ]
    (ws / "clips" / "manifest.json").write_text(
        json.dumps({"shots": entries}, ensure_ascii=False), encoding="utf-8",
    )
    (ws / "audio" / "manifest.json").write_text(
        json.dumps({"shots": [
            {"id": i, "audio": f"audio/shot_{i}.mp3", "status": "ok",
             "path": str(ws / "audio" / f"shot_{i}.mp3")} for i in (1, 2)
        ]}, ensure_ascii=False), encoding="utf-8",
    )

    render = stages.render_final()
    assert "error" not in render, render
    # 512x768 → 720 宽（高按宽高比 → 1080，偶数）
    assert render["width"] == 720
    assert render["height"] == 1080
    assert render["upscale"] == {"from": [512, 768], "to": [720, 1080]}


@pytest.mark.skipif(not HAS_FFMPEG, reason="本机未安装 ffmpeg/ffprobe")
def test_render_final_reports_missing_clips(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPT2VID_WORKSPACE", str(tmp_path))
    result = stages.render_final()
    assert "error" in result


@pytest.mark.skipif(not HAS_FFMPEG, reason="本机未安装 ffmpeg/ffprobe")
def test_quality_checks_flag_silent_voiceover(tmp_path, monkeypatch):
    """全部旁白无声（TTS 被拦截的静音占位）→ voiceover_audible 不通过。"""
    ws = _setup_workspace(tmp_path, monkeypatch)
    final = ws / "final" / "final.mp4"
    final.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=size=512x768:rate=25", "-t", "2",
         "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "2",
         "-c:v", "libx264", "-c:a", "aac", "-shortest", str(final)],
        check=True,
    )
    audio = ws / "audio" / "shot_1.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "1",
         "-c:a", "libmp3lame", str(audio)],
        check=True,
    )
    (ws / "audio" / "manifest.json").write_text(
        json.dumps({"shots": [
            {"id": 1, "audio": "audio/shot_1.mp3", "status": "ok",
             "path": str(audio)}
        ]}, ensure_ascii=False), encoding="utf-8",
    )

    report = stages.run_quality_checks()
    check = next(c for c in report["checks"] if c["name"] == "voiceover_audible")
    assert not check["passed"]
    assert "TTS" in check["detail"]
    assert not report["passed"]
