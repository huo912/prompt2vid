"""确定性阶段引擎测试。

generate_frames / generate_clips 用 mock 的 ComfyUIClient 验证批量提交
编排与清单；render / quality 依赖真实 ffmpeg（本机没有则跳过，
CI 或装有 ffmpeg 的环境会覆盖）。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from prompt2vid import stages
from prompt2vid.comfyui_client import ComfyUITaskError, parse_output_files

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

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
    """

    def __init__(
        self,
        fail_first_n: int = 0,
        fail_all: bool = False,
        fail_indices: set[int] | None = None,
    ):
        self.submitted: list[dict] = []
        self.uploads: list[Path] = []
        self._fail_first_n = fail_first_n
        self._fail_all = fail_all
        self._fail_indices = fail_indices or set()

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

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _setup_workspace(tmp_path: Path, monkeypatch, storyboard: dict | None = None) -> Path:
    monkeypatch.setenv("PROMPT2VID_WORKSPACE", str(tmp_path))
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
    # prompt = image_prompt + motion_prompt
    text = first_wf["6"]["inputs"]["text"]
    assert "autumn park, golden leaves" in text
    assert "slow zoom in" in text
    # 4s → 105 帧（25fps、8n+1 约束）
    assert first_wf["95"]["inputs"]["length"] == 105
    assert manifest["shots"][0]["length"] == 105


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


@pytest.mark.skipif(not HAS_FFMPEG, reason="本机未安装 ffmpeg/ffprobe")
def test_render_final_and_quality_checks(tmp_path, monkeypatch):
    """用 lavfi 合成假片段与假配音，验证确定性剪辑与质检全链路。"""
    import subprocess

    ws = _setup_workspace(tmp_path, monkeypatch)

    # 假首帧 + 假片段（testsrc 1s 视频）+ 假配音（sine 1s 音频）
    for shot_id in (1, 2):
        frame = ws / "frames" / f"shot_{shot_id}.png"
        frame.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "color=c=blue:s=512x768:d=0.04",
             "-frames:v", "1", str(frame)],
            check=True,
        )
        clip = ws / "clips" / f"shot_{shot_id}.webm"
        clip.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc=size=512x768:rate=25",
             "-t", "1", "-c:v", "libvpx-vp9", str(clip)],
            check=True,
        )
        audio = ws / "audio" / f"shot_{shot_id}.mp3"
        audio.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
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
def test_render_final_reports_missing_clips(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPT2VID_WORKSPACE", str(tmp_path))
    result = stages.render_final()
    assert "error" in result
