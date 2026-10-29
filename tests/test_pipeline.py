"""流水线编排测试：直接媒体模式、断点续跑、阶段停止、清理。

crew 用 fake 替身（kickoff 写 storyboard.json），媒体阶段用 fake 函数，
验证编排逻辑本身，不依赖 LLM/ComfyUI/ffmpeg。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from prompt2vid import pipeline


class FakeCrew:
    """kickoff 时写 storyboard.json（创意段）或返回发布结果（发布段）。"""

    def __init__(self, run_dir: Path, task_names: list[str]):
        self.run_dir = run_dir
        self.task_names = task_names
        self.kickoff_inputs: dict | None = None

    def kickoff(self, inputs: dict) -> str:
        self.kickoff_inputs = inputs
        if "analyze_requirement" in self.task_names:
            storyboard = {
                "shots": [
                    {"id": 1, "narration": "旁白", "image_prompt": "a",
                     "motion_prompt": "m", "duration_seconds": 4},
                ]
            }
            (self.run_dir / "storyboard.json").write_text(
                json.dumps(storyboard, ensure_ascii=False), encoding="utf-8"
            )
            return "creative done"
        return "published: dry_run"


@pytest.fixture()
def fake_crews(monkeypatch, tmp_path):
    crews: list[FakeCrew] = []

    def fake_build_crew(task_names: list[str] | None = None) -> FakeCrew:
        crew = FakeCrew(tmp_path, task_names or pipeline.CREATIVE_TASKS)
        crews.append(crew)
        return crew

    monkeypatch.setattr("prompt2vid.crew.build_crew", fake_build_crew)
    return crews


@pytest.fixture()
def fake_stages(monkeypatch):
    calls: list[str] = []
    for name in pipeline._STAGE_FUNCS:
        def make_fake(stage_name):
            def fake_stage(workspace=None):
                calls.append(stage_name)
                return {"stage": stage_name, "succeeded": 1}
            return fake_stage

        monkeypatch.setitem(pipeline._STAGE_FUNCS, name, make_fake(name))
    return calls


@pytest.fixture(autouse=True)
def _cleanup_logging():
    yield
    root = logging.getLogger()
    for handler in list(pipeline._FILE_HANDLERS):
        root.removeHandler(handler)
        handler.close()
    pipeline._FILE_HANDLERS.clear()


def test_direct_media_enabled(monkeypatch):
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("PROMPT2VID_DIRECT_MEDIA", value)
        assert pipeline.direct_media_enabled() is True
    for value in ("", "0", "false", "off"):
        monkeypatch.setenv("PROMPT2VID_DIRECT_MEDIA", value)
        assert pipeline.direct_media_enabled() is False


def test_run_pipeline_direct_media(tmp_path, fake_crews, fake_stages):
    result = pipeline.run_pipeline("做一个视频", tmp_path, direct_media=True)

    # 两段 crew：创意三任务 + 发布单任务
    assert len(fake_crews) == 2
    assert fake_crews[0].task_names == pipeline.CREATIVE_TASKS
    assert fake_crews[1].task_names == [pipeline.PUBLISH_TASK]
    assert "user_prompt" in fake_crews[0].kickoff_inputs
    # 媒体阶段全部执行
    assert fake_stages == pipeline.MEDIA_STAGES
    assert result == "published: dry_run"
    # 运行日志落盘
    assert (tmp_path / "run.log").is_file()
    assert (tmp_path / "storyboard.json").is_file()


def test_run_pipeline_direct_media_salvages_storyboard(tmp_path, fake_stages, monkeypatch):
    """创意 crew 没写文件、只在回答里贴 JSON 时，分镜表仍能被抢救落盘。"""
    shot = {"id": 1, "narration": "旁白", "image_prompt": "a", "duration_seconds": 4}

    class ChattyCreative:
        task_names = pipeline.CREATIVE_TASKS

        def __init__(self):
            self.kickoff_inputs = None

        def kickoff(self, inputs: dict) -> str:
            self.kickoff_inputs = inputs
            return "分镜表：\n```json\n" + json.dumps({"shots": [shot]}) + "\n```"

    class PublishCrew:
        def kickoff(self, inputs: dict) -> str:
            return "published: dry_run"

    def fake_build_crew(task_names=None):
        return PublishCrew() if task_names == [pipeline.PUBLISH_TASK] else ChattyCreative()

    monkeypatch.setattr("prompt2vid.crew.build_crew", fake_build_crew)

    result = pipeline.run_pipeline("做一个视频", tmp_path, direct_media=True)

    assert result == "published: dry_run"
    saved = json.loads((tmp_path / "storyboard.json").read_text(encoding="utf-8"))
    assert saved["shots"][0]["image_prompt"] == "a"


def test_run_pipeline_crew_mode(tmp_path, fake_crews, fake_stages, monkeypatch):
    monkeypatch.setenv("PROMPT2VID_DIRECT_MEDIA", "0")
    pipeline.run_pipeline("做一个视频", tmp_path)
    # crew 模式：一次完整 crew，媒体阶段不由 pipeline 直跑
    assert len(fake_crews) == 1
    assert fake_stages == []


def test_run_media_stages_stops_on_error(fake_stages, monkeypatch):
    def failing_clips(workspace=None):
        fake_stages.append("clips")
        return {"stage": "clips", "error": "boom"}

    monkeypatch.setitem(pipeline._STAGE_FUNCS, "clips", failing_clips)
    last = pipeline.run_media_stages("frames")
    # frames 执行、clips 报错后停止，voiceovers/render/quality 不执行
    assert fake_stages == ["frames", "clips"]
    assert last["error"] == "boom"


def test_run_media_stages_render_error_still_quality(fake_stages, monkeypatch):
    def failing_render(workspace=None):
        fake_stages.append("render")
        return {"stage": "render", "error": "no clips"}

    monkeypatch.setitem(pipeline._STAGE_FUNCS, "render", failing_render)
    last = pipeline.run_media_stages("voiceovers")
    # 渲染失败后质检仍然执行（如实记录失败原因）；
    # 从 voiceovers 起跑会经过 lipsync（配音之后、渲染之前）
    assert fake_stages == ["voiceovers", "lipsync", "render", "quality"]
    assert last["stage"] == "quality"


def test_run_media_stages_invalid_start():
    with pytest.raises(ValueError):
        pipeline.run_media_stages("publish")


def test_run_resume_from_render(tmp_path, fake_crews, fake_stages, monkeypatch):
    result = pipeline.run_resume(tmp_path, from_stage="render", publish=False)
    assert fake_stages == ["render", "quality"]
    assert fake_crews == []  # 未要求发布，不建 crew
    assert result["stage"] == "quality"


def test_run_resume_publish(tmp_path, fake_crews, fake_stages):
    result = pipeline.run_resume(tmp_path, from_stage="quality", publish=True)
    assert fake_stages == ["quality"]
    assert len(fake_crews) == 1
    assert fake_crews[0].task_names == [pipeline.PUBLISH_TASK]
    assert result == "published: dry_run"


def test_run_resume_skips_publish_on_error(tmp_path, fake_crews, monkeypatch, fake_stages):
    def failing_quality(workspace=None):
        fake_stages.append("quality")
        return {"stage": "quality", "error": "no final"}

    monkeypatch.setitem(pipeline._STAGE_FUNCS, "quality", failing_quality)
    result = pipeline.run_resume(tmp_path, from_stage="quality", publish=True)
    assert fake_crews == []  # 失败时不再调用发布 crew
    assert "error" in result


def test_clean_outputs(tmp_path, monkeypatch):
    base = tmp_path / "output"
    base.mkdir()
    import time as time_mod

    for i in range(4):
        run = base / f"run_{i}"
        run.mkdir()
        (run / "storyboard.json").write_text("{}", encoding="utf-8")
        time_mod.sleep(0.01)

    monkeypatch.chdir(tmp_path)
    removed = pipeline.clean_outputs(keep=2)
    assert len(removed) == 2
    remaining = sorted(p.name for p in base.glob("run_*"))
    # 保留 mtime 最新的 2 个
    assert len(remaining) == 2
    assert all(not (base / name).exists() for name in
               [r.name for r in removed])


def test_trim_publish_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state_path = Path(".douyin") / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"last_publish": None,
                    "published": [{"i": i} for i in range(30)]}),
        encoding="utf-8",
    )
    trimmed = pipeline.trim_publish_state(keep=20)
    assert trimmed == 10
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert len(data["published"]) == 20
    assert data["published"][-1] == {"i": 29}
    # 不足 keep 条时不改动
    assert pipeline.trim_publish_state(keep=20) == 0
