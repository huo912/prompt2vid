"""分步执行测试：注册表、单步分发、状态落盘、失败即停、强制重跑。

crew 用 fake 替身（kickoff 写下该步骤产物），媒体阶段用 fake 函数，
与 test_pipeline.py 同一风格，不依赖 LLM/ComfyUI/ffmpeg。
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import pytest

from prompt2vid import pipeline, steps


class FakeCrew:
    """单任务 crew 替身：kickoff 写下该步骤的产物文件。"""

    write_artifact: bool = True

    def __init__(self, task_names: list[str] | None = None):
        self.task_names = task_names or []
        self.kickoff_inputs: dict | None = None

    def kickoff(self, inputs: dict) -> str:
        self.kickoff_inputs = inputs
        task = self.task_names[0]
        artifacts = {
            "analyze_requirement": "requirement.json",
            "write_script": "script.md",
            "plan_storyboard": "storyboard.json",
        }
        if task in artifacts and self.write_artifact:
            (Path(inputs["workspace"]) / artifacts[task]).write_text(
                "{}", encoding="utf-8"
            )
        return f"{task} done"


@pytest.fixture()
def fake_crews(monkeypatch):
    crews: list[FakeCrew] = []

    def fake_build_crew(task_names=None):
        crew = FakeCrew(task_names)
        crews.append(crew)
        return crew

    monkeypatch.setattr("prompt2vid.crew.build_crew", fake_build_crew)
    return crews


@pytest.fixture()
def fake_stages(monkeypatch):
    calls: list[str] = []

    def make_fake(stage_name: str):
        def fake_stage(workspace=None):
            calls.append(stage_name)
            ws = Path(workspace or "output/run_default")
            stage_dir = ws / steps.STAGE_OUTPUT_DIRS[stage_name]
            stage_dir.mkdir(parents=True, exist_ok=True)
            (stage_dir / "manifest.json").write_text("{}", encoding="utf-8")
            return {"stage": stage_name, "succeeded": 1}

        return fake_stage

    for name in pipeline._STAGE_FUNCS:
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


# ------------------------------------------------------------ 注册表

def test_registry_matches_task_order():
    from prompt2vid.crew import TASK_ORDER

    # crew 任务全部在注册表里且相对顺序一致；sync_lips 是媒体增强步骤
    # （无 crew 任务，直跑 stages），插在配音与剪辑之间
    assert [i for i in steps.STEP_IDS if i in TASK_ORDER] == TASK_ORDER
    assert steps.STEP_IDS == [
        "analyze_requirement", "write_script", "plan_storyboard",
        "generate_images", "generate_videos", "generate_voiceover",
        "sync_lips", "edit_video", "inspect_quality", "publish_douyin",
    ]
    lipsync = steps.STEP_BY_ID["sync_lips"]
    assert lipsync["kind"] == "media"
    assert lipsync["stage"] == "lipsync"
    assert steps.STAGE_OUTPUT_DIRS["lipsync"] == "lipsync"


def test_catalog_has_labels():
    catalog = steps.catalog()
    assert catalog[0]["id"] == "analyze_requirement"
    assert catalog[0]["label"] == "需求分析"
    assert catalog[0]["kind_label"] == "创意"
    assert catalog[-1]["id"] == "publish_douyin"
    assert catalog[-1]["kind_label"] == "发布"


# ------------------------------------------------------------ 单步执行

def test_run_step_media(tmp_path, fake_stages):
    result = steps.run_step(tmp_path, "generate_images")

    assert result["status"] == "done"
    assert fake_stages == ["frames"]
    # 阶段产物落盘（steps.json 的状态记录由 run_steps 统一维护）
    assert (tmp_path / "frames" / "manifest.json").is_file()


def test_run_step_creative(tmp_path, fake_crews):
    result = steps.run_step(
        tmp_path, "analyze_requirement", prompt="做一个秋天的视频"
    )

    assert result["status"] == "done"
    assert len(fake_crews) == 1
    assert fake_crews[0].task_names == ["analyze_requirement"]
    assert fake_crews[0].kickoff_inputs["user_prompt"] == "做一个秋天的视频"
    assert (tmp_path / "requirement.json").is_file()


def test_run_step_creative_missing_artifact(tmp_path, monkeypatch):
    """LLM 没有写出期望产物时判为失败，并给出原因。"""

    class LazyCrew(FakeCrew):
        write_artifact = False

    monkeypatch.setattr("prompt2vid.crew.build_crew", lambda task_names=None: LazyCrew(task_names))

    result = steps.run_step(tmp_path, "write_script")
    assert result["status"] == "failed"
    assert "未生成期望产物 script.md" in result["error"]


def test_run_step_salvages_storyboard_from_reply(tmp_path, monkeypatch):
    """LLM 只在回答里贴出 JSON 时，从回答中抢救 storyboard.json。"""
    shot = {
        "id": 1,
        "narration": "秋天",
        "visual": "金色落叶",
        "image_prompt": "autumn leaves",
        "duration_seconds": 4,
    }

    class ChattyCrew(FakeCrew):
        write_artifact = False

        def kickoff(self, inputs: dict) -> str:
            self.kickoff_inputs = inputs
            return "分镜表如下：\n```json\n" + json.dumps({"shots": [shot]}) + "\n```"

    monkeypatch.setattr("prompt2vid.crew.build_crew", lambda task_names=None: ChattyCrew(task_names))

    result = steps.run_step(tmp_path, "plan_storyboard")
    assert result["status"] == "done"
    assert "抢救" in result["summary"]
    saved = json.loads((tmp_path / "storyboard.json").read_text(encoding="utf-8"))
    assert saved["shots"][0]["image_prompt"] == "autumn leaves"
    # 抢救回来的产物必须能被下游解析器正常加载
    from prompt2vid.storyboard import load_storyboard

    assert [s.id for s in load_storyboard(tmp_path / "storyboard.json")] == [1]


def test_run_step_salvage_rejects_unusable_reply(tmp_path, monkeypatch):
    """回答里没有可信内容时仍判失败，不写入半成品文件。"""

    class VagueCrew(FakeCrew):
        write_artifact = False

        def kickoff(self, inputs: dict) -> str:
            return "抱歉，我无法完成这个任务。"

    monkeypatch.setattr("prompt2vid.crew.build_crew", lambda task_names=None: VagueCrew(task_names))

    result = steps.run_step(tmp_path, "plan_storyboard")
    assert result["status"] == "failed"
    assert "未找到可信内容" in result["error"]
    assert not (tmp_path / "storyboard.json").exists()


def test_run_step_does_not_overwrite_existing_artifact(tmp_path, monkeypatch):
    """工具已写盘时不再走抢救路径（不覆盖工具产物）。"""
    (tmp_path / "storyboard.json").write_text('{"shots": [{"id": 9}]}', encoding="utf-8")

    class ToolCrew(FakeCrew):
        def kickoff(self, inputs: dict) -> str:
            return "```json\n{\"shots\": [{\"id\": 1}]}\n```"

    monkeypatch.setattr("prompt2vid.crew.build_crew", lambda task_names=None: ToolCrew(task_names))

    result = steps.run_step(tmp_path, "plan_storyboard")
    assert result["status"] == "done"
    saved = json.loads((tmp_path / "storyboard.json").read_text(encoding="utf-8"))
    assert saved["shots"][0]["id"] == 9


def test_run_step_force_deletes_stage_dir(tmp_path, monkeypatch):
    """强制重跑媒体步骤：产物目录在阶段函数执行前被删除。"""
    stage_dir = tmp_path / "frames"
    stage_dir.mkdir()
    (stage_dir / "shot_1.png").write_text("old", encoding="utf-8")
    seen: dict = {}

    def fake_frames(workspace=None):
        seen["frames_dir_exists_at_call"] = (Path(workspace) / "frames").is_dir()
        return {"stage": "frames", "succeeded": 1}

    monkeypatch.setitem(pipeline._STAGE_FUNCS, "frames", fake_frames)

    result = steps.run_step(tmp_path, "generate_images", force=True)
    assert result["status"] == "done"
    assert seen["frames_dir_exists_at_call"] is False


def test_run_step_unknown_raises(tmp_path):
    with pytest.raises(ValueError):
        steps.run_step(tmp_path, "no_such_step")


# ------------------------------------------------------------ 多步执行

def test_run_steps_stops_on_failure(tmp_path, fake_crews, fake_stages, monkeypatch):
    def failing_clips(workspace=None):
        fake_stages.append("clips")
        return {"stage": "clips", "error": "no frames"}

    monkeypatch.setitem(pipeline._STAGE_FUNCS, "clips", failing_clips)

    results = steps.run_steps(tmp_path, prompt="p")

    # 创意三步 + 生图成功，clips 失败后停止，后续步骤未执行
    assert [r["status"] for r in results] == ["done"] * 4 + ["failed"]
    assert fake_stages == ["frames", "clips"]
    states = json.loads((tmp_path / "steps.json").read_text(encoding="utf-8"))
    assert states["generate_videos"]["status"] == "failed"
    assert "generate_voiceover" not in states  # 未执行的步骤没有记录


def test_run_steps_order_follows_pipeline(tmp_path, fake_stages):
    # 乱序传入仍按流水线顺序执行（阶段名：clips → voiceovers → render）
    results = steps.run_steps(
        tmp_path, ["edit_video", "generate_voiceover", "generate_videos"], prompt="p"
    )
    assert fake_stages == ["clips", "voiceovers", "render"]
    assert all(r["status"] == "done" for r in results)


def test_run_steps_cancel_between_steps(tmp_path, fake_stages):
    cancel = threading.Event()
    cancel.set()  # 立即取消：一步都不执行
    results = steps.run_steps(
        tmp_path, ["generate_images", "generate_videos"], prompt="p",
        cancel_event=cancel,
    )
    assert results == []
    assert fake_stages == []


def test_run_steps_reads_prompt_from_meta(tmp_path, fake_crews):
    steps.write_meta(tmp_path, "meta 里的一句话")
    steps.run_steps(tmp_path, ["analyze_requirement"])
    assert fake_crews[0].kickoff_inputs["user_prompt"] == "meta 里的一句话"


def test_run_steps_unknown_step(tmp_path):
    with pytest.raises(ValueError):
        steps.run_steps(tmp_path, ["bad_step"])


# ------------------------------------------------------------ 状态推断

def test_step_statuses_artifact_fallback(tmp_path):
    # 没有任何记录与产物 → 全部无状态
    statuses = steps.step_statuses(tmp_path)
    assert statuses["analyze_requirement"].get("status") is None

    # 产物存在 → done（CLI 全流程跑出的历史 run）
    (tmp_path / "requirement.json").write_text("{}", encoding="utf-8")
    statuses = steps.step_statuses(tmp_path)
    assert statuses["analyze_requirement"]["status"] == "done"


def test_step_statuses_stale_running_marked_interrupted(tmp_path):
    # steps.json 残留 running（服务重启）→ interrupted
    (tmp_path / steps.STEPS_FILE).write_text(
        json.dumps({"generate_images": {"status": "running"}}), encoding="utf-8"
    )
    statuses = steps.step_statuses(tmp_path, active=False)
    assert statuses["generate_images"]["status"] == "interrupted"
    # 该 run 正在执行时保持 running
    statuses = steps.step_statuses(tmp_path, active=True)
    assert statuses["generate_images"]["status"] == "running"


def test_write_and_read_meta(tmp_path):
    steps.write_meta(tmp_path, "第一句话")
    steps.write_meta(tmp_path, "第二句话")
    meta = steps.read_meta(tmp_path)
    assert meta["prompt"] == "第二句话"
    assert meta["created_at"]  # 首次写入时间不被覆盖
    assert steps.read_prompt(tmp_path) == "第二句话"
    assert steps.read_meta(tmp_path / "not_exist") == {}
