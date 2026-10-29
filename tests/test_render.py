"""SRT 生成与渲染编排测试（不依赖真实 ffmpeg）。

build_srt 是纯函数直接验证；render_final 的编排逻辑
（清单读取、时间轴累计、配音超长自动延长、字幕对齐、报告写出）
通过 mock ffmpeg_ops 的媒体原语验证。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prompt2vid import ffmpeg_ops, stages


def test_build_srt_format(tmp_path: Path):
    dest = ffmpeg_ops.build_srt(
        [(0.0, 2.5, "秋天的公园"), (2.5, 5.0, "风吹落叶")],
        tmp_path / "final.srt",
    )
    content = dest.read_text(encoding="utf-8")
    assert "1\n00:00:00,000 --> 00:00:02,500\n秋天的公园" in content
    assert "2\n00:00:02,500 --> 00:00:05,000\n风吹落叶" in content
    # 空文本的镜不生成字幕块
    dest2 = ffmpeg_ops.build_srt([(0.0, 1.0, ""), (1.0, 2.0, "x")], tmp_path / "a.srt")
    blocks = [b for b in dest2.read_text(encoding="utf-8").split("\n\n") if b.strip()]
    assert len(blocks) == 1


def test_srt_timestamp():
    assert ffmpeg_ops._srt_timestamp(0) == "00:00:00,000"
    assert ffmpeg_ops._srt_timestamp(1.5) == "00:00:01,500"
    assert ffmpeg_ops._srt_timestamp(3661.25) == "01:01:01,250"


@pytest.fixture()
def render_workspace(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PROMPT2VID_WORKSPACE", str(tmp_path))
    storyboard = {
        "shots": [
            {"id": 1, "narration": "第一镜", "image_prompt": "a", "motion_prompt": "m",
             "duration_seconds": 2},
            {"id": 2, "narration": "第二镜", "image_prompt": "b", "motion_prompt": "m",
             "duration_seconds": 3},
        ]
    }
    (tmp_path / "storyboard.json").write_text(
        json.dumps(storyboard, ensure_ascii=False), encoding="utf-8"
    )
    # clips 清单（两个 ok 片段）
    for shot_id in (1, 2):
        clip = tmp_path / "clips" / f"shot_{shot_id}.webm"
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(b"fake")
    (tmp_path / "clips" / "manifest.json").write_text(
        json.dumps({"shots": [
            {"id": 1, "video": "clips/shot_1.webm", "status": "ok",
             "path": str(tmp_path / "clips" / "shot_1.webm")},
            {"id": 2, "video": "clips/shot_2.webm", "status": "ok",
             "path": str(tmp_path / "clips" / "shot_2.webm")},
        ]}, ensure_ascii=False), encoding="utf-8",
    )
    # audio 清单（只有镜 1，镜 2 缺失 → 静音填充）
    audio = tmp_path / "audio" / "shot_1.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"fake")
    (tmp_path / "audio" / "manifest.json").write_text(
        json.dumps({"shots": [
            {"id": 1, "audio": "audio/shot_1.mp3", "status": "ok", "path": str(audio)},
        ]}, ensure_ascii=False), encoding="utf-8",
    )
    return tmp_path


def _install_ffmpeg_mocks(monkeypatch, clip_durations: dict[int, float],
                          audio_durations: dict[int, float] | None = None):
    """安装 ffmpeg_ops mock；返回事件列表。"""
    audio_durations = audio_durations or {}
    # 仓库 .env 可能被其它测试经 load_env 写进 os.environ；过渡默认开启
    # （0.5s）会走 merge_with_transitions（未 mock），mock 测试固定关闭
    for name in ("PROMPT2VID_UPSCALE", "PROMPT2VID_SMOOTH_FPS",
                 "PROMPT2VID_TRANSITION_TYPE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PROMPT2VID_TRANSITION", "0")
    events: list[tuple] = []

    def fake_probe(media):
        path = Path(str(media))
        name = path.name
        if name.startswith(("final", "merged")):
            return {"path": str(path), "duration_seconds": 5.0, "width": 512,
                    "height": 768, "fps": 25.0, "has_audio": True}
        shot_id = int(name.split("_")[1].split(".")[0])
        if path.suffix == ".mp3":
            return {"path": str(path), "duration_seconds": audio_durations.get(shot_id, 0.0),
                    "width": 0, "height": 0, "fps": 0.0, "has_audio": True}
        return {"path": str(path), "duration_seconds": clip_durations[shot_id],
                "width": 512, "height": 768, "fps": 25.0, "has_audio": False}

    def touch(path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return path

    def fake_fit_audio(audio, duration, dest):
        events.append(("fit_audio", Path(audio).name if audio else None, round(duration, 1)))
        return touch(dest)

    def fake_normalize(clip, dest, extend_seconds=0.0):
        events.append(("normalize", Path(clip).name, round(extend_seconds, 1)))
        return touch(dest)

    def fake_concat(items, dest, **kwargs):
        return touch(dest)

    def fake_mux(video, audio, srt, dest, bgm=None, **kwargs):
        events.append(("mux", Path(video).name, Path(audio).name,
                       Path(srt).name if srt else None, bgm))
        return touch(dest)

    monkeypatch.setattr(stages.ffmpeg_ops, "probe_file", fake_probe)
    monkeypatch.setattr(stages.ffmpeg_ops, "fit_audio", fake_fit_audio)
    monkeypatch.setattr(stages.ffmpeg_ops, "normalize_clip", fake_normalize)
    monkeypatch.setattr(stages.ffmpeg_ops, "concat_media", fake_concat)
    monkeypatch.setattr(stages.ffmpeg_ops, "mux_and_burn", fake_mux)
    return events


def test_render_final_orchestration(render_workspace, monkeypatch):
    ws = render_workspace
    events = _install_ffmpeg_mocks(
        monkeypatch, clip_durations={1: 2.0, 2: 3.0}, audio_durations={1: 2.0}
    )

    report = stages.render_final()

    assert "error" not in report, report
    # 每镜音频对齐到真实片段时长；镜 2 缺音频 → None（静音）
    assert ("fit_audio", "shot_1.mp3", 2.0) in events
    assert ("fit_audio", None, 3.0) in events
    # 两个片段都被规范化（无延长）
    assert ("normalize", "shot_1.webm", 0.0) in events
    assert ("normalize", "shot_2.webm", 0.0) in events
    # 混流时字幕传入、无 BGM
    mux = [e for e in events if e[0] == "mux"][0]
    assert mux[3] == "final.srt"
    assert mux[4] is None
    # 成片与报告
    final = ws / "final" / "final.mp4"
    assert final.is_file()
    assert (ws / "final" / "render_report.json").is_file()
    assert report["final_video"] == str(final.resolve())
    assert report["audio_overruns"] == []
    # SRT 时间轴基于真实时长累计（0-2s、2-5s）
    srt = (ws / "final" / "final.srt").read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:02,000" in srt
    assert "00:00:02,000 --> 00:00:05,000" in srt
    assert "第一镜" in srt and "第二镜" in srt


def test_render_final_extends_video_when_audio_longer(render_workspace, monkeypatch):
    """配音比片段长 → 定格延长视频而不是截断配音。"""
    ws = render_workspace
    # 镜 1 配音 5s > 片段 2s → 延长 3s；镜 2 无配音
    events = _install_ffmpeg_mocks(
        monkeypatch, clip_durations={1: 2.0, 2: 3.0}, audio_durations={1: 5.0}
    )

    report = stages.render_final()

    assert "error" not in report
    # 片段 1 定格延长 3s，片段 2 不延长
    assert ("normalize", "shot_1.webm", 3.0) in events
    assert ("normalize", "shot_2.webm", 0.0) in events
    # 音频对齐到 5s（配音不被截断）
    assert ("fit_audio", "shot_1.mp3", 5.0) in events
    # 报告记录超长信息
    overruns = report["audio_overruns"]
    assert len(overruns) == 1
    assert overruns[0]["id"] == 1
    assert overruns[0]["extended_seconds"] == 3.0
    # SRT 时间轴按延长后的时长（镜 1: 0-5s，镜 2: 5-8s）
    srt = (ws / "final" / "final.srt").read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:05,000" in srt
    assert "00:00:05,000 --> 00:00:08,000" in srt


def test_render_final_with_bgm(render_workspace, monkeypatch):
    bgm_file = render_workspace / "bgm.mp3"
    bgm_file.write_bytes(b"fake")
    monkeypatch.setenv("PROMPT2VID_BGM", str(bgm_file))
    events = _install_ffmpeg_mocks(
        monkeypatch, clip_durations={1: 2.0, 2: 3.0}, audio_durations={1: 2.0}
    )

    report = stages.render_final()

    mux = [e for e in events if e[0] == "mux"][0]
    assert Path(mux[4]) == bgm_file
    assert report["bgm"] == str(bgm_file)


def test_render_final_skips_failed_clips(render_workspace, monkeypatch):
    """失败分镜应被跳过而不是让整次渲染失败。"""
    ws = render_workspace
    _install_ffmpeg_mocks(monkeypatch, clip_durations={1: 2.0})

    # 把镜 2 标记为 failed
    manifest = json.loads((ws / "clips" / "manifest.json").read_text(encoding="utf-8"))
    manifest["shots"][1]["status"] = "failed"
    (ws / "clips" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )

    report = stages.render_final()
    assert "error" not in report
    assert [s["id"] for s in report["shots"]] == [1]


def test_render_final_prefers_lipsync_outputs(render_workspace, monkeypatch):
    """lipsync/manifest.json 里 ok 的镜头用口型同步片段，其余回退原片段。"""
    ws = render_workspace
    lipsync_clip = ws / "lipsync" / "shot_1.mp4"
    lipsync_clip.parent.mkdir(parents=True, exist_ok=True)
    lipsync_clip.write_bytes(b"lipsynced")
    (ws / "lipsync" / "manifest.json").write_text(
        json.dumps({"shots": [
            {"id": 1, "video": "lipsync/shot_1.mp4", "status": "ok",
             "path": str(lipsync_clip)},
        ]}, ensure_ascii=False), encoding="utf-8",
    )

    events = _install_ffmpeg_mocks(
        monkeypatch, clip_durations={1: 2.0, 2: 3.0}, audio_durations={1: 2.0}
    )

    report = stages.render_final()

    assert "error" not in report
    # 镜 1 用口型同步片段（mp4），镜 2 无 lipsync 条目 → 原片段（webm）
    normalize_names = [e[1] for e in events if e[0] == "normalize"]
    assert "shot_1.mp4" in normalize_names
    assert "shot_2.webm" in normalize_names
    assert report["lipsync"] == {"shots": 1, "total": 2}


def test_render_final_lipsync_entry_missing_file_falls_back(
    render_workspace, monkeypatch
):
    """lipsync 清单 ok 但文件被删：回退原片段，渲染不失败。"""
    ws = render_workspace
    (ws / "lipsync").mkdir(exist_ok=True)
    (ws / "lipsync" / "manifest.json").write_text(
        json.dumps({"shots": [
            {"id": 1, "video": "lipsync/shot_1.mp4", "status": "ok",
             "path": str(ws / "lipsync" / "shot_1.mp4")},  # 文件不存在
        ]}, ensure_ascii=False), encoding="utf-8",
    )
    events = _install_ffmpeg_mocks(
        monkeypatch, clip_durations={1: 2.0, 2: 3.0}, audio_durations={1: 2.0}
    )

    report = stages.render_final()

    assert "error" not in report
    normalize_names = [e[1] for e in events if e[0] == "normalize"]
    assert "shot_1.webm" in normalize_names
    assert report["lipsync"] == {"shots": 0, "total": 2}


def test_render_final_transitions_between_shots(render_workspace, monkeypatch):
    """PROMPT2VID_TRANSITION：非末段镜头垫过渡秒、视频走 xfade 链合并。

    时长账本：offset[j] = 前片段实际时长之和 - (j+1)*transition。
    """
    events = _install_ffmpeg_mocks(
        monkeypatch, clip_durations={1: 2.0, 2: 3.0}, audio_durations={1: 2.0}
    )
    # _install_ffmpeg_mocks 默认把过渡关掉（0），这里再打开
    monkeypatch.setenv("PROMPT2VID_TRANSITION", "0.5")
    # 补 mock：xfade 合并（记录参数）
    merges: list[dict] = []

    def fake_merge(clips, offsets, duration, transition, dest):
        merges.append({
            "clips": [Path(c).name for c in clips],
            "offsets": offsets, "duration": duration,
            "transition": transition, "dest": Path(dest).name,
        })
        return Path(dest)

    monkeypatch.setattr(stages.ffmpeg_ops, "merge_with_transitions", fake_merge)

    report = stages.render_final()

    # 非末段镜头（镜 1）的规范化带 0.5s 垫片；末段（镜 2）不带
    normalizes = [(e[1], e[2]) for e in events if e[0] == "normalize"]
    assert ("shot_1.webm", 0.5) in normalizes
    assert ("shot_2.webm", 0.0) in normalizes
    # xfade 链：offset = 2.0 - 1*0.5 = 1.5（音轨时间轴的衔接点）
    assert len(merges) == 1
    assert merges[0]["clips"] == ["clip_1.mp4", "clip_2.mp4"]
    assert merges[0]["offsets"] == [1.5]
    assert merges[0]["duration"] == 0.5
    assert merges[0]["transition"] == "fade"
    assert merges[0]["dest"] == "merged.mp4"
    # 报告记录过渡信息
    assert report["transition"] == {"type": "fade", "seconds": 0.5}
    # SRT 时间轴不受过渡影响（镜 1: 0-2s，镜 2: 2-5s）
    srt = (render_workspace / "final" / "final.srt").read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:02,000" in srt
    assert "00:00:02,000 --> 00:00:05,000" in srt
