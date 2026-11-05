"""分镜表宽松解析测试：代码栅栏、字段修复、类型容错。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prompt2vid import storyboard


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_extract_json_strips_code_fence():
    text = "好的，分镜表如下：\n```json\n{\"shots\": []}\n```\n以上。"
    assert storyboard.extract_json(text) == {"shots": []}


def test_extract_json_from_surrounding_text():
    text = '前置说明 {"a": {"b": 1}} 后置说明'
    assert storyboard.extract_json(text) == {"a": {"b": 1}}


def test_load_storyboard_happy_path(tmp_path: Path):
    data = {
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
    path = _write(tmp_path / "storyboard.json", json.dumps(data, ensure_ascii=False))
    shots = storyboard.load_storyboard(path)
    assert [s.id for s in shots] == [1, 2]
    assert shots[0].image_prompt == "autumn park, golden leaves"
    assert shots[0].fixes == []


def test_load_storyboard_repairs_and_fills_defaults(tmp_path: Path, monkeypatch):
    # 其它测试可能经 load_env 把仓库 .env 写进 os.environ（8GB 档上限 121 帧），
    # 这里清掉保证断言的是默认档的内容上限
    monkeypatch.delenv("COMFYUI_LTX_MAX_FRAMES", raising=False)
    data = {
        "shots": [
            # duration 是字符串、缺 motion_prompt / image_prompt
            {"id": "3", "narration": "旁白", "visual": "画面", "duration_seconds": "12"},
            # 重复 id
            {"id": 3, "narration": "旁白2", "image_prompt": "x", "duration_seconds": 2},
        ]
    }
    path = _write(tmp_path / "storyboard.json", json.dumps(data, ensure_ascii=False))
    shots = storyboard.load_storyboard(path)
    assert len(shots) == 2
    assert shots[0].id == 3
    assert shots[0].duration_seconds == storyboard.max_shot_seconds()  # 上限 10s
    assert shots[0].motion_prompt == storyboard.DEFAULT_MOTION_PROMPT
    assert shots[0].image_prompt  # 回落到 visual
    assert shots[0].fixes  # 记录了修复项
    assert shots[1].id == 2  # 重复 id 重排


def test_load_storyboard_wrapped_in_markdown(tmp_path: Path):
    content = "```json\n" + json.dumps({"shots": [{"id": 1, "image_prompt": "a"}]}) + "\n```"
    path = _write(tmp_path / "storyboard.json", content)
    shots = storyboard.load_storyboard(path)
    assert len(shots) == 1


def test_extract_json_top_level_array_not_truncated():
    text = json.dumps([{"id": 1}, {"id": 2}])
    assert storyboard.extract_json_value(text) == [{"id": 1}, {"id": 2}]
    assert storyboard.extract_json(text) == {"shots": [{"id": 1}, {"id": 2}]}


def test_load_storyboard_top_level_array(tmp_path: Path):
    """LLM 直接把分镜写成裸数组时也要能解析（回归：曾只取到第一个分镜）。"""
    data = [
        {"id": 1, "narration": "a", "image_prompt": "shot one", "duration_seconds": 4},
        {"id": 2, "narration": "b", "image_prompt": "shot two", "duration_seconds": 3},
    ]
    path = _write(tmp_path / "storyboard.json", json.dumps(data, ensure_ascii=False))
    shots = storyboard.load_storyboard(path)
    assert [s.id for s in shots] == [1, 2]
    assert [s.image_prompt for s in shots] == ["shot one", "shot two"]


def test_load_storyboard_array_with_trailing_summary(tmp_path: Path):
    """数组末尾混入 {"total_duration_seconds": N} 时跳过该汇总项。"""
    data = [
        {"id": 1, "narration": "a", "image_prompt": "shot one", "duration_seconds": 4},
        {"total_duration_seconds": 4},
    ]
    path = _write(tmp_path / "storyboard.json", json.dumps(data, ensure_ascii=False))
    shots = storyboard.load_storyboard(path)
    assert [s.id for s in shots] == [1]


def test_load_storyboard_fenced_array_with_prose(tmp_path: Path):
    data = [{"id": 1, "narration": "a", "image_prompt": "shot one"}]
    content = "分镜表如下：\n```json\n" + json.dumps(data, ensure_ascii=False) + "\n```\n以上。"
    path = _write(tmp_path / "storyboard.json", content)
    shots = storyboard.load_storyboard(path)
    assert len(shots) == 1


def test_load_storyboard_errors(tmp_path: Path):
    # 不存在
    with pytest.raises(storyboard.StoryboardError):
        storyboard.load_storyboard(tmp_path / "missing.json")
    # 无 shots
    path = _write(tmp_path / "storyboard.json", "{}")
    with pytest.raises(storyboard.StoryboardError):
        storyboard.load_storyboard(path)
    # 空数组
    path = _write(tmp_path / "storyboard.json", "[]")
    with pytest.raises(storyboard.StoryboardError):
        storyboard.load_storyboard(path)
    # 非 JSON
    path = _write(tmp_path / "storyboard.json", "这不是 JSON")
    with pytest.raises(storyboard.StoryboardError):
        storyboard.load_storyboard(path)


def test_shot_longer_than_ltx_clip_limit_is_flagged(tmp_path: Path, monkeypatch):
    """台词超出 LTX 单镜上限时提前告警（否则成片只能靠降速/补帧硬撑）。"""
    monkeypatch.setenv("COMFYUI_LTX_MAX_FRAMES", "121")
    data = {
        "shots": [
            {"id": 1, "narration": "这是一段很长的旁白大概要念十秒钟左右才能念完",
             "image_prompt": "a", "duration_seconds": 4},
        ]
    }
    path = _write(tmp_path / "storyboard.json", json.dumps(data, ensure_ascii=False))
    shots = storyboard.load_storyboard(path)
    assert shots[0].duration_seconds > 4
    assert any("LTX 单镜上限" in fix for fix in shots[0].fixes)


def test_shot_over_speech_budget_is_flagged(tmp_path: Path):
    """台词超出时长对应的字数预算时，告警并给出预算字数。"""
    data = {
        "shots": [
            {"id": 1, "narration": "这句旁白明显超出四秒钟能念完的字数",
             "image_prompt": "a", "duration_seconds": 4},
        ]
    }
    path = _write(tmp_path / "storyboard.json", json.dumps(data, ensure_ascii=False))
    shots = storyboard.load_storyboard(path)
    budget_fixes = [f for f in shots[0].fixes if "台词预算" in f]
    assert budget_fixes, shots[0].fixes
    # 4 秒 ≈ 9 字（(4-0.4)×2.6）
    assert f"约 {storyboard.speech_budget_chars(4)} 字" in budget_fixes[0]
    assert storyboard.speech_budget_chars(4) == 9


def test_speech_budget_chars_subtracts_line_gaps():
    assert storyboard.speech_budget_chars(5) == 11
    # 两句要扣一次句间停顿（0.25s ≈ 0.65 字，5 秒时仍落在 11 字）
    assert storyboard.speech_budget_chars(5, 2) == 11
    assert storyboard.speech_budget_chars(4, 3) == 8
    assert storyboard.speech_budget_chars(0.2) == 0


def test_total_speech_over_declared_duration_is_logged(tmp_path: Path, caplog):
    data = {
        "total_duration_seconds": 6,
        "shots": [
            {"id": 1, "narration": "第一句旁白内容较长", "image_prompt": "a",
             "duration_seconds": 3},
            {"id": 2, "narration": "第二句旁白内容也较长", "image_prompt": "b",
             "duration_seconds": 3},
        ],
    }
    path = _write(tmp_path / "storyboard.json", json.dumps(data, ensure_ascii=False))
    with caplog.at_level("WARNING", logger="prompt2vid.storyboard"):
        storyboard.load_storyboard(path)
    assert any("台词总量" in r.getMessage() for r in caplog.records)


def test_read_manifest(tmp_path: Path):
    path = _write(
        tmp_path / "manifest.json",
        json.dumps({"shots": [{"id": 1, "status": "ok", "path": "x"}]}),
    )
    assert len(storyboard.read_manifest(path)) == 1
    with pytest.raises(FileNotFoundError):
        storyboard.read_manifest(tmp_path / "missing.json")


# ------------------------------------------------------------ 角色一致性

CHARACTER_DESCRIPTION = "a young woman with long black hair, wearing a red knit sweater"


def _write_with_characters(
    tmp_path: Path,
    shots: list[dict] | None = None,
    characters: list[dict] | None = None,
) -> Path:
    data = {
        "characters": characters
        if characters is not None
        else [{"id": 1, "name": "女主", "description": CHARACTER_DESCRIPTION}],
        "shots": shots
        if shots is not None
        else [
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
    }
    return _write(tmp_path / "storyboard.json", json.dumps(data, ensure_ascii=False))


def test_load_characters(tmp_path: Path):
    path = _write_with_characters(tmp_path)
    characters = storyboard.load_characters(path)
    assert list(characters) == [1]
    assert characters[1].name == "女主"
    assert characters[1].description == CHARACTER_DESCRIPTION


def test_load_characters_absent(tmp_path: Path):
    path = _write(
        tmp_path / "storyboard.json",
        json.dumps({"shots": [{"id": 1, "image_prompt": "a"}]}),
    )
    assert storyboard.load_characters(path) == {}


def test_shot_characters_and_prompt_embedding(tmp_path: Path):
    path = _write_with_characters(tmp_path)
    shots = storyboard.load_storyboard(path)

    assert shots[0].characters == [1]
    # 出场角色的外观描述逐字补入 image_prompt（LLM 漏写时代码兜底）
    assert shots[0].image_prompt == f"autumn park, golden leaves, {CHARACTER_DESCRIPTION}"
    assert any("补入角色" in f for f in shots[0].fixes)
    # 空镜不受影响
    assert shots[1].characters == []
    assert shots[1].image_prompt == "falling leaves"
    assert shots[1].fixes == []


def test_prompt_already_contains_description_not_duplicated(tmp_path: Path):
    shot = {
        "id": 1,
        "characters": [1],
        "narration": "n",
        "visual": "v",
        "image_prompt": f"girl in park, {CHARACTER_DESCRIPTION}",
        "duration_seconds": 4,
    }
    path = _write_with_characters(tmp_path, shots=[shot])
    shots = storyboard.load_storyboard(path)
    assert shots[0].image_prompt.count(CHARACTER_DESCRIPTION) == 1
    assert not any("补入角色" in f for f in shots[0].fixes)


def test_undefined_character_marker_ignored(tmp_path: Path):
    shot = {
        "id": 1,
        "characters": [9],
        "narration": "n",
        "visual": "v",
        "image_prompt": "a girl",
        "duration_seconds": 4,
    }
    path = _write_with_characters(tmp_path, shots=[shot])
    shots = storyboard.load_storyboard(path)
    assert shots[0].characters == []
    assert any("未在 characters 定义" in f for f in shots[0].fixes)


def test_character_without_description_dropped(tmp_path: Path):
    characters = [
        {"id": 1, "name": "无描述角色"},  # 无 description → 丢弃
        {"id": 2, "name": "路人", "description": "a man in a blue jacket"},
    ]
    shot = {
        "id": 1,
        "characters": [1, 2],
        "narration": "n",
        "visual": "v",
        "image_prompt": "city street",
        "duration_seconds": 4,
    }
    path = _write_with_characters(tmp_path, shots=[shot], characters=characters)
    shots = storyboard.load_storyboard(path)
    # 只有带描述的角色 2 生效，其描述补入 prompt
    assert shots[0].characters == [2]
    assert "a man in a blue jacket" in shots[0].image_prompt


def test_backward_compatible_storyboard_without_characters(tmp_path: Path):
    """旧格式（无 characters）：解析行为与此前完全一致。"""
    data = {
        "shots": [
            {
                "id": 1,
                "narration": "n",
                "visual": "v",
                "image_prompt": "autumn park",
                "motion_prompt": "slow zoom",
                "duration_seconds": 4,
            }
        ]
    }
    path = _write(tmp_path / "storyboard.json", json.dumps(data, ensure_ascii=False))
    shots = storyboard.load_storyboard(path)
    assert shots[0].characters == []
    assert shots[0].image_prompt == "autumn park"


# ------------------------------------------------------------ 角色对白


def test_character_voice_and_timbre_parsed(tmp_path: Path):
    """角色卡可带 voice（edge-tts 音色）与 timbre（IndexTTS 音库）。"""
    characters = [
        {
            "id": 1,
            "name": "玉皇大帝",
            "description": "an old emperor in golden robes",
            "voice": "zh-CN-YunjianNeural",
            "timbre": "播音中年男.mp3",
        }
    ]
    path = _write_with_characters(
        tmp_path,
        shots=[{"id": 1, "image_prompt": "palace", "duration_seconds": 4}],
        characters=characters,
    )
    characters = storyboard.load_characters(path)
    assert characters[1].voice == "zh-CN-YunjianNeural"
    assert characters[1].timbre == "播音中年男.mp3"


def test_dialogue_speaker_resolution(tmp_path: Path):
    """speaker 支持角色 id / 角色名 / 旁白。/ 未定义发言人降级为旁白。"""
    characters = [
        {"id": 1, "name": "玉皇大帝", "description": "an old emperor"},
        {"id": 2, "name": "老者", "description": "an old servant"},
    ]
    shot = {
        "id": 1,
        "characters": [1, 2],
        "image_prompt": "palace hall",
        "duration_seconds": 4,
        "dialogue": [
            {"speaker": 1, "text": "你们这届凡人，有点意思。"},
            {"speaker": "老者", "text": "陛下，人间已经变天了。"},
            {"speaker": "旁白", "text": "三千年前，他也说过这句话。"},
            {"speaker": "路人甲", "text": "谁在说话？"},
        ],
    }
    path = _write_with_characters(tmp_path, shots=[shot], characters=characters)
    shots = storyboard.load_storyboard(path)
    lines = shots[0].dialogue

    assert [line.speaker for line in lines] == [1, 2, storyboard.NARRATOR_ID,
                                                storyboard.NARRATOR_ID]
    assert lines[0].name == "玉皇大帝"
    assert lines[1].name == "老者"
    assert lines[2].is_narration and lines[2].name == ""
    # 未定义发言人 → 旁白 + 修复项（不中断出片）
    assert lines[3].is_narration
    assert any("路人甲" in f for f in shots[0].fixes)


def test_dialogue_accepts_prefixed_strings_and_dict_singleton(tmp_path: Path):
    """dialogue 也宽容接受 "角色名：台词" 字符串与单个对象。"""
    characters = [{"id": 1, "name": "女主", "description": "a young woman"}]
    shot = {
        "id": 1,
        "characters": [1],
        "image_prompt": "park",
        "duration_seconds": 4,
        "dialogue": ["女主：你来晚了。", "旁白：那年冬天，雪很大。"],
    }
    path = _write_with_characters(tmp_path, shots=[shot], characters=characters)
    shots = storyboard.load_storyboard(path)
    assert [line.speaker for line in shots[0].dialogue] == [1, storyboard.NARRATOR_ID]
    assert shots[0].dialogue[0].text == "你来晚了。"

    single = dict(shot, dialogue={"speaker": 1, "text": "走吧。"})
    path = _write_with_characters(tmp_path, shots=[single], characters=characters)
    shots = storyboard.load_storyboard(path)
    assert len(shots[0].dialogue) == 1
    assert shots[0].dialogue[0].speaker == 1


def test_dialogue_empty_lines_dropped(tmp_path: Path):
    shot = {
        "id": 1,
        "image_prompt": "park",
        "duration_seconds": 4,
        "dialogue": [{"speaker": 1, "text": "   "}, {"speaker": 1, "text": "走吧。"}],
    }
    path = _write(tmp_path / "storyboard.json", json.dumps({"shots": [shot]},
                                                          ensure_ascii=False))
    shots = storyboard.load_storyboard(path)
    assert len(shots[0].dialogue) == 1


def test_spoken_lines_falls_back_to_narration(tmp_path: Path):
    """没有 dialogue 的旧分镜：spoken_lines() 回退成一整条旁白。"""
    data = {
        "shots": [
            {
                "id": 1,
                "narration": "秋天的公园",
                "image_prompt": "autumn park",
                "duration_seconds": 4,
            }
        ]
    }
    path = _write(tmp_path / "storyboard.json", json.dumps(data, ensure_ascii=False))
    shot = storyboard.load_storyboard(path)[0]
    assert shot.has_dialogue is False
    lines = shot.spoken_lines()
    assert len(lines) == 1
    assert lines[0].is_narration
    assert lines[0].text == "秋天的公园"


def test_spoken_lines_prefers_dialogue(tmp_path: Path):
    """有 dialogue 时不再额外念 narration，避免同一镜说两遍。"""
    shot = {
        "id": 1,
        "narration": "旁白兜底文案",
        "image_prompt": "park",
        "duration_seconds": 4,
        "dialogue": [{"speaker": "旁白", "text": "他抬头看天。"}],
    }
    path = _write(tmp_path / "storyboard.json", json.dumps({"shots": [shot]},
                                                          ensure_ascii=False))
    shot = storyboard.load_storyboard(path)[0]
    assert shot.has_dialogue is True
    assert [line.text for line in shot.spoken_lines()] == ["他抬头看天。"]


def test_estimate_speech_seconds_counts_lines_and_gaps():
    lines = [storyboard.Line(speaker=1, text="一二三四五六七八"), storyboard.Line(speaker=0, text="九十")]
    expected = 10 / storyboard.SPEECH_CHARS_PER_SECOND + storyboard.SPEECH_LINE_GAP_SECONDS + storyboard.SPEECH_TAIL_SECONDS
    assert storyboard.estimate_speech_seconds(lines) == pytest.approx(expected)
    # 没有台词行时回退到整镜旁白
    assert storyboard.estimate_speech_seconds(None, "一二三四五") == pytest.approx(
        5 / storyboard.SPEECH_CHARS_PER_SECOND + storyboard.SPEECH_TAIL_SECONDS
    )
    assert storyboard.estimate_speech_seconds([], "") == 0.0


def test_shot_duration_raised_to_cover_dialogue(tmp_path: Path):
    """分镜时长给短了 → 按台词语速抬到够用（受 MAX_SHOT_SECONDS 约束）。"""
    data = {
        "shots": [
            {
                "id": 1,
                "narration": "",
                "visual": "老者在殿前说话",
                "image_prompt": "old man in palace hall",
                "motion_prompt": "slow push in",
                "duration_seconds": 3,
                "dialogue": [{"speaker": "旁白", "text": "人" * 20}],
            },
            {
                "id": 2,
                "narration": "短",
                "visual": "空镜",
                "image_prompt": "empty hall",
                "motion_prompt": "static",
                "duration_seconds": 3,
            },
        ]
    }
    path = _write(tmp_path / "storyboard.json", json.dumps(data, ensure_ascii=False))
    shots = storyboard.load_storyboard(path)
    assert shots[0].duration_seconds > 3
    assert shots[0].duration_seconds <= storyboard.MAX_SHOT_SECONDS
    assert any("时长" in fix for fix in shots[0].fixes)
    # 台词装得下的镜头不动
    assert shots[1].duration_seconds == 3


def test_max_shot_seconds_follows_ltx_frame_cap(monkeypatch):
    """单镜硬顶取「内容上限」和「LTX 帧数上限」的较小值。"""
    monkeypatch.delenv("COMFYUI_LTX_MAX_FRAMES", raising=False)
    assert storyboard.max_shot_seconds() == pytest.approx(storyboard.MAX_SHOT_SECONDS)
    monkeypatch.setenv("COMFYUI_LTX_MAX_FRAMES", "121")
    assert storyboard.max_shot_seconds() == pytest.approx(120 / 25)


def test_shot_duration_clamped_to_ltx_frame_cap(tmp_path: Path, monkeypatch):
    """8GB 档单镜只有 4.8s：分镜声明的 8s 被压到 4.8s 并给出告警。"""
    monkeypatch.setenv("COMFYUI_LTX_MAX_FRAMES", "121")
    data = {
        "shots": [
            {
                "id": 1,
                "narration": "",
                "visual": "空镜",
                "image_prompt": "empty hall",
                "motion_prompt": "static",
                "duration_seconds": 8,
            }
        ]
    }
    path = _write(tmp_path / "storyboard.json", json.dumps(data, ensure_ascii=False))
    shots = storyboard.load_storyboard(path)
    assert shots[0].duration_seconds == pytest.approx(4.8)
    assert any("4.8" in fix for fix in shots[0].fixes)
