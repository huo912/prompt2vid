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


def test_load_storyboard_repairs_and_fills_defaults(tmp_path: Path):
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
    assert shots[0].duration_seconds == 10.0  # 上限 10s
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
