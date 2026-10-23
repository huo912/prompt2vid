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


def test_load_storyboard_errors(tmp_path: Path):
    # 不存在
    with pytest.raises(storyboard.StoryboardError):
        storyboard.load_storyboard(tmp_path / "missing.json")
    # 无 shots
    path = _write(tmp_path / "storyboard.json", "{}")
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
