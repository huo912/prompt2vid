"""LLM 输出抢救测试：从最终回答中恢复缺失的创意产物。"""

from __future__ import annotations

import json

from prompt2vid import llm_output

STORYBOARD = {
    "characters": [{"id": 1, "name": "女主", "description": "a girl with long black hair"}],
    "shots": [
        {
            "id": 1,
            "narration": "秋天",
            "visual": "金色落叶",
            "image_prompt": "autumn leaves",
            "duration_seconds": 4,
        }
    ],
}


def test_strip_code_fence_with_language_tag():
    text = "```json\n{\"a\": 1}\n```"
    assert llm_output.strip_code_fence(text) == '{"a": 1}'


def test_strip_code_fence_plain_text_unchanged():
    text = "# 剧本\n\n正文"
    assert llm_output.strip_code_fence(text) == text


def test_salvage_storyboard_from_fenced_reply():
    output = "分镜表如下：\n```json\n" + json.dumps(STORYBOARD, ensure_ascii=False) + "\n```\n以上。"
    payload = llm_output.salvage_payload("storyboard.json", output)
    assert payload is not None
    assert json.loads(payload)["shots"][0]["image_prompt"] == "autumn leaves"


def test_salvage_storyboard_from_bare_array():
    output = "```json\n" + json.dumps(STORYBOARD["shots"], ensure_ascii=False) + "\n```"
    payload = llm_output.salvage_payload("storyboard.json", output)
    assert payload is not None
    assert json.loads(payload) == {"shots": STORYBOARD["shots"]}


def test_salvage_storyboard_requires_non_empty_shots():
    assert llm_output.salvage_payload("storyboard.json", '{"shots": []}') is None
    assert llm_output.salvage_payload("storyboard.json", "抱歉，我无法完成") is None


def test_salvage_json_requires_object():
    assert llm_output.salvage_payload("requirement.json", '{"topic": "秋天"}') is not None
    assert llm_output.salvage_payload("requirement.json", "{}") is None


def test_salvage_text_script_from_fenced_markdown():
    script = "# 秋日\n\n## 整体叙事\n\n一个治愈的秋天午后。\n\n## 逐镜文案\n\n### 镜头 1\n- 旁白：秋天来了。"
    payload = llm_output.salvage_payload("script.md", f"```markdown\n{script}\n```")
    assert payload == script


def test_salvage_text_rejects_small_talk():
    """只有寒暄/确认语时不要把它当成剧本写盘。"""
    assert llm_output.salvage_payload("script.md", "write_script done") is None
    assert llm_output.salvage_payload("script.md", "好的，已完成。") is None
    assert llm_output.salvage_payload("script.md", "") is None
