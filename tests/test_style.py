"""固定风格约束（PROMPT2VID_STYLE）单元测试。"""

from __future__ import annotations

from prompt2vid import style
from prompt2vid.style import (
    STYLE_PRESETS,
    apply_style,
    apply_style_negative,
    storyboard_style_hint,
    style_enabled,
    style_name,
)


def _clear_style(monkeypatch):
    monkeypatch.delenv("PROMPT2VID_STYLE", raising=False)


def test_disabled_when_unset(monkeypatch):
    _clear_style(monkeypatch)
    assert style_name() == ""
    assert style_enabled() is False
    # 原样返回，不追加任何词
    assert apply_style("a cat in the rain") == "a cat in the rain"
    assert apply_style_negative("low quality") == "low quality"
    assert storyboard_style_hint() == ""


def test_anime_preset_appends_positive_and_negative(monkeypatch):
    monkeypatch.setenv("PROMPT2VID_STYLE", "anime")
    assert style_enabled() is True

    styled = apply_style("a cat in the rain")
    assert styled.startswith("a cat in the rain, ")
    for term in ("anime style", "japanese animation", "cel shaded"):
        assert term in styled

    negative = apply_style_negative("low quality, blurry")
    assert "photorealistic" in negative
    assert "3d render" in negative
    # 原有词条保留在前
    assert negative.startswith("low quality, blurry, ")


def test_preset_name_case_insensitive(monkeypatch):
    monkeypatch.setenv("PROMPT2VID_STYLE", "  ANIME  ")
    styled = apply_style("a cat")
    assert "anime style" in styled
    # 与小写预设解析结果完全一致（大小写与首尾空白不影响）
    monkeypatch.setenv("PROMPT2VID_STYLE", "anime")
    assert apply_style("a cat") == styled


def test_term_level_dedup(monkeypatch):
    monkeypatch.setenv("PROMPT2VID_STYLE", "anime")
    # prompt 已含部分风格词时只补缺的，不整段重复
    styled = apply_style("a cat, anime style, vibrant colors, wet street")
    assert styled.count("anime style") == 1
    assert styled.count("vibrant colors") == 1
    assert "japanese animation" in styled
    # 全部风格词都已存在时原样返回
    full = STYLE_PRESETS["anime"]["positive"]
    assert apply_style(f"a cat, {full}") == f"a cat, {full}"


def test_realistic_preset_excludes_anime(monkeypatch):
    monkeypatch.setenv("PROMPT2VID_STYLE", "realistic")
    assert "photorealistic" in apply_style("a city street")
    assert "anime" in apply_style_negative("low quality")


def test_unknown_value_treated_as_literal_prompt(monkeypatch):
    """非预设名：整个值按字面英文提示词追加（自定义风格）。"""
    monkeypatch.setenv("PROMPT2VID_STYLE", "watercolor painting, soft pastel palette")
    styled = apply_style("a quiet harbor")
    assert "watercolor painting" in styled
    assert "soft pastel palette" in styled
    # 自定义风格没有预设负向词
    assert apply_style_negative("low quality") == "low quality"


def test_empty_prompt_gets_style_terms(monkeypatch):
    monkeypatch.setenv("PROMPT2VID_STYLE", "anime")
    styled = apply_style("")
    assert "anime style" in styled
    assert not styled.startswith(", ")


def test_storyboard_hint_carries_style_words(monkeypatch):
    monkeypatch.setenv("PROMPT2VID_STYLE", "anime")
    hint = storyboard_style_hint()
    assert "PROMPT2VID_STYLE=anime" in hint
    assert "anime style" in hint
    assert "image_prompt" in hint
    # 写实排斥词也要告知分镜 LLM
    assert "photorealistic" in hint
