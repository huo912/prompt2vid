"""从 LLM 最终回答中抢救期望产物（兜底，非首选路径）。

创意步骤（需求/剧本/分镜）依赖 LLM 主动调用写文件工具。小参数本地模型
经常只把内容贴在最终回答里而不调用工具，导致"未生成期望产物"。本模块
在产物缺失时做最后一次努力：从回答文本中提取内容并返回可写入的字符串。

判定从严，宁缺勿滥——救回来的内容必须是"看起来像产物"的：
- JSON 产物：能整体解析，且必需字段非空（storyboard 要求 shots 非空）；
- 文本产物：剥掉外层代码栅栏后，至少 2 行且不少于 MIN_TEXT_CHARS 字符，
  避免把 "任务完成" 这类寒暄当成剧本写进 script.md。
"""

from __future__ import annotations

import json
import re

from .storyboard import StoryboardError, extract_json_value

# 文本产物（script.md）最低可信长度：过滤 "task done" 之类的寒暄输出
MIN_TEXT_CHARS = 40

# 整个回答被一对 ``` 包裹时剥掉栅栏（含 ```markdown / ```json 等语言标记）
_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)```", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """整段文本被代码栅栏包裹时剥掉栅栏，否则原样返回（去首尾空白）。"""
    stripped = text.strip()
    match = _FENCE_RE.fullmatch(stripped)
    return match.group(1).strip() if match else stripped


def _salvage_json(artifact: str, output: str) -> str | None:
    try:
        value = extract_json_value(output)
    except StoryboardError:
        return None

    if artifact.endswith("storyboard.json"):
        # 顶层裸数组是 LLM 常见写法，包成 {"shots": [...]} 再落盘
        if isinstance(value, list):
            return json.dumps({"shots": value}, ensure_ascii=False, indent=2) if value else None
        if not isinstance(value, dict):
            return None
        shots = value.get("shots")
        if not isinstance(shots, list) or not shots:
            return None
        return json.dumps(value, ensure_ascii=False, indent=2)

    if not isinstance(value, dict) or not value:
        return None
    return json.dumps(value, ensure_ascii=False, indent=2)


def _salvage_text(output: str) -> str | None:
    text = strip_code_fence(output)
    if len(text) < MIN_TEXT_CHARS or len(text.splitlines()) < 2:
        return None
    return text


def salvage_payload(artifact: str, output: str) -> str | None:
    """尝试从 LLM 最终回答中提取 `artifact` 的内容；失败返回 None。

    Args:
        artifact: 产物文件名（如 storyboard.json / script.md），仅用扩展名判断类型。
        output: crew.kickoff() 的最终回答文本。

    Returns:
        可直接写入文件的字符串，或 None（表示无法可靠提取）。
    """
    if not output or not output.strip():
        return None
    if artifact.lower().endswith(".json"):
        return _salvage_json(artifact, output)
    return _salvage_text(output)
