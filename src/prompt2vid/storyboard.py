"""分镜表（storyboard.json）宽松解析、校验与修复。

storyboard 由 LLM 生成的 JSON，字段可能缺失、类型不规范、甚至被
markdown 代码栅栏包裹。本模块做"宽容输入"处理：
- 自动剥离 ```json 代码栅栏、从任意文本中提取首个 JSON 对象；
- 数值字段容错转换（"4" → 4）；
- 缺失字段给出合理默认值并记录修复项；
- 只有完全没有可用分镜时才抛 StoryboardError。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

STORYBOARD_FILE = "storyboard.json"

# 每镜时长与帧数预算（LTX + 8GB 显存）
MIN_SHOT_SECONDS = 1.0
MAX_SHOT_SECONDS = 10.0
DEFAULT_SHOT_SECONDS = 4.0
DEFAULT_MOTION_PROMPT = "subtle motion, gentle camera drift"


class StoryboardError(RuntimeError):
    """分镜表无法解析或没有可用分镜。"""


@dataclass
class Shot:
    """单个分镜（已规范化）。"""

    id: int
    narration: str
    visual: str
    image_prompt: str
    motion_prompt: str
    duration_seconds: float
    fixes: list[str] = field(default_factory=list)


def extract_json(text: str) -> dict:
    """从文本中提取首个 JSON 对象（容忍代码栅栏 / 前后杂文）。"""
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise StoryboardError(f"未能从内容中解析出 JSON 对象: {text[:120]!r}…")


def _to_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _normalize_shot(item: object, fallback_id: int) -> Shot | None:
    if not isinstance(item, dict):
        return None
    fixes: list[str] = []

    shot_id = _to_int(item.get("id"), fallback_id)
    if item.get("id") != shot_id:
        fixes.append("id 规范化")

    narration = str(item.get("narration") or "").strip()
    visual = str(item.get("visual") or "").strip()
    image_prompt = str(item.get("image_prompt") or "").strip()
    motion_prompt = str(item.get("motion_prompt") or "").strip()

    duration = _to_float(item.get("duration_seconds"), DEFAULT_SHOT_SECONDS)
    clamped = min(max(duration, MIN_SHOT_SECONDS), MAX_SHOT_SECONDS)
    if not math.isclose(duration, clamped):
        fixes.append(f"时长 {duration} 调整为 {clamped}")
        duration = clamped

    if not image_prompt:
        # 英文 prompt 缺失时回落到画面描述（中文也能出图，效果打折但不中断）
        image_prompt = visual or narration
        fixes.append("image_prompt 缺失，回落到 visual/narration")
    if not motion_prompt:
        motion_prompt = DEFAULT_MOTION_PROMPT
        fixes.append("motion_prompt 缺失，使用默认运动描述")

    if not (image_prompt or narration):
        return None

    return Shot(
        id=shot_id,
        narration=narration,
        visual=visual,
        image_prompt=image_prompt,
        motion_prompt=motion_prompt,
        duration_seconds=duration,
        fixes=fixes,
    )


def load_storyboard(path: str | Path) -> list[Shot]:
    """加载并规范化分镜表；不可恢复时抛 StoryboardError。"""
    path = Path(path)
    if not path.is_file():
        raise StoryboardError(f"分镜表不存在: {path}（请先由分镜师生成 storyboard.json）")
    text = path.read_text(encoding="utf-8")
    data = extract_json(text)
    raw_shots = data.get("shots")
    if not isinstance(raw_shots, list) or not raw_shots:
        raise StoryboardError("分镜表缺少非空 shots 数组")

    shots: list[Shot] = []
    seen_ids: set[int] = set()
    for idx, item in enumerate(raw_shots, start=1):
        shot = _normalize_shot(item, fallback_id=idx)
        if shot is None:
            continue
        if shot.id in seen_ids:
            shot.fixes.append(f"id {shot.id} 重复，重排为 {len(shots) + 1}")
            shot.id = len(shots) + 1
        seen_ids.add(shot.id)
        shots.append(shot)

    if not shots:
        raise StoryboardError("分镜表没有任何有效分镜")
    return shots


def read_manifest(path: str | Path) -> list[dict]:
    """读取阶段产物清单（manifest.json），返回 shots 列表（宽容解析）。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"清单不存在: {path}")
    data = extract_json(path.read_text(encoding="utf-8"))
    shots = data.get("shots")
    if not isinstance(shots, list):
        raise ValueError(f"{path} 缺少 shots 数组")
    return shots
