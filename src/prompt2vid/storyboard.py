"""分镜表（storyboard.json）宽松解析、校验与修复。

storyboard 由 LLM 生成的 JSON，字段可能缺失、类型不规范、甚至被
markdown 代码栅栏包裹。本模块做"宽容输入"处理：
- 自动剥离 ```json 代码栅栏、从任意文本中提取首个 JSON 对象；
- 数值字段容错转换（"4" → 4）；
- 缺失字段给出合理默认值并记录修复项；
- 只有完全没有可用分镜时才抛 StoryboardError。

角色一致性（顶层 characters 角色卡 + 每镜出场标记）：
- 每镜 characters 引用未定义角色 → 忽略该标记并记录修复项；
- 出场角色的 description 未逐字出现在 image_prompt → 代码兜底补入
  （仅内存修复，不改写 storyboard.json），保证文本条件跨镜一致。
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
class Character:
    """角色卡：出场人物的外观契约。

    description 是一段固定的英文外观描述（发型/脸型/服装/配饰），
    各镜 image_prompt 必须逐字包含它——措辞一致 = 文本条件一致，
    这是角色一致性的第一道防线（配合 IPAdapter 参考图为第二道）。
    """

    id: int
    name: str
    description: str


@dataclass
class Shot:
    """单个分镜（已规范化）。"""

    id: int
    narration: str
    visual: str
    image_prompt: str
    motion_prompt: str
    duration_seconds: float
    characters: list[int] = field(default_factory=list)
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


def _normalize_characters(raw: object) -> dict[int, Character]:
    """宽容解析顶层 characters 数组；缺失/损坏返回 {}（空镜场景完全兼容）。"""
    if not isinstance(raw, list):
        return {}
    characters: dict[int, Character] = {}
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        description = str(item.get("description") or "").strip()
        if not description:
            continue  # 没有外观描述的角色卡没有意义，静默丢弃
        cid = _to_int(item.get("id"), idx)
        characters[cid] = Character(
            id=cid,
            name=str(item.get("name") or "").strip(),
            description=description,
        )
    return characters


def _normalize_shot(
    item: object, fallback_id: int, characters: dict[int, Character]
) -> Shot | None:
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

    # 角色一致性契约：出场角色的外观描述必须逐字出现在 image_prompt。
    # LLM 漏写时代码兜底补入（不修改 storyboard.json 文件，仅内存修复）。
    shot_characters: list[int] = []
    raw_chars = item.get("characters")
    if isinstance(raw_chars, list):
        for value in raw_chars:
            cid = _to_int(value, 0)
            if cid in characters:
                shot_characters.append(cid)
            elif cid:
                fixes.append(f"出场角色 {cid} 未在 characters 定义，忽略")
    missing = [cid for cid in shot_characters if characters[cid].description not in image_prompt]
    if missing:
        appended = ", ".join(characters[cid].description for cid in missing)
        image_prompt = f"{image_prompt}, {appended}"
        fixes.append(f"image_prompt 补入角色 {missing} 的外观描述")

    if not (image_prompt or narration):
        return None

    return Shot(
        id=shot_id,
        narration=narration,
        visual=visual,
        image_prompt=image_prompt,
        motion_prompt=motion_prompt,
        duration_seconds=duration,
        characters=shot_characters,
        fixes=fixes,
    )


def _parse_storyboard(
    path: str | Path,
) -> tuple[list[Shot], dict[int, Character]]:
    path = Path(path)
    if not path.is_file():
        raise StoryboardError(f"分镜表不存在: {path}（请先由分镜师生成 storyboard.json）")
    text = path.read_text(encoding="utf-8")
    data = extract_json(text)
    raw_shots = data.get("shots")
    if not isinstance(raw_shots, list) or not raw_shots:
        raise StoryboardError("分镜表缺少非空 shots 数组")

    characters = _normalize_characters(data.get("characters"))
    shots: list[Shot] = []
    seen_ids: set[int] = set()
    for idx, item in enumerate(raw_shots, start=1):
        shot = _normalize_shot(item, fallback_id=idx, characters=characters)
        if shot is None:
            continue
        if shot.id in seen_ids:
            shot.fixes.append(f"id {shot.id} 重复，重排为 {len(shots) + 1}")
            shot.id = len(shots) + 1
        seen_ids.add(shot.id)
        shots.append(shot)

    if not shots:
        raise StoryboardError("分镜表没有任何有效分镜")
    return shots, characters


def load_storyboard(path: str | Path) -> list[Shot]:
    """加载并规范化分镜表；不可恢复时抛 StoryboardError。"""
    shots, _ = _parse_storyboard(path)
    return shots


def load_characters(path: str | Path) -> dict[int, Character]:
    """加载分镜表顶层 characters（角色卡）；没有角色时返回 {}。"""
    _, characters = _parse_storyboard(path)
    return characters


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
