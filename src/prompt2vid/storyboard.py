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

角色对白（每镜 dialogue 台词行）：
- 每行 {speaker, text}，speaker 可写角色 id、角色名或「旁白」；
- 未定义的发言人一律降级为旁白并记录修复项，不阻塞出片；
- 没有 dialogue 的旧分镜自动用 narration 作为唯一旁白行（完全兼容）。
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .comfyui_client import env_float, ltx_clip_seconds

logger = logging.getLogger("prompt2vid.storyboard")

STORYBOARD_FILE = "storyboard.json"

# 每镜时长与帧数预算（LTX + 显存）：内容上限之外还要服从 LTX 单镜帧数
# 上限——声明 7s 而显卡只能生成 4.8s 时，多出的配音只能靠降速/补帧吸收，
# 口型必然对不上，所以要取两者较小值（见 max_shot_seconds）
MIN_SHOT_SECONDS = 1.0
MAX_SHOT_SECONDS = 10.0
DEFAULT_SHOT_SECONDS = 4.0
DEFAULT_MOTION_PROMPT = "subtle motion, gentle camera drift"

# 台词时长估算（把「说了多久」变成「画面要拍多久」）：
# 中文 TTS 实测语速——IndexTTS-2.5 约 2.8-3.2 字/秒，edge-tts 约 4.5 字/秒，
# 取最慢的一档并留出句间停顿，宁可多给画面余量（多出来的部分由后期
# 降速/定格吸收，比反过来截断旁白安全）。
SPEECH_CHARS_PER_SECOND = 2.6
SPEECH_LINE_GAP_SECONDS = 0.25
SPEECH_TAIL_SECONDS = 0.4

# 旁白行的发言人 id（角色 id 从 1 开始，0 专门表示旁白/画外音）
NARRATOR_ID = 0

# 有台词的镜头必须能看见嘴。LatentSync 逐帧检测正脸，任何一帧检不到就
# 整镜跳过口型同步——成片里就成了"人没开口却在说话"。分镜师为了构图
# 常写"背面/出画/半张脸/虚化"，这里在解析阶段直接纠回正脸中近景。
_NO_FACE_FRAMINGS = (
    "seen from behind",
    "from behind",
    "back view",
    "back to camera",
    "back of his head",
    "back of her head",
    "over-the-shoulder",
    "over the shoulder",
    "at the very top edge of the frame",
    "at the top edge of the frame",
    "top edge of the frame",
    "at the very top of the frame",
    "at the top of the frame",
    "top of the frame",
    "bottom edge of the frame",
    "at the bottom of the frame",
    "bottom of the frame",
    "edge of the frame",
    "upper corner of the frame",
    "off-screen",
    "off screen",
    "half his face",
    "half her face",
    "half of his face",
    "half of her face",
    "only his shoulder",
    "only her shoulder",
    "only a shoulder",
    "only his back",
    "partially out of frame",
    "out of frame",
    "in soft focus",
    "silhouette",
    "silhouetted",
)

_FACE_VISIBLE_SUFFIX = (
    "front-facing medium close-up, the character's whole face clearly visible "
    "and in sharp focus, mouth and jaw unobstructed, eyes toward camera"
)

# 会让主体变小/把背景整个换掉的运镜：台词镜里一律改成慢推近。
# "slow zoom out" 喂给 LTX 会真的拉远——主体从半身变成全身、脸小到
# LatentSync 检不到（整镜跳过口型），而且拉远露出的背景是模型现编的
# （实测玉帝从红墙半身镜变成金云全身镜，同一镜内换了景别和背景）。
_SHRINK_MOVES = (
    "slow zoom out",
    "slight zoom out",
    "zoom out",
    "zoom back",
    "dolly out",
    "dolly back",
    "pull back",
    "pull out",
    "crane out",
    "crane up",
    "tilt up",
    "boom out",
)
# 拉远改用推近：主体只会变大，脸和背景都是首帧里已有的内容
_CAMERA_LOCK_MOTION = "slow zoom in, camera stays on the character"

# motion_prompt 里出现这些词才算写了"在说话"；否则 LatentSync 没有可
# 对齐的口部动作，只能生成一张不动的脸
_SPEAKING_CUES = ("lips", "speak", "talk", "mouth", "jaw")
_SPEAKING_MOTION = "the character speaks, lips and jaw moving naturally"
# 发言人写这些字样时按旁白处理
NARRATOR_ALIASES = {
    "", "旁白", "旁白/画外音", "画外音", "解说", "叙述",
    "narrator", "narration", "voiceover", "voice-over", "none", "null",
}


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
    # 音色（可选）：voice 给 edge-tts 用（如 zh-CN-YunxiNeural），
    # timbre 给 ComfyUI IndexTTS-2.5 用（插件 TimbreModel 目录里的文件名）。
    # 留空时由 tts 按角色 id 在默认音色表里轮转分配，保证不同角色不同声线。
    voice: str = ""
    timbre: str = ""


@dataclass
class Line:
    """一句台词（或一句旁白）。

    speaker 是角色 id，0 表示旁白；name 是角色的中文名（写字幕前缀用）。
    emotion 是该句的情感描述（中文短语，如「好奇地打量，语调带笑意」），
    给 IndexTTS-2.5 情感节点用；空串时读全局 PROMPT2VID_TTS_EMOTION。
    """

    speaker: int
    text: str
    name: str = ""
    emotion: str = ""

    @property
    def is_narration(self) -> bool:
        return self.speaker == NARRATOR_ID


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
    # 有序台词行；空列表 = 使用 narration 作为唯一旁白（旧分镜兼容）
    dialogue: list[Line] = field(default_factory=list)

    @property
    def has_dialogue(self) -> bool:
        return bool(self.dialogue)

    def spoken_lines(self) -> list[Line]:
        """本镜要念出来的文本行（按播放顺序）。

        有 dialogue 就用 dialogue；否则回退到 narration 单行旁白。
        这样旧 storyboard 与「整片只有旁白」的形态完全不变。
        """
        if self.dialogue:
            return list(self.dialogue)
        if self.narration:
            return [Line(speaker=NARRATOR_ID, text=self.narration)]
        return []


def estimate_speech_seconds(lines: list[Line] | None, narration: str = "") -> float:
    """估算这些台词念出来需要多少秒（用于给画面排时长）。

    中文按 SPEECH_CHARS_PER_SECOND 保守折算，行间留 SPEECH_LINE_GAP_SECONDS
    停顿，末尾留 SPEECH_TAIL_SECONDS——分镜时长必须覆盖这段语音，否则成片
    阶段只能靠降速/定格把画面拉长，口型必然对不上。
    """
    texts = [line.text.strip() for line in (lines or []) if line.text.strip()]
    if not texts and narration.strip():
        texts = [narration.strip()]
    if not texts:
        return 0.0
    seconds = sum(len(text) / SPEECH_CHARS_PER_SECOND for text in texts)
    seconds += SPEECH_LINE_GAP_SECONDS * (len(texts) - 1)
    return seconds + SPEECH_TAIL_SECONDS


def speech_budget_chars(seconds: float, line_count: int = 1) -> int:
    """给定镜头时长，反推能念完多少字（供告警/自检用，与估算互逆）。"""
    usable = seconds - SPEECH_TAIL_SECONDS
    usable -= SPEECH_LINE_GAP_SECONDS * max(0, line_count - 1)
    return max(0, int(usable * SPEECH_CHARS_PER_SECOND))


def max_shot_seconds() -> float:
    """单镜可声明的最长时长：内容上限与 LTX 实际能生成的秒数取小。

    `COMFYUI_LTX_MAX_FRAMES` 决定了每镜真正能生成多久（默认 257 帧 ≈ 10.2s，
    8GB 档压到 121 帧 ≈ 4.8s）。分镜若声明超过它，多出的配音只能靠降速/
    定格吸收，口型对不上——所以先把时长夹到可生成范围内。
    """
    return min(MAX_SHOT_SECONDS, ltx_clip_seconds())


def extract_json_value(text: str) -> object:
    """从文本中提取首个 JSON 值（容忍代码栅栏 / 前后杂文）。

    与 extract_json 的区别：顶层是数组时原样返回 list，不截断成第一个元素。
    LLM 有时会直接输出 `[{分镜1}, {分镜2}, …]`，必须先整体解析再判断结构，
    否则从第一个 `{` 起解析只会拿到单个分镜对象。
    """
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            continue
        # 首个可完整解析的 JSON 值即结果；继续向后扫会把顶层数组
        # 误判成数组里的第一个对象（正是本 bug 的成因）。
        if isinstance(value, (dict, list)):
            return value
    raise StoryboardError(f"未能从内容中解析出 JSON 对象: {text[:120]!r}…")


def extract_json(text: str) -> dict:
    """从文本中提取 JSON 对象（容忍代码栅栏 / 前后杂文）。

    顶层是数组时包装成 {"shots": [...]}，兼容旧调用方按对象取值的行为。
    """
    value = extract_json_value(text)
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {"shots": value}
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
            voice=str(item.get("voice") or "").strip(),
            timbre=str(item.get("timbre") or "").strip(),
        )
    return characters


def _resolve_speaker(
    raw: object, characters: dict[int, Character], fixes: list[str]
) -> int:
    """把台词发言人规范化为角色 id（0 = 旁白）。

    LLM 可能写角色 id（1）、角色名（"玉皇大帝"）、"旁白"，甚至写个不存在的
    名字。无法识别时一律降级为旁白并记修复项——宁可整段念出来，也不要
    因为一个字段写错导致整镜无音。
    """
    if raw is None or isinstance(raw, bool):
        return NARRATOR_ID
    if isinstance(raw, (int, float)):
        cid = _to_int(raw, 0)
        if cid == NARRATOR_ID or cid in characters:
            return cid
        fixes.append(f"台词发言人 {raw} 未在 characters 定义，按旁白处理")
        return NARRATOR_ID

    text = str(raw).strip()
    lowered = text.lower().lstrip("@#")
    if lowered in NARRATOR_ALIASES:
        return NARRATOR_ID
    if lowered.isdigit():
        cid = int(lowered)
        if cid == NARRATOR_ID or cid in characters:
            return cid
    for cid, character in characters.items():
        if character.name and character.name == text:
            return cid
    fixes.append(f"台词发言人“{text}”未在 characters 定义，按旁白处理")
    return NARRATOR_ID


def _split_speaker_text(text: str, characters: dict[int, Character]) -> tuple[object, str]:
    """从 "角色名：台词" / "角色名: 台词" 里拆出（发言人, 台词）。

    只在冒号左边确实是一个已定义角色名或「旁白」时才拆，避免把台词里
    正常的冒号误切成发言人。
    """
    for sep in ("：", ":"):
        if sep not in text:
            continue
        head, body = text.split(sep, 1)
        head = head.strip()
        body = body.strip()
        if not body:
            continue
        if head.lower() in NARRATOR_ALIASES:
            return NARRATOR_ID, body
        if any(c.name and c.name == head for c in characters.values()):
            return head, body
    return None, text.strip()


def _normalize_dialogue(
    raw: object, characters: dict[int, Character], fixes: list[str]
) -> list[Line]:
    """宽容解析每镜 dialogue：支持对象数组与 "角色名：台词" 字符串数组。"""
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    lines: list[Line] = []
    for item in raw:
        speaker_raw: object = None
        text = ""
        emotion = ""
        if isinstance(item, str):
            speaker_raw, text = _split_speaker_text(item, characters)
        elif isinstance(item, dict):
            text = str(
                item.get("text") or item.get("line") or item.get("content") or ""
            ).strip()
            speaker_raw = item.get(
                "speaker", item.get("character", item.get("role", item.get("name")))
            )
            emotion = str(item.get("emotion") or "").strip()
        if not text:
            continue
        speaker = _resolve_speaker(speaker_raw, characters, fixes)
        name = characters[speaker].name if speaker in characters else ""
        lines.append(Line(speaker=speaker, text=text, name=name, emotion=emotion))
    return lines


def _strip_no_face(text: str, hits: list[str]) -> str:
    """删掉 text 里的挡脸短语，并清理残留的重复空格/悬空标点。

    短语后面紧跟的 " and <补语>" 一并删掉：分镜师常写
    "only his shoulder and white beard visible above a cloud edge"，
    只删前半句会留下一句没有主语的残句（还会被模型当成新主体画出来）。
    """
    cleaned = text
    for phrase in hits:
        cleaned = re.sub(
            re.escape(phrase) + r"(?:\s+and\s+[^,.;]*)?",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;])", r"\1", cleaned)
    cleaned = re.sub(r",(\s*,)+", ",", cleaned)
    cleaned = re.sub(r",\s*(?=[.;])", "", cleaned)
    # 删词后剩下的悬空分词（"posture, appearing, visible above ..."）
    cleaned = re.sub(r",\s*(?:appearing|standing|seen)\b\s*,", ",", cleaned)
    return cleaned.strip(" ,")


# 首帧构图句（"Vertical over-the-shoulder high angle shot."）在挡脸纠偏后
# 已经由 _FACE_VISIBLE_SUFFIX 接管，留着会和"正脸中近景"互相打架
_FRAMING_SENTENCE = re.compile(r"^[^.]{0,60}\bshot\b[^.]*\.\s*", re.IGNORECASE)


def _make_face_visible(
    shot_id: int, image_prompt: str, motion_prompt: str, fixes: list[str]
) -> tuple[str, str]:
    """有台词的镜头：剥掉遮挡面部的构图词，并补一句口部动作。

    只对"角色真的要开口"的镜头生效——空镜/动物/纯旁白镜不动，避免把
    有意的背影设计改掉。image_prompt 与 motion_prompt 都要清：后者常写着
    "sighs at the top of the frame"，只清前者的话 LTX 还是会把脸推向边缘。
    """
    hits = [phrase for phrase in _NO_FACE_FRAMINGS if phrase in image_prompt.lower()]
    if hits:
        cleaned = _FRAMING_SENTENCE.sub("", _strip_no_face(image_prompt, hits))
        image_prompt = (
            f"{cleaned}, {_FACE_VISIBLE_SUFFIX}" if cleaned else _FACE_VISIBLE_SUFFIX
        )
        msg = (
            f"分镜 {shot_id} 有台词但构图挡脸（{', '.join(hits)}），已改为"
            f"正脸中近景——脸不在画面里，口型同步会整镜跳过"
        )
        fixes.append(msg)
        logger.warning("%s", msg)

    motion_hits = [
        phrase for phrase in _NO_FACE_FRAMINGS if phrase in motion_prompt.lower()
    ]
    if motion_hits:
        motion_prompt = _strip_no_face(motion_prompt, motion_hits)
        fixes.append(
            f"分镜 {shot_id} 的 motion_prompt 含挡脸构图词"
            f"（{', '.join(motion_hits)}），已删除"
        )

    shrink_hits = [p for p in _SHRINK_MOVES if p in motion_prompt.lower()]
    if shrink_hits:
        cleaned = _strip_no_face(motion_prompt, shrink_hits)
        motion_prompt = (
            f"{_CAMERA_LOCK_MOTION}, {cleaned}" if cleaned else _CAMERA_LOCK_MOTION
        )
        msg = (
            f"分镜 {shot_id} 有台词但运镜在拉远（{', '.join(shrink_hits)}），"
            f"已改为慢推近——拉远后主体变小，口型同步会整镜跳过"
        )
        fixes.append(msg)
        logger.warning("%s", msg)

    if not any(cue in motion_prompt.lower() for cue in _SPEAKING_CUES):
        motion_prompt = (
            f"{motion_prompt}, {_SPEAKING_MOTION}" if motion_prompt else _SPEAKING_MOTION
        )
        fixes.append(f"分镜 {shot_id} 有台词但 motion_prompt 无口部动作，已补入")
    return image_prompt, motion_prompt


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
    ceiling = max_shot_seconds()
    clamped = min(max(duration, MIN_SHOT_SECONDS), ceiling)
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

    # 台词行：speaker 支持 id / 角色名 / 旁白；解析不了的降级为旁白
    dialogue = _normalize_dialogue(item.get("dialogue"), characters, fixes)

    # 时长下限要覆盖台词：分镜时长给短了，成片阶段只能靠降速/定格把画面
    # 撑到配音长度（嘴型对不上、画面发卡）。这里按估算语速抬到够用的长度，
    # 仍受 max_shot_seconds()（内容上限 & LTX 帧数上限）约束。
    needed = estimate_speech_seconds(dialogue, narration)
    if needed > duration:
        bumped = min(max(needed, MIN_SHOT_SECONDS), ceiling)
        if bumped > duration + 0.05:
            fixes.append(
                f"时长 {duration:g} 提升到 {bumped:.1f}（覆盖台词约 {needed:.1f}s）"
            )
            # 抬时长是兜底，不是解法：分镜原定时长里的台词本来就念不完，
            # 应精简台词或拆镜（提示词里给了字数预算）
            line_count = len(dialogue) or (1 if narration else 0)
            budget = speech_budget_chars(duration, line_count)
            tried = sum(len(line.text.strip()) for line in dialogue) if dialogue else len(narration)
            msg = (
                f"分镜 {shot_id} 台词 {tried} 字约 {needed:.1f}s，超出 {duration:g}s "
                f"的台词预算（约 {budget} 字），已把时长抬到 {bumped:.1f}s；"
                f"更稳妥的写法是精简台词或多拆一镜"
            )
            fixes.append(msg)
            logger.warning("%s", msg)
            duration = bumped

    # 分镜时长够了也不代表画面拍得出：LTX 单镜帧数上限（显存预算）或单镜
    # 内容上限决定的秒数才是硬顶。台词超出的部分只能靠降速/补帧吸收，口型
    # 必然对不上，这里提前告警（不截断台词，交给人决定删词还是拆镜）。
    if needed > ceiling + 0.05:
        if ceiling < MAX_SHOT_SECONDS - 1e-6:
            reason = f"LTX 单镜上限 {ceiling:.1f}s（COMFYUI_LTX_MAX_FRAMES）"
        else:
            reason = f"单镜上限 {ceiling:.1f}s（MAX_SHOT_SECONDS）"
        msg = (
            f"分镜 {shot_id} 台词约 {needed:.1f}s 超出{reason}，"
            f"成片需降速/补帧覆盖配音（口型会错位），建议精简台词或拆镜"
        )
        fixes.append(msg)
        logger.warning("%s", msg)

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
    # 有台词的发言人必须出现在该镜画面里，否则成片是"画外人在说话"，
    # 口型同步也找不到那张脸（补进 characters 后 description 会自动补进
    # image_prompt，至少保证人真的在画面中）
    absent_speakers = sorted({
        line.speaker for line in dialogue
        if line.speaker in characters and line.speaker not in shot_characters
    })
    if absent_speakers:
        shot_characters.extend(absent_speakers)
        msg = (
            f"分镜 {shot_id} 的发言人 "
            f"{'、'.join(str(cid) for cid in absent_speakers)} 未列入 characters，"
            f"已补入（否则只有声音没有画面）"
        )
        fixes.append(msg)
        logger.warning("%s", msg)
    missing = [cid for cid in shot_characters if characters[cid].description not in image_prompt]
    if missing:
        appended = ", ".join(characters[cid].description for cid in missing)
        image_prompt = f"{image_prompt}, {appended}"
        fixes.append(f"image_prompt 补入角色 {missing} 的外观描述")

    # 口播镜的画面必须看得见嘴：LatentSync 逐帧找正脸，一帧失败就整镜
    # 跳过口型同步，成片里那个人就会"闭着嘴说话"。
    speaking = [line for line in dialogue if line.speaker in characters]
    if speaking:
        image_prompt, motion_prompt = _make_face_visible(
            shot_id, image_prompt, motion_prompt, fixes
        )
        speakers = {line.speaker for line in speaking}
        if len(speakers) > 1:
            names = "、".join(characters[cid].name or str(cid) for cid in sorted(speakers))
            msg = (
                f"分镜 {shot_id} 有 {len(speakers)} 个角色同时说话（{names}），"
                f"一帧只能驱动一张脸的嘴型（其余人会动嘴但不对齐），"
                f"建议拆成一镜一人"
            )
            fixes.append(msg)
            logger.warning("%s", msg)

    if not image_prompt and dialogue:
        # 只有台词的镜头也要有画面：用台词原文兜底（仅防丢镜）
        image_prompt = " ".join(line.text for line in dialogue)
        fixes.append("image_prompt 缺失，回落到台词文本")

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
        dialogue=dialogue,
    )


def _parse_storyboard(
    path: str | Path,
) -> tuple[list[Shot], dict[int, Character]]:
    path = Path(path)
    if not path.is_file():
        raise StoryboardError(f"分镜表不存在: {path}（请先由分镜师生成 storyboard.json）")
    text = path.read_text(encoding="utf-8")
    data = extract_json_value(text)
    # 兼容两种顶层结构：
    #   1) {"characters": [...], "shots": [...]}（推荐，提示词约定的格式）
    #   2) [{分镜1}, {分镜2}, …]（LLM 常见偷懒写法，无角色卡）
    if isinstance(data, list):
        raw_shots: object = data
        raw_characters: object = None
    elif isinstance(data, dict):
        raw_shots = data.get("shots")
        raw_characters = data.get("characters")
    else:
        raise StoryboardError("分镜表顶层必须是 JSON 对象或分镜数组")
    if not isinstance(raw_shots, list) or not raw_shots:
        raise StoryboardError("分镜表缺少非空 shots 数组")

    characters = _normalize_characters(raw_characters)
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

    total_speech = sum(estimate_speech_seconds(s.dialogue, s.narration) for s in shots)
    declared = 0.0
    if isinstance(data, dict):
        declared = _to_float(data.get("total_duration_seconds"), 0.0)
    if declared <= 0:
        declared = sum(s.duration_seconds for s in shots)
    if declared > 0 and total_speech > declared + 0.5:
        logger.warning(
            "分镜表台词总量约 %.1fs，超过声明的总时长 %.1fs——成片会被顶长，"
            "质检 duration_match（±%gs）可能不过；请精简台词或增加镜头",
            total_speech,
            declared,
            env_float("PROMPT2VID_DURATION_TOLERANCE", 3.0),
        )
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
