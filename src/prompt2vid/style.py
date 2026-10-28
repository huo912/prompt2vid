"""固定风格约束（PROMPT2VID_STYLE）。

为什么需要：image_prompt 由分镜 LLM 撰写，风格随模型与温度漂移，
同一部片子里可能一半镜头写实、一半镜头动漫。这里提供两层硬约束，
让全片风格钉死在目标风格上：

1. 提示词层（comfyui_client 的工作流构建器调用）：每镜文生图 /
   图生视频的正向提示词自动追加风格正向词，负向提示词自动追加
   风格排斥词——LLM 忘写风格词、或写实系 checkpoint（如 RealVisXL）
   把画面拉向照片感时，这一层兜底生效；
2. 分镜层（crew.build_tasks 注入）：给 plan_storyboard 任务描述
   追加风格约束段，让 LLM 从写角色卡和画面描述起就围绕目标风格
   组织内容（构图、上色、光影的审美方向与提示词层一致）。

配置：PROMPT2VID_STYLE=anime（预设名，见 STYLE_PRESETS，大小写不敏感）；
设为其它非空值时整个值按字面英文提示词追加（自定义风格，逗号分隔词条）；
留空 / 未设置 = 关闭全部风格约束。
"""

from __future__ import annotations

import os

STYLE_VAR = "PROMPT2VID_STYLE"

# 风格预设：positive 追加到每镜正向提示词，negative 追加到负向提示词。
# 词条按逗号切分后与现有提示词去重合并，已存在的词条不会重复追加。
STYLE_PRESETS: dict[str, dict[str, str]] = {
    # 动漫：赛璐璐上色 + 干净线稿 + 高饱和色彩；负向排除写实/照片/3D，
    # 防止写实系 checkpoint（RealVisXL 等）把画面拉回照片质感
    "anime": {
        "positive": (
            "anime style, japanese animation, 2d cel shaded, clean line art, "
            "vibrant colors, detailed anime illustration"
        ),
        "negative": (
            "photorealistic, photograph, photo, realistic, 3d render, cgi, "
            "semi-realistic"
        ),
    },
    # 写实：电影感摄影（RealVisXL 系 checkpoint 的原生方向）
    "realistic": {
        "positive": (
            "photorealistic, cinematic film still, 35mm photography, "
            "natural lighting, shallow depth of field"
        ),
        "negative": (
            "anime, cartoon, illustration, painting, 2d, cel shaded, drawing"
        ),
    },
}


def style_name() -> str:
    """当前配置的风格名（原样返回，未配置时为空串）。"""
    return os.getenv(STYLE_VAR, "").strip()


def _preset() -> dict[str, str] | None:
    """解析当前风格：预设名 → 预设词条；未知值 → 字面自定义正向提示词。

    返回 {"positive": ..., "negative": ...}；未配置风格时 None。
    """
    raw = style_name()
    if not raw:
        return None
    preset = STYLE_PRESETS.get(raw.lower())
    if preset is not None:
        return preset
    return {"positive": raw, "negative": ""}


def _terms(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def _merge_terms(text: str, terms: list[str]) -> str:
    """把 terms 中尚未出现的词条追加到 text（词条级、大小写不敏感去重）。"""
    if not terms:
        return text
    existing = {t.strip().lower() for t in text.split(",") if t.strip()}
    fresh = [t for t in terms if t.lower() not in existing]
    if not fresh:
        return text
    return f"{text}, {', '.join(fresh)}" if text else ", ".join(fresh)


def apply_style(text: str) -> str:
    """正向提示词追加风格正向词（未配置风格时原样返回）。"""
    preset = _preset()
    if preset is None:
        return text
    return _merge_terms(text, _terms(preset["positive"]))


def apply_style_negative(text: str) -> str:
    """负向提示词追加风格排斥词（未配置风格时原样返回）。"""
    preset = _preset()
    if preset is None:
        return text
    return _merge_terms(text, _terms(preset["negative"]))


def style_enabled() -> bool:
    return _preset() is not None


def storyboard_style_hint() -> str:
    """注入 plan_storyboard 任务描述的风格约束段（未配置风格时为空串）。

    与提示词层的区别：这一层不改变生成参数，只约束 LLM 写分镜时的
    审美方向，避免 image_prompt 的画面组织（构图/上色/光影描述）与
    追加的风格词互相矛盾。
    """
    preset = _preset()
    if preset is None:
        return ""
    lines = [
        "",
        "全片视觉风格硬性约束（PROMPT2VID_STYLE="
        f"{style_name()}，优先级高于用户描述里与之冲突的风格要求）：",
        "- image_prompt 的画面组织必须围绕目标风格展开，风格关键词为："
        f"{preset['positive']}（可按镜头微调措辞，不得偏离方向）；",
    ]
    if preset["negative"]:
        lines.append(
            f"- image_prompt 禁止出现与目标风格冲突的词：{preset['negative']}；"
        )
    lines.append(
        "- characters 的 description 与 visual 也按该风格的审美撰写"
        "（如动漫风格＝赛璐璐上色、干净线稿的审美），保证全片风格统一；"
    )
    lines.append(
        "- 图像/视频生成阶段还会在提示词层自动强制追加以上风格词，"
        "此约束是为了让你写的画面描述与最终风格一致，不要在 image_prompt "
        "里堆砌与风格无关的画质词来替代它。"
    )
    return "\n".join(lines)
