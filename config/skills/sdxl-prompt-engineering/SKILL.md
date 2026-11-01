---
name: sdxl-prompt-engineering
description: 编写 SDXL 文生图 image_prompt 的规范——三段式结构、权重语法、竖屏 768x1344 构图、必须避免的抽象词与失败模式。当需要为分镜写首帧图英文提示词，或画面出现构图平淡、风格漂移、元素缺失、肢体崩坏时使用。
license: MIT
metadata:
  author: prompt2vid
  version: "1.0"
compatibility: ComfyUI + SDXL 系 checkpoint（sd_xl_base / RealVisXL / Animagine）
---

# SDXL 文生图提示词规范

`image_prompt` 是给画师的施工图：先写死角色卡原文，再写这一镜发生了什么，
最后写镜头与光。**空镜不写人物，人物镜不写抽象情绪。**

## 三段式结构（按此顺序拼接，用逗号分句）

1. **主体段**：逐字复制本镜出场角色的 `description`（见 `character-consistency`），
   多个角色按 id 顺序用 `, ` 连接。没有角色就跳过本段。
2. **动作 / 场景段**：这一镜的具体动作、位置、环境道具。
   用具象名词 + 能画出来的动词：`holding a red umbrella` ✅ / `feeling lonely` ❌
3. **镜头段**：景别 + 机位 + 光线 + 色彩。
   - 景别（稳定性由高到低）：`medium shot` > `medium close-up` > `close-up` > `wide shot`
   - 竖屏构图：`vertical composition, centered subject`；地平线放在画面下 1/3
   - 光线：`soft afternoon light` / `cinematic rim light` / `warm golden hour`
   - 色彩：`muted warm palette` / `high saturation`

示例（单角色、中景）：

```
a 22-year-old woman with shoulder-length black hair, round face, wearing a
cream wool coat and a red scarf, standing in a park holding a paper bag,
scattered ginkgo leaves on the ground, medium shot, vertical composition,
soft afternoon light, muted warm palette
```

## 权重语法

- 强调 `(term:1.2)`，一般不超过 1.4；弱化 `(term:0.8)`
- 一镜最多 2 处权重，多了画面失衡
- **风格词不要加权重**——风格由 `PROMPT2VID_STYLE` 与 LoRA 统一控制

## 必须避免

| 反模式 | 原因 | 改法 |
| --- | --- | --- |
| `beautiful` / `amazing` / `emotional` / `soulful` | 不可画 | 换成可见细节：`tear on cheek`、`hand on chest` |
| 整段中文 | SDXL 文本编码器对中文无效 | 全英文 |
| 一句话 80+ 词 | 后半段语义被稀释 | 拆短句，逗号分隔 |
| `full body shot, running, jumping` | 四肢最容易崩 | `medium shot, standing still` |
| `two people fighting`、`crowd of fans` | 多主体混脸 | 单主体镜头，背景写成 `blurred crowd in background` |
| 要求画面出现文字/招牌 | SDXL 不会拼写 | 文字交给后期字幕 |
| `holding chopsticks`、`typing` | 手指精细动作 | 手里拿大件物品或空手 |

## 竖屏 768x1344 构图要点

- 人脸放在画面上 1/3（抖音顶部 UI 会遮挡）
- 主体占画面高度 1/2 ~ 2/3，不要贴边
- 避免超广角大场景：竖屏里容易显得空

## 与风格约束的分工

`PROMPT2VID_STYLE` 会在你写的 prompt 之后**自动追加**风格正 / 负词（代码层
`style.py`），所以：

- 不要在 prompt 里重复风格词（不算错，但浪费 token）
- 不要写与当前档位冲突的风格词（anime 档位写 `photorealistic` 会互相拉扯）

## 写完自检

- [ ] 全英文？
- [ ] 有出场角色时，角色 `description` 原样逐字出现？
- [ ] 有具体景别词？
- [ ] 有光线 / 色彩词？
- [ ] 没有抽象情绪词、没有文字要求？
- [ ] 没有全身快速动作、没有手指特写？

词汇表见 [references/prompt-vocab.md](references/prompt-vocab.md)。
