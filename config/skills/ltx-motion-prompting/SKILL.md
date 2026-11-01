---
name: ltx-motion-prompting
description: 为 LTX-Video 图生视频撰写 motion_prompt（英文）的规范与词表，含时长↔帧数换算、可用/禁用运动词与四类必崩镜头的规避写法。当需要把分镜写成图生视频运动描述，或片段出现肢体变形、人物走形、镜头乱动时使用。
license: MIT
metadata:
  author: prompt2vid
  version: "1.0"
compatibility: ComfyUI LTX-Video（2B 0.9.5 / 13B 0.9.7）
---

# LTX-Video 运动提示词规范

`motion_prompt` 只描述**镜头内**的运动，不推进剧情、不描述新出现的主体。

## 硬性规则

1. 只写英文，1~2 个短句，不超过 25 个词。
2. 时长 3~5 秒。帧数按 `8n+1` 换算（代码自动算，你只需保证时长合理）：

   | 时长 | 25fps 帧数 |
   | --- | --- |
   | 3s | 81 |
   | 4s | 101 |
   | 5s | 121 |

3. 运动幅度小、速度慢：`slow zoom in`、`gentle pan left`、`subtle drift`、
   `soft handheld sway`、`slow dolly forward`。
4. 主体微动优先：`gently blinks`、`hair sways slightly`、`steam rises slowly`、
   `leaves fall slowly`。
5. 首帧的主体必须在运动中保持同一个主体——不要在 motion 里引入新角色。

## 四类必崩镜头与规避写法

| 类型 | 崩法 | 规避 |
| --- | --- | --- |
| 全身快速位移 | 腿部拉长/消失、多出肢体 | 改成原地微动：`slow pan right, the girl stands still, street lamps flicker` |
| 手部特写动作 | 手指融合、数量错乱 | 手离镜头或持大件：`holds a paper bag, slow zoom out` |
| 转身 / 回头 | 脸部重塑、五官漂移 | 用镜头绕行代替人物转身：`slow orbit around the subject` |
| 多人同框互动 | 换脸、身体粘连 | 单人镜头 + 背景虚化人物：`blurred pedestrians pass by in background` |

## 禁用词

`run`、`sprint`、`jump`、`dance`、`fight`、`wave hand`、`turn around`、
`walk towards camera`、`open mouth wide`、`fast zoom`、`rapid pan`、
`camera shake`、`handheld running`

## 推荐句式

```
<camera move>, <subject micro-motion>, <environment motion>
```

例：

```
slow zoom in, the woman gently blinks, dust motes drifting in the light
gentle pan left, steam rises from the mug, curtain sways slightly
subtle drift, the husky tilts its head slightly, tail sways slowly
```

## 与首帧的关系

- motion 里重复首帧已有的元素是安全的（强化一致性）
- motion 里出现首帧没有的元素 = 画面突变，禁止
- 运动方向不要与首帧构图冲突（首帧主体在右侧，就别写 `pan left` 把主体推出画面）

词表见 [references/motion-vocab.md](references/motion-vocab.md)。
