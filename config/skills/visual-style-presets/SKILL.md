---
name: visual-style-presets
description: 全片视觉风格预设词表（anime / realistic / 自定义）与对应 checkpoint、LoRA、画幅档位的关系。当需要判定用户描述属于哪种风格、填写 requirement.json 的 style 字段、或画面风格在片段间漂移时使用。
license: MIT
metadata:
  author: prompt2vid
  version: "1.0"
compatibility: 与 src/prompt2vid/style.py 的 STYLE_PRESETS 同步
---

# 视觉风格预设

风格有**两层**，不要混淆：

1. **提示词层**（`PROMPT2VID_STYLE`）：代码自动给每镜正向/负向提示词追加风格词，
   并把风格约束注入分镜任务。不需要任何模型文件。
2. **模型层**（`COMFYUI_T2I_CHECKPOINT` + `COMFYUI_T2I_LORA` + `COMFYUI_LTX_LORA`）：
   底模与 LoRA 决定画面质感的"底色"。

## 预设对照

| 预设名 | 提示词正向（代码自动追加） | 负向（自动追加） | 适配底模 / LoRA |
| --- | --- | --- | --- |
| `anime` | `anime style, japanese animation, 2d cel shaded, clean line art, vibrant colors, detailed anime illustration` | `photorealistic, photograph, photo, realistic, 3d render, cgi, semi-realistic` | RealVisXL / SDXL Base + `anime.safetensors`（强度 1.0~2.0）；视频侧 13B 可加新海诚 LoRA |
| `realistic` | `photorealistic, cinematic film still, 35mm photography, natural lighting, shallow depth of field` | `anime, cartoon, illustration, painting, 2d, cel shaded, drawing` | RealVisXL V5.0 原生方向，不加 LoRA |
| 自定义 | 整个值按字面追加到正向提示词 | 空 | 自行保证与底模不冲突 |

## 填写 `requirement.json` 的 `style` 字段

`style` 是给人看的中文描述（如 `治愈系、暖色调、日系动漫`），
**不需要也不应该**改成上面预设名——风格由 `.env` 决定，这里只忠实反映
用户描述与画面取向。但必须做到：

- 与 `PROMPT2VID_STYLE` 不冲突（anime 档位不要写"真实照片质感"）
- 用具象取向词：`暖色调`、`低饱和`、`高对比`、`手绘感`，避免"高级感"这类空词

## 风格漂移的排查顺序

1. 看 `.env` 的 `PROMPT2VID_STYLE` 是否与需求一致（未设置 = 无约束）
2. 看每镜 `image_prompt` 结尾是否残留与档位冲突的媒介词
   （anime 档位里出现 `photograph`、`35mm`）
3. 看 `run.log` 里是否有 `LoRA ... 缺失，自动降级` 的告警（缺文件时只剩提示词约束）
4. 视频侧风格主要由**首帧图**传递；若首帧一致而片段风格漂移，检查
   `COMFYUI_LTX_LORA` 是否与 13B 底模匹配

## 分镜写作时的落地要求

- 风格词交给代码追加，**分镜的 image_prompt 只写内容与镜头**（见 `sdxl-prompt-engineering`）
- 同一部片里不要混用媒介：一半写 `35mm film still`、一半写 `watercolor` = 风格撕裂
- 光照与色彩取向全片统一（要么全是 `soft afternoon light`，要么全是 `neon glow`）

各档位完整配置见 [references/style-matrix.md](references/style-matrix.md)。
