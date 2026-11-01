---
name: delivery-qa-spec
description: 成片交付质检口径——quality_report.json 每项检查的判定标准、失败时的归因与修复建议、汇报格式。当需要执行或解读成片质检、判断能否发布、或质检未通过需要定位原因时使用。
license: MIT
metadata:
  author: prompt2vid
  version: "1.0"
compatibility: 与 src/prompt2vid/stages.py 的 run_quality_checks 实现一一对应
---

# 成片质检口径

## 1. 执行方式

调用阶段工具 `run_quality_checks`（无参数），它会在
`report/quality_report.json` 写出结果并返回同样的 JSON。**不要自己用 shell
或 ffmpeg 复核**——判定逻辑已在代码里，重复计算只会引入分歧。

## 2. 六项检查的含义与判定线

| 检查名 | 判定线 | 不合格常见原因 |
| --- | --- | --- |
| `final_video_exists` | `final/final.mp4` 存在且非空 | 渲染阶段失败，看 `final/render_report.json` |
| `duration_match` | 成片时长与需求/分镜时长差 ≤ 2s | 片段数不足、某镜生成失败被跳过、旁白超长被截断 |
| `resolution_ok` | 宽高均 ≥ 448 **且** 像素数 ≥ `PROMPT2VID_MIN_CLIP_PIXELS`（默认 512×768×0.9） | 首帧画幅太小、LTX 像素预算设得过低 |
| `audio_track_present` | 成片含音频轨 | 混流参数错误、所有音频缺失 |
| `voiceover_audible` | 至少一条旁白峰值音量 > -50dB | TTS 全部失败（网络/节点缺失），只剩静音占位 |
| `assets_complete` | frames / clips / audio 产物齐全 | 某阶段部分失败，重跑该阶段即可 |

> `resolution_ok` **不是**"必须 1080p"：竖屏 480×832 是合法档位（2B 模型）。

## 3. 汇报格式（必须包含）

1. 结论：`通过` / `未通过`
2. 逐项：`检查名 — passed 值 — detail 原文`（detail 里有具体数字，直接引用）
3. 未通过项：**归因 + 可执行修复**（见下表），不要只说"失败了"

## 4. 归因与修复对照

| 失败项 | 归因 | 修复动作 |
| --- | --- | --- |
| `final_video_exists` | 渲染中断 | `prompt2vid --resume <run> --only edit_video --force` |
| `duration_match` | 分镜时长与需求不符 / 片段缺失 | 先查 `clips/manifest.json` 的 failed 列表，再补跑该阶段 |
| `resolution_ok` | 首帧画幅不符竖屏档位 | 检查 `COMFYUI_T2I_WIDTH/HEIGHT` 与 `COMFYUI_LTX_MAX_PIXELS` |
| `audio_track_present` | 音频全缺 | 先跑配音阶段，再重跑渲染 |
| `voiceover_audible` | TTS 未产出真实语音 | 装 IndexTTS-2.5 节点 / `pip install piper-tts` / 配 TTS 网关，见 `docs/LOCAL_SETUP.md` |
| `assets_complete` | 某阶段部分失败 | 按 detail 里列出的缺失文件补跑对应阶段 |

## 5. 与发布的关系

- `passed=false` → **不要发布**，先汇报并给出修复建议
- `passed=true` → 可以进入发布步骤
- 允许的例外：仅 `resolution_ok` 因档位设定失败且用户明确接受时，可在汇报里
  标注"已知偏差，用户确认后发布"

## 6. 不要做的事

- 不要篡改 `quality_report.json` 让检查"通过"
- 不要重复调用 `run_quality_checks` 期望结果变化（结果是确定性的）
- 不要把 `voiceover_audible` 的失败说成"音轨缺失"（两者含义不同）
