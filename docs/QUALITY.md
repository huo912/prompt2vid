# 画质调优指南（按显卡档位）

> 目标：同一句话描述，在三档显卡上都拿到该档位能达到的最好画面。
> 各档位的完整配置见根目录 `.env.4060` / `.env.4090` / `.env.5090`，
> 切换方式见 README「多 GPU 档位配置」。本文解释**每档为什么这么配**、
> 还有哪些可开的画质杠杆，以及常见画质问题的对症下药。

## 画质链路：参数在哪些环节生效

一句话成片经过 5 道质量关卡，每一道都有自己的质量参数：

| 环节 | 决定什么 | 关键参数 |
|------|----------|----------|
| ① LTX 文本编码 | 提示词理解 | `COMFYUI_T5`（fp8 必装）、`COMFYUI_T5_DEVICE` |
| ② 文生图（首帧） | 构图/风格/人物颜值上限 | `COMFYUI_T2I_*`（checkpoint、步数、CFG、LoRA、画幅 768x1344） |
| ③ 图生视频（LTX） | 运动质量与分辨率 | `COMFYUI_LTX_CHECKPOINT`、`COMFYUI_LTX_MAX_PIXELS`、cfg/步数（按模型自动） |
| ④ 中转（webm→mp4） | 中间产物保真 | `COMFYUI_WEBM_CRF`（默认 20）、`PROMPT2VID_CLIP_CRF`（默认 16） |
| ⑤ 成片渲染 | 最终交付清晰度 | `PROMPT2VID_UPSCALE`、`PROMPT2VID_VIDEO_CRF`（默认 18）、`PROMPT2VID_SMOOTH_FPS` |

首帧质量是视频质量的天花板：LTX 是图生视频，构图、人物长相、风格
基本由首帧决定，视频阶段只负责"动起来"。所以**低显存档优先把 T2I
参数拉满（它便宜），再用渲染阶段补分辨率**。

## 三档参数矩阵

| | 4060（8GB） | 4090（24GB） | 5090（32GB） |
|---|---|---|---|
| 图生视频模型 | LTX 2B 0.9.5 | LTX 13B 0.9.7 distilled fp8 | 同 4090（可换 dev fp8） |
| cfg / 步数（自动） | 3.0 / 10 | 1.0 / 8 | 1.0 / 8（dev: 3.0 / 30） |
| LTX 生成分辨率 | 约 480x832 | 约 576x1024 | 约 704x1248（720p 档） |
| 文生图 | SDXL Base，25 步，CFG 7 | RealVisXL V5，30 步，CFG 6 | 同 4090 |
| 角色一致性 IPAdapter | 关（显存不够） | 开 | 开 |
| 成片输出 | 720x1248（渲染放大） | 576x1024（可开放大） | 704x1248 原生 |
| T5 位置 | CPU | CPU | GPU |

## 各档位画质杠杆

### 4060（8GB）——"生成小、渲染补"

显存决定了 LTX 只能跑 2B 且分辨率约 480x832，这是硬上限（13B fp8
权重就有 15.7GB，放不下）。本档策略是**生成端省下的显存换成首帧质量，
分辨率缺口在渲染阶段补**：

- 首帧 768x1344 全尺寸生成（SDXL 在 8GB 上跑 25 步毫无压力），LTX
  从高清首帧起画，比直接低分辨率生成再放大干净；
- `PROMPT2VID_UPSCALE=720`（默认开）：渲染阶段 lanczos + 轻锐化放大
  到 720 宽。这是本档最重要的观感杠杆——480 宽直接投流会被平台
  播放器粗暴拉伸，主动放大 + 锐化明显更好；不新增细节但观感接近原生；
- `PROMPT2VID_SMOOTH_FPS=50`（可选开）：纯 CPU 补帧，不占显存。
  运镜/平移镜头流畅度提升肉眼可见；快速打斗类画面可能形变，按题材取舍；
- IPAdapter 关闭是对的：8GB 上 T2I+IPAdapter 大概率 OOM，角色一致性
  交给分镜表 `characters` 角色卡的文字契约（默认层）；
- 不要尝试把 `COMFYUI_LTX_MAX_PIXELS` 拉过 45 万：2B 在 8GB 上
  121 帧 × 更高分辨率会 OOM，重试也救不回来。

### 4090（24GB）——性价比最高的档

13B distilled fp8 + 576x1024 是本档甜点位：

- `COMFYUI_LTX_MAX_PIXELS=921600`（720p）在 24GB 上偏激进：13B 权重
  15.7GB + 720p×121 帧的激活值容易顶满，**个别镜头 OOM 再考虑**，
  预检会提示这个风险；
- 想要 720 宽输出又不冒 OOM 风险：保持 589824 + 开
  `PROMPT2VID_UPSCALE=720`（×1.25 放大痕迹很小）；
- T5 留 CPU：T5 只在每镜开头跑一次（几秒），搬上 GPU 省不了多少时间，
  却挤占 13B 的激活空间；
- 写实题材本档原生优势（RealVisXL V5）；动漫题材建议把
  `COMFYUI_T2I_CHECKPOINT` 换成 `animagine-xl-3.1.safetensors`
  （cagliostrolab/animagine-xl-3.1，6.9GB，动漫原生 SDXL），比
  "写实底模 + 动漫 LoRA 硬拗"风格纯正得多。

### 5090（32GB）——把预算花满

- `COMFYUI_LTX_MAX_PIXELS=921600`（默认开）：约 704x1248，达到
  720p 档；个别镜头 OOM 时回调 589824；
- 极致画质可换 dev 底模 `ltxv-13b-0.9.7-dev-fp8.safetensors`
  （下载命令见 `.env.5090` 头部）：cfg 3.0 / 30 步自动适配，画质是
  LTX 参考级，代价是单镜耗时约为蒸馏版 3-4 倍。注意动漫 LoRA
  （shinkai）通常按 distilled 训练，换 dev 后需重调
  `COMFYUI_LTX_LORA_STRENGTH`；
- T5 上 GPU（`COMFYUI_T5_DEVICE=default`，默认配置已开）。

## 通用质量参数（所有档位）

| 变量 | 默认 | 说明 |
|------|------|------|
| `COMFYUI_WEBM_CRF` | 20 | LTX 中转 WEBM 的 VP9 CRF。旧版本是 32，渐变/暗部色带明显；20 足够干净且文件仍小 |
| `PROMPT2VID_CLIP_CRF` | 16 | 中转 mp4 的 x264 CRF（concat 前）。它是中间产物还要再转一次，保持高质量避免两代有损叠加 |
| `PROMPT2VID_VIDEO_CRF` | 18 | 成片 x264 CRF。抖音会再压一次，给它越干净的源越好；追求体积可放到 20 |
| `PROMPT2VID_X264_PRESET` | medium | 成片编码预设。`medium` 比 `fast` 同画质省 10-15% 码率；赶时间可回 `fast` |
| `PROMPT2VID_UPSCALE` | 空 | 成片放大目标（`720` 或 `720x1280`）。lanczos + 轻锐化，字幕按最终分辨率烧录；源已达标自动跳过 |
| `PROMPT2VID_UPSCALE_SHARPEN` | 0.3 | 放大后锐化强度，0 关闭。动漫线条对锐化敏感，出现白边就调低 |
| `PROMPT2VID_SMOOTH_FPS` | 空 | CPU 补帧（minterpolate）。建议 50；快速运动画面可能形变 |
| `COMFYUI_T2I_CFG` | 7 | SDXL Base 用 7；RealVisXL 等写实系用 6 左右 |
| `COMFYUI_T2I_SAMPLER` / `_SCHEDULER` | dpmpp_2m / karras | 一般无需改 |
| `COMFYUI_LTX_MAX_SHIFT` / `_BASE_SHIFT` | 2.05 / 0.95 | LTX 噪声调度（官方 simple 工作流按 512-768p 调校）。720p 长镜头若出现结构漂移，可尝试 max_shift 2.2-2.4 |
| `PROMPT2VID_SEED` | 随机 | 固定种子做 A/B 对比（逐镜 +1），调参时务必固定，否则分不清是参数还是运气 |

## 配音（TTS）可用性与音质

旁白由 `tts.py` 的引擎链产出（`PROMPT2VID_TTS_ENGINE=auto` 按序尝试，
**每级降级都会写 run.log warning**——音色变差一定能发现，不再静默换）：

1. **IndexTTS-2.5**（ComfyUI 本地推理，中文自然度最佳；自带 59 个音色，
   支持克隆自定义声音，约需 12GB 显存。安装见
   [LOCAL_SETUP.md §7](LOCAL_SETUP.md)）；
2. **edge-tts**（微软公网语音，公司网络拦 WSS 时 403）；
3. **OpenAI 兼容网关**（`PROMPT2VID_TTS_BASE_URL` 指向 `/audio/speech` 接口）；
4. **piper 本地离线**（`pip install piper-tts`，音色 onnx 放 `models/tts/`，
   或 `PROMPT2VID_TTS_PIPER_MODEL` 显式指定）；
5. 静音占位（流水线不中断，但质检会报 `voiceover_audible` 不通过）。

### 音色与自然度调校（IndexTTS-2.5）

| 变量 | 默认 | 说明 |
|------|------|------|
| `PROMPT2VID_TTS_TIMBRE` | 甜美女声.mp3 | 插件自带音色（TimbreModel 目录共 59 个：新闻女声/说书人/御姐配音/温润男声/播音中年男/花甲奶奶…）；完整列表看 ComfyUI 节点下拉框 |
| `PROMPT2VID_TTS_REF_AUDIO` | 空 | 指向 3-15 秒干净人声（wav/mp3）时**克隆该音色**，优先于 timbre |
| `PROMPT2VID_TTS_SPEED` | 1.0 | 语速（>1 变慢，<1 变快） |
| `PROMPT2VID_TTS_SILENCE_MS` | 200 | 句间停顿毫秒：长句旁白的自然换气感，嫌拖沓可降到 100 |
| `PROMPT2VID_TTS_KEEP_CACHED` | true | 模型留显存逐镜免重载；低显存与其它阶段抢卡时设 false |

### 公司网络拦 WSS（edge-tts 403）时

首选**装 IndexTTS-2.5**（本地推理，不吃网络，见上）；轻量替代——本地
piper 离线配音（音质明显机械，仅救急）：

```bash
pip install piper-tts -i https://pypi.tuna.tsinghua.edu.cn/simple
mkdir -p models/tts
# 中文女声音色（约 63MB，走 hf-mirror）：
curl -L -o models/tts/zh_CN-huayan-medium.onnx \
  https://hf-mirror.com/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx
curl -L -o models/tts/zh_CN-huayan-medium.onnx.json \
  https://hf-mirror.com/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json
```

其它音色（男声等）在 [rhasspy/piper-voices](https://hf-mirror.com/rhasspy/piper-voices)
按 `zh/zh_CN/<名字>/medium/` 路径下载，`models/tts/` 只放一个 onnx 时自动启用。

想让 edge-tts 走代理：云机装个 HTTP 代理（如 `gost`/`tinyproxy`），
`ssh -L 8888:127.0.0.1:<代理端口> <云机>` 后设 `EDGE_TTS_PROXY=http://127.0.0.1:8888`。

> 注意：`models/` 已在 .gitignore（音色文件 60MB+，不要提交进仓库）。

## 口型同步（LatentSync）

配音阶段之后的增强媒体阶段：用每镜配音驱动片段人物唇形，嘴动与旁白
对上；配音比片段长时自动往返补帧（旁白零截断），渲染优先使用
`lipsync/shot_<id>.mp4`。约需 20GB 显存（4090/5090 档可用），安装见
[LOCAL_SETUP.md §8](LOCAL_SETUP.md)。

| 变量 | 默认 | 说明 |
|------|------|------|
| `PROMPT2VID_LIPSYNC` | auto | `auto`（节点齐全才跑）/ `on`（缺节点报错）/ `off` |
| `PROMPT2VID_LIPSYNC_EXPRESSION` | 1.5 | 唇形表现力（guidance scale）：1.0-1.5 平静语速自然；2.0-3.0 演讲式夸张 |
| `PROMPT2VID_LIPSYNC_STEPS` | 20 | 推理步数：20 够用；30-50 更慢更稳 |
| `PROMPT2VID_LIPSYNC_CRF` | 19 | 产物 mp4 的 x264 CRF |

> **重要**：LatentSync 只处理**真人正脸**——`PROMPT2VID_STYLE=anime` 的
> 动漫脸、侧脸、远景小脸、无人脸镜头会失败，该镜自动保留原片段
> （`lipsync/manifest.json` 记录原因，渲染不受影响）。想要口型生效：
> 用 `realistic` 风格，且分镜画面包含清晰的人物面部（近景/中景）。

## 镜头过渡（PROMPT2VID_TRANSITION）

默认镜头间 0.5s 叠化（`fade`），不再硬切。实现方式：非末段镜头用定格
末帧多垫过渡秒数，xfade 的重叠正好吃掉垫片——**音轨与字幕时间轴完全
不变**，旁白在叠化期间自然延续。相关变量：

| 变量 | 默认 | 说明 |
|------|------|------|
| `PROMPT2VID_TRANSITION` | 0.5 | 过渡秒数；`0` 关闭回硬切；自动按最短镜头一半封顶 |
| `PROMPT2VID_TRANSITION_TYPE` | fade | xfade 过渡名：`fadeblack`（黑场）/ `wipeleft`（左擦）/ `circleopen`（圆形展开）/ `slideleft` 等 |

节奏快的视频可降到 0.3；治愈系/风景类可到 0.8 更绵长。

## 进阶选项（需额外装模型/节点，按需启用）

1. **LTX 官方超分（真·分辨率翻倍）**：Lightricks 提供专用超分模型
   `ltxv-spatial-upscaler-0.9.7.safetensors`（分辨率 ×2）与
   `ltxv-temporal-upscaler-0.9.8.safetensors`（帧率 ×2），配合
   ComfyUI 内置的 `LTXVpyramidUpscale` 节点做二段式生成：先低分辨率
   出片，再超分到 720p/1080p。比 ffmpeg 放大多出真实细节，代价是
   显存与耗时（4090/5090 适用）。目前流水线未内置该阶段，可在
   ComfyUI 里对 `clips/shot_<id>.webm` 手动跑，或等后续版本集成。
2. **RIFE 补帧（GPU 加速补帧）**：装 ComfyUI-Frame-Interpolation
   自定义节点后用 RIFE 模型补帧，质量与速度都优于 ffmpeg minterpolate，
   适合批量开启；不装节点就用 `PROMPT2VID_SMOOTH_FPS` 的 CPU 方案。
3. **动漫原生底模**：`animagine-xl-3.1.safetensors`
   （cagliostrolab/animagine-xl-3.1）替代 SDXL Base/RealVisXL，动漫
   题材的首帧质量（线稿、上色、脸）提升立竿见影，显存占用与 SDXL 相同，
   三档通用。
4. **升级 0.9.8**：Lightricks 已发布 `ltxv-13b-0.9.8-distilled-fp8.safetensors`
   等新版模型（下载命令把 `.env` 里的文件名替换即可）；0.9.8 相比
   0.9.7 在文字渲染与运动稳定性上有改进，工作流节点兼容。

## 常见画质问题对症

| 症状 | 原因 | 处理 |
|------|------|------|
| 成片无旁白（有字幕没人声） | edge-tts 被网络拦截（403）→ 静音占位 | 本地 piper 离线配音（见上文 §配音）；质检的 `voiceover_audible` 会明确报出 |
| 成片有持续底噪/嗡声 | 旧版 loudnorm 把静音占位的量化噪声满增益放大 | 已修复（静音输入直接输出纯静音）；老运行用 `--resume --from render` 重渲染 |
| 镜头切换生硬 | 渲染默认硬切 | `PROMPT2VID_TRANSITION`（默认已开 0.5s 叠化，见上文 §镜头过渡） |
| 暗部/天空出现色带 | 中转 WEBM CRF 32 旧默认 | 已默认改 20；老运行目录用 `--force` 重跑 clips |
| 成片比预览糊 | 两代 CRF 23 转码叠加 | 已分层为 16/18；确认没把 CRF 手动调高 |
| 画面卡顿不跟手 | 25fps 原生帧率 | `PROMPT2VID_SMOOTH_FPS=50`（或 RIFE） |
| 人物被裁/比例失真 | 首帧与视频画幅不一致 | 保持 `COMFYUI_T2I_WIDTH/HEIGHT` 默认，勿手设 `COMFYUI_LTX_WIDTH` |
| 手指/肢体崩坏 | LTX 常见问题 | 已默认注入肢体稳定提示词（`COMFYUI_LTX_PROMPT_SUFFIX`）；13B 档明显好于 2B；换镜别硬重试，改运动描述 |
| 全片风格不统一 | LLM 提示词漂移 | `PROMPT2VID_STYLE` + LoRA（两层硬约束，见 style.py） |
| 换了参数看不出差别 | 种子随机 | `PROMPT2VID_SEED=42` 固定后 A/B |
| 抖音上传后画质暴跌 | 源分辨率太低 | 保证源 ≥544 宽（开 `PROMPT2VID_UPSCALE=720`），CRF ≤18 |
