# 配音升级（IndexTTS-2.5）+ 口型同步（LatentSync 1.6）实施方案

## 现状与方案结论

现状（已探明）：配音用 edge-tts，网络被拦时**静默降级**到 piper（明显机械）甚至静音占位（`tts.py:134` 的 `except: pass` 吞掉一切错误）；画面是 SDXL 首帧 + LTX 文本驱动图生视频，**与配音完全无关，人物不动嘴**，配音只是旁白铺在画面上。

两个能力都经现有 ComfyUI HTTP 管线接入（复用 `ComfyUIClient` 的提交/轮询/下载/上传，**不新增常驻服务、不新增 Python 依赖**）：

1. **配音**：接入 **IndexTTS-2.5**（B 站开源，中文自然度第一梯队，支持 59 个自带音色 + 自定义音色克隆），经 `chenpipi0807/ComfyUI-Index-TTS` 节点（772★，活跃维护）。约 12GB 显存。
2. **口型同步**：新增媒体阶段，用 **LatentSync 1.6**（字节跳动）对每镜 LTX 片段做音频驱动唇形重绘，经 `ShmuelRonen/ComfyUI-LatentSyncWrapper`（963★）+ `VideoHelperSuite`。约 20GB 显存，**4090 24GB 正好够**。

已核实的关键节点规格（API JSON 的 class_type / 输入 / 输出）：
- `IndexTTS25BaseNode`：必填 `text`、`reference_audio`(AUDIO)、`lang`(ZH)、`duration_factor`；可选 `interval_silence_ms`(默认200，句间停顿更自然)、`cache_control`；输出 `(AUDIO, seed, subtitle)`
- `TimbreAudioLoader`：`audio_file` 从插件自带 TimbreModel 目录选音色（含"甜美女声/新闻女声/说书人"等 59 个）
- `LatentSyncNode`：`images`(IMAGE)、`audio`(AUDIO)、`seed`、`lips_expression`(1.5)、`inference_steps`(20)；输出 `(IMAGE, AUDIO)`；要求 25fps（我们 LTX 默认就是 25fps）
- `VideoLengthAdjuster`：`mode=pingpong` 在配音更长时把视频往返补帧到配音长度（**不裁旁白**），`mode=normal` 处理视频更长的情况
- `VHS_LoadVideo`（force_rate=25 → IMAGE 帧）、`VHS_VideoCombine`（输出注册为 **"gifs"** 键——现有 `parse_output_files` 已兼容）；`SaveAudio` 输出 "audio" 键（需补支持）

## 代码改动

### 1. `src/prompt2vid/comfyui_client.py`
- `parse_output_files`：输出键元组 `("images","gifs","videos","images_preview")` 增加 `"audio"`（SaveAudio 产物）
- `upload_image` 泛化出 `upload_file`（音频/视频同样走 `POST /upload/image`，ComfyUI 不限扩展名）
- 新增 `node_exists(name)`（object_info 判定，供引擎探测/preflight/阶段 gating）
- 新增 `build_tts_workflow(text, filename_prefix, ...)`：`TimbreAudioLoader → IndexTTS25CacheControlNode(keep_models_cached) → IndexTTS25BaseNode → SaveAudio`；配置了自定义参考音频时改用 `LoadAudio`（上传后引用）
- 新增 `build_lipsync_workflow(clip_name, audio_name, mode, ...)`：`VHS_LoadVideo(force_rate=25) + LoadAudio → VideoLengthAdjuster(mode, fps=25) → LatentSyncNode → VHS_VideoCombine(format="video/h264-mp4")`
- 两个构建器的输入按服务端 `object_info` 自适应（只送节点实际声明的输入，兼容 VHS 版本差异——沿用 `resolve_checkpoint` 的防御式风格）

### 2. `src/prompt2vid/tts.py`（回退链重构）
- 新链：**comfyui(IndexTTS-2.5) → edge-tts → 网关 → piper → 静音**；`PROMPT2VID_TTS_ENGINE=auto|comfyui|edge|gateway|piper`（默认 auto：探测一次 IndexTTS 节点在不在，结果进程内缓存）
- **每级降级都 `logger.warning`**（修复静默降级——这正是"配音不自然"的可能原因之一）
- ComfyUI 引擎：提交 → wait → 下载 wav → ffmpeg 转 192k mp3（抽取与 piper 共用的转码 helper），保持 `audio/shot_<id>.mp3` 契约不变
- 新 env：`PROMPT2VID_TTS_TIMBRE`（默认 `甜美女声.mp3`）、`PROMPT2VID_TTS_REF_AUDIO`（自定义克隆参考音频，优先）、`PROMPT2VID_TTS_LANG=ZH`、`PROMPT2VID_TTS_SPEED=1.0`、`PROMPT2VID_TTS_SILENCE_MS=200`、`PROMPT2VID_TTS_KEEP_CACHED=true`（4090 上模型留显存，逐镜免重载）

### 3. `src/prompt2vid/stages.py`
- 新增 `sync_lips` 阶段（**配音之后、渲染之前**，不动现有阶段顺序）：
  - gating：`PROMPT2VID_LIPSYNC=auto|on|off`（默认 auto：LatentSync + VHS 节点齐全才跑，否则告警跳过、直接用原片段）
  - 逐镜：上传 `clips/shot_<id>.webm` 与 `audio/shot_<id>.mp3` → ffprobe 探测两者时长 → 配音更长选 `mode=pingpong`（补帧到配音长，旁白零截断）否则 `normal` → 批量提交（复用 `_batch_run`）→ 下载到 `lipsync/shot_<id>.mp4` + manifest
  - 失败降级：单镜失败（无正脸/动漫脸/OOM）**保留原片段**并记入 manifest；已有产物跳过（断点续跑）
- `render_final`：读 clips manifest 后用 lipsync manifest 覆盖 ok 镜头的视频路径（时长本来就按 ffprobe 实测，无需其他改动）；`render_report.json` 增加 lipsync 统计
- 新增 `LIPSYNC_DIR = "lipsync"` 常量

### 4. `pipeline.py` + `steps.py`
- `MEDIA_STAGES` 在 voiceovers 与 render 之间插入 `"lipsync"`；`_STAGE_FUNCS` 注册 `sync_lips`
- STEPS 注册表插入 `sync_lips`（标签"口型同步"，media，artifact `lipsync/manifest.json`，requires clips+audio manifest）；`STAGE_OUTPUT_DIRS` 加 lipsync 目录

### 5. `preflight.py`
非阻断检查：6 个自定义节点存在性（`IndexTTS25BaseNode`/`TimbreAudioLoader`/`LatentSyncNode`/`VideoLengthAdjuster`/`VHS_LoadVideo`/`VHS_VideoCombine`）+ 报告实际选用的 TTS 引擎；缺节点时给一行安装指引。

### 6. 配置与文档
- `.env.example`、`.env.4090`、`.env.5090`：新增全部变量（**4090/5090 默认全开**）；`.env.4060`：TTS 开、`PROMPT2VID_LIPSYNC=off`（LatentSync 需 ~20GB）
- `README.md` 阶段表/环境变量表；`docs/LOCAL_SETUP.md` 新增安装章节（两个 custom node 的 git clone + pip install + `TTS2_5_download.py` 模型下载（支持 hf-mirror）+ LatentSync 模型自动下载/整包两条路 + 各显卡显存对照）；`docs/QUALITY.md` 配音章节改写

### 7. 测试（沿用现有 mock 模式，无真实 ComfyUI/ffmpeg 自动 skip）
- `test_comfyui_workflow.py`：两个新 workflow 构建器（结构/链接/env 覆盖/JSON 可序列化/音色与参考音频两种形态）
- `test_comfyui_client.py`：`parse_output_files` 支持 "audio" 键；`upload_file`
- `test_tts.py`：comfyui 引擎成功路径、auto 探测失败回退 edge、降级告警（caplog）
- `test_stages.py`：sync_lips 成功/节点缺失跳过/单镜失败降级/断点续跑（FakeComfyUIClient 走真实 parse_output_files）
- `test_render.py`：渲染优先使用 lipsync 产物；`test_preflight.py`/`test_steps.py`：新检查与注册表

## 关键风险与对策
- LatentSync 只处理**真人正脸**；动漫风格（PROMPT2VID_STYLE=anime）或无脸镜头会失败 → 逐镜自动降级保原片段，manifest 如实记录
- LatentSync 1.6 模型 ~7GB（HF 下载，可能需镜像）→ 文档写明自动下载与 Google Drive 整包两条路
- 显存：阶段串行 + ComfyUI 自动换页；4090 全开，4060 默认只开 TTS
- 自定义节点版本差异 → object_info 运行时自适应输入

## 验收标准
- pytest 全绿；
- 4090 真机 `.env.4090` 跑完整 run：成片人物口型与配音对齐、配音为 IndexTTS-2.5；未装口型节点时流水线照常出片（带告警）。

## 用户侧一次性安装（将写入 LOCAL_SETUP.md）
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/chenpipi0807/ComfyUI-Index-TTS.git
pip install -r ComfyUI-Index-TTS/requirements.txt
python ComfyUI-Index-TTS/TTS2_5_download.py        # 模型 → models/IndexTTS-2.5
git clone https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper.git
pip install -r ComfyUI-LatentSyncWrapper/requirements.txt   # 首次运行自动下模型
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git   # 如未装
```