# 本地运行配置指南（ComfyUI 页面可访问之后）

> 适用环境：**RTX 4060 8GB 显存 + 64GB 内存**，Windows，ComfyUI 已启动并可通过
> `http://127.0.0.1:8188` 打开工作流页面。
>
> ComfyUI 页面能打开只说明**服务本体**就绪，距离 `prompt2vid` 一键跑通还差 4 件事：
> **① 模型文件（本文重点）② ffmpeg ③ .env 配置核对 ④ ComfyUI 启动参数调优**。
> 每完成一步都可以用 `prompt2vid --check-only` 复查，全部 OK 即可直接运行。

```
快速自查（在项目根目录执行）：
prompt2vid --check-only
```

预检会逐项报告：ffmpeg/ffprobe、ComfyUI 可达性、两个 checkpoint、T5 文本编码器、
LTXV 节点是否齐备，并对每个 NG 项给出修复提示。本文就是把这些 NG 项逐个变 OK 的操作手册。

---

## 1. 硬件预算总览（8GB 显存 / 64GB 内存）

| 生成阶段 | 模型 | 显存占用 | 内存占用 | 8GB 结论 |
|---------|------|---------|---------|----------|
| 文生图 SDXL（768×1344，30 步） | sd_xl_base_1.0（fp16 权重 ~6.9GB） | ~6.5-7GB | 少量 | ✅ 可跑（紧张但可行，ComfyUI 自动启用注意力优化；OOM 见 §5.3） |
| 图生视频 LTX 2B（512×768，≤121 帧） | ltx-video-2b-v0.9.5（fp16 权重 ~2GB） | ~3-4GB | 少量 | ✅ 富余 |
| LTX 文本编码 | T5-XXL **fp8**（放 CPU，配置默认） | **0**（不走显卡） | ~5GB | ✅ 64GB 内存富余 |
| 两个阶段切换 | — | ComfyUI 自动卸载旧模型换新模型 | 卸载时 ~7GB | ✅ 64GB 内存足够做 offload 缓冲 |

**磁盘预算**：模型合计约 **13.8GB**；每次运行的产物（frames/clips/audio/final）
每 10 秒成片约 100-300MB，建议预留 50GB+ 空闲磁盘。

**时间预算（粗估）**：每镜文生图 15-35s（25 步）；每镜图生视频
（41-105 帧、**10 步蒸馏采样**）约 0.5-2 分钟；一条 5 镜、约 15 秒的
短视频，端到端约 8-20 分钟（`--direct-media` 直连模式再省数分钟 LLM
往返）。

> **关键约束 1**：T5 必须用 **fp8 量化版**（`t5xxl_fp8_e4m3fn.safetensors`）。
> fp16 版约 9GB+，加载即 OOM。fp8 版放在 CPU 上跑（本项目工作流默认
> `COMFYUI_T5_DEVICE=cpu`），编码速度足够（每镜只编码一次）。
>
> **关键约束 2**：LTX 帧数必须是 **8n+1**（9/17/…/121），且 8GB 显存下建议
> 单镜 ≤121 帧（约 4.8s @25fps）。本项目代码已自动按时长换算并强制这两个约束，
> 无需手工干预（`COMFYUI_LTX_MAX_FRAMES` 可调）。

---

## 2. 模型清单（必装 3 个）

三个模型与本项目工作流模板（`src/prompt2vid/comfyui_client.py`）及 `.env`
默认值一一对应，**文件名保持下表原名即可，无需改任何配置**：

| # | 模型 | HF 仓库 | 文件名 | 大小 | 放置路径（ComfyUI 安装目录下） | 对应 .env 变量（默认值即文件名） |
|---|------|---------|--------|------|-------------------------------|----------------------------------|
| 1 | SDXL Base 1.0（分镜首帧文生图） | `stabilityai/stable-diffusion-xl-base-1.0` | `sd_xl_base_1.0.safetensors` | ~6.9GB | `ComfyUI/models/checkpoints/` | `COMFYUI_T2I_CHECKPOINT=sd_xl_base_1.0.safetensors` |
| 2 | LTX-Video 2B 0.9.5 distilled（图生视频） | `Lightricks/LTX-Video` | `ltx-video-2b-v0.9.5.safetensors` | ~2.0GB | `ComfyUI/models/checkpoints/` | `COMFYUI_LTX_CHECKPOINT=ltx-video-2b-v0.9.5.safetensors` |
| 3 | T5-XXL fp8（LTX 文本编码器） | `comfyanonymous/flux_text_encoders` | `t5xxl_fp8_e4m3fn.safetensors` | ~4.9GB | `ComfyUI/models/text_encoders/` | `COMFYUI_T5=t5xxl_fp8_e4m3fn.safetensors` |

说明：

- **不需要单独下载 VAE**：LTX 的 VAE 内置于其 checkpoint；SDXL 的 VAE 内置于
  base 模型。也不需要 refiner、LoRA、上采样器。
- **SDXL 与 LTX 是两个独立 checkpoint**，都放 `models/checkpoints/`，ComfyUI
  按工作流里的 `ckpt_name` 精确匹配文件名加载。
- **T5 属于文本编码器目录**（`models/text_encoders/`），不是 checkpoints。
  工作流用 `CLIPLoader(type=ltxv)` 加载它，ComfyUI 会在 `text_encoders/` 找。
- 若想换用其他模型（见 §6 可选替换），只需改 `.env` 对应变量为实际文件名，
  代码无需改动。

以 Windows 便携版 ComfyUI 为例，最终目录结构应为：

```
D:\ComfyUI_windows_portable\
└── ComfyUI\
    └── models\
        ├── checkpoints\
        │   ├── sd_xl_base_1.0.safetensors        (~6.9GB)
        │   └── ltx-video-2b-v0.9.5.safetensors   (~2.0GB)
        └── text_encoders\
            └── t5xxl_fp8_e4m3fn.safetensors      (~4.9GB)
```

---

## 3. 下载命令（huggingface-cli + hf-mirror 镜像）

> **网络提示**：本机所在网络直连 huggingface.co 会超时，全程走
> **hf-mirror.com 镜像**（实测可达）。huggingface-cli 支持断点续传，
> 下载中断后重跑同一条命令即可接着下。

### 3.1 安装 HF 命令行工具

```powershell
pip install -U "huggingface_hub[cli]"
huggingface-cli version   # 验证可用
```

### 3.2 设置镜像端点（PowerShell，同一会话内生效）

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

> 每次新开 PowerShell 窗口都要先执行这一句再下载。想一劳永逸可写入系统
> 环境变量（"编辑系统环境变量" → 新建 `HF_ENDPOINT`）。

### 3.3 下载三个模型（把 `<ComfyUI模型目录>` 替换为你的实际路径）

```powershell
# 1. SDXL Base 1.0 → models/checkpoints/
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 `
  sd_xl_base_1.0.safetensors `
  --local-dir "D:\ComfyUI_windows_portable\ComfyUI\models\checkpoints"

# 2. LTX-Video 2B 0.9.5 → models/checkpoints/
huggingface-cli download Lightricks/LTX-Video `
  ltx-video-2b-v0.9.5.safetensors `
  --local-dir "D:\ComfyUI_windows_portable\ComfyUI\models\checkpoints"

# 3. T5-XXL fp8 → models/text_encoders/
huggingface-cli download comfyanonymous/flux_text_encoders `
  t5xxl_fp8_e4m3fn.safetensors `
  --local-dir "D:\ComfyUI_windows_portable\ComfyUI\models\text_encoders"
```

> 新版工具的等价命令是 `hf download <仓库> <文件> --local-dir <目录>`，
> 参数完全一致。

### 3.4 可选：开启 hf_transfer 多线程加速

```powershell
pip install -U hf_transfer
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"   # 与 HF_ENDPOINT 同一会话设置
```

> 若加速模式下镜像报错/速度异常，去掉该变量回到普通模式即可（普通模式
> 也支持断点续传）。

### 3.5 备选：浏览器/下载工具直链

不想用命令行时，可直接用直链下载（支持断点续传的下载器更佳），下载后
按 §2 的路径手工放置：

```
https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
https://hf-mirror.com/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.5.safetensors
https://hf-mirror.com/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors
```

### 3.6 下载后核验文件大小

```powershell
Get-ChildItem "D:\ComfyUI_windows_portable\ComfyUI\models\checkpoints", `
             "D:\ComfyUI_windows_portable\ComfyUI\models\text_encoders" | `
  Select-Object Name, @{n='GB';e={[math]::Round($_.Length/1GB,2)}}
```

| 文件 | 期望大小 |
|------|---------|
| sd_xl_base_1.0.safetensors | ≈ 6.94 GB |
| ltx-video-2b-v0.9.5.safetensors | ≈ 1.99 GB |
| t5xxl_fp8_e4m3fn.safetensors | ≈ 4.89 GB |

大小明显偏小说明下载不完整，删除后重新执行下载命令（会自动续传/重下）。

---

## 4. 验证模型就位

三种方式任选，推荐用最后一条一键核验：

1. **ComfyUI 页面**：刷新浏览器，新建 CheckpointLoaderSimple 节点，下拉框应
   出现 `sd_xl_base_1.0.safetensors` 与 `ltx-video-2b-v0.9.5.safetensors`；
   CLIPLoader 节点下拉框应出现 `t5xxl_fp8_e4m3fn.safetensors`。
2. **API**：
   ```powershell
   curl.exe http://127.0.0.1:8188/object_info/CheckpointLoaderSimple
   curl.exe http://127.0.0.1:8188/object_info/CLIPLoader
   ```
   返回的 JSON 中应包含对应文件名。
3. **本项目预检**（同时检查模型与 ffmpeg）：
   ```powershell
   prompt2vid --check-only
   ```

> 若页面下拉框没出现新模型：ComfyUI 对模型目录有缓存，刷新页面或重启
> ComfyUI 即可；确认文件放在 `ComfyUI/models/` 而不是 portable 根目录。

---

## 5. 模型之外的其余配置

### 5.1 ffmpeg（必装，本机当前缺失）

剪辑/质检阶段依赖 `ffmpeg` 与 `ffprobe`（预检已报 NG 的第一项）。Windows 安装：

1. 从 <https://www.gyan.dev/ffmpeg/builds/> 下载 **ffmpeg-release-essentials.zip**
   （约 80MB，含 ffmpeg.exe 与 ffprobe.exe）；
2. 解压到固定目录，如 `C:\ffmpeg`（结构为 `C:\ffmpeg\bin\ffmpeg.exe`）；
3. 加入 PATH："编辑系统环境变量" → Path → 新建 `C:\ffmpeg\bin`；
4. **重开 PowerShell** 验证：
   ```powershell
   ffmpeg -version
   ffprobe -version
   ```

> 任一来源的静态构建均可（BtbN GitHub Releases 的 win64 gpl 构建也行），
> 唯一要求：`ffmpeg` 和 `ffprobe` 两个 exe 都在 PATH 里。
> Docker/k8s 部署无需此步（镜像内置）。

### 5.2 .env 配置核对

项目根目录 `.env`（本地运行形态）：

| 变量 | 应填值 | 说明 |
|------|--------|------|
| `COMFYUI_URL` | `http://127.0.0.1:8188` | 本地 ComfyUI（k8s 才用 `...svc.cluster.local`），当前已正确 |
| `COMFYUI_T2I_CHECKPOINT` | `sd_xl_base_1.0.safetensors` | 与 §2 文件名一致，默认即正确 |
| `COMFYUI_LTX_CHECKPOINT` | `ltx-video-2b-v0.9.5.safetensors` | 同上 |
| `COMFYUI_T5` | `t5xxl_fp8_e4m3fn.safetensors` | 同上 |
| `COMFYUI_T5_DEVICE` | `cpu` | 8GB 显存必配（默认已是），T5 走内存 |
| `COMFYUI_RETRIES` | `2` | 单镜生成失败自动重试次数 |
| `COMFYUI_LTX_FPS` / `COMFYUI_LTX_MAX_FRAMES` | `25` / `121` | 帧率与单镜帧数上限（8n+1 自动保证） |
| `PROMPT2VID_MODEL` | 如 `openai/qwen3.8-27b` | **注意加 LiteLLM 提供商前缀**（自建 OpenAI 兼容网关用 `openai/` 前缀），否则 LiteLLM 可能无法路由 |
| `PROMPT2VID_LLM_BASE_URL` / `_API_KEY` | 网关地址 / 密钥 | 你已配置好内网网关，无需改动 |
| `DOUYIN_DRY_RUN` | `true` | 无抖音凭据时保持 true，只记录不真实上传 |
| `PROMPT2VID_TTS_*` | （可选） | edge-tts 被公司代理拦截时配一个 OpenAI 兼容 `/audio/speech` 网关；不配则自动落静音占位，流水线不中断 |

### 5.3 ComfyUI 启动参数建议（8GB）

- **默认参数即可跑通**。文生图与图生视频分开提交（整批一起进 ComfyUI
  队列串行执行），ComfyUI 会在切换时自动把上一个模型从显存卸载
  （64GB 内存做 offload 缓冲非常充裕）。
- 若文生图（768×1344）偶发 OOM，给启动命令加 `--lowvram`
  （便携版改 `run_nvidia_gpu.bat` 里的参数；速度会略降，但保证不 OOM）：
  ```
  python main.py --lowvram
  ```
- 无需 `--disable-smart-memory`、`--cpu-vae` 等激进选项。

**字幕中文字体注意（Windows 本地）**：Docker 镜像内置 `fonts-noto-cjk`，
但本地 Windows + gyan.dev 构建的 ffmpeg 烧录中文字幕依赖系统字体，
极少数情况下 libass 找不到 CJK 字体会输出方块。成片出来后请人工检查
`final/final.mp4` 的字幕；若有乱码，在 `.env` 显式指定字体：
```
PROMPT2VID_SUBTITLE_STYLE=FontName=Microsoft YaHei,FontSize=18,Outline=1,MarginV=40
```

**性能与复现（可选调优）**：
- `COMFYUI_LTX_STEPS=10`（默认）与 `COMFYUI_T2I_STEPS=25`（默认）：
  LTX 0.9.5 为蒸馏版，8-13 步即可，30 步只会白慢 2-3 倍；
- `PROMPT2VID_SEED=42`：固定种子跑对比实验（逐镜自动 +1）；
- `PROMPT2VID_DIRECT_MEDIA=true` 或 CLI `--direct-media`：LLM 只做创意
  三阶段 + 发布，媒体阶段由代码直跑，每次 run 省 6 次 LLM 往返；
- `PROMPT2VID_BGM=<音频文件>`：可选背景音乐（循环低音量混入）。

### 5.4 ComfyUI 版本要求

需要 **2024-11 之后支持 LTX-Video 0.9.5 的版本**（含 LTXVImgToVideo /
LTXVScheduler / SaveWEBM 等节点）。你的 ComfyUI 能打开页面不代表节点齐备——
`prompt2vid --check-only` 会专门检查 LTXV 节点，版本过旧会提示升级。

---

## 6. 可选模型替换（非必需）

保持默认即可跑通；想换只在** aesthetics / 风格**上做文章时参考（换后需同步改
`.env` 对应变量为实际文件名）：

| 用途 | 可选模型 | 仓库/文件 | 8GB 显存适配 |
|------|---------|-----------|--------------|
| 文生图替换 SDXL | Juggernaut XL 9（写实风更好） | `RunDiffusion/Juggernaut-X-v9` → `Juggernaut-X_v9_RunDiffusionPhoto_v2.safetensors`，放 `checkpoints/` | ✅ 同为 SDXL 架构，占用相当 |
| 文生图替换 SDXL | DreamShaper XL / 动漫风 SDXL 系 | 任一 `*_xl_*.safetensors` | ✅ 同上 |
| 图生视频 | LTX 0.9.7 系列 | `Lightricks/LTX-Video`（0.9.7 distilled/dev 文件名不同，且新版本部分节点参数有变化） | ⚠️ 需同步调整工作流模板，当前流水线按 0.9.5 验证，不建议现阶段替换 |

> 谨慎项：LTX 新版本（0.9.7+）节点参数与 0.9.5 不完全兼容，本项目工作流
> 模板与单测均按 **0.9.5** 校验，换版本前需要改 `comfyui_client.py`。

---

## 7. 全部就绪后：一键跑通

```powershell
# 1. 终检（应全绿）
prompt2vid --check-only

# 2. 正式运行（默认自动预检）
prompt2vid "做一个关于秋天的治愈系短视频"

# 3. 推荐日常用法：直连媒体模式（更快更省 token）
prompt2vid "做一个关于秋天的治愈系短视频" --direct-media

# 4. 中断后续跑（已生成的帧/片段/配音自动跳过）
prompt2vid --resume output/run_<时间戳> --from render --publish

# 5. 磁盘治理：清理旧运行目录
prompt2vid --clean --keep 5
```

每次运行的排障日志在 `output/run_<时间戳>/run.log`（各阶段开始/结束、
耗时、失败原因）。产物在 `output/run_<时间戳>/`：`frames/`（分镜首帧）→
`clips/`（分镜视频）→ `audio/`（配音）→ `final/final.mp4`（成片 + 字幕）→
`report/quality_report.json`（质检报告）→ `.douyin/state.json`（dry-run
发布记录）。

### 常见问题速查

| 症状 | 原因与处置 |
|------|-----------|
| 预检报 `ComfyUI 不可达` | ComfyUI 未启动或端口非 8188；核对 `.env` 的 `COMFYUI_URL` |
| 预检报 checkpoint/T5 缺失 | 文件名或目录不对（§2/§3.6）；刷新 ComfyUI 页面或重启 |
| 预检报 LTXV 节点缺失 | ComfyUI 版本过旧，升级到 master |
| 文生图 CUDA OOM | 启动参数加 `--lowvram`（§5.3）；确认 T5 在 CPU（`.env` `COMFYUI_T5_DEVICE=cpu`） |
| 下载超时/中断 | 确认 `$env:HF_ENDPOINT` 已设；重跑同一命令自动续传；或改用 §3.5 直链 |
| 配音无声 | edge-tts 被代理拦截 → 配 `PROMPT2VID_TTS_BASE_URL`，或接受静音占位（流水线不中断） |
| LLM 调用失败 | 模型名加 `openai/` 前缀；确认网关地址与 key；内网网关已被自动加入 NO_PROXY |

---

## 附：一次性的完整下载脚本（PowerShell）

把路径替换成你的 ComfyUI 实际目录后整段执行：

```powershell
$ComfyUI = "D:\ComfyUI_windows_portable\ComfyUI"   # ← 改成你的路径

pip install -U "huggingface_hub[cli]"
$env:HF_ENDPOINT = "https://hf-mirror.com"

huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 `
  sd_xl_base_1.0.safetensors --local-dir "$ComfyUI\models\checkpoints"
huggingface-cli download Lightricks/LTX-Video `
  ltx-video-2b-v0.9.5.safetensors --local-dir "$ComfyUI\models\checkpoints"
huggingface-cli download comfyanonymous/flux_text_encoders `
  t5xxl_fp8_e4m3fn.safetensors --local-dir "$ComfyUI\models\text_encoders"

# 核对大小
Get-ChildItem "$ComfyUI\models\checkpoints", "$ComfyUI\models\text_encoders" |
  Select-Object Name, @{n='GB';e={[math]::Round($_.Length/1GB,2)}}

# 本项目终检
prompt2vid --check-only
```

k8s 部署的模型下载见 [deploy/comfyui/MODELS.md](../deploy/comfyui/MODELS.md)。
