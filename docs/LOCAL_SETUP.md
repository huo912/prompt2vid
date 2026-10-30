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
>
> **可选增强的显存门槛**：配音升级 IndexTTS-2.5 约 12GB（§7），口型同步
> LatentSync 约 20GB（§8）——8GB 档两者都建议关闭（配音走回退链、
> `PROMPT2VID_LIPSYNC=off`）；24GB+（4090/5090 档）两者都开。

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
  base 模型。也不需要 refiner、上采样器。
- **动漫风格 LoRA（可选，48MB）**：`ntc-ai/SDXL-LoRA-slider.anime` 的
  `anime.safetensors` 放 `models/loras/`，配合 `.env` 的
  `PROMPT2VID_STYLE=anime` + `COMFYUI_T2I_LORA=anime.safetensors` 使用；
  文件不下载也不报错（自动降级为纯提示词风格约束）。
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
        ├── loras\                                 (可选，动漫风格)
        │   └── anime.safetensors                  (~48MB)
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

# 4.（可选）动漫风格 LoRA → models/loras/
huggingface-cli download ntc-ai/SDXL-LoRA-slider.anime `
  anime.safetensors `
  --local-dir "D:\ComfyUI_windows_portable\ComfyUI\models\loras"
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

#### 不想装系统包？两种免 root 方案

1. **随包二进制（推荐，离线可用）**：
   ```bash
   pip install -e ".[local-media]"   # 即 imageio-ffmpeg，自带静态 ffmpeg
   ```
   `imageio-ffmpeg` 不含 `ffprobe`，但程序探测媒体时**会自动回退解析
   ffmpeg 的流信息**，产出字段（时长/分辨率/fps/音轨）完全一致，
   剪辑与质检照常工作。
2. **手动指定路径**（自定义构建、容器内固定路径等）：
   ```bash
   PROMPT2VID_FFMPEG=/path/to/ffmpeg
   PROMPT2VID_FFPROBE=/path/to/ffprobe   # 可选，缺失时走 ffmpeg 回退
   ```
   程序查找顺序为：环境变量 → `PATH` → `~/.local/bin`、`/usr/local/bin`
   等常见目录 → 随包轮子（`static-ffmpeg` 含 ffprobe，`imageio-ffmpeg`
   仅 ffmpeg）。两处都找不到时会报错并提示设置哪个变量。

### 5.2 .env 配置核对

项目根目录 `.env`（本地运行形态）：

| 变量 | 应填值 | 说明 |
|------|--------|------|
| `COMFYUI_URL` | `http://127.0.0.1:8188` | 本地 ComfyUI（k8s 才用 `...svc.cluster.local`），当前已正确 |
| `COMFYUI_T2I_CHECKPOINT` | `sd_xl_base_1.0.safetensors` | 与 §2 文件名一致，默认即正确 |
| `COMFYUI_LTX_CHECKPOINT` | `ltx-video-2b-v0.9.5.safetensors` | 同上 |
| `COMFYUI_T5` | `t5xxl_fp8_e4m3fn.safetensors` | 同上 |
| `COMFYUI_T5_DEVICE` | `cpu` | 8GB 显存必配（默认已是），T5 走内存 |
| `PROMPT2VID_STYLE` | `anime` | 全片固定风格（动漫）。分镜提示词 + 每镜正负向提示词两层自动追加风格词；留空关闭，改为 `realistic` 可切写实 |
| `COMFYUI_T2I_LORA` / `COMFYUI_T2I_LORA_STRENGTH` | `anime.safetensors` / `1.5` | 文生图动漫 LoRA（§3.3 第 4 步下载，48MB）；滑块类 LoRA 权重建议 1.0-2.0；文件缺失自动降级为不挂 |
| `COMFYUI_LTX_LORA` | （空） | 视频侧 LoRA：LTX 2B 无现成动漫 LoRA，留空即可（视频风格由动漫首帧传递）；13B 档推荐见 docs/deploy.md §7 |
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
- `COMFYUI_LTX_STEPS` / `COMFYUI_LTX_CFG` 留空即可按模型自动选择：
  distilled → cfg 1.0 / 8 步；13B dev → cfg 3.0 / 30 步；2B 0.9.5 →
  cfg 3.0 / 10 步。手动设了才会覆盖自动档位；
- `COMFYUI_T2I_STEPS`：SDXL 默认 25 步（dpmpp_2m）；
- `COMFYUI_T2I_WIDTH` / `COMFYUI_T2I_HEIGHT`：首帧画幅，默认 768x1344；
  视频画幅会自动跟随首帧宽高比并取 32 的倍数，避免人物被裁变形；
- `PROMPT2VID_UPSCALE=720`：成片 lanczos 放大到 720 宽（`.env` 已默认
  开启）——8GB 档 LTX 只能生成约 480x832，渲染阶段补到 720 宽；
- `PROMPT2VID_SMOOTH_FPS=50`：可选 CPU 补帧，运镜更顺滑（快速运动
  可能形变，默认关闭）；
- `PROMPT2VID_SEED=42`：固定种子跑对比实验（逐镜自动 +1）；
- `PROMPT2VID_DIRECT_MEDIA=true` 或 CLI `--direct-media`：LLM 只做创意
  三阶段 + 发布，媒体阶段由代码直跑，每次 run 省 6 次 LLM 往返；
- `PROMPT2VID_BGM=<音频文件>`：可选背景音乐（循环低音量混入）。

> 更多画质杠杆（转码 CRF 分层、超分、RIFE、换底模）与对症速查见
> [QUALITY.md](QUALITY.md)。

### 5.4 ComfyUI 版本要求

需要 **2024-11 之后支持 LTX-Video 0.9.5 的版本**（含 LTXVImgToVideo /
LTXVScheduler / SaveWEBM 等节点）。你的 ComfyUI 能打开页面不代表节点齐备——
`prompt2vid --check-only` 会专门检查 LTXV 节点，版本过旧会提示升级。

---

## 6. 角色一致性（可选：IPAdapter 参考图注入）

> 作用：让分镜里反复出场的人物（主角）在每一镜都长得一样。分两层生效：
> ① **默认生效**——分镜规划时 LLM 产出 `characters` 角色卡，固定英文外观
> 描述逐字进入每镜 image_prompt（`storyboard.py` 解析时兜底补入）；
> ② **`COMFYUI_IPADAPTER=true` 时生效**——生图阶段先为每个角色生成一张
> **定妆照**（`frames/character_ref_<id>.png`），出镜分镜的首帧生成通过
> IPAdapter 注入该参考。第 ② 层需要按本节额外安装。

### 6.1 安装自定义节点

ComfyUI 原生没有 IPAdapter 节点，需要安装
[ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)：

```powershell
cd "<ComfyUI目录>\custom_nodes"
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
# 然后重启 ComfyUI
```

### 6.2 下载两个模型（约 3GB，走 hf-mirror）

| 模型 | HF 仓库 | 文件 | 放置路径 | 大小 |
|------|---------|------|---------|------|
| IPAdapter SDXL plus-face | `h94/IP-Adapter` | `ip-adapter-plus-face_sdxl_vit-h.safetensors` | `models/ipadapter/` | ~0.5GB |
| CLIP Vision ViT-H 图像编码器 | `h94/IP-Adapter` | `models/image_encoder/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` | `models/clip_vision/` | ~2.5GB |

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 1. IPAdapter adapter 权重 → models/ipadapter/
huggingface-cli download h94/IP-Adapter `
  ip-adapter-plus-face_sdxl_vit-h.safetensors `
  --local-dir "<ComfyUI模型目录>\ipadapter"

# 2. CLIP Vision 编码器 → models/clip_vision/
huggingface-cli download h94/IP-Adapter `
  models/image_encoder/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors `
  --local-dir "<ComfyUI模型目录>\clip_vision_tmp"
# 下载器会保留子目录结构，把文件挪到最终位置：
Move-Item "<ComfyUI模型目录>\clip_vision_tmp\models\image_encoder\CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors" `
  "<ComfyUI模型目录>\clip_vision\"
```

### 6.3 开启与验证

`.env` 中取消注释：

```ini
COMFYUI_IPADAPTER=true
# 可选微调：
# COMFYUI_IPADAPTER_PRESET=PLUS FACE (portraits)
# COMFYUI_IPADAPTER_WEIGHT=0.8
```

验证：跑一个带人物的 prompt（如"一个红衣少女在公园捡银杏"），
`run.log` 应出现「角色 … 定妆照已生成」，`frames/` 下多出
`character_ref_1.png`，出镜分镜的工作流携带 IPAdapter 节点。

> **容错**：未安装节点/模型不会报错中断——相关分镜自动回退为无参考生成
> （run.log 有 warning），所以可以先不开、以后随时补装。
>
> **显存**：adapter + CLIP Vision 常驻约 +1.5GB；SDXL 生图阶段峰值仍在
> 8GB 内（ComfyUI 自动权重 offload）。
>
> **增强**：需要更强的人脸身份保持时可换 FACEID 系列预设
> （`COMFYUI_IPADAPTER_PRESET=FACEID PLUS v2`），但要求 ComfyUI 环境
> 另装 insightface（`pip install insightface onnxruntime`）并下载对应
> FaceID 模型；默认的 `PLUS FACE (portraits)` 无额外依赖。

---

## 7. 配音升级（可选：IndexTTS-2.5 本地高音质配音）

> 作用：把配音从 edge-tts（微软公网语音）升级为 **IndexTTS-2.5**（B 站
> 开源，中文自然度第一梯队）：自带 59 个音色（甜美女声/新闻女声/说书人/
> 御姐配音/温润男声/播音中年男…），也支持用一段 3-15 秒的录音**克隆你
> 自己的声音**；句间自动停顿让旁白有自然换气感。约需 **12GB 显存**。
>
> 未安装时配音自动走回退链（edge-tts → OpenAI 网关 → piper → 静音），
> 每次降级 `run.log` 都有 warning——不再静默换音色。

### 7.1 安装自定义节点与模型

```powershell
cd "<ComfyUI目录>\custom_nodes"
git clone https://github.com/chenpipi0807/ComfyUI-Index-TTS.git

# 安装依赖（便携版用 python_embeded 的 pip）
pip install -r ComfyUI-Index-TTS\requirements.txt

# 下载 IndexTTS-2.5 模型（约 8-10GB → models/IndexTTS-2.5/，脚本内置 hf-mirror 支持）
python ComfyUI-Index-TTS\TTS2_5_download.py

# 重启 ComfyUI
```

> ⚠ **下载脚本对单步失败不中断**（打 `✗` 后继续跑其余步骤）：跑完务必看
> 末尾的校验输出，确认 6 个核心权重 `feat1.pt` / `feat2.pt` / `gpt.pth` /
> `codec.pth` / `s2mel.pth` / `wav2vec2bert_stats.pt` 都落在
> `models/IndexTTS-2.5/` 下——大文件最容易超时失败，恰好就是这批；缺了
> 它们节点会在推理时报 `FileNotFoundError: IndexTTS-2.5 missing files`。
> 定向补齐（Linux 服务器示例，Windows 把 `export` 换成 `set`）：
>
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> huggingface-cli download IndexTeam/IndexTTS-2.5 \
>   config.yaml codec.pth feat1.pt feat2.pt gpt.pth s2mel.pth \
>   wav2vec2bert_stats.pt multilingual_zh_ja_yue_char_del.tiktoken \
>   --local-dir "<ComfyUI目录>/models/IndexTTS-2.5"
> ```
>
> 注意：脚本会在模型目录里再留一份 `hf_cache/`（约双倍占盘），全部
> 校验通过后可删；克隆自定义音色时若报缺 `campplus_cn_common.bin`，
> 同样方式从 `funasr/campplus` 仓库补到模型根目录。

### 7.2 配置与验证

`.env` 无需改动（默认 `PROMPT2VID_TTS_ENGINE=auto`：探测到节点即用）。
想换音色或克隆自定义声音：

```ini
# 插件自带音色（TimbreModel 目录里的文件名，完整列表见该目录）
PROMPT2VID_TTS_TIMBRE=新闻女声.WAV
# 或者：克隆自己的声音（3-15 秒干净人声，wav/mp3 均可，优先于上面）
PROMPT2VID_TTS_REF_AUDIO=D:\voices\my_voice.wav
# 语速（>1 变慢）/ 句间停顿 / 模型留显存（默认开，整批逐镜免重载）
PROMPT2VID_TTS_SPEED=1.0
PROMPT2VID_TTS_SILENCE_MS=200
PROMPT2VID_TTS_KEEP_CACHED=true
```

验证：`prompt2vid --check-only` 应出现 `配音（IndexTTS-2.5）: OK`；
正式跑一次后听 `audio/shot_*.mp3`，`run.log` 无降级告警。

> **容错**：节点未安装/服务未启动 → 预检与 run.log 告警一次，配音走
> 回退链，流水线不中断。
>
> **显存**：推理峰值约 12GB；8GB 卡放不下（4060 档请继续用回退链），
> 24GB+ 无压力。`PROMPT2VID_TTS_KEEP_CACHED=false` 可让模型每次推理后
> 卸载，与其它阶段错峰用显存（代价是逐镜重载，变慢）。

---

## 8. 口型同步（可选：LatentSync 唇形驱动）

> 作用：新增的媒体阶段（配音与剪辑之间）把每镜片段与配音上传 ComfyUI，
> **LatentSync 1.6**（字节跳动）对人物嘴部做音频驱动重绘——人物"嘴动"
> 与旁白对上。配音比片段长时自动往返补帧延长（旁白零截断），渲染阶段
> 优先使用口型同步产物。约需 **20GB 显存**（4090/24GB 档可用，8-12GB
> 卡请在 `.env` 保持 `PROMPT2VID_LIPSYNC=off`）。
>
> ⚠ **只处理真人正脸**：动漫脸（`PROMPT2VID_STYLE=anime`）/ 侧脸 / 远景
> 小脸 / 无人脸镜头会失败——单镜失败自动保留原片段并在
> `lipsync/manifest.json` 记录，流水线不中断。想要口型生效，请用
> `realistic` 风格且分镜画面包含清晰人物面部。

### 8.1 安装自定义节点

```powershell
cd "<ComfyUI目录>\custom_nodes"

# LatentSync 包装节点
git clone https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper.git
pip install -r ComfyUI-LatentSyncWrapper\requirements.txt

# 视频加载/合成（很多环境已装；没有就补）
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
pip install -r ComfyUI-VideoHelperSuite\requirements.txt

# 重启 ComfyUI；首次运行时 LatentSync 模型（约 7GB）会自动从
# HuggingFace ByteDance/LatentSync-1.6 下载（国内先设 HF_ENDPOINT 镜像），
# 也可用该仓库 README 提供的 Google Drive 预配置整包
```

> ⚠ **两个环境前提**（缺一会在运行期报错）：
> 1. **PATH 里要有 `ffmpeg` 命令**：LatentSyncWrapper 启动节点时直接调
>    `ffmpeg -version` 检测——`imageio-ffmpeg` 的内置二进制**不算数**。
>    Windows 参考 §5.1 安装；Linux 服务器 `apt-get install -y ffmpeg`。
> 2. **上传的片段必须带音轨**：VHS_LoadVideo 的懒加载音频输出会在
>    ComfyUI 内存缓存扫描时提取音轨，无音轨视频直接崩（"Output file
>    does not contain any stream"）。LTX 生成的片段天然无声——流水线
>    已在上传前自动补静音轨（`ffmpeg_ops.ensure_audio_track`，幂等），
>    无需手工处理；仅手工往 ComfyUI input 塞视频时需注意。

### 8.2 配置与验证

`.env.4090` / `.env.5090` 默认已开启（`PROMPT2VID_LIPSYNC=auto`：节点
齐全才跑）。可调项：

```ini
PROMPT2VID_LIPSYNC=auto        # auto（默认）/ on（缺节点报错）/ off
PROMPT2VID_LIPSYNC_EXPRESSION=1.5   # 唇形表现力：1.0-1.5 自然，2.0-3.0 夸张
PROMPT2VID_LIPSYNC_STEPS=20    # 推理步数：20 够用，30-50 更慢更稳
PROMPT2VID_LIPSYNC_CRF=19      # 产物 mp4 质量（越小越清晰）
```

验证：`prompt2vid --check-only` 应出现 `口型同步（LatentSync）: OK`；
正式跑一次后 `lipsync/` 目录出现 `shot_*.mp4`，成片人物说话时嘴部与
配音对齐（`render_report.json` 的 `lipsync.shots` 记录生效镜头数）。

> **容错**：auto 模式缺节点 → 预检提示、渲染直接用原片段；单镜无人脸
> 失败 → 该镜保留原片段，其余镜头照常。
>
> **显存**：推理峰值约 20GB；与 LTX/SDXL 缓存共存时 ComfyUI 会自动
> 换页，若 OOM 可重启 ComfyUI 后单独重跑 `sync_lips` 步骤
> （`prompt2vid --resume output/run_xxx --only sync_lips`）。

---

## 9. 可选模型替换（非必需）

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

## 10. 全部就绪后：一键跑通

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
`clips/`（分镜视频）→ `audio/`（配音）→ `lipsync/`（口型同步片段，装好
节点时）→ `final/final.mp4`（成片 + 字幕）→ `report/quality_report.json`
（质检报告）→ `.douyin/state.json`（dry-run 发布记录）。

### 常见问题速查

| 症状 | 原因与处置 |
|------|-----------|
| 预检报 `ComfyUI 不可达` | ComfyUI 未启动或端口非 8188；核对 `.env` 的 `COMFYUI_URL` |
| 预检报 checkpoint/T5 缺失 | 文件名或目录不对（§2/§3.6）；刷新 ComfyUI 页面或重启 |
| 预检报 LTXV 节点缺失 | ComfyUI 版本过旧，升级到 master |
| 文生图 CUDA OOM | 启动参数加 `--lowvram`（§5.3）；确认 T5 在 CPU（`.env` `COMFYUI_T5_DEVICE=cpu`） |
| 下载超时/中断 | 确认 `$env:HF_ENDPOINT` 已设；重跑同一命令自动续传；或改用 §3.5 直链 |
| 配音无声/音色不对 | `run.log` 里的降级告警会说明实际用了哪个引擎：装 IndexTTS 节点（§7）、配 `PROMPT2VID_TTS_BASE_URL`、或设 `EDGE_TTS_PROXY` 走代理 |
| 口型同步整段被跳过 | 预检/日志提示缺节点 → 装 LatentSyncWrapper + VideoHelperSuite（§8）；或 `.env` 设 `PROMPT2VID_LIPSYNC=off` 明确关闭 |
| 个别镜头没有口型 | 该镜无真人正脸（动漫脸/远景/侧脸），LatentSync 无法处理 → 已自动保留原片段（`lipsync/manifest.json` 有记录），属预期行为 |
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
