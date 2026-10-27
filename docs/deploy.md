# ComfyUI 流水线配置指南（零基础版）

> 适用对象：**ComfyUI 理解程度为零**的读者。读完即可完成本项目所需的全部
> ComfyUI 侧配置。模型下载命令见 [LOCAL_SETUP.md](LOCAL_SETUP.md)（§3 下载命令、
> §6 角色一致性），本文专注"哪个节点用哪个模型、文件放哪里"。

## 核心结论：不需要在 ComfyUI 里"搭建"任何流水线

本项目把 ComfyUI 当**无头渲染引擎**用：两张工作流（文生图、图生视频）已经
写成代码（`src/prompt2vid/comfyui_client.py`），运行时通过 HTTP API 直接
提交给 ComfyUI 执行。ComfyUI 那个浏览器页面从头到尾不需要打开来配置。

只需建立一个概念：**工作流里的"节点"靠文件名引用模型**——代码把文件名传给
节点，节点去 ComfyUI 的 `models/` 目录下找同名文件。所以你要做的全部
"配置"就是：**把模型文件放到正确目录、文件名一字不差**。

## 一、模型文件放置总表（核心配置，就这一张）

以 ComfyUI 安装目录为根（如 `D:\ComfyUI_windows_portable\ComfyUI\`）：

| # | 文件名（必须一字不差） | 大小 | 放到 | 用途 | `.env` 对应变量（默认值=文件名，不改即可） |
|---|---|---|---|---|---|
| 1 | `sd_xl_base_1.0.safetensors` | ~6.9GB | `models/checkpoints/` | 分镜首帧文生图 + 角色定妆照 | `COMFYUI_T2I_CHECKPOINT` |
| 2 | `ltx-video-2b-v0.9.5.safetensors` | ~2.0GB | `models/checkpoints/` | 首帧转视频片段 | `COMFYUI_LTX_CHECKPOINT` |
| 3 | `t5xxl_fp8_e4m3fn.safetensors` | ~4.9GB | `models/text_encoders/` | 视频阶段的文字编码器（必须 fp8 版，fp16 版会 OOM） | `COMFYUI_T5` |
| 4（可选） | `ip-adapter-plus-face_sdxl_vit-h.safetensors` | ~0.5GB | `models/ipadapter/` | 角色一致性：参考注入适配器 | 无需配置（节点按预设名自动加载） |
| 5（可选） | `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` | ~2.5GB | `models/clip_vision/` | 角色一致性：参考图的图像编码器 | 无需配置 |

前 3 个必装（约 13.8GB）；4、5 只在想开启角色一致性时才装（约 3GB）。
下载命令（hf-mirror 镜像，PowerShell）见 [LOCAL_SETUP.md](LOCAL_SETUP.md)
§3.3 与 §6.2，可直接复制。

## 二、工作流① 文生图——节点与模型对照

代码在 `comfyui_client.py` 的 `build_t2i_workflow`，生成的节点链：

| 节点 | 作用 | 需要你配置什么 |
|---|---|---|
| CheckpointLoaderSimple | 加载 SDXL 主模型 | **文件 #1** 放对位置即可 |
| CLIPTextEncode（正/负） | 提示词 → 文本向量 | 无（用 SDXL 自带的 CLIP） |
| EmptyLatentImage | 768×1344 竖屏空白画布 | 无 |
| KSampler | 采样出图（25 步，代码传参） | 无 |
| VAEDecode → SaveImage | 解码、保存 PNG | 无 |
| LoadImage（仅角色一致性开启时出现） | 载入角色定妆照 | 无（定妆照由代码自动上传，不用你管） |
| IPAdapterUnifiedLoader（同上） | 按预设 `PLUS FACE (portraits)` 自动加载适配器 | **文件 #4、#5** 放对位置即可 |
| IPAdapter（同上） | 把定妆照特征注入采样（权重 0.8，代码传参） | 无 |

## 三、工作流② 图生视频——节点与模型对照

代码在 `build_i2v_workflow`，节点链对照 ComfyUI 官方 LTX 模板：

| 节点 | 作用 | 需要你配置什么 |
|---|---|---|
| CheckpointLoaderSimple | 加载 LTX-Video 主模型（自带 VAE，无需单独下载） | **文件 #2** |
| CLIPLoader | 加载 T5 文本编码器（`type=ltxv`、`device=cpu` 由代码传，省显存） | **文件 #3** |
| CLIPTextEncode（正/负） | 运动/画面提示词编码 | 无 |
| LoadImage → LTXVPreprocess | 载入首帧图 | 无（自动上传） |
| LTXVImgToVideo / LTXVConditioning / LTXVScheduler / KSamplerSelect / SamplerCustom | LTX 采样管线（帧数、步数全由代码传参） | 无，但**要求 ComfyUI 是 2024-11 之后的版本**（自带这些 LTXV 节点） |
| VAEDecode → SaveWEBM | 输出 webm 片段 | 无 |

## 四、自定义节点（唯一需要"安装"的东西，且可选）

ComfyUI 原生没有 IPAdapter 节点。要开角色一致性才需要装这一项：

```powershell
cd "<ComfyUI目录>\custom_nodes"
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
# 然后重启 ComfyUI
```

不装也不会报错：相关分镜自动回退为无参考生成，run.log 里有 warning。

## 五、从零开始的操作顺序

1. **装 ComfyUI**（Windows 便携版解压即用，或 git clone；版本须 2024-11 之后）；
2. **放 3 个必装模型**（上表 #1-#3，目录别放错：checkpoints / checkpoints / text_encoders）；
3. **（可选）装 IPAdapter 节点 + 模型 #4 #5**；
4. **启动 ComfyUI**——默认参数即可；文生图偶发 OOM 才加 `--lowvram`
   （便携版改 `run_nvidia_gpu.bat` 里的参数）；
5. **核对项目 `.env`**：`COMFYUI_URL=http://127.0.0.1:8188`，三个模型变量
   保持默认（=文件名）；开了角色一致性则加 `COMFYUI_IPADAPTER=true`；
6. **一键体检**：`prompt2vid --check-only`——它直接问 ComfyUI 要模型清单，
   逐项验证：ffmpeg/ffprobe、ComfyUI 可达、SDXL 就位、LTX 就位、T5 就位、
   LTXV 节点存在。**全 OK 即配置完成，不需要手动验证任何节点**；
7. 试跑：`prompt2vid "一个红衣少女在公园捡银杏" --direct-media`。此时可以
   打开 `http://127.0.0.1:8188` 看队列里任务在跑（纯围观，无需操作）。

## 六、出错时对照表

| 症状 | 原因 |
|---|---|
| 预检 NG「T2I/LTX checkpoint」 | 文件 #1/#2 没放 `models/checkpoints/`，或文件名与 `.env` 不一致 |
| 预检 NG「T5 文本编码器」 | 文件 #3 没放 `models/text_encoders/`，或下了 fp16 版（文件名不同） |
| 预检 NG「LTXV 节点」 | ComfyUI 版本太旧，升级 |
| 预检 NG「ComfyUI 可达」 | ComfyUI 没启动，或端口/`COMFYUI_URL` 不对 |
| run.log 出现「角色 … 定妆照生成/上传失败…回退」 | IPAdapter 节点或模型 #4/#5 未就位（不影响出片，只是没有参考） |

## 七、5090 高质量档：模型下载清单（第 2 档升级）

> 适用：租用 5090（32GB）跑高质量 1 分钟视频。本文第 1-6 节的标准档模型
> （SDXL Base + LTX 2B + T5）继续保留，旧文件不删——`.env` 一行即可在
> 快速档/高质量档之间来回切换。所有文件名均已对照 hf-mirror 仓库逐一核实。

### 7.1 准备（一次性，云机 shell）

```bash
pip install -U "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com          # 国内镜像
MODELS=/root/autodl-tmp/ComfyUI/models            # ← 改成你的 ComfyUI models 实际路径
mkdir -p "$MODELS"/{checkpoints,text_encoders,ipadapter,clip_vision}
```

### 7.2 模型清单（5 个文件，新增约 26GB）

| # | 用途 | 文件（已核实） | 大小 |
|---|---|---|---|
| ① | 首帧文生图（写实系升级） | `RealVisXL_V5.0_fp16.safetensors` | 6.94GB |
| ② | 图生视频（13B 升级） | `ltxv-13b-0.9.7-distilled-fp8.safetensors` | 15.7GB |
| ③ | 角色一致性 adapter | `ip-adapter-plus-face_sdxl_vit-h.safetensors` | 848MB |
| ④ | 角色一致性编码器 | `model.safetensors`（**下载后必须重命名**） | 2.53GB |
| ⑤ | 视频文本编码 | `t5xxl_fp8_e4m3fn.safetensors`（标准档已有则跳过） | 4.9GB |

### 7.3 下载命令

```bash
# ① 首帧文生图：RealVisXL V5.0（写实系，SDXL 架构，HF 免登录直下）
huggingface-cli download SG161222/RealVisXL_V5.0 \
  RealVisXL_V5.0_fp16.safetensors \
  --local-dir "$MODELS/checkpoints"

# ② 图生视频：LTX-Video 13B 蒸馏版 fp8（5090 首选）
huggingface-cli download Lightricks/LTX-Video \
  ltxv-13b-0.9.7-distilled-fp8.safetensors \
  --local-dir "$MODELS/checkpoints"

# ③ IPAdapter 适配器（注意：在 sdxl_models/ 子目录下）
huggingface-cli download h94/IP-Adapter \
  sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors \
  --local-dir /tmp/p2v-dl
mv /tmp/p2v-dl/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors \
   "$MODELS/ipadapter/"

# ④ CLIP Vision 编码器（仓库内文件名是 model.safetensors，必须重命名成统一加载器识别的名字）
huggingface-cli download h94/IP-Adapter \
  models/image_encoder/model.safetensors \
  --local-dir /tmp/p2v-dl
mv /tmp/p2v-dl/models/image_encoder/model.safetensors \
   "$MODELS/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
rm -rf /tmp/p2v-dl

# ⑤ T5（标准档已有就跳过；没有则下载）
[ -f "$MODELS/text_encoders/t5xxl_fp8_e4m3fn.safetensors" ] || \
huggingface-cli download comfyanonymous/flux_text_encoders \
  t5xxl_fp8_e4m3fn.safetensors \
  --local-dir "$MODELS/text_encoders"
```

### 7.4 下载后验证

```bash
ls -lh "$MODELS"/checkpoints/ "$MODELS"/ipadapter/ "$MODELS"/clip_vision/ "$MODELS"/text_encoders/
# 预期看到：
#   checkpoints/RealVisXL_V5.0_fp16.safetensors            6.9G
#   checkpoints/ltxv-13b-0.9.7-distilled-fp8.safetensors    16G
#   ipadapter/ip-adapter-plus-face_sdxl_vit-h.safetensors  848M
#   clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors 2.5G
#   text_encoders/t5xxl_fp8_e4m3fn.safetensors             4.9G
```

### 7.5 启用（项目 `.env` 三行 + 体检）

```ini
COMFYUI_T2I_CHECKPOINT=RealVisXL_V5.0_fp16.safetensors
COMFYUI_LTX_CHECKPOINT=ltxv-13b-0.9.7-distilled-fp8.safetensors
COMFYUI_IPADAPTER=true
# 首帧画幅（视频画幅会跟随首帧自动换算，不要固定 512x768）
COMFYUI_T2I_WIDTH=768
COMFYUI_T2I_HEIGHT=1344
# 13B 单帧像素预算：576x1024；想上 720p 可改 921600（约 704x1248）
COMFYUI_LTX_MAX_PIXELS=589824
```

```bash
prompt2vid --check-only     # 本地执行，会真实连云机验证模型就位
```

切回快速档：把 checkpoint 三行改回 `sd_xl_base_1.0.safetensors` /
`ltx-video-2b-v0.9.5.safetensors` / 注释掉 IPADAPTER 即可，同时把
`COMFYUI_LTX_MAX_PIXELS` 改回 `393216` 或直接删掉（会按 2B 自动选择）。

### 7.6 备注

- **断点续传**：任何一条中断后，重跑同一条命令即可接着下；
- **变体选择**：追求极限质量可换 `ltxv-13b-0.9.7-dev-fp8.safetensors`
  （同 15.7GB，步数多、更慢）；画质上限更高有 bf16 版
  `ltxv-13b-0.9.7-distilled.safetensors`（28.6GB，5090 32GB 也能装）；
- **13B 采样参数已自动化**：客户端按 checkpoint 文件名自动选择采样档位
  （distilled → cfg 1.0 / 8 步；13B dev → cfg 3.0 / 30 步；2B → cfg 3.0 /
  10 步），不再把 cfg 写死 3.0。需要手动微调时设
  `COMFYUI_LTX_CFG` / `COMFYUI_LTX_STEPS` 覆盖；
- **人物变形的主要根因**：首帧是 768x1344，而旧版视频工作流固定
  512x768（宽高比 0.571 → 0.667），LTX 会把首帧裁切/挤压后再生成，人物
  头脚被裁、肢体比例失真。现在 `generate_clips` 会读取首帧真实宽高，
  换算成 32 倍数、符合显存预算的画幅（2B → 480x832，13B → 576x1024）；
- **为什么是 RealVisXL 而不是 Juggernaut XL**：Juggernaut 的 HF 仓库受限
  登录（401），Civitai 下载需注册 + token；RealVisXL V5.0 是写实系同档
  水平且免登录直下。坚持要 Juggernaut 的话在 Civitai 生成 API Token 后：
  ```bash
  wget -O "$MODELS/checkpoints/Juggernaut-X_v9_RunDiffusionPhoto_v2.safetensors" \
    "https://civitai.com/api/download/models/<版本ID>?token=<你的TOKEN>"
  ```
- **本地 Windows 下载**：同样命令在 PowerShell 里跑，先执行
  `$env:HF_ENDPOINT = "https://hf-mirror.com"`；

## 八、AutoDL 租用 5090：从零安装 ComfyUI（含无卡模式，保姆级）

> 前提：采用「本地编排 + 云端 GPU」形态——**云机只跑 ComfyUI**，
> prompt2vid / LLM / 剪辑都在你本地电脑上跑，云机不需要装本项目代码。

### 8.1 租用实例与选镜像

1. 注册 [autodl.com](https://www.autodl.com) 并充值（先充 ¥20 足够跑通全流程）；
2. 「算力市场」筛选 **RTX 5090（32GB）**；区域优先选单价低的（西北等），
   同款卡不同区差 10~20%；
3. **镜像选择（最关键的一步）**：选带 **PyTorch 2.7+ / CUDA 12.8（cu128）**
   的框架镜像。5090 是 Blackwell 架构，旧 torch（cu124 及更早）没有对应
   内核，启动即报 `no kernel image is available`。若镜像列表没有 cu128 版，
   选 Miniconda / CUDA 12.8 基础镜像，后面 8.5 步会手动装 torch；
4. 数据盘默认即可（`/root/autodl-tmp`，关机保留；免费额度与扩容单价以
   平台页面为准，模型 + ComfyUI 约 30~45GB）；
5. 创建实例，记下控制台显示的 **SSH 登录指令**（形如
   `ssh -p 12345 root@west-1.autodl.com`）。

### 8.2 无卡模式开机（装环境不花 GPU 钱）

**无卡模式 = 以约 1 折价格开机，看得见 CPU/内存/磁盘但看不见 GPU**，
适合装环境、下模型这类不需要显卡的工作。

1. 实例创建后先**立即关机**（控制台「关机」按钮；新实例按量计费从开机
   起算，别空烧）；
2. 在关机状态点「开机」时选择**「无卡模式开机」**（约 ¥0.1~0.5/小时，
   具体以页面显示为准）；
3. 开机后 `nvidia-smi` 报错或没有输出是**正常现象**——此模式下 GPU 不可见，
   不影响 pip 安装和模型下载。

### 8.3 连入云机

```bash
# 本地终端（PowerShell / Git Bash 均可），指令在 AutoDL 控制台「复制 ssh 登录指令」
ssh -p 12345 root@west-1.autodl.com   # 端口和主机换成你自己的
```

不想用 SSH 也可以：控制台点 **JupyterLab**，在网页终端里执行同样的命令。

### 8.4 加速与目录约定（一次设好）

```bash
# AutoDL 免费学术加速：加速 GitHub / HuggingFace 等访问（本会话有效）
source /etc/network_turbo
# 用完关闭（避免影响其他下载）：unset http_proxy && unset https_proxy

# 约定：一切持久化内容放数据盘 /root/autodl-tmp（关机保留），
# 系统盘 /root 下的内容在释放实例/迁移时会丢
cd /root/autodl-tmp
```

### 8.5 安装 ComfyUI（无卡模式下完成）

```bash
# 1) 克隆 ComfyUI（GitHub 慢就先 source /etc/network_turbo）
cd /root/autodl-tmp
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 2) 验证 torch 是否满足 5090 要求（无卡模式下也可执行，只查版本不碰 GPU）
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# 要求：torch >= 2.7 且 cuda 为 12.8。不满足就重装（无需 GPU，无卡模式可做）：
pip install -U torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

# 3) 装 ComfyUI 依赖（已装好的 torch 会被自动复用，不会被降级）
pip install -r requirements.txt

# 4) 装 PyAV —— 本项目图生视频工作流的 SaveWEBM 节点输出 webm 必需
pip install av

# 5)（可选，角色一致性用）装 IPAdapter 自定义节点
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git \
  custom_nodes/ComfyUI_IPAdapter_plus
```

### 8.6 下载模型（无卡模式下完成）

按 [§一 模型放置总表](#一模型文件放置总表核心配置就这一张) 下载标准档
3 个模型；上高质量档则按 [§七 5090 高质量档下载清单](#七5090-高质量档模型下载清单第-2-档升级)
下载 5 个文件——两节的命令在云机 shell 里原样可用（记得先
`export HF_ENDPOINT=https://hf-mirror.com`）。

### 8.7 关机 → 有卡开机 → 启动 ComfyUI

```bash
# 1) 环境装完，控制台「关机」（无卡计费停止）
# 2) 重新开机（这次选「正常开机」，即有卡模式）
# 3) SSH 连入后验证 GPU 与 torch：
nvidia-smi                                          # 应显示 RTX 5090 32GB
python -c "import torch; print(torch.cuda.is_available())"   # 应输出 True

# 4) 启动 ComfyUI（32GB 显存，无需任何省显存参数）
cd /root/autodl-tmp/ComfyUI
nohup python main.py --listen 127.0.0.1 --port 8188 > comfyui.log 2>&1 &
tail -f comfyui.log        # 看到 "To see the GUI go to: http://127.0.0.1:8188" 即成功
```

> `nohup ... &` 让 ComfyUI 在你断开 SSH 后继续运行；只监听 127.0.0.1，
> 不暴露公网（配合下一步的 SSH 隧道使用）。

### 8.8 本地打通与体检

```bash
# 本地电脑常驻这条隧道（端口/主机换成你的）：
ssh -N -L 8188:127.0.0.1:8188 -p 12345 root@west-1.autodl.com

# 浏览器打开 http://127.0.0.1:8188 能看到 ComfyUI 界面（纯围观用）
# 项目体检（本地执行，.env 保持 COMFYUI_URL=http://127.0.0.1:8188 不变）：
prompt2vid --check-only
# 六项全 OK → 可以直接跑片
```

### 8.9 省钱与环境复用

- **开关机节奏**：创意三步（LLM）在云机关机状态跑；媒体五步开机跑
  （1 分钟视频约 40~70 分钟）；跑完立即关机。控制台可设**定时关机**兜底；
- **个人镜像**：环境装好后控制台「更多 → 保存镜像」——以后开新实例
  （同区域）直接选这个镜像，ComfyUI/依赖全免重装，只需挂回模型；
- **模型放数据盘**（`/root/autodl-tmp`）或 AutoDL 网盘，关机/换实例不丢；
- 详见 README「租用 GPU」相关讨论：轻度使用月成本 ¥10~30。

### 8.10 常见问题

| 症状 | 原因与处理 |
|---|---|
| ComfyUI 启动即 `no kernel image is available` | torch 是 cu124 及更早——回 8.5 第 2 步装 cu128 版 |
| 无卡模式下 `nvidia-smi` 报错 | 正常，无卡模式本来就没有 GPU；装环境不受影响 |
| `git clone` GitHub 超时 | 先 `source /etc/network_turbo`；仍不行用镜像 `git clone https://ghproxy.net/https://github.com/...` |
| 本地打不开 127.0.0.1:8188 | 隧道没建或断了：重跑 8.8 的 ssh -N 命令；确认云机 ComfyUI 进程活着（`tail comfyui.log`） |
| `--check-only` 报模型不存在 | 模型没下全或目录放错——对照 §一 总表 / §7.4 验证输出 |
| SaveWEBM 节点报错 | 没装 `pip install av`（8.5 第 4 步） |
| 断开 SSH 后 ComfyUI 停了 | 没用 nohup/tmux——按 8.7 第 4 步的命令启动 |

### 8.11 seetacloud 实例（SSH 隧道）专用配置

本节以 `ssh -p 44264 root@connect.westd.seetacloud.com` 为例，适用于
**云机只跑 ComfyUI、本地跑 prompt2vid** 的形态。与 AutoDL 的唯一区别
是登录主机/端口，其余步骤同 8.5–8.9。

#### 1）本地 `.env`：只改 `COMFYUI_URL`

```ini
# 云机 ComfyUI 只监听 127.0.0.1:8188，通过 SSH 隧道映射到本机同端口
COMFYUI_URL=http://127.0.0.1:8188
```

模型文件名必须与云机上实际文件一致；不确定时先列目录再填：

```bash
ssh -p 44264 root@connect.westd.seetacloud.com \
  'ls /root/autodl-tmp/ComfyUI/models/checkpoints /root/autodl-tmp/ComfyUI/models/text_encoders'
```

```ini
COMFYUI_T2I_CHECKPOINT=sd_xl_base_1.0.safetensors
COMFYUI_LTX_CHECKPOINT=ltx-video-2b-v0.9.5.safetensors
COMFYUI_T5=t5xxl_fp8_e4m3fn.safetensors
```

> 32GB 显存（5090/4090 等）跑 13B 高质量档时，把上面三行换成 §7.5 的
> `RealVisXL_V5.0_fp16.safetensors` / `ltxv-13b-0.9.7-distilled-fp8.safetensors`，
> 并加上 `COMFYUI_T2I_WIDTH=768` / `COMFYUI_T2I_HEIGHT=1344` /
> `COMFYUI_LTX_MAX_PIXELS=589824`；还可把 `COMFYUI_T5_DEVICE` 从 `cpu`
> 改为 `default`（T5 上 GPU，更快）。换模型后务必跑一次 `prompt2vid --check-only`。
>
> 文件名填错也不必慌：生成前客户端会查 `/object_info/CheckpointLoaderSimple`，
> 配置的 checkpoint 不在服务端候选列表时自动改用同类模型（T2I 跳过 LTX 视频
> 模型），并打印 `WARNING ... 自动改用 xxx`。这只是兜底，预检仍会把真实
> 文件名差异标出来，建议按 `--check-only` 的结果对齐 `.env`。

#### 2）本地常驻 SSH 隧道

```bash
# 前台保持（Ctrl+C 即断开）；端口/主机换成你自己的
ssh -N -L 8188:127.0.0.1:8188 -p 44264 root@connect.westd.seetacloud.com
```

建议写进 `~/.ssh/config`，以后一条 `ssh gpu-tunnel` 即可，并带心跳防止
空闲断线：

```sshconfig
Host gpu-tunnel
    HostName connect.westd.seetacloud.com
    Port 44264
    User root
    LocalForward 8188 127.0.0.1:8188
    ServerAliveInterval 30
    ServerAliveCountMax 3
    ExitOnForwardFailure yes
```

```bash
ssh -N gpu-tunnel
```

> 想免密码登录先执行 `ssh-copy-id -p 44264 root@connect.westd.seetacloud.com`。
> 本机 8188 已被占用时（如本地 ComfyUI），把 `-L 18188:127.0.0.1:8188`
> 并同步把 `.env` 的 `COMFYUI_URL` 改成 `http://127.0.0.1:18188`。

#### 3）云机侧：确认 ComfyUI 已启动

```bash
# SSH 进云机后（环境/模型按 8.5–8.6 准备）
curl -s http://127.0.0.1:8188/system_stats | head -c 200   # 有 JSON 即正常
# 没启动则：
cd /root/autodl-tmp/ComfyUI
nohup python main.py --listen 127.0.0.1 --port 8188 > comfyui.log 2>&1 &
tail -f comfyui.log
```

若平台提供公网端口映射，也可以跳过隧道，直接把 `COMFYUI_URL` 设为平台
给出的 `https://...` 地址（需 ComfyUI 以 `--listen 0.0.0.0` 启动）。

#### 4）本地验证

```bash
# 隧道建好后，本地应能打开 ComfyUI 页面
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8188/system_stats   # 200

prompt2vid --check-only
# 六项全 OK 即配置完成；模型项 NG 时按输出的“可用: ...”改名 .env 变量
```

> 注意：SSH 隧道断开后 `COMFYUI_URL` 立刻不可达，预检会报“ComfyUI 可达 NG”。
> 先重连隧道再跑，不需要改任何代码或 k8s 清单。云机关机后隧道也会失效，
> 重新开机后主机名/端口可能变化，记得同步更新 `.env` 与隧道命令。

## 一句话总结

**ComfyUI 侧的全部工作 = 放 3（~5）个文件 + （可选）clone 一个节点包 +
启动**，其余由项目代码通过 API 完成；`prompt2vid --check-only` 是验证配置
是否正确的唯一入口。



hf download stabilityai/stable-diffusion-xl-base-1.0 sd_xl_base_1.0.safetensors
  --local-dir "$MODELS/checkpoints"

hf download Lightricks/LTX-Video ltx-video-2b-v0.9.5.safetensors
  --local-dir "$MODELS/checkpoints"

hf download comfyanonymous/flux_text_encoders t5xxl_fp8_e4m3fn.safetensors
  --local-dir "$MODELS/text_encoders"



ssh -p 44264 root@connect.westd.seetacloud.com