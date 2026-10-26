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
```

```bash
prompt2vid --check-only     # 本地执行，会真实连云机验证模型就位
```

切回快速档：把这三行改回 `sd_xl_base_1.0.safetensors` /
`ltx-video-2b-v0.9.5.safetensors` / 注释掉 IPADAPTER 即可。

### 7.6 备注

- **断点续传**：任何一条中断后，重跑同一条命令即可接着下；
- **变体选择**：追求极限质量可换 `ltxv-13b-0.9.7-dev-fp8.safetensors`
  （同 15.7GB，步数多、更慢）；画质上限更高有 bf16 版
  `ltxv-13b-0.9.7-distilled.safetensors`（28.6GB，5090 32GB 也能装）；
- **为什么是 RealVisXL 而不是 Juggernaut XL**：Juggernaut 的 HF 仓库受限
  登录（401），Civitai 下载需注册 + token；RealVisXL V5.0 是写实系同档
  水平且免登录直下。坚持要 Juggernaut 的话在 Civitai 生成 API Token 后：
  ```bash
  wget -O "$MODELS/checkpoints/Juggernaut-X_v9_RunDiffusionPhoto_v2.safetensors" \
    "https://civitai.com/api/download/models/<版本ID>?token=<你的TOKEN>"
  ```
- **本地 Windows 下载**：同样命令在 PowerShell 里跑，先执行
  `$env:HF_ENDPOINT = "https://hf-mirror.com"`；
- 13B 的最佳采样参数（cfg 等）与 2B 有差异，见 README「角色一致性/模型
  切换」相关说明或联系项目维护者确认参数化改造进度。

## 一句话总结

**ComfyUI 侧的全部工作 = 放 3（~5）个文件 + （可选）clone 一个节点包 +
启动**，其余由项目代码通过 API 完成；`prompt2vid --check-only` 是验证配置
是否正确的唯一入口。
