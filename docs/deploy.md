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

## 一句话总结

**ComfyUI 侧的全部工作 = 放 3（~5）个文件 + （可选）clone 一个节点包 +
启动**，其余由项目代码通过 API 完成；`prompt2vid --check-only` 是验证配置
是否正确的唯一入口。
