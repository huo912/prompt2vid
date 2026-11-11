# ComfyUI 模型下载与放置说明

## 需要的模型（RTX 4060 8GB 显存预算）

| 模型 | 用途 | 放置路径（PVC `comfyui-models` 内） | 大小 | 下载地址（hf-mirror 镜像） |
|------|------|-----------------------------------|------|--------------------------|
| SDXL Base 1.0 | 分镜首帧文生图 | `checkpoints/sd_xl_base_1.0.safetensors` | ~6.9GB | https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0 |
| LTX-Video 2B 0.9.5 | 图生视频 | `checkpoints/ltx-video-2b-v0.9.5.safetensors` | ~2.0GB | https://hf-mirror.com/Lightricks/LTX-Video |
| T5-XXL fp8 | LTX 文本编码器 | `text_encoders/t5xxl_fp8_e4m3fn.safetensors` | ~4.9GB | https://hf-mirror.com/comfyanonymous/flux_text_encoders |
| IPAdapter SDXL plus-face（可选） | 角色一致性：首帧参考注入 | `ipadapter/ip-adapter-plus-face_sdxl_vit-h.safetensors` | ~0.5GB | https://hf-mirror.com/h94/IP-Adapter |
| CLIP Vision ViT-H（可选） | IPAdapter 图像编码器 | `clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors` | ~2.5GB | https://hf-mirror.com/h94/IP-Adapter（`models/image_encoder/` 下） |

> **网络提示**：公司网络直连 huggingface.co 会 504 超时，上表已换成
> hf-mirror.com 镜像（实测可达）。也可用 modelscope.cn 等国内源。

> **角色一致性（可选）**：镜像已内置 ComfyUI_IPAdapter_plus 自定义节点；
> 下载上表最后两个模型并在 prompt2vid 的 `.env` 设 `COMFYUI_IPADAPTER=true`
> 后，带角色的分镜首帧会注入角色定妆照参考。未下载时相关分镜自动回退为
> 无参考生成，不影响流水线。详见 docs/LOCAL_SETUP.md 第 6 节。

> **显存警告**：T5 必须用 fp8 量化版（`t5xxl_fp8_e4m3fn.safetensors`）。
> fp16 版约 9GB+，8GB 显存必 OOM。`COMFYUI_T5_DEVICE=cpu`（默认）可把文本
> 编码放到内存（64GB 宿主机内存充足），进一步省显存。

## MiniMax-H3 后端（`COMFYUI_VIDEO_BACKEND=minimax_h3`）

上表是 LTX 后端的最小集。切到 MiniMax-H3 后图生视频改用另一套模型（约
46GB，4090/24GB 下的 int8_convrot + nvfp4_awq 组合），一键脚本会自动下载、
软链进 ComfyUI 模型目录并校验 safetensors 头与文件大小（幂等，已就绪的
文件跳过）：

```bash
# 大文件落在共享存储，软链进 ComfyUI——不占系统盘
COMFYUI_DIR=/root/autodl-tmp/ComfyUI \
P2V_H3_MODEL_ROOT=/autodl-fs/data/models \
bash deploy/comfyui/download_minimax_h3_models.sh
```

| 文件 | 放置路径 | 大小 |
|------|----------|------|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `diffusion_models/` | ~20.9GB |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `text_encoders/` | ~15.7GB |
| `minimax_h3_video_vae_int8_convrot.safetensors` | `vae/` | ~2.8GB |
| `minimax_h3_audio_vae_fp32.safetensors` | `vae/` | ~0.6GB |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | `loras/` | ~2.0GB |
| `minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` | `loras/` | ~2.0GB |
| `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors` | `latent_upscale_models/` | ~0.69GB |

> 仓库是 `Comfy-Org/MiniMax-H3`；要求 ComfyUI ≥ 0.30.0。
>
> 最后那个放大模型来自另一个仓库 `LBH-123-AI/Minimax_h3_latent_Upscaler`，
> 由 ComfyUI 核心节点 `MiniMaxH3LatentUpscaleWithModel` 使用（`COMFYUI_H3_LATENT_UPSCALE=2`
> 时接采样后、解码前），治"低分出片糊"。选 3D conv v1（345M 参数）的原因：
> 它在作者训练过的 2×/2.25×/2.5×/4× 上更强（2× 时 PSNR 高 11%），而默认正好是 2×；
> 2D conv v1（182M）在其余倍率更好、且锐度分 t1–t6 可调（3D 的锐度≈t1 固定）。
> 软链后必须重启 ComfyUI——模型目录只在启动时扫描，断链时 loader 候选
> 里不会出现该文件。配置见 [`.env.minimax-h3.4090`](../../.env.minimax-h3.4090)。

### 空间布局（二选一）

| 布局 | 做法 | 适用 |
|------|------|------|
| 网盘 + 软链 | 默认：6 个文件都落在 `P2V_H3_MODEL_ROOT/minimax-h3`，`ComfyUI/models/` 里只是软链 | 数据盘小、要保留 LTX 后端 |
| 真实文件放数据盘 | 把 `diffusion_models/` + `text_encoders/` + `vae/`（约 40GB）直接 `mv` 进 `ComfyUI/models/`，LoRA 继续软链 | 只跑 H3，加载最快 |

下载脚本会识别"`ComfyUI/models` 里已是真实文件"的布局并跳过对应下载。

#### 只跑 H3 时的 50GB 数据盘预算

H3 全量 46GB + 文生图 checkpoint 6.9GB ≈ 53GB，装不进 AutoDL 50GB 数据盘，
所以要把 H3 用不到的部分挪到文件存储（`/autodl-fs/data`）：

| 项目 | 大小 | 处理 |
|------|------|------|
| H3 diffusion（`fl2va_pruned_int8_convrot`） | 20.9GB | 数据盘真实文件（加载最重） |
| H3 文本编码器（`qwen3vl_32b_nvfp4_awq`） | 15.7GB | 数据盘真实文件（加载最重） |
| `RealVisXL_V5.0_fp16`（`COMFYUI_T2I_CHECKPOINT`） | 6.9GB | **必须保留**，H3 档仍用它出首帧 |
| H3 video / audio VAE | 3.4GB | 数据盘真实文件 |
| H3 turbo LoRA ×2 | 4.0GB | 文件存储 + 软链 |
| H3 latent 放大模型 | 0.69GB | 文件存储 + 软链（`latent_upscale_models/`） |
| `ltxv-13b-*`、`t5xxl_*`、`ipadapter/`+`clip_vision/`、动漫 LoRA | 24GB | 文件存储：`/autodl-fs/data/models/unused/` |
| `ComfyUI-LatentSyncWrapper-main/checkpoints` | 5.6GB | 文件存储 + 整目录软链 |

- `unused/` 里带 `MANIFEST.txt` 与 `restore.sh`，切回 LTX/IPAdapter 档时
  `bash /autodl-fs/data/models/unused/restore.sh` 一键软链回来。
- LatentSync 的路径是相对 `base_dir` 解析的，所以直接软链整个 `checkpoints`
  目录即可，无需改代码；`PROMPT2VID_LIPSYNC=auto` 仍然生效。

模型目录只在启动时扫描，改完布局后用仓库脚本启动（先 `cd` 到 ComfyUI 根目录，
避免 `can't open file '.../output/main.py'`）：

```bash
bash deploy/comfyui/start_comfyui.sh          # GPU 启动，等 /system_stats 返回 200
bash deploy/comfyui/start_comfyui.sh status   # 查状态
bash deploy/comfyui/start_comfyui.sh stop     # 停
bash deploy/comfyui/start_comfyui.sh --cpu    # AutoDL 无卡模式：只能验证节点，跑不了推理
```

> AutoDL 无卡（CPU-only）模式下 `/usr/bin/nvidia-smi` 是 0 字节占位文件、
> `/dev/nvidia*` 不存在，此时 ComfyUI 能起、loader 下拉框能看到模型，
> 但 `torch.cuda.is_available()` 为 `False`，必须回控制台关机再以 GPU 模式开机。


## 下载方法

### 方法一：临时 Pod 下载（推荐）

```bash
kubectl -n comfyui run model-downloader --rm -it \
  --image=python:3.11-slim --restart=Never \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "model-downloader",
        "image": "python:3.11-slim",
        "command": ["bash", "-c", "pip install -q huggingface_hub && HF_ENDPOINT=https://hf-mirror.com python -c \"from huggingface_hub import hf_hub_download; import shutil; [shutil.move(hf_hub_download(repo_id=r, filename=f, local_dir=\"/dl\"), \"/dl/\" + f) for r, f in [(\"stabilityai/stable-diffusion-xl-base-1.0\", \"sd_xl_base_1.0.safetensors\"), (\"Lightricks/LTX-Video\", \"ltx-video-2b-v0.9.5.safetensors\"), (\"comfyanonymous/flux_text_encoders\", \"t5xxl_fp8_e4m3fn.safetensors\")]]"],
        "volumeMounts": [{"name": "models", "mountPath": "/dl"}]
      }],
      "volumes": [{"name": "models", "persistentVolumeClaim": {"claimName": "comfyui-models"}}]
    }
  }'
```

下载完成后按上表路径移动：

```bash
kubectl -n comfyui exec -it deploy/comfyui -- bash -c '
mkdir -p /app/ComfyUI/models/checkpoints /app/ComfyUI/models/text_encoders
mv /app/ComfyUI/models/sd_xl_base_1.0.safetensors /app/ComfyUI/models/checkpoints/ 2>/dev/null
mv /app/ComfyUI/models/ltx-video-2b-v0.9.5.safetensors /app/ComfyUI/models/checkpoints/ 2>/dev/null
mv /app/ComfyUI/models/t5xxl_fp8_e4m3fn.safetensors /app/ComfyUI/models/text_encoders/ 2>/dev/null
'
```

### 方法二：宿主机直接拷贝（单节点最简单）

```bash
# 找到 PVC 在宿主机的路径（local-path provisioner 通常在）
# /var/local-path-provisioner/<pvc-uid>/
kubectl -n comfyui get pvc comfyui-models -o jsonpath='{.spec.volumeName}'
kubectl get pv <volume-name> -o jsonpath='{.spec.local.path}'

# 直接在宿主机放置文件
mkdir -p <path>/checkpoints <path>/text_encoders
wget -O <path>/checkpoints/sd_xl_base_1.0.safetensors https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
wget -O <path>/checkpoints/ltx-video-2b-v0.9.5.safetensors https://hf-mirror.com/Lightricks/LTX-Video/resolve/main/ltx-video-2b-v0.9.5.safetensors
wget -O <path>/text_encoders/t5xxl_fp8_e4m3fn.safetensors https://hf-mirror.com/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors
```

## 验证

```bash
# ComfyUI 就绪后，检查模型列表
curl http://comfyui.comfyui.svc.cluster.local:8188/object_info/CheckpointLoaderSimple
# 应包含 sd_xl_base_1.0.safetensors 与 ltx-video-2b-v0.9.5.safetensors

curl http://comfyui.comfyui.svc.cluster.local:8188/object_info/CLIPLoader
# 应包含 t5xxl_fp8_e4m3fn.safetensors
```
