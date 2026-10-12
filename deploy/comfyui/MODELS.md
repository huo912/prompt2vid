# ComfyUI 模型下载与放置说明

## 需要的模型（RTX 4060 8GB 显存预算）

| 模型 | 用途 | 放置路径（PVC `comfyui-models` 内） | 大小 | 下载地址（hf-mirror 镜像） |
|------|------|-----------------------------------|------|--------------------------|
| SDXL Base 1.0 | 分镜首帧文生图 | `checkpoints/sd_xl_base_1.0.safetensors` | ~6.9GB | https://hf-mirror.com/stabilityai/stable-diffusion-xl-base-1.0 |
| LTX-Video 2B 0.9.5 | 图生视频 | `checkpoints/ltx-video-2b-v0.9.5.safetensors` | ~2.0GB | https://hf-mirror.com/Lightricks/LTX-Video |
| T5-XXL fp8 | LTX 文本编码器 | `text_encoders/t5xxl_fp8_e4m3fn.safetensors` | ~4.9GB | https://hf-mirror.com/comfyanonymous/flux_text_encoders |

> **网络提示**：公司网络直连 huggingface.co 会 504 超时，上表已换成
> hf-mirror.com 镜像（实测可达）。也可用 modelscope.cn 等国内源。

> **显存警告**：T5 必须用 fp8 量化版（`t5xxl_fp8_e4m3fn.safetensors`）。
> fp16 版约 9GB+，8GB 显存必 OOM。`COMFYUI_T5_DEVICE=cpu`（默认）可把文本
> 编码放到内存（64GB 宿主机内存充足），进一步省显存。

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
