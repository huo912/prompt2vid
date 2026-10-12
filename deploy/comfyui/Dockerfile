# ComfyUI GPU 镜像（LTX-Video 就绪）
# 基础镜像：CUDA 12.4 运行时 + PyTorch（cu124）
# ComfyUI 固定到 master commit e8665ca8e4089ab917567cdb491308e1397663d4
# （该版本 nodes_lt.py 内置 LTX 节点、SaveWEBM 可用）

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# git（clone ComfyUI）+ ffmpeg（PyAV 依赖）+ 基础工具
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 固定 commit 的 ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI \
    && cd /app/ComfyUI \
    && git checkout e8665ca8e4089ab917567cdb491308e1397663d4

WORKDIR /app/ComfyUI

# 依赖 + PyAV（SaveWEBM/SaveVideo 需要）
RUN pip install -r requirements.txt \
    && pip install av

# 模型与输出通过卷挂载：
#   PVC models  -> /app/ComfyUI/models
#   PVC output  -> /app/ComfyUI/output
#   PVC input   -> /app/ComfyUI/input
# 见 deployment.yaml 与 MODELS.md

EXPOSE 8188

CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
