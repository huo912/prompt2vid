#!/usr/bin/env bash
# MiniMax-H3 模型自动下载 + 软链（24GB 档，RTX 4090）
#
# 用途：给 ComfyUI 准备 MiniMax-H3（视频+音频联合生成）所需的 6 个模型文件。
# 选型针对 24GB 显存 + torch cu130：
#   diffusion_models  pruned_int8_convrot（21GB，官方模板默认档；cu130 可用 int8
#                     旋转量化，比 fp8_scaled 精度更高、比 w6a8 更快）
#   text_encoders     qwen3vl_32b nvfp4_awq（15.7GB，官方说明不要求 Blackwell；
#                     bf16 51.5GB / int8 27.1GB 都塞不进 24GB）
#   vae               video int8_convrot（2.8GB）+ audio fp32（0.6GB）
#   loras             fl2v turbo 8step（默认）+ 4step 768p（快速档）
#   latent_upscale_models
#                     minimax_h3_latent_upscaler_3d_conv_v1_bf16（0.69GB，
#                     第三个仓库 LBH-123-AI/Minimax_h3_latent_Upscaler）；
#                     ComfyUI 核心节点 MiniMaxH3LatentUpscaleWithModel 用它
#                     在 latent 域 2 倍放大，治低分生成糊
#
# 大文件放文件存储 /autodl-fs/data，再软链进 ComfyUI（数据盘只有 50GB）。
# 幂等：已下载且校验通过的文件会跳过；aria2c 支持断点续传。
# 也兼容"已把大文件搬进数据盘（ComfyUI/models 下是真实文件）"的机器：
# 这类文件会被识别并跳过，不会重复下载 20GB。
#
# 用法：
#   bash deploy/comfyui/download_minimax_h3_models.sh
#   COMFYUI_DIR=/root/autodl-tmp/ComfyUI bash ...
#   P2V_H3_MODEL_ROOT=/autodl-fs/data/models bash ...
set -euo pipefail

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
REPO="${P2V_H3_REPO:-Comfy-Org/MiniMax-H3}"
MODEL_ROOT="${P2V_H3_MODEL_ROOT:-/autodl-fs/data/models}"
H3_DIR="$MODEL_ROOT/minimax-h3"
COMFYUI_DIR="${COMFYUI_DIR:-/root/autodl-tmp/ComfyUI}"
CONNECTIONS="${P2V_H3_CONNECTIONS:-8}"

# 潜空间放大模型（治低分模糊；ComfyUI 核心节点 LatentUpscaleModelLoader 吃它）
UPSCALE_REPO="${P2V_H3_UPSCALE_REPO:-LBH-123-AI/Minimax_h3_latent_Upscaler}"
UPSCALE_SRC="minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
UPSCALE_REL="latent_upscale_models/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"

FILES="
diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
vae/minimax_h3_video_vae_int8_convrot.safetensors
vae/minimax_h3_audio_vae_fp32.safetensors
loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors
loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors
"

# 云机镜像常只有 conda 里的 python（没有 python3），逐个探测
PY_BIN=""
for cand in python3 python "$HOME/miniconda3/bin/python" /root/miniconda3/bin/python; do
    if command -v "$cand" >/dev/null 2>&1; then
        PY_BIN="$cand"
        break
    fi
done
if [ -z "$PY_BIN" ]; then
    echo "找不到 python，无法校验 safetensors（装一个或设 PATH）" >&2
    exit 1
fi

verify_safetensors() {
    "$PY_BIN" - "$1" <<'PY'
import json, os, struct, sys

path = sys.argv[1]
size = os.path.getsize(path)
with open(path, "rb") as fh:
    (header_len,) = struct.unpack("<Q", fh.read(8))
    header = json.loads(fh.read(header_len))
end = 0
for key, value in header.items():
    if key == "__metadata__":
        continue
    end = max(end, value["data_offsets"][1])
expected = 8 + header_len + end
if expected != size:
    sys.exit(f"truncated: header expects {expected} bytes, file has {size}")
PY
}

mkdir -p "$H3_DIR"
for sub in diffusion_models text_encoders vae loras latent_upscale_models; do
    mkdir -p "$H3_DIR/$sub" "$COMFYUI_DIR/models/$sub"
done

failed=0

# fetch_one <相对路径> <下载 URL>
fetch_one() {
    rel="$1"
    url="$2"
    dest="$H3_DIR/$rel"
    link="$COMFYUI_DIR/models/$rel"
    if [ -f "$dest" ] && verify_safetensors "$dest" >/dev/null 2>&1; then
        echo "[skip] $rel 已就绪（$(du -h "$dest" | cut -f1)）"
        return 0
    fi
    if [ ! -L "$link" ] && [ -f "$link" ] && verify_safetensors "$link" >/dev/null 2>&1; then
        echo "[skip] $rel 已是 ComfyUI/models 下的真实文件（$(du -h "$link" | cut -f1)）"
        return 0
    fi
    echo "[get ] $rel"
    if command -v aria2c >/dev/null 2>&1; then
        aria2c -x "$CONNECTIONS" -s "$CONNECTIONS" -k 1M \
            --continue=true --allow-overwrite=true --auto-file-renaming=false \
            --file-allocation=none --retry-wait=5 --max-tries=20 \
            --summary-interval=30 --console-log-level=warn \
            -d "$(dirname "$dest")" -o "$(basename "$dest")" "$url"
    else
        curl -fL --retry 20 --retry-delay 5 -C - -o "$dest" "$url"
    fi
    if ! verify_msg=$(verify_safetensors "$dest" 2>&1); then
        echo "[FAIL] $rel 校验不通过（$verify_msg）" >&2
        return 1
    fi
    echo "[done] $rel  $(du -h "$dest" | cut -f1)"
    return 0
}

for rel in $FILES; do
    fetch_one "$rel" "$HF_ENDPOINT/$REPO/resolve/main/$rel" || failed=1
done
fetch_one "$UPSCALE_REL" "$HF_ENDPOINT/$UPSCALE_REPO/resolve/main/$UPSCALE_SRC" || failed=1

echo "--- 软链进 ComfyUI（ComfyUI 只在启动时扫目录，软链后必须重启）"
for rel in $FILES $UPSCALE_REL; do
    dest="$H3_DIR/$rel"
    link="$COMFYUI_DIR/models/$rel"
    [ -f "$dest" ] || continue
    if [ -e "$link" ] && [ ! -L "$link" ]; then
        echo "[warn] $link 是真实文件，未覆盖" >&2
        continue
    fi
    ln -sfn "$dest" "$link"
    echo "[link] $link -> $dest"
done

echo "--- 校验软链可读"
for rel in $FILES $UPSCALE_REL; do
    link="$COMFYUI_DIR/models/$rel"
    if [ -r "$link" ]; then
        echo "[ ok ] $rel"
    else
        echo "[bad ] $rel 不可读" >&2
        failed=1
    fi
done

echo "--- 磁盘"
df -h "$MODEL_ROOT" "$COMFYUI_DIR" 2>/dev/null | sed 's/  */ /g'

if [ "$failed" -ne 0 ]; then
    echo "存在失败项，重跑本脚本可续传（已完成的文件会跳过）" >&2
    exit 1
fi
echo "MiniMax-H3 模型就绪，重启 ComfyUI 后即可在 UNETLoader / CLIPLoader / VAELoader / LatentUpscaleModelLoader 中看到"
