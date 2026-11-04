mv /autodl-fs/data/ltxv-13b-0.9.7-dev-fp8.safetensors.part /autodl-fs/data/ltxv-13b-0.9.7-dev-fp8.safetensors
ln -sf /autodl-fs/data/ltxv-13b-0.9.7-dev-fp8.safetensors /root/autodl-tmp/ComfyUI/models/checkpoints/ltxv-13b-0.9.7-dev-fp8.safetensors
ls -la /root/autodl-tmp/ComfyUI/models/checkpoints/ | grep dev
/root/miniconda3/bin/python - <<'PYEOF'
import json, urllib.request
info = json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info/CheckpointLoaderSimple", timeout=30))
ckpts = []
for spec in (info.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name") or []):
    if isinstance(spec, list):
        ckpts = spec
        break
print("dev 可见:", "ltxv-13b-0.9.7-dev-fp8.safetensors" in ckpts)
print("全部:", ckpts)
PYEOF
