/root/miniconda3/bin/python - <<'PYEOF'
import json, urllib.request
info = json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info/IndexTTS25BaseNode", timeout=15))
node = info.get("IndexTTS25BaseNode", {})
print("=== IndexTTS25BaseNode 输入 ===")
for group in ("required", "optional"):
    for k, v in (node.get("input", {}).get(group) or {}).items():
        spec = v[0] if isinstance(v, list) and v else v
        print(f"[{group}] {k}: {spec}")
print("=== 其它 IndexTTS 节点 ===")
allinfo = json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=20))
for name in allinfo:
    if "index" in name.lower() or "tts" in name.lower():
        print(" ", name)
PYEOF
echo '=== 服务器现有 LTX 模型 ==='
ls -la /root/autodl-tmp/ComfyUI/models/checkpoints/ /root/autodl-tmp/ComfyUI/models/loras/ 2>/dev/null | grep -v '^d' | grep -v total
