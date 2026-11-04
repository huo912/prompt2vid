/root/miniconda3/bin/python - <<'PYEOF'
import json, urllib.request
for name in ("IndexTTS25EmotionVectorNode", "IndexTTS25EmotionTextNode"):
    info = json.load(urllib.request.urlopen(f"http://127.0.0.1:8188/object_info/{name}", timeout=15))
    node = info.get(name, {})
    print(f"=== {name} ===")
    for group in ("required", "optional"):
        for k, v in (node.get("input", {}).get(group) or {}).items():
            spec = v[0] if isinstance(v, list) and v else v
            print(f"[{group}] {k}: {spec}")
    print("outputs:", node.get("output_name"), "|", node.get("output"))
PYEOF
