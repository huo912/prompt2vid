/root/miniconda3/bin/python - <<'PYEOF'
import json, urllib.request
info = json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info/IndexTTS25BaseNode", timeout=15))
node = info.get("IndexTTS25BaseNode", {})
for group in ("required", "optional"):
    for k, v in (node.get("input", {}).get(group) or {}).items():
        if isinstance(v, list) and len(v) > 1 and isinstance(v[1], dict):
            print(f"{k}: default={v[1].get('default')}")
        elif isinstance(v, list):
            print(f"{k}: {v[0]}")
PYEOF
