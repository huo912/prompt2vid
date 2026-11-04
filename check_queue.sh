/root/miniconda3/bin/python - <<'PYEOF'
import json, urllib.request
h = json.load(urllib.request.urlopen("http://127.0.0.1:8188/history", timeout=15))
rows = []
for pid, info in h.items():
    prompt = info.get("prompt")
    if not (isinstance(prompt, list) and len(prompt) > 3 and isinstance(prompt[3], dict)):
        continue
    create_ms = prompt[3].get("create_time", 0)
    if create_ms < 1793777000000:  # ~17:08 之后提交的任务
        continue
    nodes = prompt[2]
    shot = err = ""
    for n in nodes.values():
        if isinstance(n, dict):
            if n.get("class_type") == "VHS_LoadVideo":
                shot = n.get("inputs", {}).get("video", "?")
    st = info.get("status", {})
    if st.get("status_str") == "error":
        for m in st.get("messages", []):
            if "execution_error" in str(m[0]):
                err = str(m[1].get("exception_message", ""))[:100]
    rows.append((shot, st.get("status_str"), err))
for r in sorted(rows):
    print(r)
PYEOF
