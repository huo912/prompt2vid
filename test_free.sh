echo '=== 显存占用（当前）==='
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo '=== 调用 /free 卸载全部模型 ==='
curl -s -m 60 -X POST http://127.0.0.1:8188/free -H 'Content-Type: application/json' -d '{"unload": true, "free_memory": true}'
echo ''
sleep 8
echo '=== 显存占用（卸载后）==='
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
