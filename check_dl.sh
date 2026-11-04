df -h /root/autodl-tmp | tail -1
echo '=== 检查 hf-mirror 上 dev 模型是否可下载 ==='
curl -sI -m 20 "https://hf-mirror.com/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.7-dev-fp8.safetensors" | head -8
