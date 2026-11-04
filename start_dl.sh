if [ -f /autodl-fs/data/ltxv-13b-0.9.7-dev-fp8.safetensors ]; then
  echo "already exists"
  ls -la /autodl-fs/data/ltxv-13b-0.9.7-dev-fp8.safetensors
  exit 0
fi
nohup curl -L -o /autodl-fs/data/ltxv-13b-0.9.7-dev-fp8.safetensors.part \
  "https://hf-mirror.com/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.7-dev-fp8.safetensors" \
  > /tmp/dev_model_dl.log 2>&1 &
echo "download started, pid $!"
sleep 10
ls -la /autodl-fs/data/ | grep ltxv
