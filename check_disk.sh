du -sh /root/autodl-tmp/ComfyUI/models/* 2>/dev/null | sort -rh | head -10
echo '=== HF 缓存 ==='
du -sh /root/.cache/huggingface/hub/* 2>/dev/null | sort -rh | head -8
echo '=== 其它大目录 ==='
du -sh /root/autodl-tmp/ComfyUI/output /root/autodl-tmp/ComfyUI/input /root/.cache/pip /root/autodl-tmp/*.log 2>/dev/null | sort -rh
echo '=== 全部挂载点 ==='
df -h | grep -v tmpfs | grep -v overlay
