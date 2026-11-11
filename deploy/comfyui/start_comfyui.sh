#!/usr/bin/env bash
# ComfyUI 启动脚本（AutoDL 4090 / MiniMax-H3 档）
# 用法：
#   bash deploy/comfyui/start_comfyui.sh            # 正常 GPU 启动
#   bash deploy/comfyui/start_comfyui.sh --cpu      # 无卡模式（只能验证节点，跑不了推理）
#   bash deploy/comfyui/start_comfyui.sh stop       # 停止
#   bash deploy/comfyui/start_comfyui.sh status      # 查看状态
#
# 之前 `python main.py` 从 ComfyUI/output 下执行会报
# "can't open file '.../output/main.py'"——本脚本固定先 cd 到 ComfyUI 根目录。
set -uo pipefail

COMFYUI_DIR="${COMFYUI_DIR:-/root/autodl-tmp/ComfyUI}"
PY="${PY:-/root/miniconda3/bin/python}"
PIDFILE="$COMFYUI_DIR/comfyui.pid"
LOG="$COMFYUI_DIR/comfyui.log"
PORT="${PORT:-8188}"

[ -x "$PY" ] || { echo "找不到 python：$PY" >&2; exit 1; }
[ -f "$COMFYUI_DIR/main.py" ] || { echo "找不到 $COMFYUI_DIR/main.py" >&2; exit 1; }

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

# pidfile 可能丢失（比如手工 nohup 起的），按命令行兜底找本端口的 ComfyUI
pid_by_cmdline() { pgrep -f "main.py --listen 127.0.0.1 --port $PORT" 2>/dev/null | head -n 1; }

ACT="${1:-start}"
case "$ACT" in
  stop)
    pids=$(cat "$PIDFILE" 2>/dev/null; pid_by_cmdline)
    pids=$(echo "$pids" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u)
    if [ -n "$pids" ]; then
      kill $pids 2>/dev/null
      for _ in $(seq 1 20); do
        alive=0
        for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done
        [ "$alive" = 0 ] && break
        sleep 1
      done
      kill -9 $pids 2>/dev/null || true
      echo "已停止 ComfyUI (pid $(echo $pids))"
    else
      echo "ComfyUI 未在运行"
    fi
    rm -f "$PIDFILE"
    exit 0
    ;;
  status)
    if running; then
      echo "运行中 pid=$(cat "$PIDFILE") url=http://127.0.0.1:$PORT"
      curl -s -m 5 -o /dev/null -w "HTTP %{http_code}\n" "http://127.0.0.1:$PORT/system_stats" || true
    else
      echo "未运行"
    fi
    exit 0
    ;;
  start) shift || true ;;
  *) echo "未知参数：$ACT（可用：start / stop / status）" >&2; exit 2 ;;
esac

if running; then echo "已在运行 pid=$(cat "$PIDFILE")，先执行 $0 stop"; exit 0; fi

cd "$COMFYUI_DIR" || exit 1
echo "启动 ComfyUI：$PY main.py --listen 127.0.0.1 --port $PORT $*"
setsid nohup "$PY" main.py --listen 127.0.0.1 --port "$PORT" "$@" > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
echo "pid=$(cat "$PIDFILE")  日志：$LOG"

for i in $(seq 1 180); do
  code=$(curl -s -m 3 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/system_stats" 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "就绪：http://127.0.0.1:$PORT  （用时 ${i}s）"
    exit 0
  fi
  if ! running; then echo "进程已退出，最后 30 行日志：" >&2; tail -30 "$LOG" >&2; exit 1; fi
  sleep 1
done
echo "180s 未就绪，最后 30 行日志：" >&2; tail -30 "$LOG" >&2; exit 1
