# ============================================================
# prompt2vid agent 镜像：CrewAI 流水线 + A2A 服务器
# 构建：docker build -t <registry>/prompt2vid:latest .
# ============================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg：剪辑工具依赖；fonts-noto-cjk：烧录中文字幕依赖
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config

RUN pip install .

EXPOSE 8081

# 端口默认 8081；kagent harness 会注入 KAGENT_PORT 环境变量自动覆盖
CMD ["python", "-m", "prompt2vid.server", "--host", "0.0.0.0"]
