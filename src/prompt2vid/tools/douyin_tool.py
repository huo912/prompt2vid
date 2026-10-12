"""抖音发布工具：开放平台 API + dry-run + 状态跟踪。

真实发布走抖音开放平台「视频发布」接口：
1. POST /api/douyin/v1/video/create_video/ （创建视频任务，返回 upload_url）
2. PUT upload_url（分片上传视频文件）
3. POST /api/douyin/v1/video/complete_video/ （完成上传并发布）

DOUYIN_DRY_RUN=true（默认）时不真实上传，仅把发布记录写入 .douyin/state.json。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
from crewai.tools import tool

OPEN_API_BASE = "https://open.douyin.com"


def _workspace() -> Path:
    ws = os.getenv("PROMPT2VID_WORKSPACE", "").strip()
    path = Path(ws) if ws else Path("output") / "run_default"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_path() -> Path:
    state = os.getenv("DOUYIN_STATE_FILE", "").strip()
    return Path(state) if state else Path(".douyin") / "state.json"


def _load_state() -> dict:
    path = _state_path()
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"last_publish": None, "published": []}


def _save_state(state: dict) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _is_dry_run() -> bool:
    return os.getenv("DOUYIN_DRY_RUN", "true").strip().lower() != "false"


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = _workspace() / path
    return path


def _publish_real(video: Path, title: str, tags: list[str]) -> dict:
    """真实发布：抖音开放平台视频发布 API。"""
    client_key = os.getenv("DOUYIN_CLIENT_KEY", "").strip()
    client_secret = os.getenv("DOUYIN_CLIENT_SECRET", "").strip()
    access_token = os.getenv("DOUYIN_ACCESS_TOKEN", "").strip()
    if not (client_key and client_secret and access_token):
        raise RuntimeError(
            "缺少抖音凭据：请设置 DOUYIN_CLIENT_KEY / DOUYIN_CLIENT_SECRET / "
            "DOUYIN_ACCESS_TOKEN，或保持 DOUYIN_DRY_RUN=true"
        )

    headers = {"access-token": access_token}
    text = title + " " + " ".join(t if t.startswith("#") else f"#{t}" for t in tags)

    with httpx.Client(timeout=120.0) as client:
        # 1. 创建视频任务
        resp = client.post(
            f"{OPEN_API_BASE}/api/douyin/v1/video/create_video/",
            headers=headers,
            json={
                "video": {"video_id": "", "width": 0, "height": 0},
                "text": text,
            },
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        upload_url = data.get("upload_url", "")
        video_id = data.get("video_id", "")
        if not upload_url:
            raise RuntimeError(f"创建视频任务失败: {data}")

        # 2. 上传视频文件
        with video.open("rb") as f:
            up = client.put(
                upload_url,
                headers={"Content-Type": "video/mp4"},
                content=f.read(),
            )
        up.raise_for_status()

        # 3. 完成上传并发布
        resp = client.post(
            f"{OPEN_API_BASE}/api/douyin/v1/video/complete_video/",
            headers=headers,
            json={"video_id": video_id},
        )
        resp.raise_for_status()
        result = resp.json().get("data", {})

    item_id = result.get("item_id", "")
    return {
        "status": "published",
        "title": title,
        "tags": tags,
        "item_id": item_id or None,
        "published_at": datetime.now(timezone.utc).isoformat(),
    }


@tool("Publish video to Douyin")
def publish(video_path: str, title: str, tags: str) -> str:
    """发布视频到抖音（DOUYIN_DRY_RUN=true 时仅记录状态）。

    Args:
        video_path: 成片路径（如 final/final.mp4）。
        title: 抖音标题（≤ 30 字）。
        tags: 话题标签，逗号分隔（如 "秋天,治愈,风景"）。

    Returns:
        JSON 字符串：status（published/dry_run）、title、item_id、published_at。
    """
    video = _resolve(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"成片不存在: {video}")

    tag_list = [t.strip().lstrip("#") for t in tags.split(",") if t.strip()]
    now = datetime.now(timezone.utc).isoformat()

    if _is_dry_run():
        result = {
            "status": "dry_run",
            "title": title,
            "tags": tag_list,
            "item_id": None,
            "published_at": now,
            "video": str(video.resolve()),
        }
    else:
        result = _publish_real(video, title, tag_list)

    state = _load_state()
    state["last_publish"] = result
    state["published"].append(result)
    _save_state(state)
    return json.dumps(result, ensure_ascii=False)


TOOLS = [publish]
