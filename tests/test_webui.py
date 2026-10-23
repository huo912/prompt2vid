"""Web 控制台 API 测试：步骤目录 / 运行管理 / 单步执行 / 日志 / 产物。

执行链路用 fake run_steps 替身，验证 API 编排本身，不依赖
LLM/ComfyUI/ffmpeg。
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from prompt2vid import steps, webui


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(webui, "OUTPUT_BASE", tmp_path / "output")
    monkeypatch.setattr(webui, "manager", webui.RunManager())
    app = FastAPI()
    webui.register_webui(app)
    return TestClient(app)


@pytest.fixture()
def fake_run_steps(monkeypatch):
    calls: list[dict] = []

    def _fake(run_dir, step_ids=None, prompt=None, force=False, cancel_event=None):
        calls.append(
            {
                "run_dir": Path(run_dir),
                "steps": list(step_ids or []),
                "prompt": prompt,
                "force": force,
            }
        )
        (Path(run_dir) / "storyboard.json").write_text("{}", encoding="utf-8")
        return [{"step": i, "status": "done"} for i in (step_ids or [])]

    monkeypatch.setattr("prompt2vid.steps.run_steps", _fake)
    return calls


def _create_run(client, prompt="做一个秋天的治愈系短视频", auto_run=None):
    body = {"prompt": prompt}
    if auto_run is not None:
        body["auto_run"] = auto_run
    return client.post("/api/runs", json=body)


def _run_dir(run_id: str) -> Path:
    return webui.OUTPUT_BASE / run_id


# ------------------------------------------------------------ 步骤目录 / 页面

def test_steps_catalog(client):
    res = client.get("/api/steps")
    assert res.status_code == 200
    catalog = res.json()["steps"]
    assert len(catalog) == 9
    assert catalog[0]["id"] == "analyze_requirement"
    assert catalog[0]["kind_label"] == "创意"
    assert catalog[6]["stage"] == "render"  # edit_video


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "调试控制台" in res.text


# ------------------------------------------------------------ 运行管理

def test_create_and_list_run(client):
    res = _create_run(client)
    assert res.status_code == 201
    run_id = res.json()["run_id"]
    assert run_id.startswith("run_")
    assert _run_dir(run_id).is_dir()
    assert steps.read_prompt(_run_dir(run_id)) == "做一个秋天的治愈系短视频"

    res = client.get("/api/runs")
    runs = res.json()["runs"]
    assert len(runs) == 1
    assert runs[0]["id"] == run_id
    assert runs[0]["prompt"] == "做一个秋天的治愈系短视频"
    assert runs[0]["total"] == 9


def test_create_run_rejects_blank_prompt(client):
    assert client.post("/api/runs", json={"prompt": "   "}).status_code == 422


def test_run_detail_artifact_fallback(client):
    run_id = _create_run(client).json()["run_id"]
    (_run_dir(run_id) / "requirement.json").write_text("{}", encoding="utf-8")

    res = client.get(f"/api/runs/{run_id}")
    assert res.status_code == 200
    detail = res.json()
    by_id = {s["id"]: s for s in detail["steps"]}
    assert by_id["analyze_requirement"]["status"] == "done"
    assert by_id["analyze_requirement"]["artifact_exists"] is True
    assert by_id["write_script"].get("status") is None
    assert detail["execution"] is None


def test_unknown_run_404(client):
    assert client.get("/api/runs/run_nope").status_code == 404
    assert client.get("/api/runs/run_nope/log").status_code == 404


# ------------------------------------------------------------ 单步执行

def test_execute_steps(client, fake_run_steps):
    run_id = _create_run(client).json()["run_id"]

    res = client.post(
        f"/api/runs/{run_id}/steps", json={"steps": ["generate_images"]}
    )
    assert res.status_code == 202
    webui.manager.wait(timeout=5)

    assert len(fake_run_steps) == 1
    assert fake_run_steps[0]["steps"] == ["generate_images"]
    assert fake_run_steps[0]["run_dir"] == _run_dir(run_id).resolve()
    # fake 替身写下的产物能被详情识别
    detail = client.get(f"/api/runs/{run_id}").json()
    by_id = {s["id"]: s for s in detail["steps"]}
    assert by_id["plan_storyboard"]["artifact_exists"] is True


def test_execute_steps_prompt_persisted(client, fake_run_steps):
    run_id = _create_run(client).json()["run_id"]
    res = client.post(
        f"/api/runs/{run_id}/steps",
        json={"steps": ["analyze_requirement"], "prompt": "新的一句话"},
    )
    assert res.status_code == 202
    webui.manager.wait(timeout=5)

    assert fake_run_steps[0]["prompt"] == "新的一句话"
    assert steps.read_prompt(_run_dir(run_id)) == "新的一句话"


def test_execute_unknown_step_422(client):
    run_id = _create_run(client).json()["run_id"]
    res = client.post(f"/api/runs/{run_id}/steps", json={"steps": ["bad_step"]})
    assert res.status_code == 422


def test_execute_busy_returns_409(client, monkeypatch):
    gate = threading.Event()
    started = threading.Event()

    def blocking_run_steps(run_dir, step_ids=None, prompt=None,
                           force=False, cancel_event=None):
        started.set()
        gate.wait(timeout=5)
        return []

    monkeypatch.setattr("prompt2vid.steps.run_steps", blocking_run_steps)

    run_id = _create_run(client, auto_run=["generate_images"]).json()["run_id"]
    assert started.wait(3)

    # 同一运行 / 另一运行的执行请求都被拒绝
    assert client.post(
        f"/api/runs/{run_id}/steps", json={"steps": ["generate_voiceover"]}
    ).status_code == 409
    other = _create_run(client, prompt="另一个").json()["run_id"]
    assert client.post(
        f"/api/runs/{other}/steps", json={"steps": ["generate_images"]}
    ).status_code == 409

    # 详情能看到执行中状态
    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["execution"]["active"] is True
    assert detail["execution"]["run_id"] == run_id

    gate.set()
    webui.manager.wait(timeout=5)


def test_cancel_run(client, monkeypatch):
    gate = threading.Event()
    started = threading.Event()

    def blocking_run_steps(run_dir, step_ids=None, prompt=None,
                           force=False, cancel_event=None):
        started.set()
        gate.wait(timeout=5)
        return []

    monkeypatch.setattr("prompt2vid.steps.run_steps", blocking_run_steps)

    run_id = _create_run(client, auto_run=["generate_images"]).json()["run_id"]
    assert started.wait(3)

    res = client.post(f"/api/runs/{run_id}/cancel")
    assert res.status_code == 200
    assert "取消" in res.json()["message"]

    gate.set()
    webui.manager.wait(timeout=5)
    # 结束后再取消 → 409
    assert client.post(f"/api/runs/{run_id}/cancel").status_code == 409


def test_auto_run_all(client, fake_run_steps):
    res = _create_run(client, auto_run=["all"])
    assert res.status_code == 201
    assert res.json()["started"] is True
    webui.manager.wait(timeout=5)
    assert fake_run_steps[0]["steps"] == steps.STEP_IDS


# ------------------------------------------------------------ 日志与产物

def test_log_tail(client):
    run_id = _create_run(client).json()["run_id"]
    (_run_dir(run_id) / "run.log").write_text(
        "\n".join(f"line{i}" for i in range(10)), encoding="utf-8"
    )
    res = client.get(f"/api/runs/{run_id}/log?tail=3")
    assert res.status_code == 200
    assert res.json()["lines"] == ["line7", "line8", "line9"]


def test_log_missing_file(client):
    run_id = _create_run(client).json()["run_id"]
    res = client.get(f"/api/runs/{run_id}/log")
    assert res.status_code == 200
    assert res.json()["lines"] == []


def test_file_serving(client):
    run_id = _create_run(client).json()["run_id"]
    final_dir = _run_dir(run_id) / "final"
    final_dir.mkdir(parents=True)
    (final_dir / "final.mp4").write_bytes(b"fake-video")

    res = client.get(f"/api/runs/{run_id}/files/final/final.mp4")
    assert res.status_code == 200
    assert res.content == b"fake-video"

    # 不存在的文件
    assert client.get(
        f"/api/runs/{run_id}/files/final/nope.mp4"
    ).status_code == 404


def test_file_traversal_blocked(client):
    run_id = _create_run(client).json()["run_id"]
    secret = webui.OUTPUT_BASE / "secret.txt"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("secret", encoding="utf-8")

    # %2F 编码绕过客户端路径归一化，服务端 resolve 后必须拒绝
    res = client.get(f"/api/runs/{run_id}/files/..%2Fsecret.txt")
    assert res.status_code == 403
    # 带路径分隔符的 run_id 同样拒绝
    assert client.get("/api/runs/..%2F..%2Fetc/log").status_code == 404
