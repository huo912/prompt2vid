"""ComfyUI 客户端测试：批量等待、退避轮询、种子复现（MockTransport，无真实服务）。"""

from __future__ import annotations

import httpx
import pytest

from prompt2vid import comfyui_client
from prompt2vid.comfyui_client import ComfyUITaskError, ComfyUIClient, next_seed


def _completed_entry() -> dict:
    return {"outputs": {}, "status": {"completed": True}}


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch):
    """轮询间隔压到毫秒级，测试不真睡。"""
    monkeypatch.setattr(comfyui_client, "POLL_START", 0.01)
    monkeypatch.setattr(comfyui_client, "POLL_MAX", 0.02)


def _history_handler(states: dict[str, list[dict]]):
    """每个 prompt_id 一串状态：先若干次空，再完成或错误。"""

    def handler(request: httpx.Request) -> httpx.Response:
        pid = request.url.path.rsplit("/", 1)[1]
        states.setdefault(pid, [])
        entry = states[pid].pop(0) if states[pid] else None
        if entry is None:
            return httpx.Response(200, json={})
        if isinstance(entry, Exception):
            # 返回 error 状态的 history 条目
            return httpx.Response(200, json={
                pid: {"status": {"completed": False, "status_str": "error",
                                 "messages": [str(entry)]}}
            })
        return httpx.Response(200, json={pid: entry})

    return handler


def test_wait_polls_until_completed():
    states = {"pid-1": [{}, {}, _completed_entry()]}
    client = ComfyUIClient(
        base_url="http://comfyui.test",
        http=httpx.Client(transport=httpx.MockTransport(_history_handler(states))),
    )
    entry = client.wait("pid-1", timeout=5)
    assert entry["status"]["completed"] is True


def test_wait_raises_task_error():
    states = {"pid-1": [RuntimeError("node exploded")]}
    client = ComfyUIClient(
        base_url="http://comfyui.test",
        http=httpx.Client(transport=httpx.MockTransport(_history_handler(states))),
    )
    with pytest.raises(ComfyUITaskError):
        client.wait("pid-1", timeout=5)


def test_wait_all_collects_errors_and_successes():
    states = {
        "pid-a": [RuntimeError("boom")],
        "pid-b": [{}, _completed_entry()],
    }
    client = ComfyUIClient(
        base_url="http://comfyui.test",
        http=httpx.Client(transport=httpx.MockTransport(_history_handler(states))),
    )
    results = client.wait_all(["pid-a", "pid-b"], timeout=5)
    # 失败任务不炸整批
    assert isinstance(results["pid-a"], ComfyUITaskError)
    assert results["pid-b"]["status"]["completed"] is True


def test_wait_all_timeout():
    states = {"pid-a": []}  # 永远不完成
    client = ComfyUIClient(
        base_url="http://comfyui.test",
        http=httpx.Client(transport=httpx.MockTransport(_history_handler(states))),
    )
    with pytest.raises(TimeoutError):
        client.wait_all(["pid-a"], timeout=0.1)


def test_wait_all_empty_list():
    client = ComfyUIClient(
        base_url="http://comfyui.test",
        http=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404))),
    )
    assert client.wait_all([]) == {}


def test_next_seed_env_reproducible(monkeypatch):
    monkeypatch.setenv("PROMPT2VID_SEED", "42")
    first = [next_seed() for _ in range(3)]
    assert first == [42, 43, 44]


def test_next_seed_random_without_env(monkeypatch):
    monkeypatch.delenv("PROMPT2VID_SEED", raising=False)
    seeds = {next_seed() for _ in range(5)}
    assert len(seeds) == 5


def test_workflow_steps_env_override(monkeypatch):
    from prompt2vid.comfyui_client import build_i2v_workflow, build_t2i_workflow

    wf_t2i = build_t2i_workflow(prompt="a", negative_prompt="b")
    assert wf_t2i["3"]["inputs"]["steps"] == 25  # 默认 25 步

    wf_i2v = build_i2v_workflow(image_name="x.png", prompt="a", negative_prompt="b")
    assert wf_i2v["71"]["inputs"]["steps"] == 10  # LTX distilled 默认 10 步

    monkeypatch.setenv("COMFYUI_T2I_STEPS", "20")
    monkeypatch.setenv("COMFYUI_LTX_STEPS", "13")
    assert build_t2i_workflow(prompt="a", negative_prompt="b")["3"]["inputs"]["steps"] == 20
    assert build_i2v_workflow("x.png", "a", "b")["71"]["inputs"]["steps"] == 13
