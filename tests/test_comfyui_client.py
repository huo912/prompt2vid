"""ComfyUI 客户端测试：批量等待、退避轮询、种子复现、history 解析（MockTransport，无真实服务）。"""

from __future__ import annotations

import httpx
import pytest

from prompt2vid import comfyui_client
from prompt2vid.comfyui_client import (
    ComfyUITaskError,
    ComfyUIClient,
    next_seed,
    parse_output_files,
)


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


# ------------------------------------------------------------ history 输出解析

def test_parse_output_files_direct_format():
    """回归：images 直接挂在节点下（经典文档格式 / 2025+ 新版 master 格式）。

    此前 outputs() 只认 {"ui": {...}} 包装，直接格式解析为空 →
    云机（新版 ComfyUI）上全部分镜"任务完成但没有输出文件"。
    """
    entry = {
        "outputs": {
            "9": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]},
        },
        "status": {"completed": True, "status_str": "success"},
    }
    assert parse_output_files(entry) == [
        {"filename": "out.png", "subfolder": "", "type": "output"}
    ]


def test_parse_output_files_ui_wrapped_format():
    """旧变体：{"ui": {"images": [...]}} 包装。"""
    entry = {
        "outputs": {
            "9": {"ui": {"images": [{"filename": "out.png", "subfolder": "s", "type": "temp"}]}},
        },
        "status": {"completed": True},
    }
    assert parse_output_files(entry) == [
        {"filename": "out.png", "subfolder": "s", "type": "temp"}
    ]


def test_parse_output_files_video_keys_and_dedup():
    """SaveWEBM 的 gifs/videos 键、双格式并存去重、坏条目跳过。"""
    entry = {
        "outputs": {
            "86": {
                "gifs": [{"filename": "clip.webm", "type": "output"}],
                "videos": [{"filename": "clip.webm", "type": "output"}],
            },
            "9": {"images": [{"filename": "f.png"}, "bad-item", None]},
        },
    }
    assert parse_output_files(entry) == [
        {"filename": "clip.webm", "subfolder": "", "type": "output"},
        {"filename": "f.png", "subfolder": "", "type": "output"},
    ]


def test_wait_accepts_status_str_success():
    """status 里只有 status_str=success（无 completed 字段）也视为完成。"""
    states = {
        "pid-1": [
            {},
            {"outputs": {"9": {"images": [{"filename": "a.png"}]}},
             "status": {"status_str": "success"}},
        ]
    }
    client = ComfyUIClient(
        base_url="http://comfyui.test",
        http=httpx.Client(transport=httpx.MockTransport(_history_handler(states))),
    )
    entry = client.wait("pid-1", timeout=5)
    assert parse_output_files(entry) == [
        {"filename": "a.png", "subfolder": "", "type": "output"}
    ]


def test_wait_status_none_does_not_crash():
    """status 为 None（新版 task_done 允许）不抛 AttributeError，继续轮询。"""
    states = {"pid-1": [{"outputs": {}, "status": None}, _completed_entry()]}
    client = ComfyUIClient(
        base_url="http://comfyui.test",
        http=httpx.Client(transport=httpx.MockTransport(_history_handler(states))),
    )
    assert client.wait("pid-1", timeout=5)["status"]["completed"] is True


def test_submit_rejection_includes_body():
    """工作流被拒（如节点未安装）时，错误信息带响应体便于定位。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": "prompt outputs failed validation",
                "node_errors": {"ref_loader": "IPAdapterUnifiedLoader not found"},
            },
        )

    client = ComfyUIClient(
        base_url="http://comfyui.test",
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(RuntimeError, match="IPAdapterUnifiedLoader"):
        client.submit({"3": {"class_type": "X"}})
