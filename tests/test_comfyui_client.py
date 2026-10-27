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

    # 其它测试可能 load_dotenv 把 .env 写进 os.environ（可能已切到 13B），
    # 这里显式清理，验证“无覆盖时的内置默认值”。
    for name in (
        "COMFYUI_LTX_CHECKPOINT",
        "COMFYUI_LTX_STEPS",
        "COMFYUI_LTX_CFG",
        "COMFYUI_T2I_STEPS",
    ):
        monkeypatch.delenv(name, raising=False)

    wf_t2i = build_t2i_workflow(prompt="a", negative_prompt="b")
    assert wf_t2i["3"]["inputs"]["steps"] == 25  # 默认 25 步

    wf_i2v = build_i2v_workflow(image_name="x.png", prompt="a", negative_prompt="b")
    assert wf_i2v["71"]["inputs"]["steps"] == 10  # LTX 2B 0.9.5 默认 10 步
    assert wf_i2v["72"]["inputs"]["cfg"] == 3.0

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


# --------------------------------------------------- checkpoint 文件名自动校准

def _checkpoint_info_handler(ckpts: list[str]):
    """Mock /object_info/CheckpointLoaderSimple，返回候选 checkpoint 列表。

    返回结构与真实 ComfyUI 0.21.x 一致：
    `input.required.ckpt_name = [候选列表, {"tooltip": ...}]`
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/object_info/CheckpointLoaderSimple":
            return httpx.Response(
                200,
                json={
                    "CheckpointLoaderSimple": {
                        "input": {
                            "required": {
                                "ckpt_name": [ckpts, {"tooltip": "The name..."}]
                            }
                        }
                    }
                },
            )
        return httpx.Response(404)

    return handler


def _client_with_checkpoints(ckpts: list[str]) -> ComfyUIClient:
    return ComfyUIClient(
        base_url="http://comfyui.test",
        http=httpx.Client(transport=httpx.MockTransport(_checkpoint_info_handler(ckpts))),
    )


CKPTS = [
    "RealVisXL_V5.0_fp16.safetensors",
    "ltx-video-2b-v0.9.5.safetensors",
    "ltxv-13b-0.9.7-distilled-fp8.safetensors",
]


def test_resolve_checkpoint_keeps_configured_when_available(monkeypatch):
    monkeypatch.setenv("COMFYUI_T2I_CHECKPOINT", "RealVisXL_V5.0_fp16.safetensors")
    assert _client_with_checkpoints(CKPTS).resolve_checkpoint("t2i") == (
        "RealVisXL_V5.0_fp16.safetensors"
    )


def test_resolve_checkpoint_falls_back_to_sdxl_like_model(monkeypatch):
    """回归：配置 sd_xl_base_1.0 而云机只有 RealVisXL/LTX → 自动选 RealVisXL。

    旧行为是每个分镜都 HTTP 400 value_not_in_list，整段产不出图。
    """
    monkeypatch.setenv("COMFYUI_T2I_CHECKPOINT", "sd_xl_base_1.0.safetensors")
    assert _client_with_checkpoints(CKPTS).resolve_checkpoint("t2i") == (
        "RealVisXL_V5.0_fp16.safetensors"
    )


def test_resolve_ltx_checkpoint_picks_ltx_family(monkeypatch):
    """配置的 LTX 名不可用时，优先 13B distilled（质量档），而不是 2B。"""
    monkeypatch.setenv("COMFYUI_LTX_CHECKPOINT", "missing-ltx.safetensors")
    assert _client_with_checkpoints(CKPTS).resolve_checkpoint("ltx") == (
        "ltxv-13b-0.9.7-distilled-fp8.safetensors"
    )


def test_resolve_ltx_checkpoint_falls_back_to_2b_when_only_2b(monkeypatch):
    monkeypatch.setenv("COMFYUI_LTX_CHECKPOINT", "missing-ltx.safetensors")
    only_2b = [
        "RealVisXL_V5.0_fp16.safetensors",
        "ltx-video-2b-v0.9.5.safetensors",
    ]
    assert _client_with_checkpoints(only_2b).resolve_checkpoint("ltx") == (
        "ltx-video-2b-v0.9.5.safetensors"
    )


def test_resolve_checkpoint_returns_configured_when_query_fails(monkeypatch):
    """object_info 不可用时保持原配置（旧版 ComfyUI 不能被新逻辑拖挂）。"""
    monkeypatch.setenv("COMFYUI_T2I_CHECKPOINT", "sd_xl_base_1.0.safetensors")
    client = ComfyUIClient(
        base_url="http://comfyui.test",
        http=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404))),
    )
    assert client.resolve_checkpoint("t2i") == "sd_xl_base_1.0.safetensors"


def test_workflow_checkpoint_override():
    from prompt2vid.comfyui_client import build_i2v_workflow, build_t2i_workflow

    wf_t2i = build_t2i_workflow(prompt="a", checkpoint="RealVisXL_V5.0_fp16.safetensors")
    assert wf_t2i["44"]["inputs"]["ckpt_name"] == "RealVisXL_V5.0_fp16.safetensors"
    wf_i2v = build_i2v_workflow(
        "x.png", "a", checkpoint="ltxv-13b-0.9.7-distilled-fp8.safetensors"
    )
    assert wf_i2v["44"]["inputs"]["ckpt_name"] == "ltxv-13b-0.9.7-distilled-fp8.safetensors"


def test_combo_values_handles_all_schema_variants():
    """回归：真实 ComfyUI 0.21.x 把输入放在 input.required 下且候选外包一层配置。

    旧解析只认 input.<field> 的扁平写法，导致 preflight 显示"可用: 无"，
    自动校准也拿不到候选列表。
    """
    from prompt2vid.comfyui_client import combo_values

    nested = {
        "input": {
            "required": {
                "ckpt_name": [["a.safetensors", "b.safetensors"], {"tooltip": "x"}]
            }
        }
    }
    assert combo_values(nested, "ckpt_name") == ["a.safetensors", "b.safetensors"]
    # CLIPLoader 的真实写法：只有候选列表，没有配置尾项
    assert combo_values(
        {"input": {"required": {"clip_name": [["t5xxl_fp8_e4m3fn.safetensors"]]}}},
        "clip_name",
    ) == ["t5xxl_fp8_e4m3fn.safetensors"]
    assert combo_values({"input": {"ckpt_name": ["c.safetensors"]}}, "ckpt_name") == [
        "c.safetensors"
    ]
    assert combo_values(
        {"input": {"ckpt_name": {"value": ["d.safetensors"], "tooltip": "x"}}},
        "ckpt_name",
    ) == ["d.safetensors"]
