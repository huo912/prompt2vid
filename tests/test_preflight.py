"""preflight 预检测试：用 httpx.MockTransport 模拟 ComfyUI 响应。"""

from __future__ import annotations

import httpx

from prompt2vid import preflight


def _transport(handler):
    return httpx.MockTransport(handler)


def test_all_checks_pass(monkeypatch):
    monkeypatch.setattr(
        "prompt2vid.preflight.shutil.which", lambda name: f"/usr/bin/{name}"
    )
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/system_stats":
            return httpx.Response(200, json={"system": {}})
        if path == "/object_info/CheckpointLoaderSimple":
            return httpx.Response(200, json={
                "CheckpointLoaderSimple": {
                    "input": {"ckpt_name": [
                        "sd_xl_base_1.0.safetensors",
                        "ltx-video-2b-v0.9.5.safetensors",
                    ]}
                }
            })
        if path == "/object_info/CLIPLoader":
            return httpx.Response(200, json={
                "CLIPLoader": {
                    "input": {"clip_name": ["t5xxl_fp8_e4m3fn.safetensors"]}
                }
            })
        if path in ("/object_info/LTXVImgToVideo", "/object_info/SaveWEBM"):
            return httpx.Response(200, json={path.rsplit("/", 1)[1]: {"input": {}}})
        return httpx.Response(404)

    result = preflight.check_environment(
        base_url="http://comfyui.test", transport=_transport(handler)
    )
    assert result.passed, result.format_report()


def test_comfyui_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    result = preflight.check_environment(
        base_url="http://comfyui.test", transport=_transport(handler)
    )
    assert not result.passed
    # ComfyUI 不可达时，其余检查直接跳过（不误报模型缺失）
    names = [c.name for c in result.checks]
    assert any("ComfyUI" in n for n in names)
    assert not any("checkpoint" in n for n in names)


def test_missing_models_reported_with_hints():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/system_stats":
            return httpx.Response(200, json={})
        if path == "/object_info/CheckpointLoaderSimple":
            return httpx.Response(200, json={
                "CheckpointLoaderSimple": {"input": {"ckpt_name": ["other.safetensors"]}}
            })
        if path == "/object_info/CLIPLoader":
            return httpx.Response(200, json={
                "CLIPLoader": {"input": {"clip_name": []}}
            })
        if path in ("/object_info/LTXVImgToVideo", "/object_info/SaveWEBM"):
            return httpx.Response(200, json={path.rsplit("/", 1)[1]: {"input": {}}})
        return httpx.Response(404)

    result = preflight.check_environment(
        base_url="http://comfyui.test", transport=_transport(handler)
    )
    assert not result.passed
    failures = result.failures()
    assert any("T2I checkpoint" in c.name for c in failures)
    assert any("LTX checkpoint" in c.name for c in failures)
    assert any("T5" in c.name for c in failures)
    assert all(c.hint for c in failures if "T5" in c.name or "checkpoint" in c.name)


def test_combo_values_supports_dict_format():
    """新版 ComfyUI 的 combo 候选可能是 {value: [...]} 格式。"""
    spec = {"input": {"ckpt_name": {"value": ["a.safetensors"], "tooltip": "x"}}}
    assert preflight._combo_values(spec, "ckpt_name") == ["a.safetensors"]
    spec2 = {"input": {"ckpt_name": ["b.safetensors"]}}
    assert preflight._combo_values(spec2, "ckpt_name") == ["b.safetensors"]
