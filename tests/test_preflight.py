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
    # 不受本地 .env 影响：固定成本测试期望的模型名
    monkeypatch.setenv("COMFYUI_T2I_CHECKPOINT", "sd_xl_base_1.0.safetensors")
    monkeypatch.setenv("COMFYUI_LTX_CHECKPOINT", "ltx-video-2b-v0.9.5.safetensors")
    monkeypatch.setenv("COMFYUI_T5", "t5xxl_fp8_e4m3fn.safetensors")

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
    # 真实 ComfyUI 0.21.x：input.required 嵌套 + [候选, 配置]
    spec3 = {"input": {"required": {"ckpt_name": [["c.safetensors"], {"tooltip": "x"}]}}}
    assert preflight._combo_values(spec3, "ckpt_name") == ["c.safetensors"]


def test_lora_check_optional_and_style_reported(monkeypatch):
    """LoRA 预检是可选项：缺失只标 NG 不拦截；风格约束有提示项。"""
    monkeypatch.setattr(
        "prompt2vid.preflight.shutil.which", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setenv("COMFYUI_T2I_CHECKPOINT", "sd_xl_base_1.0.safetensors")
    monkeypatch.setenv("COMFYUI_LTX_CHECKPOINT", "ltx-video-2b-v0.9.5.safetensors")
    monkeypatch.setenv("COMFYUI_T5", "t5xxl_fp8_e4m3fn.safetensors")
    monkeypatch.setenv("COMFYUI_T2I_LORA", "anime.safetensors")
    monkeypatch.setenv("PROMPT2VID_STYLE", "anime")
    monkeypatch.delenv("COMFYUI_LTX_LORA", raising=False)

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
                "CLIPLoader": {"input": {"clip_name": ["t5xxl_fp8_e4m3fn.safetensors"]}}
            })
        if path in ("/object_info/LTXVImgToVideo", "/object_info/SaveWEBM"):
            return httpx.Response(200, json={path.rsplit("/", 1)[1]: {"input": {}}})
        if path in ("/object_info/LoraLoader", "/object_info/LoraLoaderModelOnly"):
            # loras 目录为空 → 配置的 LoRA 缺失
            return httpx.Response(
                200, json={path.rsplit("/", 1)[1]: {"input": {"lora_name": []}}}
            )
        return httpx.Response(404)

    result = preflight.check_environment(
        base_url="http://comfyui.test", transport=_transport(handler)
    )
    # LoRA 缺失：检查项存在且 NG，但 required=False 不拦截预检
    lora_checks = [c for c in result.checks if "LoRA" in c.name]
    assert lora_checks and all(not c.ok for c in lora_checks)
    assert result.passed, result.format_report()
    # 风格约束有对应提示项
    assert any("风格约束" in c.name for c in result.checks)
