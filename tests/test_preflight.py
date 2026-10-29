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


def _gpu_handler(vram_bytes: int):
    """带 CUDA 设备的 system_stats + 模型齐备的 object_info mock。"""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/system_stats":
            return httpx.Response(200, json={
                "system": {},
                "devices": [{
                    "name": "NVIDIA GeForce RTX Test", "type": "cuda",
                    "vram_total": vram_bytes,
                }],
            })
        if path == "/object_info/CheckpointLoaderSimple":
            return httpx.Response(200, json={
                "CheckpointLoaderSimple": {"input": {"ckpt_name": [
                    "sd_xl_base_1.0.safetensors",
                    "ltx-video-2b-v0.9.5.safetensors",
                    "ltxv-13b-0.9.7-distilled-fp8.safetensors",
                ]}}
            })
        if path == "/object_info/CLIPLoader":
            return httpx.Response(200, json={
                "CLIPLoader": {"input": {"clip_name": ["t5xxl_fp8_e4m3fn.safetensors"]}}
            })
        if path in ("/object_info/LTXVImgToVideo", "/object_info/SaveWEBM",
                    "/object_info/LoraLoader", "/object_info/LoraLoaderModelOnly"):
            return httpx.Response(200, json={path.rsplit("/", 1)[1]: {"input": {}}})
        return httpx.Response(404)

    return handler


def test_gpu_checks_warn_on_overspec(monkeypatch):
    """8GB 显存 + 13B + 720p 预算：档位校对警告出现但不拦截预检。"""
    monkeypatch.setattr(
        "prompt2vid.preflight.shutil.which", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setenv("COMFYUI_T2I_CHECKPOINT", "sd_xl_base_1.0.safetensors")
    monkeypatch.setenv("COMFYUI_LTX_CHECKPOINT", "ltxv-13b-0.9.7-distilled-fp8.safetensors")
    monkeypatch.setenv("COMFYUI_T5", "t5xxl_fp8_e4m3fn.safetensors")
    monkeypatch.setenv("COMFYUI_LTX_MAX_PIXELS", "921600")

    result = preflight.check_environment(
        base_url="http://comfyui.test",
        transport=_transport(_gpu_handler(8 * 2**30)),
    )
    assert result.passed, result.format_report()  # 警告全部 required=False
    names = [c.name for c in result.checks]
    assert any("GPU 显存" in n for n in names)
    warn_13b = next(c for c in result.checks if "LTX 13B" in c.name)
    assert not warn_13b.ok and not warn_13b.required
    assert "2B" in warn_13b.hint
    warn_pixels = next(c for c in result.checks if "分辨率预算" in c.name)
    assert not warn_pixels.ok and not warn_pixels.required


def test_gpu_checks_suggest_upgrade_when_vram_rich(monkeypatch):
    """32GB 显存 + 2B（预算 393216）：提示可以上 720p（信息项）。"""
    monkeypatch.setattr(
        "prompt2vid.preflight.shutil.which", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setenv("COMFYUI_T2I_CHECKPOINT", "sd_xl_base_1.0.safetensors")
    monkeypatch.setenv("COMFYUI_LTX_CHECKPOINT", "ltx-video-2b-v0.9.5.safetensors")
    monkeypatch.setenv("COMFYUI_T5", "t5xxl_fp8_e4m3fn.safetensors")
    monkeypatch.delenv("COMFYUI_LTX_MAX_PIXELS", raising=False)

    result = preflight.check_environment(
        base_url="http://comfyui.test",
        transport=_transport(_gpu_handler(32 * 2**30)),
    )
    assert result.passed
    hint = next(c for c in result.checks if "分辨率预算" in c.name)
    assert hint.ok and "921600" in hint.detail


# ------------------------------------------------------------ IndexTTS / LatentSync


def _full_handler(extra_nodes: set[str] | None = None):
    """模型齐备 + 可选自定义节点齐备的 ComfyUI mock。"""
    extra = extra_nodes or set()
    node_names = {
        "CheckpointLoaderSimple", "CLIPLoader", "LTXVImgToVideo", "SaveWEBM",
        "LoraLoader", "LoraLoaderModelOnly",
    } | extra

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/system_stats":
            return httpx.Response(200, json={"system": {}})
        if path.startswith("/object_info/"):
            name = path.rsplit("/", 1)[1]
            if name not in node_names:
                return httpx.Response(404)
            body: dict = {"input": {}}
            if name == "CheckpointLoaderSimple":
                body = {"input": {"ckpt_name": [
                    "sd_xl_base_1.0.safetensors",
                    "ltx-video-2b-v0.9.5.safetensors",
                ]}}
            elif name == "CLIPLoader":
                body = {"input": {"clip_name": ["t5xxl_fp8_e4m3fn.safetensors"]}}
            elif name == "TimbreAudioLoader":
                body = {"input": {"audio_file": ["甜美女声.mp3", "说书人.WAV"]}}
            return httpx.Response(200, json={name: body})
        return httpx.Response(404)

    return handler


_TTS_NODES = {"IndexTTS25BaseNode", "TimbreAudioLoader"}
_LIPSYNC_NODES = {"LatentSyncNode", "VideoLengthAdjuster",
                  "VHS_LoadVideo", "VHS_VideoCombine"}


def test_preflight_reports_indextts_engine(monkeypatch):
    monkeypatch.setattr(
        "prompt2vid.preflight.shutil.which", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setenv("COMFYUI_T2I_CHECKPOINT", "sd_xl_base_1.0.safetensors")
    monkeypatch.setenv("COMFYUI_LTX_CHECKPOINT", "ltx-video-2b-v0.9.5.safetensors")
    monkeypatch.setenv("COMFYUI_T5", "t5xxl_fp8_e4m3fn.safetensors")
    monkeypatch.delenv("PROMPT2VID_TTS_ENGINE", raising=False)
    monkeypatch.delenv("PROMPT2VID_TTS_TIMBRE", raising=False)
    monkeypatch.delenv("PROMPT2VID_TTS_REF_AUDIO", raising=False)
    monkeypatch.delenv("PROMPT2VID_TTS_BASE_URL", raising=False)
    monkeypatch.delenv("PROMPT2VID_LIPSYNC", raising=False)

    result = preflight.check_environment(
        base_url="http://comfyui.test",
        transport=_transport(_full_handler(_TTS_NODES | _LIPSYNC_NODES)),
    )
    assert result.passed, result.format_report()
    # IndexTTS 主引擎与口型同步节点都报 OK
    tts_check = next(c for c in result.checks if "IndexTTS-2.5" in c.name)
    assert tts_check.ok and "甜美女声" in tts_check.detail
    lipsync_check = next(c for c in result.checks if "LatentSync" in c.name)
    assert lipsync_check.ok


def test_preflight_warns_on_missing_lipsync_nodes(monkeypatch):
    """auto 模式缺 LatentSync 节点：NG 但 required=False，预检仍通过。"""
    monkeypatch.setattr(
        "prompt2vid.preflight.shutil.which", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setenv("COMFYUI_T2I_CHECKPOINT", "sd_xl_base_1.0.safetensors")
    monkeypatch.setenv("COMFYUI_LTX_CHECKPOINT", "ltx-video-2b-v0.9.5.safetensors")
    monkeypatch.setenv("COMFYUI_T5", "t5xxl_fp8_e4m3fn.safetensors")
    monkeypatch.setenv("PROMPT2VID_LIPSYNC", "auto")

    result = preflight.check_environment(
        base_url="http://comfyui.test",
        transport=_transport(_full_handler(_TTS_NODES)),
    )
    assert result.passed, result.format_report()
    lipsync_check = next(c for c in result.checks if "LatentSync" in c.name)
    assert not lipsync_check.ok
    assert not lipsync_check.required
    assert "auto" in lipsync_check.detail or "跳过" in lipsync_check.detail


def test_preflight_blocks_when_lipsync_on_but_missing(monkeypatch):
    """on 模式缺节点是配置错误：required=True 阻断预检。"""
    monkeypatch.setattr(
        "prompt2vid.preflight.shutil.which", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setenv("COMFYUI_T2I_CHECKPOINT", "sd_xl_base_1.0.safetensors")
    monkeypatch.setenv("COMFYUI_LTX_CHECKPOINT", "ltx-video-2b-v0.9.5.safetensors")
    monkeypatch.setenv("COMFYUI_T5", "t5xxl_fp8_e4m3fn.safetensors")
    monkeypatch.setenv("PROMPT2VID_LIPSYNC", "on")

    result = preflight.check_environment(
        base_url="http://comfyui.test",
        transport=_transport(_full_handler(set())),
    )
    assert not result.passed
    lipsync_check = next(c for c in result.checks if "LatentSync" in c.name)
    assert lipsync_check.required and not lipsync_check.ok
    assert "LOCAL_SETUP" in lipsync_check.hint


def test_preflight_reports_unknown_timbre(monkeypatch):
    """配置的音色不在插件自带列表：NG + 提示可选音色与参考音频。"""
    monkeypatch.setattr(
        "prompt2vid.preflight.shutil.which", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setenv("COMFYUI_T2I_CHECKPOINT", "sd_xl_base_1.0.safetensors")
    monkeypatch.setenv("COMFYUI_LTX_CHECKPOINT", "ltx-video-2b-v0.9.5.safetensors")
    monkeypatch.setenv("COMFYUI_T5", "t5xxl_fp8_e4m3fn.safetensors")
    monkeypatch.setenv("PROMPT2VID_TTS_TIMBRE", "不存在的音色.wav")
    monkeypatch.delenv("PROMPT2VID_TTS_REF_AUDIO", raising=False)

    result = preflight.check_environment(
        base_url="http://comfyui.test",
        transport=_transport(_full_handler(_TTS_NODES | _LIPSYNC_NODES)),
    )
    tts_check = next(c for c in result.checks if "IndexTTS-2.5" in c.name)
    assert not tts_check.ok
    assert "不存在的音色.wav" in tts_check.detail
    assert "PROMPT2VID_TTS_REF_AUDIO" in tts_check.hint
