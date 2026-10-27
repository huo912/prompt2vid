"""ComfyUI 工作流构建单元测试（不依赖真实 ComfyUI 服务）。"""

from __future__ import annotations

import json

from prompt2vid.comfyui_client import (
    build_i2v_workflow,
    build_t2i_workflow,
    fit_ltx_size,
    image_dimensions,
    ipadapter_enabled,
    ltx_max_pixels,
    ltx_sampling_defaults,
)


def test_t2i_workflow_structure():
    wf = build_t2i_workflow(
        prompt="autumn park, golden leaves",
        negative_prompt="blurry",
        width=768,
        height=1344,
        seed=42,
    )
    # 关键节点齐全
    assert wf["44"]["class_type"] == "CheckpointLoaderSimple"
    assert wf["3"]["class_type"] == "KSampler"
    assert wf["8"]["class_type"] == "VAEDecode"
    assert wf["9"]["class_type"] == "SaveImage"
    # 连线正确：KSampler 用 checkpoint 的 MODEL/CLIP/VAE
    assert wf["3"]["inputs"]["model"] == ["44", 0]
    assert wf["6"]["inputs"]["clip"] == ["44", 1]
    assert wf["8"]["inputs"]["vae"] == ["44", 2]
    # 参数
    assert wf["3"]["inputs"]["seed"] == 42
    assert wf["70"]["inputs"]["width"] == 768
    assert wf["70"]["inputs"]["height"] == 1344
    assert wf["6"]["inputs"]["text"] == "autumn park, golden leaves"
    assert wf["7"]["inputs"]["text"] == "blurry"


def test_i2v_workflow_structure():
    wf = build_i2v_workflow(
        image_name="shot_1.png",
        prompt="autumn park, slow zoom in",
        negative_prompt="",
        width=512,
        height=768,
        length=41,
        seed=42,
        checkpoint="ltx-video-2b-v0.9.5.safetensors",
    )
    # LTX 节点链齐全（对照官方 ltxv_image_to_video_simple.0.9.5.json）
    assert wf["38"]["class_type"] == "CLIPLoader"
    assert wf["38"]["inputs"]["type"] == "ltxv"
    assert wf["95"]["class_type"] == "LTXVImgToVideo"
    assert wf["69"]["class_type"] == "LTXVConditioning"
    assert wf["71"]["class_type"] == "LTXVScheduler"
    assert wf["73"]["class_type"] == "KSamplerSelect"
    assert wf["72"]["class_type"] == "SamplerCustom"
    assert wf["82"]["class_type"] == "LTXVPreprocess"
    assert wf["86"]["class_type"] == "SaveWEBM"
    # SamplerCustom 输入键（API 格式，cfg 为 Float）
    sampler_inputs = wf["72"]["inputs"]
    assert set(sampler_inputs) == {
        "add_noise", "noise_seed", "cfg", "model",
        "positive", "negative", "sampler", "sigmas", "latent_image",
    }
    assert sampler_inputs["cfg"] == 3.0
    # latent 取自 LTXVImgToVideo 的输出 2（LATENT）；LTXVConditioning 只有
    # positive/negative 两个输出，接它的 index 2 会被 ComfyUI 预校验拒绝
    assert sampler_inputs["latent_image"] == ["95", 2]
    assert sampler_inputs["positive"] == ["69", 0]
    assert sampler_inputs["negative"] == ["69", 1]
    # LTXVImgToVideo 参数
    i2v_inputs = wf["95"]["inputs"]
    assert i2v_inputs["width"] == 512
    assert i2v_inputs["height"] == 768
    assert i2v_inputs["length"] == 41
    assert i2v_inputs["image"] == ["82", 0]
    # 首帧图上传名
    assert wf["78"]["inputs"]["image"] == "shot_1.png"
    # 帧率一致性：LTXVConditioning 与 SaveWEBM 都用 25fps
    assert wf["69"]["inputs"]["frame_rate"] == wf["86"]["inputs"]["fps"] == 25.0


def test_i2v_13b_distilled_uses_low_cfg_and_steps(monkeypatch):
    """13B distilled 用 cfg=1.0 / 8 步；2B 仍保持 cfg=3.0 / 10 步。"""
    # 本机 .env 可能被其它测试加载进 os.environ，先清掉显式覆盖
    monkeypatch.delenv("COMFYUI_LTX_CFG", raising=False)
    monkeypatch.delenv("COMFYUI_LTX_STEPS", raising=False)
    distilled = "ltxv-13b-0.9.7-distilled-fp8.safetensors"
    assert ltx_sampling_defaults(distilled) == (1.0, 8)
    assert ltx_sampling_defaults("ltx-video-2b-v0.9.5.safetensors") == (3.0, 10)
    assert ltx_sampling_defaults("ltxv-13b-0.9.7-dev-fp8.safetensors") == (3.0, 30)

    wf = build_i2v_workflow(
        "x.png", "a", width=512, height=768, checkpoint=distilled
    )
    assert wf["72"]["inputs"]["cfg"] == 1.0
    assert wf["71"]["inputs"]["steps"] == 8

    # 显式环境变量仍然可以覆盖自动档位
    monkeypatch.setenv("COMFYUI_LTX_CFG", "1.5")
    monkeypatch.setenv("COMFYUI_LTX_STEPS", "12")
    wf = build_i2v_workflow(
        "x.png", "a", width=512, height=768, checkpoint=distilled
    )
    assert wf["72"]["inputs"]["cfg"] == 1.5
    assert wf["71"]["inputs"]["steps"] == 12


def test_ltx_size_keeps_frame_aspect_ratio():
    """首帧 768x1344 不能再被塞进 512x768（会裁/压人物）。"""
    width, height = fit_ltx_size(768, 1344, 512 * 768)
    assert width % 32 == 0 and height % 32 == 0
    assert width * height <= int(512 * 768 * 1.02)
    assert abs((width / height) - (768 / 1344)) / (768 / 1344) <= 0.02
    # 不应小于 2B 档的旧输出（512x768），否则等于降质
    assert width >= 448 and height >= 768

    # 13B 预算更大，分辨率应更高
    w13, h13 = fit_ltx_size(768, 1344, 576 * 1024)
    assert w13 % 32 == 0 and h13 % 32 == 0
    assert w13 * h13 > width * height
    assert abs((w13 / h13) - (768 / 1344)) / (768 / 1344) <= 0.02


def test_ltx_max_pixels_defaults_by_model(monkeypatch):
    monkeypatch.delenv("COMFYUI_LTX_MAX_PIXELS", raising=False)
    assert ltx_max_pixels("ltx-video-2b-v0.9.5.safetensors") == 512 * 768
    assert ltx_max_pixels("ltxv-13b-0.9.7-distilled-fp8.safetensors") == 576 * 1024
    monkeypatch.setenv("COMFYUI_LTX_MAX_PIXELS", "720")
    assert ltx_max_pixels("ltxv-13b-0.9.7-distilled-fp8.safetensors") == 720


def test_i2v_default_size_follows_t2i_aspect(monkeypatch):
    monkeypatch.delenv("COMFYUI_LTX_WIDTH", raising=False)
    monkeypatch.delenv("COMFYUI_LTX_HEIGHT", raising=False)
    monkeypatch.delenv("COMFYUI_LTX_MAX_PIXELS", raising=False)
    monkeypatch.setenv("COMFYUI_T2I_WIDTH", "768")
    monkeypatch.setenv("COMFYUI_T2I_HEIGHT", "1344")
    monkeypatch.setenv("COMFYUI_LTX_CHECKPOINT", "ltx-video-2b-v0.9.5.safetensors")
    wf = build_i2v_workflow("x.png", "a")
    width = wf["95"]["inputs"]["width"]
    height = wf["95"]["inputs"]["height"]
    assert width % 32 == 0 and height % 32 == 0
    assert abs((width / height) - (768 / 1344)) / (768 / 1344) <= 0.02


def test_image_dimensions_reads_png_header(tmp_path):
    png = tmp_path / "frame.png"
    header = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        + (768).to_bytes(4, "big")
        + (1344).to_bytes(4, "big")
    )
    png.write_bytes(header)  # 只需头部，函数不会读像素数据
    assert image_dimensions(png) == (768, 1344)
    assert image_dimensions(tmp_path / "missing.png") is None
    (tmp_path / "not_a_png.bin").write_bytes(b"not a png")
    assert image_dimensions(tmp_path / "not_a_png.bin") is None


# 各节点实际输出槽数量（对照 ComfyUI 0.21.1 的 RETURN_TYPES）。
# 引用超出槽数的连线会被 ComfyUI 预校验拒绝（HTTP 400 list index out of range），
# 这类错误本地就该拦住，不必等真实服务报错。
_NODE_OUTPUT_SLOTS = {
    "CLIPLoader": 1,
    "CLIPTextEncode": 1,
    "CheckpointLoaderSimple": 3,
    "LoadImage": 2,
    "LTXVPreprocess": 1,
    "LTXVImgToVideo": 3,
    "LTXVConditioning": 2,
    "LTXVScheduler": 1,
    "KSamplerSelect": 1,
    "SamplerCustom": 2,
    "VAEDecode": 1,
    "SaveWEBM": 0,
}


def test_i2v_workflow_links_within_output_slots():
    """每条连线引用的输出槽必须真实存在（回归：latent 误接 LTXVConditioning 的 2 号槽）。"""
    wf = build_i2v_workflow("x.png", "a")
    for node_id, node in wf.items():
        assert node["class_type"] in _NODE_OUTPUT_SLOTS, node["class_type"]
        for name, value in node["inputs"].items():
            if not (isinstance(value, list) and len(value) == 2):
                continue
            src_id, slot = value
            src_class = wf[src_id]["class_type"]
            slots = _NODE_OUTPUT_SLOTS[src_class]
            assert slot < slots, (
                f"节点 {node_id}({node['class_type']}).{name} 引用了 "
                f"{src_id}({src_class}) 的输出槽 {slot}，但该节点只有 {slots} 个输出"
            )


def test_workflows_are_json_serializable():
    for wf in (
        build_t2i_workflow("a", "b"),
        build_i2v_workflow("x.png", "a", "b"),
        build_t2i_workflow("a", reference={"image_name": "ref.png"}),
    ):
        json.dumps(wf)  # 不抛异常即可


# ------------------------------------------------------------ IPAdapter 参考注入

def test_t2i_workflow_without_reference_unchanged():
    """无参考时与原工作流完全一致（向后兼容）。"""
    wf = build_t2i_workflow(prompt="a", seed=1)
    assert not any(key.startswith("ref_") for key in wf)
    assert wf["3"]["inputs"]["model"] == ["44", 0]


def test_t2i_workflow_with_reference():
    wf = build_t2i_workflow(
        prompt="a",
        seed=1,
        reference={
            "image_name": "character_ref_1.png",
            "weight": 0.9,
            "preset": "PLUS FACE (portraits)",
        },
    )
    assert wf["ref_image"]["class_type"] == "LoadImage"
    assert wf["ref_image"]["inputs"]["image"] == "character_ref_1.png"
    assert wf["ref_loader"]["class_type"] == "IPAdapterUnifiedLoader"
    assert wf["ref_loader"]["inputs"]["ipadapter"] == "PLUS FACE (portraits)"
    assert wf["ref_loader"]["inputs"]["model"] == ["44", 0]
    assert wf["ref_apply"]["class_type"] == "IPAdapter"
    assert wf["ref_apply"]["inputs"]["weight"] == 0.9
    assert wf["ref_apply"]["inputs"]["image"] == ["ref_image", 0]
    assert wf["ref_apply"]["inputs"]["model"] == ["ref_loader", 0]
    # KSampler 的 model 改接 adapter 输出，其余连线不变
    assert wf["3"]["inputs"]["model"] == ["ref_apply", 0]
    assert wf["8"]["inputs"]["samples"] == ["3", 0]
    assert wf["9"]["inputs"]["images"] == ["8", 0]


def test_t2i_workflow_reference_defaults_from_env(monkeypatch):
    monkeypatch.setenv("COMFYUI_IPADAPTER_PRESET", "FACEID PLUS v2")
    monkeypatch.setenv("COMFYUI_IPADAPTER_WEIGHT", "0.6")
    wf = build_t2i_workflow(prompt="a", reference={"image_name": "ref.png"})
    assert wf["ref_loader"]["inputs"]["ipadapter"] == "FACEID PLUS v2"
    assert wf["ref_apply"]["inputs"]["weight"] == 0.6


def test_t2i_workflow_empty_reference_ignored():
    """reference 缺 image_name 时不注入（生成阶段失败回退的防御）。"""
    wf = build_t2i_workflow(prompt="a", reference={"image_name": ""})
    assert not any(key.startswith("ref_") for key in wf)


def test_ipadapter_enabled(monkeypatch):
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("COMFYUI_IPADAPTER", value)
        assert ipadapter_enabled() is True
    for value in ("", "0", "false", "off"):
        monkeypatch.setenv("COMFYUI_IPADAPTER", value)
        assert ipadapter_enabled() is False
