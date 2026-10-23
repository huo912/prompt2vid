"""ComfyUI 工作流构建单元测试（不依赖真实 ComfyUI 服务）。"""

from __future__ import annotations

import json

from prompt2vid.comfyui_client import (
    build_i2v_workflow,
    build_t2i_workflow,
    ipadapter_enabled,
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
    # LTXVConditioning 输出 3 个：positive(0)/negative(1)/LATENT(2)
    assert sampler_inputs["latent_image"] == ["69", 2]
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
