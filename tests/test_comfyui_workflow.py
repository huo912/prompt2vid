"""ComfyUI 工作流构建单元测试（不依赖真实 ComfyUI 服务）。"""

from __future__ import annotations

import json

from prompt2vid.comfyui_client import (
    build_i2v_workflow,
    build_t2i_workflow,
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
    ):
        json.dumps(wf)  # 不抛异常即可
