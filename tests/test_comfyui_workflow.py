"""ComfyUI 工作流构建单元测试（不依赖真实 ComfyUI 服务）。"""

from __future__ import annotations

import json

import pytest

from prompt2vid.comfyui_client import (
    build_i2v_workflow,
    build_t2i_workflow,
    fit_ltx_size,
    image_dimensions,
    ipadapter_enabled,
    ltx_max_pixels,
    ltx_sampling_defaults,
)


@pytest.fixture(autouse=True)
def _no_style_no_lora(monkeypatch):
    """基线：风格与 LoRA 关闭，保证既有精确断言不受本机 .env 影响。

    其他测试（如 test_crew_assembly）可能把仓库 .env 加载进 os.environ，
    其中含 PROMPT2VID_STYLE / COMFYUI_*_LORA。
    """
    for name in (
        "PROMPT2VID_STYLE",
        "COMFYUI_T2I_LORA",
        "COMFYUI_T2I_LORA_STRENGTH",
        "COMFYUI_LTX_LORA",
        "COMFYUI_LTX_LORA_STRENGTH",
    ):
        monkeypatch.delenv(name, raising=False)


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


def test_t2i_sampling_params_from_env(monkeypatch):
    """COMFYUI_T2I_CFG / _SAMPLER / _SCHEDULER 覆盖默认 7.0/dpmpp_2m/karras。"""
    for name in ("COMFYUI_T2I_CFG", "COMFYUI_T2I_SAMPLER",
                 "COMFYUI_T2I_SCHEDULER"):
        monkeypatch.delenv(name, raising=False)
    wf = build_t2i_workflow(prompt="a", seed=1)
    assert wf["3"]["inputs"]["cfg"] == 7.0
    assert wf["3"]["inputs"]["sampler_name"] == "dpmpp_2m"
    assert wf["3"]["inputs"]["scheduler"] == "karras"

    monkeypatch.setenv("COMFYUI_T2I_CFG", "6")
    monkeypatch.setenv("COMFYUI_T2I_SAMPLER", "dpmpp_2m_sde")
    monkeypatch.setenv("COMFYUI_T2I_SCHEDULER", "sgm_uniform")
    wf = build_t2i_workflow(prompt="a", seed=1)
    assert wf["3"]["inputs"]["cfg"] == 6.0
    assert wf["3"]["inputs"]["sampler_name"] == "dpmpp_2m_sde"
    assert wf["3"]["inputs"]["scheduler"] == "sgm_uniform"


def test_i2v_webm_crf_and_scheduler_knobs(monkeypatch):
    """WEBM CRF 默认 20（旧版 32 压缩痕迹重）；shift/采样器可配。"""
    for name in ("COMFYUI_WEBM_CRF", "COMFYUI_LTX_MAX_SHIFT",
                 "COMFYUI_LTX_BASE_SHIFT", "COMFYUI_LTX_SAMPLER"):
        monkeypatch.delenv(name, raising=False)
    wf = build_i2v_workflow("x.png", "a", width=512, height=768)
    assert wf["86"]["inputs"]["crf"] == 20.0
    assert wf["71"]["inputs"]["max_shift"] == 2.05
    assert wf["71"]["inputs"]["base_shift"] == 0.95
    assert wf["73"]["inputs"]["sampler_name"] == "euler"

    monkeypatch.setenv("COMFYUI_WEBM_CRF", "24")
    monkeypatch.setenv("COMFYUI_LTX_MAX_SHIFT", "2.3")
    monkeypatch.setenv("COMFYUI_LTX_BASE_SHIFT", "1.1")
    monkeypatch.setenv("COMFYUI_LTX_SAMPLER", "euler_ancestral")
    wf = build_i2v_workflow("x.png", "a", width=512, height=768)
    assert wf["86"]["inputs"]["crf"] == 24.0
    assert wf["71"]["inputs"]["max_shift"] == 2.3
    assert wf["71"]["inputs"]["base_shift"] == 1.1
    assert wf["73"]["inputs"]["sampler_name"] == "euler_ancestral"


def test_i2v_frame_rate_follows_ltx_fps_env(monkeypatch):
    """COMFYUI_LTX_FPS 同时驱动 LTXVConditioning 与 SaveWEBM（回退 bug）。"""
    monkeypatch.delenv("COMFYUI_LTX_FPS", raising=False)
    wf = build_i2v_workflow("x.png", "a", width=512, height=768)
    assert wf["69"]["inputs"]["frame_rate"] == wf["86"]["inputs"]["fps"] == 25.0

    monkeypatch.setenv("COMFYUI_LTX_FPS", "30")
    wf = build_i2v_workflow("x.png", "a", width=512, height=768)
    assert wf["69"]["inputs"]["frame_rate"] == wf["86"]["inputs"]["fps"] == 30.0


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
    "EmptyLatentImage": 1,
    "IPAdapter": 1,
    "IPAdapterUnifiedLoader": 2,
    "KSampler": 1,
    "LoadImage": 2,
    "LoraLoader": 2,
    "LoraLoaderModelOnly": 1,
    "LTXVPreprocess": 1,
    "LTXVImgToVideo": 3,
    "LTXVConditioning": 2,
    "LTXVScheduler": 1,
    "KSamplerSelect": 1,
    "SamplerCustom": 2,
    "VAEDecode": 1,
    "SaveImage": 0,
    "SaveWEBM": 0,
    # IndexTTS-2.5（chenpipi0807/ComfyUI-Index-TTS）
    "TimbreAudioLoader": 1,          # AUDIO
    "LoadAudio": 1,                  # AUDIO（ComfyUI 核心）
    "IndexTTS25CacheControlNode": 1, # cache_control DICT
    "IndexTTS25BaseNode": 3,         # audio / seed / subtitle
    "SaveAudio": 0,
    # LatentSync 口型同步（ShmuelRonen/ComfyUI-LatentSyncWrapper + VHS）
    "VHS_LoadVideo": 4,              # IMAGE / frame_count / audio / video_info
    "VideoLengthAdjuster": 2,        # IMAGE / AUDIO
    "LatentSyncNode": 2,             # IMAGE / AUDIO
    "VHS_VideoCombine": 1,           # VHS_FILENAMES
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


# ------------------------------------------------------------ 固定风格约束

def test_t2i_workflow_applies_style(monkeypatch):
    """PROMPT2VID_STYLE=anime：正向追加风格词，负向追加排斥词。"""
    monkeypatch.setenv("PROMPT2VID_STYLE", "anime")
    wf = build_t2i_workflow(prompt="a cat", negative_prompt="", seed=1)
    assert wf["6"]["inputs"]["text"].startswith("a cat, ")
    assert "anime style" in wf["6"]["inputs"]["text"]
    # 负向：默认词保留 + 风格排斥词追加
    assert "bad anatomy" in wf["7"]["inputs"]["text"]
    assert "photorealistic" in wf["7"]["inputs"]["text"]


def test_i2v_workflow_applies_style(monkeypatch):
    monkeypatch.setenv("PROMPT2VID_STYLE", "anime")
    wf = build_i2v_workflow("x.png", "a cat, slow zoom in")
    assert "anime style" in wf["6"]["inputs"]["text"]
    assert "a cat, slow zoom in" in wf["6"]["inputs"]["text"]
    assert "3d render" in wf["7"]["inputs"]["text"]


def test_workflow_style_disabled_by_default():
    """未配置风格时提示词原样进入工作流（向后兼容）。"""
    wf = build_t2i_workflow(prompt="a cat", seed=1)
    assert wf["6"]["inputs"]["text"] == "a cat"
    assert "photorealistic" not in wf["7"]["inputs"]["text"]


# ------------------------------------------------------------ LoRA 风格模型

def test_t2i_workflow_with_lora_param():
    """显式传 LoRA 名：插入 LoraLoader，文本编码与采样器接到 LoRA 之后。"""
    wf = build_t2i_workflow(prompt="a", seed=1, lora="anime.safetensors")
    assert wf["lora"]["class_type"] == "LoraLoader"
    assert wf["lora"]["inputs"]["lora_name"] == "anime.safetensors"
    assert wf["lora"]["inputs"]["strength_model"] == 0.8  # 默认权重
    assert wf["lora"]["inputs"]["strength_clip"] == 0.8
    assert wf["lora"]["inputs"]["model"] == ["44", 0]
    assert wf["lora"]["inputs"]["clip"] == ["44", 1]
    # 连线改接 LoRA 输出
    assert wf["6"]["inputs"]["clip"] == ["lora", 1]
    assert wf["7"]["inputs"]["clip"] == ["lora", 1]
    assert wf["3"]["inputs"]["model"] == ["lora", 0]
    # VAE 仍来自 checkpoint
    assert wf["8"]["inputs"]["vae"] == ["44", 2]


def test_t2i_workflow_lora_from_env_and_strength(monkeypatch):
    monkeypatch.setenv("COMFYUI_T2I_LORA", "anime.safetensors")
    monkeypatch.setenv("COMFYUI_T2I_LORA_STRENGTH", "1.5")
    wf = build_t2i_workflow(prompt="a", seed=1)
    assert wf["lora"]["inputs"]["lora_name"] == "anime.safetensors"
    assert wf["lora"]["inputs"]["strength_model"] == 1.5


def test_t2i_workflow_lora_empty_string_disables(monkeypatch):
    """空串（resolve_lora 降级的返回值）= 显式关闭 LoRA，不读环境变量。"""
    monkeypatch.setenv("COMFYUI_T2I_LORA", "anime.safetensors")
    wf = build_t2i_workflow(prompt="a", seed=1, lora="")
    assert "lora" not in wf
    assert wf["3"]["inputs"]["model"] == ["44", 0]


def test_t2i_workflow_lora_plus_reference_chain():
    """LoRA 与 IPAdapter 叠加：Checkpoint → LoRA → IPAdapter → KSampler。"""
    wf = build_t2i_workflow(
        prompt="a",
        seed=1,
        lora="anime.safetensors",
        reference={"image_name": "ref.png"},
    )
    assert wf["lora"]["class_type"] == "LoraLoader"
    assert wf["ref_loader"]["class_type"] == "IPAdapterUnifiedLoader"
    # IPAdapter 接 LoRA 之后的 MODEL，而不是 checkpoint 直出
    assert wf["ref_loader"]["inputs"]["model"] == ["lora", 0]
    # 文本编码仍接 LoRA 的 CLIP
    assert wf["6"]["inputs"]["clip"] == ["lora", 1]
    assert wf["3"]["inputs"]["model"] == ["ref_apply", 0]


def test_i2v_workflow_with_lora_param():
    """视频侧 LoRA：LoraLoaderModelOnly 只接 MODEL，SamplerCustom 接其后。"""
    wf = build_i2v_workflow(
        "x.png", "a", width=512, height=768, lora="shinkai.safetensors"
    )
    assert wf["lora"]["class_type"] == "LoraLoaderModelOnly"
    assert wf["lora"]["inputs"]["lora_name"] == "shinkai.safetensors"
    assert wf["lora"]["inputs"]["strength_model"] == 0.8
    assert wf["lora"]["inputs"]["model"] == ["44", 0]
    assert wf["72"]["inputs"]["model"] == ["lora", 0]
    # T5 文本编码不经 LoRA（LTX 的 CLIP 由独立 CLIPLoader 提供）
    assert wf["6"]["inputs"]["clip"] == ["38", 0]


def test_i2v_workflow_lora_from_env(monkeypatch):
    monkeypatch.setenv("COMFYUI_LTX_LORA", "shinkai.safetensors")
    monkeypatch.setenv("COMFYUI_LTX_LORA_STRENGTH", "0.9")
    wf = build_i2v_workflow("x.png", "a", width=512, height=768)
    assert wf["lora"]["inputs"]["lora_name"] == "shinkai.safetensors"
    assert wf["lora"]["inputs"]["strength_model"] == 0.9
    assert wf["72"]["inputs"]["model"] == ["lora", 0]


def test_workflow_links_within_slots_with_lora():
    """挂上 LoRA / 风格后的工作流连线也必须落在真实输出槽内。"""
    for wf in (
        build_t2i_workflow("a", lora="anime.safetensors"),
        build_t2i_workflow(
            "a", lora="anime.safetensors", reference={"image_name": "ref.png"}
        ),
        build_i2v_workflow("x.png", "a", lora="shinkai.safetensors"),
    ):
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


# ------------------------------------------------------------ IndexTTS 配音


def test_tts_workflow_structure_with_bundled_timbre(monkeypatch):
    from prompt2vid.comfyui_client import build_tts_workflow

    for name in ("PROMPT2VID_TTS_TIMBRE", "PROMPT2VID_TTS_LANG",
                 "PROMPT2VID_TTS_SPEED", "PROMPT2VID_TTS_SILENCE_MS",
                 "PROMPT2VID_TTS_KEEP_CACHED", "PROMPT2VID_TTS_REF_AUDIO"):
        monkeypatch.delenv(name, raising=False)

    wf = build_tts_workflow("秋天的公园")

    # 自带音色：TimbreAudioLoader → (缓存控制) → IndexTTS25BaseNode → SaveAudio
    assert wf["timbre"]["class_type"] == "TimbreAudioLoader"
    assert wf["timbre"]["inputs"]["audio_file"] == "甜美女声.mp3"
    assert wf["cache"]["class_type"] == "IndexTTS25CacheControlNode"
    assert wf["cache"]["inputs"]["keep_models_cached"] is True
    tts_inputs = wf["tts"]["inputs"]
    assert wf["tts"]["class_type"] == "IndexTTS25BaseNode"
    assert tts_inputs["text"] == "秋天的公园"
    assert tts_inputs["reference_audio"] == ["timbre", 0]
    assert tts_inputs["cache_control"] == ["cache", 0]
    assert tts_inputs["lang"] == "ZH"
    assert tts_inputs["duration_factor"] == 1.0
    assert tts_inputs["interval_silence_ms"] == 200
    assert wf["save"]["inputs"]["audio"] == ["tts", 0]
    # JSON 可序列化（/prompt 的 body）
    json.dumps(wf, ensure_ascii=False)


def test_tts_workflow_reference_audio_and_env(monkeypatch):
    """自定义参考音频走核心 LoadAudio（声音克隆），且不挂缓存节点。"""
    from prompt2vid.comfyui_client import build_tts_workflow

    monkeypatch.setenv("PROMPT2VID_TTS_LANG", "EN")
    monkeypatch.setenv("PROMPT2VID_TTS_SPEED", "1.2")
    monkeypatch.setenv("PROMPT2VID_TTS_SILENCE_MS", "300")
    monkeypatch.setenv("PROMPT2VID_TTS_KEEP_CACHED", "false")

    wf = build_tts_workflow("hello", reference_audio="my_voice.wav")

    assert wf["timbre"]["class_type"] == "LoadAudio"
    assert wf["timbre"]["inputs"]["audio"] == "my_voice.wav"
    assert "cache" not in wf
    assert "cache_control" not in wf["tts"]["inputs"]
    assert wf["tts"]["inputs"]["lang"] == "EN"
    assert wf["tts"]["inputs"]["duration_factor"] == 1.2
    assert wf["tts"]["inputs"]["interval_silence_ms"] == 300


def test_tts_workflow_links_within_output_slots():
    from prompt2vid.comfyui_client import build_tts_workflow

    for wf in (
        build_tts_workflow("你好"),
        build_tts_workflow("hello", reference_audio="ref.wav"),
    ):
        for node_id, node in wf.items():
            assert node["class_type"] in _NODE_OUTPUT_SLOTS, node["class_type"]
            for name, value in node["inputs"].items():
                if not (isinstance(value, list) and len(value) == 2):
                    continue
                src_id, slot = value
                src_class = wf[src_id]["class_type"]
                assert slot < _NODE_OUTPUT_SLOTS[src_class]


# ------------------------------------------------------------ LatentSync 口型同步


def test_lipsync_workflow_structure(monkeypatch):
    from prompt2vid.comfyui_client import build_lipsync_workflow

    for name in ("PROMPT2VID_LIPSYNC_EXPRESSION", "PROMPT2VID_LIPSYNC_STEPS",
                 "PROMPT2VID_LIPSYNC_CRF", "COMFYUI_LTX_FPS"):
        monkeypatch.delenv(name, raising=False)

    wf = build_lipsync_workflow("shot_1.webm", "shot_1.mp3", seed=7)

    # VHS_LoadVideo(25fps) + LoadAudio → VideoLengthAdjuster → LatentSync → VHS_VideoCombine
    assert wf["load_video"]["class_type"] == "VHS_LoadVideo"
    assert wf["load_video"]["inputs"]["video"] == "shot_1.webm"
    assert wf["load_video"]["inputs"]["force_rate"] == 25.0
    assert wf["load_audio"]["class_type"] == "LoadAudio"
    assert wf["load_audio"]["inputs"]["audio"] == "shot_1.mp3"
    adjust = wf["adjust"]
    assert adjust["class_type"] == "VideoLengthAdjuster"
    assert adjust["inputs"]["mode"] == "pingpong"
    assert adjust["inputs"]["fps"] == 25.0
    assert adjust["inputs"]["images"] == ["load_video", 0]
    assert adjust["inputs"]["audio"] == ["load_audio", 0]
    sync = wf["latentsync"]
    assert sync["class_type"] == "LatentSyncNode"
    assert sync["inputs"]["images"] == ["adjust", 0]
    assert sync["inputs"]["audio"] == ["adjust", 1]
    assert sync["inputs"]["seed"] == 7
    assert sync["inputs"]["lips_expression"] == 1.5
    assert sync["inputs"]["inference_steps"] == 20
    save = wf["save"]
    assert save["class_type"] == "VHS_VideoCombine"
    assert save["inputs"]["images"] == ["latentsync", 0]
    assert save["inputs"]["frame_rate"] == 25.0
    assert save["inputs"]["format"] == "video/h264-mp4"
    assert save["inputs"]["save_output"] is True
    assert save["inputs"]["crf"] == 19
    json.dumps(wf, ensure_ascii=False)


def test_lipsync_workflow_env_overrides(monkeypatch):
    from prompt2vid.comfyui_client import build_lipsync_workflow

    monkeypatch.setenv("PROMPT2VID_LIPSYNC_EXPRESSION", "2.2")
    monkeypatch.setenv("PROMPT2VID_LIPSYNC_STEPS", "30")
    monkeypatch.setenv("PROMPT2VID_LIPSYNC_CRF", "17")

    wf = build_lipsync_workflow("c.webm", "a.mp3", mode="normal")

    assert wf["adjust"]["inputs"]["mode"] == "normal"
    assert wf["latentsync"]["inputs"]["lips_expression"] == 2.2
    assert wf["latentsync"]["inputs"]["inference_steps"] == 30
    assert wf["save"]["inputs"]["crf"] == 17


def test_lipsync_workflow_links_within_output_slots():
    from prompt2vid.comfyui_client import build_lipsync_workflow

    wf = build_lipsync_workflow("c.webm", "a.mp3")
    for node_id, node in wf.items():
        assert node["class_type"] in _NODE_OUTPUT_SLOTS, node["class_type"]
        for name, value in node["inputs"].items():
            if not (isinstance(value, list) and len(value) == 2):
                continue
            src_id, slot = value
            src_class = wf[src_id]["class_type"]
            assert slot < _NODE_OUTPUT_SLOTS[src_class]
