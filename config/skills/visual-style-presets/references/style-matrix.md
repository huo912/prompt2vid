# 显卡档位 × 风格 配置矩阵

对应项目根目录的三份档位配置（`--env-file 4060|4090|5090`）。

## 4060 档（8GB，`.env.4060`）

```ini
PROMPT2VID_STYLE=anime
COMFYUI_T2I_CHECKPOINT=sd_xl_base_1.0.safetensors
COMFYUI_LTX_CHECKPOINT=ltx-video-2b-v0.9.5.safetensors
COMFYUI_T2I_WIDTH=768
COMFYUI_T2I_HEIGHT=1344
COMFYUI_LTX_MAX_PIXELS=393216        # 480x832
COMFYUI_T2I_LORA=anime.safetensors
COMFYUI_T2I_LORA_STRENGTH=1.5
COMFYUI_LTX_LORA=                    # 2B 无现成动漫 LoRA，留空
COMFYUI_IPADAPTER=false
```

特点：出片快；风格靠 SDXL Base + 动漫滑块 LoRA + 提示词层三重约束。

## 4090 / 5090 档（24GB / 32GB）

```ini
PROMPT2VID_STYLE=anime
COMFYUI_T2I_CHECKPOINT=RealVisXL_V5.0_fp16.safetensors
COMFYUI_LTX_CHECKPOINT=ltxv-13b-0.9.7-distilled-fp8.safetensors
COMFYUI_T2I_WIDTH=768
COMFYUI_T2I_HEIGHT=1344
COMFYUI_LTX_MAX_PIXELS=589824        # 576x1024
COMFYUI_T2I_LORA=anime.safetensors
COMFYUI_T2I_LORA_STRENGTH=1.5
COMFYUI_LTX_LORA=shinkai-anime-style-lora-v1_58000.safetensors
COMFYUI_LTX_LORA_STRENGTH=0.8~0.9
COMFYUI_IPADAPTER=true
```

特点：写实系底模靠 LoRA 拉向动漫，因此**负向词里的 `photorealistic`
必须保留**（代码自动追加），否则会回到照片感。

## 只想要写实片

```ini
PROMPT2VID_STYLE=realistic
COMFYUI_T2I_CHECKPOINT=RealVisXL_V5.0_fp16.safetensors
COMFYUI_T2I_LORA=
COMFYUI_LTX_LORA=
```

## 自定义风格（例如国风水墨）

```ini
PROMPT2VID_STYLE=chinese ink painting, sumi-e, rice paper texture, muted ink wash
```

此时代码把整串字面追加到每镜正向提示词，负向为空——**需要排斥的风格
（如 `photograph`）自己写进 `COMFYUI_NEGATIVE_PROMPT`**。

## 片段间风格漂移的档位级排查

| 现象 | 检查 |
| --- | --- |
| 一半写实一半动漫 | `COMFYUI_T2I_LORA` 是否加载成功（`run.log` 有降级告警）或者 LoRA 强度过高导致随机 |
| 首帧动漫、片段写实 | `COMFYUI_LTX_LORA` 是否就位；13B 底模与 LoRA 版本是否匹配 |
| 全片偏灰 | `PROMPT2VID_STYLE` 未设置；或负向词里误加了 `vibrant` 相关词 |
| 人物每镜换脸 | 不是风格问题，见 `character-consistency` skill |
