"""运行前环境预检：ComfyUI 可达性 / 模型就位 / LTX 节点 / ffmpeg。

目标：把"配置好 ComfyUI 后直接运行"变成可验证的事实——配置缺失时
在启动几秒内给出可操作的错误清单，而不是跑了几分钟、生成了半个
分镜表之后才在某个工具里炸掉。

CLI：prompt2vid "描述" 默认自动预检；也可 `prompt2vid --check-only`
单独体检。检查项：
1. ffmpeg 可用（ffprobe 可选：缺失时探测回退解析 ffmpeg 输出）；
2. ComfyUI 可达（GET /system_stats）；
3. T2I checkpoint（CheckpointLoaderSimple 的 ckpt_name 候选中）；
4. LTX checkpoint（同上）；
5. T5 文本编码器（CLIPLoader 的 clip_name 候选中）；
6. LTXV 节点存在（ComfyUI 版本不能太旧）。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import Any

import httpx

from .comfyui_client import (
    DEFAULT_COMFYUI_URL,
    DEFAULT_LTX_CHECKPOINT,
    DEFAULT_T2I_CHECKPOINT,
    DEFAULT_T5,
    combo_values,
    env_str,
)
from .ffmpeg_ops import resolve_tool
from .style import style_enabled, style_name

POLL_TIMEOUT = 8.0


@dataclass
class Check:
    """单项检查结果。"""

    name: str
    ok: bool
    detail: str = ""
    hint: str = ""
    required: bool = True


@dataclass
class PreflightResult:
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks if c.required)

    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.required and not c.ok]

    def format_report(self) -> str:
        lines = ["===== 环境预检 ====="]
        for c in self.checks:
            mark = "OK " if c.ok else "NG "
            lines.append(f"{mark}{c.name}: {c.detail}")
            if not c.ok and c.hint:
                lines.append(f"    -> {c.hint}")
        if self.passed:
            lines.append("预检通过，可以开始运行。")
        else:
            lines.append("预检未通过，请按上面的提示修复后重试。")
        return "\n".join(lines)


def _combo_values(node_spec: dict[str, Any], field_name: str) -> list[str]:
    """从 object_info 的节点定义里取 combo 候选值（兼容各版 schema）。"""
    return combo_values(node_spec, field_name)


def _fetch_node_info(client: httpx.Client, node: str, base_url: str) -> dict[str, Any]:
    """GET /object_info/<node>，返回该节点定义；404 时返回空。"""
    resp = client.get(f"{base_url}/object_info/{node}")
    if resp.status_code != 200:
        return {}
    return resp.json()


def check_environment(
    base_url: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> PreflightResult:
    """执行全部预检项，返回 PreflightResult。"""
    comfyui_url = (base_url or env_str("COMFYUI_URL", DEFAULT_COMFYUI_URL)).rstrip("/")
    result = PreflightResult()
    checks = result.checks

    # 1. ffmpeg / ffprobe（ffmpeg 必需；ffprobe 缺失时探测会回退到 ffmpeg 解析）
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool) or resolve_tool(tool)
        ok = path is not None
        detail = path or "未找到"
        if not ok and tool == "ffprobe" and resolve_tool("ffmpeg"):
            ok = True
            detail = "未找到独立 ffprobe，探测将回退用 ffmpeg 解析（功能等价）"
        checks.append(
            Check(
                name=tool,
                ok=ok,
                detail=detail,
                hint=f"请安装 {tool} 并加入 PATH（Docker 镜像已内置，本地需自行安装）",
            )
        )

    # 2. ComfyUI 可达性
    comfyui_ok = False
    try:
        with httpx.Client(timeout=POLL_TIMEOUT, transport=transport) as client:
            resp = client.get(f"{comfyui_url}/system_stats")
            comfyui_ok = resp.status_code == 200
            detail = resp.text[:80]
    except Exception as exc:  # noqa: BLE001 - 任何网络错误都归为不可达
        detail = str(exc)
    checks.append(
        Check(
            name="ComfyUI 可达",
            ok=comfyui_ok,
            detail=detail if comfyui_ok else f"{comfyui_url} 不可达（{detail}）",
            hint="请确认 ComfyUI 已启动，且 .env 中 COMFYUI_URL 正确（本地默认 http://127.0.0.1:8188）",
        )
    )

    # 3-6. 依赖 ComfyUI 的检查
    if comfyui_ok:
        t2i_ckpt = env_str("COMFYUI_T2I_CHECKPOINT", DEFAULT_T2I_CHECKPOINT)
        ltx_ckpt = env_str("COMFYUI_LTX_CHECKPOINT", DEFAULT_LTX_CHECKPOINT)
        t5 = env_str("COMFYUI_T5", DEFAULT_T5)
        try:
            with httpx.Client(timeout=POLL_TIMEOUT, transport=transport) as client:
                loader = _fetch_node_info(client, "CheckpointLoaderSimple", comfyui_url)
                clip_loader = _fetch_node_info(client, "CLIPLoader", comfyui_url)
                ltx_node = _fetch_node_info(client, "LTXVImgToVideo", comfyui_url)
                save_webm = _fetch_node_info(client, "SaveWEBM", comfyui_url)
                lora_loader = _fetch_node_info(client, "LoraLoader", comfyui_url)
                lora_model_only = _fetch_node_info(
                    client, "LoraLoaderModelOnly", comfyui_url
                )

            ckpts = _combo_values(loader.get("CheckpointLoaderSimple", {}), "ckpt_name")
            clips = _combo_values(clip_loader.get("CLIPLoader", {}), "clip_name")
            loras = _combo_values(lora_loader.get("LoraLoader", {}), "lora_name")
            loras_model_only = _combo_values(
                lora_model_only.get("LoraLoaderModelOnly", {}), "lora_name"
            )

            checks.append(
                Check(
                    name=f"T2I checkpoint（{t2i_ckpt}）",
                    ok=t2i_ckpt in ckpts,
                    detail=f"可用: {', '.join(ckpts[:5]) or '无'}",
                    hint="把 SDXL checkpoint 放入 ComfyUI models/checkpoints/，或修改 .env 的 COMFYUI_T2I_CHECKPOINT",
                )
            )
            checks.append(
                Check(
                    name=f"LTX checkpoint（{ltx_ckpt}）",
                    ok=ltx_ckpt in ckpts,
                    detail=f"可用: {', '.join(ckpts[:5]) or '无'}",
                    hint="把 LTX-Video checkpoint 放入 ComfyUI models/checkpoints/，或修改 .env 的 COMFYUI_LTX_CHECKPOINT",
                )
            )
            checks.append(
                Check(
                    name=f"T5 文本编码器（{t5}）",
                    ok=t5 in clips,
                    detail=f"可用: {', '.join(clips[:5]) or '无'}",
                    hint="把 t5xxl fp8 放入 ComfyUI models/text_encoders/，或修改 .env 的 COMFYUI_T5",
                )
            )
            checks.append(
                Check(
                    name="LTXV 节点",
                    ok=bool(ltx_node) and bool(save_webm),
                    detail="LTXVImgToVideo / SaveWEBM 已注册" if ltx_node and save_webm else "ComfyUI 缺少 LTXV 节点",
                    hint="请升级 ComfyUI 到支持 LTX-Video 0.9.5 的版本（master）",
                )
            )
            # LoRA 是风格增强件：缺失时运行期自动降级为无 LoRA 生成，
            # 只提示不拦截（风格词约束仍在，画面不会跑偏回默认风格）
            t2i_lora = env_str("COMFYUI_T2I_LORA", "")
            if t2i_lora:
                checks.append(
                    Check(
                        name=f"T2I LoRA（{t2i_lora}）",
                        ok=t2i_lora in loras,
                        detail=f"可用: {', '.join(loras[:5]) or '无'}",
                        hint="把 LoRA 放入 ComfyUI models/loras/，或清空 .env 的 COMFYUI_T2I_LORA（将退回纯提示词风格约束）",
                        required=False,
                    )
                )
            ltx_lora = env_str("COMFYUI_LTX_LORA", "")
            if ltx_lora:
                checks.append(
                    Check(
                        name=f"LTX LoRA（{ltx_lora}）",
                        ok=ltx_lora in loras_model_only,
                        detail=f"可用: {', '.join(loras_model_only[:5]) or '无'}",
                        hint="把 LTX LoRA 放入 ComfyUI models/loras/，或清空 .env 的 COMFYUI_LTX_LORA",
                        required=False,
                    )
                )
            if style_enabled():
                checks.append(
                    Check(
                        name=f"风格约束（{style_name()}）",
                        ok=True,
                        detail="每镜提示词将自动追加风格词（见 style.py 预设）",
                        required=False,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - object_info 异常
            checks.append(Check(name="ComfyUI object_info", ok=False, detail=str(exc),
                                hint="检查 ComfyUI 版本是否过旧"))

    return result


def run_preflight(base_url: str | None = None) -> bool:
    """执行预检并打印报告；返回是否通过。"""
    from .env_config import load_env, loaded_env_file

    load_env()  # 幂等：确保独立调用时（不经 CLI）环境配置也已就位
    result = check_environment(base_url)
    env_file = loaded_env_file()
    if env_file is not None:
        print(f"配置文件: {env_file}")
    print(result.format_report())
    return result.passed
