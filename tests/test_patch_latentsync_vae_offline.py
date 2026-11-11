"""LatentSync VAE 离线补丁脚本的回归测试。

`patch_latentsync_vae_offline.py` 把 `scripts/inference.py` 里会联网请求
`stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/vae/config.json`
的构造方式换成只读本地 `checkpoints/vae/`。这里用假的 inference.py 覆盖
两种入口形态（上游原版 / 之前打过 "FIXED" 补丁的版本），确保打补丁后
不再残留联网调用、且幂等。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "deploy"
    / "comfyui"
    / "patch_latentsync_vae_offline.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("patch_latentsync_vae_offline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


UPSTREAM = '''import argparse
import os
import torch
from diffusers import AutoencoderKL, DDIMScheduler


def main(config, args):
    print("hello")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)

    # Set VAE configuration
    vae.config.scaling_factor = 0.18215
    return vae
'''

EARLIER_PATCH = '''import argparse
import os
import torch
from diffusers import AutoencoderKL, DDIMScheduler


def main(config, args):
    print("hello")
    # FIXED: Load VAE locally with proper path resolution
    base_dir = args.extension_dir
    vae_locations = [
        os.path.join(base_dir, "checkpoints", "vae", "sd-vae-ft-mse.safetensors"),
        os.path.join(base_dir, "checkpoints", "vae"),
    ]
    vae = None
    for vae_path in vae_locations:
        if os.path.exists(vae_path):
            vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=dtype)

    # Set VAE configuration
    vae.config.scaling_factor = 0.18215
    return vae
'''


@pytest.mark.parametrize("source", [UPSTREAM, EARLIER_PATCH])
def test_patch_removes_network_vae_loading(tmp_path: Path, source: str) -> None:
    module = _load_module()
    target = tmp_path / "scripts" / "inference.py"
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")

    assert module.patch(target) is True
    text = target.read_text(encoding="utf-8")

    assert "PATCHED_VAE_OFFLINE" in text
    assert "sd-vae-ft-mse\", torch_dtype=dtype" not in text
    assert 'from_pretrained("stabilityai/sd-vae-ft-mse"' not in text
    assert "vae = load_local_vae(args, dtype)" in text
    # 辅助函数必须定义在 main 之前，否则运行期 NameError
    assert text.index("def load_local_vae(") < text.index("def main(config, args):")
    compile(text, str(target), "exec")


def test_patch_is_idempotent(tmp_path: Path) -> None:
    module = _load_module()
    target = tmp_path / "scripts" / "inference.py"
    target.parent.mkdir(parents=True)
    target.write_text(UPSTREAM, encoding="utf-8")

    assert module.patch(target) is True
    first = target.read_text(encoding="utf-8")
    assert module.patch(target) is False
    assert target.read_text(encoding="utf-8") == first


def test_patch_reports_missing_anchor(tmp_path: Path) -> None:
    module = _load_module()
    target = tmp_path / "scripts" / "inference.py"
    target.parent.mkdir(parents=True)
    target.write_text("def main(config, args):\n    return None\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        module.patch(target)
