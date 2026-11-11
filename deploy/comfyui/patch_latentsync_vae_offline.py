#!/usr/bin/env python3
"""Stop LatentSync from asking the Hugging Face Hub for the SD1.5 VAE config.

`scripts/inference.py` builds the VAE with

    AutoencoderKL.from_single_file("checkpoints/vae/sd-vae-ft-mse.safetensors", ...)

and `from_single_file()` cannot infer an architecture from raw weights alone, so
diffusers fetches the matching diffusers config from the Hub first:

    HEAD https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/vae/config.json

On an offline box (AutoDL containers cannot route to huggingface.co) that call
retries 5 times with exponential backoff and then hangs/blows up - even though
`checkpoints/vae/config.json` + the weights are sitting right there on disk.
The visible symptom is `sync_lips` stuck at "running" and a log full of
`[Errno 101] Network is unreachable ... vae/config.json`.

This patch moves the VAE loading into a module-level `load_local_vae()` helper
that:
  * prefers the diffusers-style folder (`checkpoints/vae/`, which already has
    `config.json` + `diffusion_pytorch_model.safetensors`) and loads it with
    `local_files_only=True` - zero network;
  * only touches a bare `.safetensors` as a last resort, handing over the local
    `config.json` explicitly so diffusers has no reason to call the Hub;
  * falls back to a freshly-built AutoencoderKL with SD1.5 defaults, so a
    missing checkpoint degrades instead of hanging.

Idempotent: re-running it is a no-op.  Run it after (re)installing
ComfyUI-LatentSyncWrapper, then restart ComfyUI.

Usage:
    python patch_latentsync_vae_offline.py [--node-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "PATCHED_VAE_OFFLINE"

HELPER = '''# PATCHED_VAE_OFFLINE: resolve the SD1.5 VAE from the local checkout only.
# `from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae")`
# and `from_single_file(<bare .safetensors>)` both phone home to the Hugging Face
# Hub for the diffusers config; on an offline machine that is five retries and
# then a hang.  Everything we need is already under checkpoints/vae/.
def load_local_vae(args, dtype):
    base_dir = getattr(args, "extension_dir", None)
    if not base_dir:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = str(base_dir)

    vae_dir = os.path.join(base_dir, "checkpoints", "vae")
    config_json = os.path.join(vae_dir, "config.json")
    single_files = [
        os.path.join(vae_dir, "sd-vae-ft-mse.safetensors"),
        os.path.join(base_dir, "checkpoints", "sd-vae-ft-mse.safetensors"),
    ]

    # 1) diffusers folder: config.json + weights, fully offline.
    if os.path.isfile(config_json):
        try:
            print(f"Attempting to load VAE from directory: {vae_dir}")
            vae = AutoencoderKL.from_pretrained(vae_dir, torch_dtype=dtype, local_files_only=True)
            print(f"\\u2713 Successfully loaded VAE from: {vae_dir}")
            return vae
        except Exception as exc:
            print(f"Failed to load VAE from directory {vae_dir}: {exc}")

    # 2) bare single-file VAE - hand diffusers the local config so it never asks
    #    the Hub which architecture to use.
    for path in single_files:
        if not os.path.isfile(path):
            continue
        try:
            print(f"Attempting to load VAE from safetensors file: {path}")
            kwargs = {"torch_dtype": dtype, "local_files_only": True}
            if os.path.isfile(config_json):
                kwargs["config"] = config_json
            vae = AutoencoderKL.from_single_file(path, **kwargs)
            print(f"\\u2713 Successfully loaded VAE from: {path}")
            return vae
        except Exception as exc:
            print(f"Failed to load VAE from {path}: {exc}")

    # 3) Nothing usable on disk: degrade instead of retrying the network.
    print("Local VAE not found in any location, creating VAE with standard configuration")
    print(f"Searched: {[config_json] + single_files}")
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        act_fn="silu",
        latent_channels=4,
        norm_num_groups=32,
        sample_size=512,
    ).to(dtype=dtype)
    print("\\u26a0\\ufe0f  Using default VAE configuration - drop sd-vae-ft-mse into checkpoints/vae/ for better results")
    return vae


'''

CALL_BLOCK = '''    # PATCHED_VAE_OFFLINE: local-only VAE resolution (see load_local_vae above).
    vae = load_local_vae(args, dtype)

'''

# Where the VAE construction starts, depending on how the file currently looks.
START_MARKERS = (
    "    # FIXED: Load VAE locally with proper path resolution",
    "    # PATCHED_VAE_OFFLINE: local-only VAE resolution",
    '    vae = AutoencoderKL.from_pretrained(',
)

END_MARKER = "    # Set VAE configuration"


def patch(target: Path) -> bool:
    source = target.read_text(encoding="utf-8")
    if MARKER in source:
        print(f"already patched: {target}")
        return False

    start = -1
    for marker in START_MARKERS:
        start = source.find(marker)
        if start != -1:
            break
    if start == -1:
        raise SystemExit(
            "error: could not find the VAE construction block; the upstream "
            "scripts/inference.py probably changed - patch it by hand"
        )

    end = source.find(END_MARKER, start)
    if end == -1:
        raise SystemExit(f"error: could not find {END_MARKER!r} after the VAE block")

    source = source[:start] + CALL_BLOCK + source[end:]

    main_def = source.index("def main(config, args):")
    source = source[:main_def] + HELPER + source[main_def:]

    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(source)
    print(f"patched: {target}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--node-dir",
        default="/root/autodl-tmp/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper-main",
        help="path to the ComfyUI-LatentSyncWrapper-main checkout",
    )
    args = parser.parse_args()
    target = Path(args.node_dir) / "scripts" / "inference.py"
    if not target.is_file():
        sys.exit(f"error: {target} not found; pass --node-dir pointing at ComfyUI-LatentSyncWrapper-main")
    changed = patch(target)
    compile(target.read_text(encoding="utf-8"), str(target), "exec")
    print("syntax OK")
    if changed:
        print("restart ComfyUI for the change to take effect")


if __name__ == "__main__":
    main()
