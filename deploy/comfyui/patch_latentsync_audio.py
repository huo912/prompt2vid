#!/usr/bin/env python3
"""Fix ComfyUI-LatentSyncWrapper on torchaudio >= 2.9 (missing TorchCodec).

Symptom (ComfyUI console / prompt queue error):

    ImportError: TorchCodec is required for save_with_torchcodec.
    File ".../ComfyUI-LatentSyncWrapper/nodes.py", line 569, in inference
        torchaudio.save(audio_path, waveform_cpu, sample_rate)

torchaudio >= 2.9 dropped the built-in ffmpeg/sox backends: `torchaudio.save`
now dispatches to TorchCodec and dies when the `torchcodec` wheel is absent
(it has to match the exact torch build, so it is often unavailable on rented
GPU images). LatentSyncWrapper only uses that call to dump the driving audio
to a temp `.wav`; ffmpeg/decord read it back.

This script rewrites that single line so it falls back to a stdlib `wave`
writer (PCM16) when torchaudio cannot save. Idempotent, keeps a timestamped
backup, touches nothing else.

Usage (on the machine that runs ComfyUI):

    python3 patch_latentsync_audio.py                  # auto-detect nodes.py
    python3 patch_latentsync_audio.py /path/to/nodes.py

Then restart ComfyUI.
"""

from __future__ import annotations

import datetime as _dt
import re
import shutil
import sys
from pathlib import Path

DEFAULT_NODES = (
    "/root/autodl-tmp/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/nodes.py"
)

# Search a few common install layouts when no path is given.
CANDIDATES = (
    DEFAULT_NODES,
    "/root/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/nodes.py",
    "/app/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/nodes.py",
    "./ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/nodes.py",
)

MARKER = "prompt2vid fix: torchaudio>=2.9 needs torchcodec"

CALL_RE = re.compile(
    r"^(?P<indent>[ \t]*)torchaudio\.save\(\s*audio_path\s*,\s*"
    r"waveform_cpu\s*,\s*sample_rate\s*\)[ \t]*$",
    re.MULTILINE,
)


def _replacement(indent: str) -> str:
    p = indent
    return "\n".join(
        [
            f"{p}try:",
            f"{p}    torchaudio.save(audio_path, waveform_cpu, sample_rate)",
            f"{p}except (ImportError, ModuleNotFoundError, RuntimeError):",
            f"{p}    # {MARKER}.",
            f"{p}    # torchaudio>=2.9 dropped the ffmpeg/sox backends and defaults",
            f"{p}    # to TorchCodec, which is often missing or ABI-mismatched.",
            f"{p}    # Write a PCM16 WAV with the stdlib instead.",
            f"{p}    import wave",
            f"{p}    _p2v = waveform_cpu.detach().cpu().float().numpy()",
            f"{p}    if _p2v.ndim == 2:",
            f"{p}        _p2v = _p2v.T  # (channels, samples) -> (samples, channels)",
            f"{p}    else:",
            f"{p}        _p2v = _p2v.reshape(-1, 1)",
            f"{p}    if _p2v.shape[1] > 2:",
            f"{p}        _p2v = _p2v.mean(axis=1, keepdims=True)",
            f'{p}    with wave.open(audio_path, "wb") as _p2v_fh:',
            f"{p}        _p2v_fh.setnchannels(_p2v.shape[1])",
            f"{p}        _p2v_fh.setsampwidth(2)",
            f"{p}        _p2v_fh.setframerate(int(sample_rate))",
            f"{p}        _p2v_fh.writeframes(",
            f'{p}            (np.clip(_p2v, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()',
            f"{p}        )",
        ]
    )


def _env_report() -> None:
    print(f"[info] python = {sys.executable}")
    try:
        import torch  # noqa: F401
        import torchaudio

        print(f"[info] torch = {torch.__version__}, torchaudio = {torchaudio.__version__}")
    except Exception as exc:  # pragma: no cover - depends on host env
        print(f"[info] torch/torchaudio not importable here ({exc})")
    try:
        import torchcodec

        print(f"[info] torchcodec = {getattr(torchcodec, '__version__', '?')}")
    except Exception:
        print("[info] torchcodec = MISSING (expected; this patch removes the need)")


def _find_nodes(argv: list[str]) -> Path | None:
    if len(argv) > 1:
        return Path(argv[1]).expanduser()
    for cand in CANDIDATES:
        path = Path(cand)
        if path.is_file():
            return path
    return None


def patch(path: Path) -> int:
    """Return number of patched calls, 0 if already patched, -1 if not found."""
    source = path.read_text(encoding="utf-8")
    if MARKER in source:
        print(f"[skip] already patched: {path}")
        return 0

    patched, count = CALL_RE.subn(lambda m: _replacement(m.group("indent")), source)
    if count == 0:
        return -1

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.bak-{stamp}")
    shutil.copy2(path, backup)
    path.write_text(patched, encoding="utf-8")
    print(f"[ok] {count} torchaudio.save() call(s) patched in {path}")
    print(f"[ok] backup written to {backup}")
    return count


REMAINING_RE = re.compile(r"torchaudio\.(save|load)\(")


def _warn_remaining(path: Path) -> None:
    """Warn about torchaudio.save/load calls that TorchCodec would still break."""
    lines = path.read_text(encoding="utf-8").splitlines()
    leftovers: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        if not REMAINING_RE.search(line):
            continue
        # Patched calls sit directly under a `try:` line.
        prev = lines[idx - 1].strip() if idx else ""
        if prev != "try:":
            leftovers.append((idx + 1, line.strip()))
    for lineno, text in leftovers:
        print(f"[warn] {path.name}:{lineno} still calls TorchCodec directly: {text}")
    if leftovers:
        print("[warn] the same try/except fallback pattern applies to those calls")


def scan(root: Path) -> int:
    """Report every torchaudio.save/load() call site under `root`."""
    found = 0
    for py in sorted(root.rglob("*.py")):
        for lineno, line in enumerate(py.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines(), 1):
            if REMAINING_RE.search(line):
                found += 1
                print(f"{py}:{lineno}: {line.strip()}")
    if not found:
        print(f"[ok] no torchaudio.save()/load() call sites under {root}")
    else:
        print(f"[info] {found} call site(s) under {root} — each needs a "
              f"TorchCodec-free fallback")
    return 0


def main(argv: list[str]) -> int:
    _env_report()
    if len(argv) > 1 and argv[1] == "--scan":
        root = Path(argv[2]).expanduser() if len(argv) > 2 else Path(
            "/root/autodl-tmp/ComfyUI/custom_nodes"
        )
        return scan(root)

    target = _find_nodes(argv)
    if target is None or not target.is_file():
        print(
            "[error] nodes.py not found. Pass it explicitly, e.g.\n"
            "        python3 patch_latentsync_audio.py "
            "/root/autodl-tmp/ComfyUI/custom_nodes/ComfyUI-LatentSyncWrapper/nodes.py",
            file=sys.stderr,
        )
        return 2

    result = patch(target)
    if result < 0:
        print(
            f"[error] no 'torchaudio.save(audio_path, waveform_cpu, sample_rate)' "
            f"line in {target} — the wrapper version differs; patch it by hand:\n"
            f"        replace that call with the block documented in "
            f"docs/LOCAL_SETUP.md §8.3",
            file=sys.stderr,
        )
        return 1

    _warn_remaining(target)
    print("[next] restart ComfyUI, then re-run the lipsync stage:")
    print("       cd /root/autodl-tmp/ComfyUI && "
          "nohup python main.py --listen 127.0.0.1 --port 8188 > comfyui.log 2>&1 &")
    print("       prompt2vid --resume output/run_xxx --only sync_lips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
