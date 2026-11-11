#!/usr/bin/env python3
"""Make LatentSync tolerate frames where insightface finds no face.

Upstream `lipsync_pipeline.affine_transform_video()` aborts the whole run on the
first frame whose largest detected face is too small / too blurry / too
side-on (see `latentsync/utils/face_detector.py`: w<50, h<80, 0.2<=w/h<=1.5,
det_score>=0.5).  On AI-generated clips that happens constantly - a single bad
frame kills an otherwise fine job.

This patch:
  * probes every frame, remembering which ones failed instead of raising;
  * fills the failed frames with the nearest successfully-detected frame's
    face crop / box / affine matrix (printed as a warning);
  * only raises - with an actionable message - when *no* frame has a face.

Idempotent: re-running it is a no-op.  Run it after (re)installing
ComfyUI-LatentSyncWrapper.

Usage:
    python patch_latentsync_face_detect.py [--node-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "PATCHED_FACE_GAP_FILL"

NEW_FUNC = '''    def affine_transform_video(self, video_frames: np.ndarray):
        # PATCHED_FACE_GAP_FILL: upstream aborts on the first frame insightface
        # misses. Probe every frame instead, then fill the gaps from the
        # nearest detected frame so one bad frame cannot kill the whole run.
        print(f"Affine transforming {len(video_frames)} faces...")
        results = []
        for frame in tqdm.tqdm(video_frames):
            try:
                results.append(self.image_processor.affine_transform(frame))
            except RuntimeError:
                results.append(None)

        valid = [i for i, result in enumerate(results) if result is not None]
        if not valid:
            raise RuntimeError(
                f"Face not detected in any of the {len(video_frames)} frames. LatentSync needs a "
                "clearly visible frontal face in most frames (>=50x80 px, detection score >= 0.5, "
                "aspect ratio between 0.2 and 1.5). Regenerate the clip with the character facing "
                "the camera and filling more of the frame, or disable lipsync "
                "(PROMPT2VID_LIPSYNC=false)."
            )

        missing = len(video_frames) - len(valid)
        if missing:
            print(
                f"[LatentSync] WARNING: face not detected in {missing}/{len(video_frames)} frames; "
                "reusing the nearest detected face position for those frames."
            )
            for index, result in enumerate(results):
                if result is None:
                    nearest = min(valid, key=lambda j: abs(j - index))
                    results[index] = results[nearest]

        faces = torch.stack([result[0] for result in results])
        boxes = [result[1] for result in results]
        affine_matrices = [result[2] for result in results]
        return faces, boxes, affine_matrices
'''


def patch(target: Path) -> bool:
    source = target.read_text(encoding="utf-8")
    if MARKER in source:
        print(f"already patched: {target}")
        return False

    start = source.index("    def affine_transform_video(self, video_frames: np.ndarray):")
    end = source.index("    def restore_video(", start)
    source = source[:start] + NEW_FUNC + "\n" + source[end:]
    target.write_text(source, encoding="utf-8")
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
    target = Path(args.node_dir) / "latentsync" / "pipelines" / "lipsync_pipeline.py"
    if not target.is_file():
        sys.exit(f"error: {target} not found; pass --node-dir pointing at ComfyUI-LatentSyncWrapper-main")
    changed = patch(target)
    compile(target.read_text(encoding="utf-8"), str(target), "exec")
    print("syntax OK")
    if changed:
        print("restart ComfyUI for the change to take effect")


if __name__ == "__main__":
    main()
