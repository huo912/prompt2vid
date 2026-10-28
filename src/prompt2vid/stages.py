"""确定性阶段引擎：把"循环 + 重试 + 清单"从 LLM 手里下沉到代码。

此前每个媒体 agent 需要自己逐镜循环调用工具、维护 manifest.json，
受 max_iter 限制（镜头一多就截断）、易编造路径、慢且贵。现在每个阶段
是一个无参数的确定性函数：自动从工作目录读取上一步产物，逐镜执行、
失败重试、写清单、返回 JSON 汇总。LLM agent 只负责调用一次阶段工具
并把汇总汇报给下游。

阶段产物契约（均位于工作目录）：
- storyboard.json  分镜表（分镜师产出，storyboard.load_storyboard 解析）
- frames/manifest.json  {"shots": [{"id", "image", "status", "error", "path"}]}
- clips/manifest.json   {"shots": [{"id", "video", "status", "error", "path", "length"}]}
- audio/manifest.json   {"shots": [{"id", "audio", "status", "path"}]}
- final/final.mp4 + final/final.srt + final/render_report.json
- report/quality_report.json
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable

from . import ffmpeg_ops, tts
from .comfyui_client import (
    ComfyUITaskError,
    ComfyUIClient,
    build_i2v_workflow,
    build_t2i_workflow,
    default_t2i_size,
    env_int,
    env_str,
    image_dimensions,
    ipadapter_enabled,
    resolve_ltx_size,
    run_text_to_image,
)
from .storyboard import (
    Character,
    Shot,
    StoryboardError,
    extract_json,
    load_characters,
    load_storyboard,
    read_manifest,
)
from .workspace import resolve_in_workspace, workspace_root

logger = logging.getLogger("prompt2vid.stages")

STORYBOARD_FILE = "storyboard.json"
REQUIREMENT_FILE = "requirement.json"
FRAMES_DIR = "frames"
CLIPS_DIR = "clips"
AUDIO_DIR = "audio"
FINAL_DIR = "final"
REPORT_DIR = "report"

# LTX 帧数约束：length = 8n+1；帧率与上限可配
DEFAULT_LTX_FPS = 25
DEFAULT_LTX_MAX_FRAMES = 121
MIN_FRAMES = 9  # 8*1+1

# LTX 对人物肢体的稳定性提示：模型本身没有 anatomy 先验，正向提示词里
# 补一句能明显减少手部/四肢在运动中被「脑补」成多余肢体。
DEFAULT_LTX_PROMPT_SUFFIX = (
    "consistent anatomy, natural body proportions, stable limbs, detailed hands"
)


def _ltx_prompt(shot: Shot) -> str:
    """构造图生视频正向提示词：场景 + 运动 + 肢体稳定约束。"""
    parts = [
        part.strip(" ,")
        for part in (shot.image_prompt, shot.motion_prompt)
        if part and part.strip()
    ]
    text = ", ".join(parts)
    suffix = env_str("COMFYUI_LTX_PROMPT_SUFFIX", DEFAULT_LTX_PROMPT_SUFFIX)
    if suffix and suffix.lower() not in text.lower():
        text = f"{text}, {suffix}" if text else suffix
    return text


def frames_for_duration(seconds: float) -> int:
    """把每镜时长换算为 LTX 帧数（8n+1，且不超过显存预算上限）。"""
    fps = env_int("COMFYUI_LTX_FPS", DEFAULT_LTX_FPS)
    max_frames = max(MIN_FRAMES, env_int("COMFYUI_LTX_MAX_FRAMES", DEFAULT_LTX_MAX_FRAMES))
    target = max(MIN_FRAMES, math.ceil(seconds * fps))
    length = 8 * math.ceil((target - 1) / 8) + 1
    if length > max_frames:
        length = 8 * ((max_frames - 1) // 8) + 1
    return max(MIN_FRAMES, length)


def _retries() -> int:
    return max(0, env_int("COMFYUI_RETRIES", 2))


def _batch_run(
    jobs: list[Shot],
    make_workflow: Callable[[Shot], dict],
    dest_of: Callable[[Shot], Path],
    client: ComfyUIClient,
) -> tuple[dict[int, Path], dict[int, str], list[Shot]]:
    """把一批分镜的工作流一次性提交给 ComfyUI 并统一等待收集。

    ComfyUI 自带 GPU 执行队列：整批提交后串行执行，省掉逐镜
    "提交→轮询→下载"的调度空隙。返回 (成功 id→路径, 失败 id→错误,
    需重试分镜)。单个任务失败不影响同批其他任务。
    """
    ok: dict[int, Path] = {}
    errors: dict[int, str] = {}
    retry: list[Shot] = []
    prompt_ids: dict[str, Shot] = {}

    for shot in jobs:
        try:
            pid = client.submit(make_workflow(shot))
            prompt_ids[pid] = shot
        except Exception as exc:  # noqa: BLE001 - 提交失败进入本批重试
            errors[shot.id] = str(exc)
            retry.append(shot)

    results = client.wait_all(list(prompt_ids)) if prompt_ids else {}
    for pid, shot in prompt_ids.items():
        entry = results[pid]
        if isinstance(entry, ComfyUITaskError):
            errors[shot.id] = str(entry)
            retry.append(shot)
            continue
        files = client.outputs(entry)
        if not files:
            errors[shot.id] = "任务完成但没有输出文件"
            retry.append(shot)
            continue
        try:
            ok[shot.id] = client.download(files[0], dest_of(shot))
        except Exception as exc:  # noqa: BLE001 - 下载失败进入本批重试
            errors[shot.id] = str(exc)
            retry.append(shot)
    return ok, errors, retry


def _write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _summary(stage: str, entries: list[dict], manifest_rel: str) -> dict:
    ok = [e for e in entries if e.get("status") == "ok"]
    failed = [e for e in entries if e.get("status") != "ok"]
    summary: dict[str, Any] = {
        "stage": stage,
        "total": len(entries),
        "succeeded": len(ok),
        "failed": len(failed),
        "manifest": manifest_rel,
    }
    if failed:
        summary["failed_shots"] = [
            {"id": e["id"], "error": e.get("error", "")} for e in failed
        ]
        if not ok:
            # 全批失败时把首个具体原因带进 error，避免只看到笼统提示
            first = next((e.get("error", "") for e in failed if e.get("error")), "")
            detail = " ".join(str(first).split())[:240]
            summary["error"] = (
                f"全部 {len(entries)} 个分镜都失败，请检查 ComfyUI 与模型配置"
                + (f"；首个错误: {detail}" if detail else "")
            )
    return summary


def _reference_prompt(character: Character) -> str:
    """角色定妆照 prompt：外观契约 + 人像构图提示（1024 方图，供 IPAdapter 用）。"""
    return (
        f"{character.description}, character reference sheet, upper body portrait, "
        "detailed face, neutral expression, looking at viewer, "
        "plain clean background, sharp focus, high detail"
    )


def _prepare_character_refs(
    ws: Path,
    characters: dict[int, Character],
    client: ComfyUIClient,
    checkpoint: str | None = None,
    lora: str | None = None,
) -> dict[int, dict]:
    """为每个角色生成/复用一张定妆照并上传，返回 char_id → 参考图注入参数。

    已有的定妆照直接复用（断点续跑）；某个角色生成或上传失败只警告，
    对应分镜回退为无参考生成——参考注入是锦上添花，不能阻塞出图。

    checkpoint / lora 与分镜首帧生成保持一致：定妆照走同一条模型链
    （含风格约束与 LoRA），IPAdapter 注入的参考才不会和成片风格打架。
    """
    refs: dict[int, dict] = {}
    for cid, character in sorted(characters.items()):
        dest = ws / FRAMES_DIR / f"character_ref_{cid}.png"
        try:
            if not (dest.is_file() and dest.stat().st_size > 0):
                run_text_to_image(
                    prompt=_reference_prompt(character),
                    dest=dest,
                    width=1024,
                    height=1024,
                    client=client,
                    checkpoint=checkpoint,
                    lora=lora,
                )
                logger.info("frames: 角色 %s 定妆照已生成: %s",
                            character.name or cid, dest.name)
            refs[cid] = {"image_name": client.upload_image(dest)}
        except Exception as exc:  # noqa: BLE001 - 参考图失败不阻塞出图
            logger.warning(
                "frames: 角色 %s 定妆照生成/上传失败，相关分镜将以无参考生成: %s",
                character.name or cid, exc,
            )
    return refs


def _shot_reference(
    shot: Shot, refs: dict[int, dict]
) -> dict | None:
    """取该镜第一个已就绪的角色参考（多角色镜头以主角色为准）。"""
    for cid in shot.characters:
        if cid in refs:
            return refs[cid]
    return None


def generate_frames(workspace: str | Path | None = None) -> dict:
    """阶段 1：为分镜表每个分镜生成首帧图（ComfyUI SDXL 文生图）。

    整批工作流一次性提交给 ComfyUI（GPU 队列串行执行），统一收集后
    对失败分镜重试（每轮重新提交）。

    角色一致性（COMFYUI_IPADAPTER 开启且分镜表定义了 characters 时）：
    先为每个角色生成一张定妆照，出镜分镜的首帧生成通过 IPAdapter
    注入该参考；带参考失败的分镜在重试轮自动降级为无参考。
    """
    ws = workspace_root(workspace)
    try:
        shots = load_storyboard(ws / STORYBOARD_FILE)
    except StoryboardError as exc:
        return {"stage": "frames", "error": str(exc)}
    characters: dict[int, Character] = {}
    if ipadapter_enabled():
        try:
            characters = load_characters(ws / STORYBOARD_FILE)
        except StoryboardError as exc:
            return {"stage": "frames", "error": str(exc)}

    def dest_of(shot: Shot) -> Path:
        return ws / FRAMES_DIR / f"shot_{shot.id}.png"

    entries: list[dict] = []
    ok: dict[int, Path] = {}
    errors: dict[int, str] = {}
    pending: list[Shot] = []
    for shot in shots:
        dest = dest_of(shot)
        # 已存在且非空的图跳过（断点续跑）
        if dest.is_file() and dest.stat().st_size > 0:
            ok[shot.id] = dest
        else:
            pending.append(shot)

    if pending:
        with ComfyUIClient() as client:
            # 配置的 checkpoint 名在云机上可能不存在（如只下了 RealVisXL）：
            # 开局按 ComfyUI 实际模型列表校准一次，避免每镜都 HTTP 400 全失败
            t2i_checkpoint = client.resolve_checkpoint("t2i")
            # LoRA 同理校准（缺失时降级为无 LoRA，流水线不中断）
            t2i_lora = client.resolve_lora("t2i")
            refs: dict[int, dict] = {}
            if characters:
                refs = _prepare_character_refs(
                    ws, characters, client, checkpoint=t2i_checkpoint, lora=t2i_lora
                )
                if not refs:
                    logger.warning("frames: 没有可用的角色参考图，全部按无参考生成")
            shot_by_id = {s.id: s for s in shots}
            # 带参考失败过的分镜：重试轮降级为无参考（IPAdapter 节点/
            # 模型未装好时自动回到纯文生图，流水线不中断）
            degraded: set[int] = set()

            def make_workflow(shot: Shot) -> dict:
                reference = None if shot.id in degraded else _shot_reference(shot, refs)
                return build_t2i_workflow(
                    prompt=shot.image_prompt,
                    reference=reference,
                    checkpoint=t2i_checkpoint,
                    lora=t2i_lora,
                )

            for attempt in range(_retries() + 1):
                if not pending:
                    break
                if attempt:
                    logger.info("frames: 第 %d 次重试 %d 个分镜", attempt, len(pending))
                batch_ok, batch_errors, pending = _batch_run(
                    pending, make_workflow, dest_of, client
                )
                ok.update(batch_ok)
                errors.update(batch_errors)
                for sid in batch_errors:
                    if sid not in degraded and _shot_reference(shot_by_id[sid], refs):
                        degraded.add(sid)
                        logger.warning(
                            "frames: 分镜 %s 带参考生成失败，重试时降级为无参考",
                            sid,
                        )
                if batch_ok:
                    logger.info("frames: 本批成功 %d / 失败 %d",
                                len(batch_ok), len(batch_errors))

    for shot in shots:
        if shot.id in ok:
            entries.append(_ok_entry(shot.id, "image", f"{FRAMES_DIR}/shot_{shot.id}.png", ok[shot.id]))
        else:
            entries.append(_failed_entry(shot.id, errors.get(shot.id, "未生成")))

    manifest_rel = f"{FRAMES_DIR}/manifest.json"
    _write_json(ws / manifest_rel, {"shots": entries})
    return _summary("frames", entries, manifest_rel)


def generate_clips(workspace: str | Path | None = None) -> dict:
    """阶段 2：把每个首帧图转为视频片段（ComfyUI LTX-Video 图生视频）。

    首帧图先逐个上传，然后整批工作流一次性提交给 ComfyUI，统一收集后
    对失败分镜重试。
    """
    ws = workspace_root(workspace)
    try:
        storyboard = {s.id: s for s in load_storyboard(ws / STORYBOARD_FILE)}
    except StoryboardError as exc:
        return {"stage": "clips", "error": str(exc)}
    try:
        frames = read_manifest(ws / FRAMES_DIR / "manifest.json")
    except (FileNotFoundError, ValueError) as exc:
        return {"stage": "clips", "error": f"请先生成首帧图: {exc}"}

    image_of: dict[int, Path] = {}
    for frame in frames:
        if frame.get("status") != "ok":
            continue
        shot_id = int(frame.get("id", 0))
        if shot_id in storyboard:
            image_of[shot_id] = resolve_in_workspace(frame.get("path", ""), ws)

    # 首帧画幅是视频画幅的唯一依据：读不到（如测试假文件）才退回默认画幅
    frame_w, frame_h = default_t2i_size()
    for image_path in image_of.values():
        dims = image_dimensions(image_path)
        if dims:
            frame_w, frame_h = dims
            break

    def dest_of(shot: Shot) -> Path:
        return ws / CLIPS_DIR / f"shot_{shot.id}.webm"

    ok: dict[int, Path] = {}
    lengths: dict[int, int] = {}
    sizes: dict[int, tuple[int, int]] = {}
    errors: dict[int, str] = {}
    pending: list[Shot] = []
    for shot in storyboard.values():
        dest = dest_of(shot)
        if shot.id not in image_of:
            continue  # 首帧图缺失的分镜直接不进清单
        if dest.is_file() and dest.stat().st_size > 0:
            ok[shot.id] = dest  # 断点续跑
            lengths[shot.id] = frames_for_duration(shot.duration_seconds)
        else:
            pending.append(shot)

    if pending:
        with ComfyUIClient() as client:
            # 同上：按服务端实际模型列表校准 LTX checkpoint 文件名与 LoRA
            ltx_checkpoint = client.resolve_checkpoint("ltx")
            ltx_lora = client.resolve_lora("ltx")
            for attempt in range(_retries() + 1):
                if not pending:
                    break
                if attempt:
                    logger.info("clips: 第 %d 次重试 %d 个分镜", attempt, len(pending))

                def make_workflow(shot: Shot) -> dict:
                    # 上传在提交前串行完成；失败会作为该镜本轮错误进入重试
                    uploaded = client.upload_image(image_of[shot.id])
                    length = frames_for_duration(shot.duration_seconds)
                    lengths[shot.id] = length
                    # 逐镜按首帧真实画幅换算 LTX 画布（32 倍数 + 显存预算），
                    # 保证首帧不被 LTXVImgToVideo 裁切/挤压
                    shot_w, shot_h = image_dimensions(image_of[shot.id]) or (frame_w, frame_h)
                    width, height = resolve_ltx_size(shot_w, shot_h, ltx_checkpoint)
                    sizes[shot.id] = (width, height)
                    return build_i2v_workflow(
                        image_name=uploaded,
                        prompt=_ltx_prompt(shot),
                        width=width,
                        height=height,
                        length=length,
                        checkpoint=ltx_checkpoint,
                        lora=ltx_lora,
                    )

                batch_ok, batch_errors, pending = _batch_run(
                    pending, make_workflow, dest_of, client
                )
                ok.update(batch_ok)
                errors.update(batch_errors)
                if batch_ok:
                    logger.info("clips: 本批成功 %d / 失败 %d",
                                len(batch_ok), len(batch_errors))

    entries: list[dict] = []
    for shot in sorted(storyboard.values(), key=lambda s: s.id):
        if shot.id in ok:
            entry = _ok_entry(shot.id, "video", f"{CLIPS_DIR}/shot_{shot.id}.webm", ok[shot.id])
            entry["length"] = lengths.get(shot.id)
            if shot.id in sizes:
                entry["width"], entry["height"] = sizes[shot.id]
            entries.append(entry)
        elif shot.id in errors:
            entries.append(_failed_entry(shot.id, errors[shot.id]))

    manifest_rel = f"{CLIPS_DIR}/manifest.json"
    _write_json(ws / manifest_rel, {"shots": entries})
    return _summary("clips", entries, manifest_rel)


def generate_voiceovers(workspace: str | Path | None = None) -> dict:
    """阶段 3：为每个分镜生成中文配音（edge-tts，失败自动回退）。"""
    ws = workspace_root(workspace)
    try:
        shots = load_storyboard(ws / STORYBOARD_FILE)
    except StoryboardError as exc:
        return {"stage": "voiceovers", "error": str(exc)}

    entries: list[dict] = []
    for shot in shots:
        dest = ws / AUDIO_DIR / f"shot_{shot.id}.mp3"
        if dest.is_file() and dest.stat().st_size > 0:
            entries.append(_ok_entry(shot.id, "audio", f"{AUDIO_DIR}/shot_{shot.id}.mp3", dest))
            continue
        try:
            path = tts.synthesize_audio(text=shot.narration or "。", dest=dest)
            entries.append(_ok_entry(shot.id, "audio", f"{AUDIO_DIR}/shot_{shot.id}.mp3", path))
        except Exception as exc:  # noqa: BLE001 - TTS 已有静音回退，此处兜底
            entries.append(_failed_entry(shot.id, str(exc)))

    manifest_rel = f"{AUDIO_DIR}/manifest.json"
    _write_json(ws / manifest_rel, {"shots": entries})
    return _summary("voiceovers", entries, manifest_rel)


def _bgm_path() -> Path | None:
    """可选背景音乐（PROMPT2VID_BGM 指向音频文件时启用，循环低音量混入）。"""
    raw = env_str("PROMPT2VID_BGM", "")
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None


def render_final(workspace: str | Path | None = None) -> dict:
    """阶段 4：确定性剪辑——拼接、逐镜音频对齐、SRT 字幕、混流烧录。

    字幕时间轴与音轨时长都基于每个片段的真实时长（ffprobe 探测），
    不依赖 LLM 手写时间轴。
    """
    ws = workspace_root(workspace)
    try:
        clips = read_manifest(ws / CLIPS_DIR / "manifest.json")
    except (FileNotFoundError, ValueError) as exc:
        return {"stage": "render", "error": f"缺少视频片段清单: {exc}"}
    try:
        audios = {
            int(a.get("id", 0)): resolve_in_workspace(a.get("path", a.get("audio", "")), ws)
            for a in read_manifest(ws / AUDIO_DIR / "manifest.json")
            if a.get("status") in (None, "ok")
        }
    except (FileNotFoundError, ValueError):
        audios = {}
    try:
        storyboard = {s.id: s for s in load_storyboard(ws / STORYBOARD_FILE)}
    except StoryboardError:
        storyboard = {}

    ok_clips = [c for c in clips if c.get("status") == "ok"]
    ok_clips.sort(key=lambda c: int(c.get("id", 0)))
    if not ok_clips:
        return {"stage": "render", "error": "没有可用的视频片段，无法剪辑"}

    work_dir = ws / FINAL_DIR / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. 逐镜：探测真实时长 → 配音超长时延长视频（定格末帧）而非截断配音 →
    #    音频对齐到该时长（响度归一）→ 规范化转码
    normalized: list[Path] = []
    fitted_audio: list[Path] = []
    timings: list[tuple[float, float, str]] = []
    audio_overruns: list[dict] = []
    cursor = 0.0
    for clip in ok_clips:
        shot_id = int(clip.get("id", 0))
        clip_path = resolve_in_workspace(clip.get("path", clip.get("video", "")), ws)
        try:
            duration = ffmpeg_ops.probe_file(clip_path)["duration_seconds"]
        except (FileNotFoundError, RuntimeError) as exc:
            return {"stage": "render", "error": f"探测片段 {clip_path} 失败: {exc}"}
        duration = max(0.1, duration)

        # 配音真实时长（探测失败按缺失处理）
        audio_path = audios.get(shot_id)
        audio_duration = 0.0
        if audio_path is not None and audio_path.is_file():
            try:
                audio_duration = ffmpeg_ops.probe_file(audio_path)["duration_seconds"]
            except (RuntimeError, FileNotFoundError):
                audio_duration = 0.0

        # 配音比片段长 → 用定格末帧延长视频，保证旁白不被截断
        effective = max(duration, audio_duration)
        extend = effective - duration
        if extend > 0.3:
            logger.info(
                "render: 分镜 %d 配音 %.1fs 超过片段 %.1fs，定格延长 %.1fs",
                shot_id, audio_duration, duration, extend,
            )
            audio_overruns.append(
                {
                    "id": shot_id,
                    "clip_seconds": round(duration, 2),
                    "audio_seconds": round(audio_duration, 2),
                    "extended_seconds": round(extend, 2),
                }
            )

        fitted = ffmpeg_ops.fit_audio(
            audio_path, effective, work_dir / f"audio_{shot_id}.wav"
        )
        fitted_audio.append(fitted)

        norm = ffmpeg_ops.normalize_clip(
            clip_path, work_dir / f"clip_{shot_id}.mp4", extend_seconds=extend
        )
        normalized.append(norm)

        shot = storyboard.get(shot_id)
        narration = shot.narration if shot else ""
        if narration:
            timings.append((cursor, cursor + effective, narration))
        cursor += effective

    # 2. 拼接视频与音轨
    merged = ffmpeg_ops.concat_media(normalized, ws / FINAL_DIR / "merged.mp4")
    voiceover = ffmpeg_ops.concat_media(
        fitted_audio, ws / FINAL_DIR / "voiceover.wav", reencode_audio=True
    )

    # 3. SRT 字幕（基于真实时间轴）
    srt = None
    if timings:
        srt = ffmpeg_ops.build_srt(timings, ws / FINAL_DIR / "final.srt")

    # 4. 混流 + 烧录字幕一步完成（可选 BGM 背景音乐）
    bgm = _bgm_path()
    final = ffmpeg_ops.mux_and_burn(
        merged, voiceover, srt, ws / FINAL_DIR / "final.mp4", bgm=bgm
    )

    probe = ffmpeg_ops.probe_file(final)
    report = {
        "stage": "render",
        "final_video": str(final.resolve()),
        "srt": str(srt.resolve()) if srt else None,
        "bgm": str(bgm) if bgm else None,
        "duration_seconds": probe["duration_seconds"],
        "width": probe["width"],
        "height": probe["height"],
        "has_audio": probe["has_audio"],
        "audio_overruns": audio_overruns,
        "shots": [
            {"id": int(c.get("id", 0)), "video": c.get("video", c.get("path", ""))}
            for c in ok_clips
        ],
    }
    _write_json(ws / FINAL_DIR / "render_report.json", report)
    return report


def run_quality_checks(workspace: str | Path | None = None) -> dict:
    """阶段 5：质检——成片存在/时长/分辨率/音轨 + 分镜产物齐全。"""
    ws = workspace_root(workspace)
    final = ws / FINAL_DIR / "final.mp4"

    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    final_exists = final.is_file() and final.stat().st_size > 0
    add("final_video_exists", final_exists, str(final.resolve()))

    duration = 0.0
    probe: dict = {}
    if final_exists:
        try:
            probe = ffmpeg_ops.probe_file(final)
            duration = probe["duration_seconds"]
        except RuntimeError:
            probe = {}

    expected = _expected_duration(ws)
    if expected:
        add(
            "duration_match",
            abs(duration - expected) <= 2.0,
            f"成片 {duration:.1f}s vs 需求 {expected:.1f}s（允许 ±2s）",
        )
    # 画幅现在跟随首帧宽高比（如 768x1344 → 480x832 / 576x1024），
    # 不能再用“宽 ≥ 512”判定，否则合法的竖屏 480x832 会被误判为不达标。
    video_w = int(probe.get("width", 0) or 0)
    video_h = int(probe.get("height", 0) or 0)
    min_pixels = env_int("PROMPT2VID_MIN_CLIP_PIXELS", int(512 * 768 * 0.9))
    resolution_ok = (
        video_w >= 448
        and video_h >= 448
        and video_w * video_h >= min_pixels
    )
    add(
        "resolution_ok",
        resolution_ok,
        f"{video_w}x{video_h}（要求 ≥448 边长且 ≥{min_pixels} 像素）",
    )
    add("audio_track_present", probe.get("has_audio", False),
        f"has_audio={probe.get('has_audio', False)}")

    missing_assets = _missing_assets(ws)
    add("assets_complete", not missing_assets,
        "全部产物存在" if not missing_assets else "; ".join(missing_assets[:5]))

    passed = all(c["passed"] for c in checks)
    report = {
        "passed": passed,
        "checks": checks,
        "summary": "质检通过" if passed else "质检未通过，详见 checks",
        "final_video": str(final.resolve()) if final.is_file() else None,
    }
    _write_json(ws / REPORT_DIR / "quality_report.json", report)
    return report


def _expected_duration(ws: Path) -> float:
    """从 requirement.json / storyboard.json 宽松读取期望时长。"""
    for name, key in (
        (REQUIREMENT_FILE, "duration_seconds"),
        (STORYBOARD_FILE, "total_duration_seconds"),
    ):
        path = ws / name
        if not path.is_file():
            continue
        try:
            data = extract_json(path.read_text(encoding="utf-8"))
            value = float(data.get(key, 0))  # type: ignore[arg-type]
            if value > 0:
                return value
        except (StoryboardError, TypeError, ValueError):
            continue
    return 0.0


def _missing_assets(ws: Path) -> list[str]:
    """核验三个 manifest 中每个 ok 条目的文件真实存在。"""
    missing: list[str] = []
    for rel in (f"{FRAMES_DIR}/manifest.json", f"{CLIPS_DIR}/manifest.json",
                f"{AUDIO_DIR}/manifest.json"):
        path = ws / rel
        if not path.is_file():
            missing.append(f"{rel} 缺失")
            continue
        try:
            for entry in read_manifest(path):
                if entry.get("status") not in (None, "ok"):
                    continue
                asset = resolve_in_workspace(entry.get("path", ""), ws)
                if not asset.is_file():
                    missing.append(f"{rel}: 缺 {entry.get('id')}")
        except ValueError as exc:
            missing.append(f"{rel}: {exc}")
    return missing


def _ok_entry(shot_id: int, kind: str, rel: str, path: Path) -> dict:
    """manifest 条目：kind 为 image / video / audio。"""
    return {
        "id": shot_id,
        kind: rel,
        "status": "ok",
        "path": str(Path(path).resolve()),
    }


def _failed_entry(shot_id: int, error: str) -> dict:
    return {"id": shot_id, "status": "failed", "error": error}
