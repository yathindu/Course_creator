"""
Full-lesson real cost comparison: runs every image/audio/video task from both
lesson JSONs (the same task list collect_media_tasks() builds for
generate_lesson_media.py and the Streamlit app) through both backends for
real, and records per-task cost.

HF's API returns no cost field, so HF costs are priced from its published
per-unit rates (HF_IMAGE_COST / HF_VIDEO_COST below). OpenRouter's cost is
real and billed, parsed from its tool results ("(cost: $X)").

Resumable: each item skips generation if its out_path already exists, so a
partial/interrupted run can continue without double-spending. Skipped items
still count toward the cost total, since that money was already spent in an
earlier run. Per-item try/except means one failure never aborts the rest.

Usage:
    python calculate_full_lesson_costs.py
Writes cost_run_huggingface/<lesson_id>/..., cost_run_openrouter/<lesson_id>/...,
and a combined cost_report.json at the root.
"""

import json
import sys
from pathlib import Path

import professional.generate_lesson_media as hf
import professional.openrouter_pipeline as orp

HF_IMAGE_COST = 0.025  # FLUX.1-dev
HF_VIDEO_COST = 0.15  # Wan2.2-TI2V-5B, ~5s silent clip

# Only used for items already generated in an earlier interrupted run, where the
# real "(cost: $X)" string wasn't captured -- these are openrouter_pipeline.py's
# own verified default rates (Gemini 2.5 Flash Image; Veo 3.1 Lite at its default
# duration=8/with_audio=True), not estimates.
OR_IMAGE_COST_FALLBACK = 0.0387
OR_VIDEO_COST_FALLBACK = 0.40

LESSONS = ["courseware_output/lessons-introduction_to_marketing_concepts_foundations.json"]


def _parse_or_cost(result_str):
    if "cost: $" not in result_str:
        return 0.0
    try:
        return float(result_str.split("cost: $")[1].rstrip(")"))
    except ValueError:
        return 0.0


def run_hf(lesson, out_root):
    language = hf.resolve_language(lesson.get("medium"))
    image_tasks, audio_tasks, video_tasks = hf.collect_media_tasks(lesson)
    items, total = [], 0.0

    for t in image_tasks:
        style = t["style"] or "illustration"
        out_path = str(out_root / t["out_path"])
        if Path(out_path).exists():
            items.append({"activity_id": t["activity_id"], "kind": "image", "path": out_path,
                          "cost": HF_IMAGE_COST, "status": "already_existed"})
            total += HF_IMAGE_COST
            print(f"  [HF image] {t['activity_id']} -> SKIPPED (already exists), counted ${HF_IMAGE_COST}", flush=True)
            continue
        try:
            path = hf.generate_image.invoke({"subject": t["subject"], "style": style, "out_path": out_path})
            items.append({"activity_id": t["activity_id"], "kind": "image", "path": path, "cost": HF_IMAGE_COST})
            total += HF_IMAGE_COST
            print(f"  [HF image] {t['activity_id']} -> ${HF_IMAGE_COST}", flush=True)
        except Exception as e:
            items.append({"activity_id": t["activity_id"], "kind": "image", "status": "failed", "error": str(e)})
            print(f"  [HF image] {t['activity_id']} FAILED: {e}", flush=True)

    for t in audio_tasks:
        out_path = str(out_root / t["out_path"])
        if Path(out_path).exists():
            items.append({"activity_id": t["activity_id"], "kind": "audio", "path": out_path,
                          "cost": 0.0, "status": "already_existed"})
            print(f"  [HF audio] {t['activity_id']} -> SKIPPED (already exists)", flush=True)
            continue
        try:
            path = hf.generate_audio.invoke({"text": t["text"], "language": language, "out_path": out_path})
            items.append({"activity_id": t["activity_id"], "kind": "audio", "path": path, "cost": 0.0})
            print(f"  [HF audio] {t['activity_id']} -> $0 (gTTS)", flush=True)
        except Exception as e:
            items.append({"activity_id": t["activity_id"], "kind": "audio", "status": "failed", "error": str(e)})
            print(f"  [HF audio] {t['activity_id']} FAILED: {e}", flush=True)

    for t in video_tasks:
        out_path = str(out_root / t["out_path"])
        if Path(out_path).exists():
            items.append({"activity_id": t["activity_id"], "kind": "video", "path": out_path,
                          "cost": HF_VIDEO_COST, "status": "already_existed"})
            total += HF_VIDEO_COST
            print(f"  [HF video] {t['activity_id']} -> SKIPPED (already exists), counted ${HF_VIDEO_COST}", flush=True)
            continue
        try:
            path = hf.generate_video.invoke({"scene_description": t["scene_description"], "out_path": out_path})
            items.append({"activity_id": t["activity_id"], "kind": "video", "path": path, "cost": HF_VIDEO_COST})
            total += HF_VIDEO_COST
            print(f"  [HF video] {t['activity_id']} -> ${HF_VIDEO_COST}", flush=True)
        except Exception as e:
            items.append({"activity_id": t["activity_id"], "kind": "video", "status": "failed", "error": str(e)})
            print(f"  [HF video] {t['activity_id']} FAILED: {e}", flush=True)

    return items, total


def run_openrouter(lesson, out_root):
    language = hf.resolve_language(lesson.get("medium"))
    image_tasks, audio_tasks, video_tasks = hf.collect_media_tasks(lesson)
    items, total = [], 0.0

    for t in image_tasks:
        style = t["style"] or "illustration"
        out_path = str(out_root / t["out_path"])
        if Path(out_path).exists():
            items.append({"activity_id": t["activity_id"], "kind": "image", "path": out_path,
                          "cost": OR_IMAGE_COST_FALLBACK, "status": "already_existed"})
            total += OR_IMAGE_COST_FALLBACK
            print(f"  [OR image] {t['activity_id']} -> SKIPPED (already exists), counted ${OR_IMAGE_COST_FALLBACK}", flush=True)
            continue
        try:
            result = orp.generate_image.invoke({"subject": t["subject"], "style": style, "out_path": out_path})
            cost = _parse_or_cost(result)
            items.append({"activity_id": t["activity_id"], "kind": "image", "result": result, "cost": cost})
            total += cost
            print(f"  [OR image] {t['activity_id']} -> {result}", flush=True)
        except Exception as e:
            items.append({"activity_id": t["activity_id"], "kind": "image", "status": "failed", "error": str(e)})
            print(f"  [OR image] {t['activity_id']} FAILED: {e}", flush=True)

    for t in audio_tasks:
        out_path = str(out_root / t["out_path"])
        if Path(out_path).exists():
            items.append({"activity_id": t["activity_id"], "kind": "audio", "path": out_path,
                          "cost": 0.0, "status": "already_existed"})
            print(f"  [OR audio] {t['activity_id']} -> SKIPPED (already exists)", flush=True)
            continue
        try:
            result = orp.generate_audio.invoke({"text": t["text"], "language": language, "out_path": out_path})
            items.append({"activity_id": t["activity_id"], "kind": "audio", "result": result, "cost": 0.0})
            print(f"  [OR audio] {t['activity_id']} -> {result}", flush=True)
        except Exception as e:
            items.append({"activity_id": t["activity_id"], "kind": "audio", "status": "failed", "error": str(e)})
            print(f"  [OR audio] {t['activity_id']} FAILED: {e}", flush=True)

    for t in video_tasks:
        out_path = str(out_root / t["out_path"])
        if Path(out_path).exists():
            items.append({"activity_id": t["activity_id"], "kind": "video", "path": out_path,
                          "cost": OR_VIDEO_COST_FALLBACK, "status": "already_existed"})
            total += OR_VIDEO_COST_FALLBACK
            print(f"  [OR video] {t['activity_id']} -> SKIPPED (already exists), counted ${OR_VIDEO_COST_FALLBACK}", flush=True)
            continue
        try:
            result = orp.generate_video.invoke({"scene_description": t["scene_description"], "out_path": out_path})
            cost = _parse_or_cost(result)
            items.append({"activity_id": t["activity_id"], "kind": "video", "result": result, "cost": cost})
            total += cost
            print(f"  [OR video] {t['activity_id']} -> {result}", flush=True)
        except Exception as e:
            items.append({"activity_id": t["activity_id"], "kind": "video", "status": "failed", "error": str(e)})
            print(f"  [OR video] {t['activity_id']} FAILED: {e}", flush=True)

    return items, total


def main():
    lessons = sys.argv[1:] or LESSONS
    grand = {"huggingface": 0.0, "openrouter": 0.0}
    full_report = {}

    for lesson_path in lessons:
        lesson = json.loads(Path(lesson_path).read_text(encoding="utf-8"))
        lesson_id = lesson["id"]
        print(f"\n=== {lesson_id} ({lesson_path}) ===")

        print("-- Hugging Face --")
        hf_items, hf_total = run_hf(lesson, Path("cost_run_huggingface") / lesson_id)
        print(f"HF total for {lesson_id}: ${hf_total:.4f}")

        print("-- OpenRouter --")
        or_items, or_total = run_openrouter(lesson, Path("cost_run_openrouter") / lesson_id)
        print(f"OpenRouter total for {lesson_id}: ${or_total:.4f}")

        grand["huggingface"] += hf_total
        grand["openrouter"] += or_total
        full_report[lesson_id] = {
            "huggingface": {"items": hf_items, "total": round(hf_total, 4)},
            "openrouter": {"items": or_items, "total": round(or_total, 4)},
        }

    full_report["grand_total"] = {k: round(v, 4) for k, v in grand.items()}
    full_report["grand_total"]["combined"] = round(grand["huggingface"] + grand["openrouter"], 4)
    Path("cost_report.json").write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== GRAND TOTAL ===")
    print(f"Hugging Face: ${grand['huggingface']:.4f}")
    print(f"OpenRouter:   ${grand['openrouter']:.4f}")
    print(f"Combined:     ${grand['huggingface'] + grand['openrouter']:.4f}")
    print("\nFull report saved to cost_report.json")


if __name__ == "__main__":
    main()
