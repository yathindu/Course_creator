"""
Reads a lesson JSON (matching the g2_maths_l1 / g2_english_l1 schema) and
generates every image/audio/video asset its activities reference but don't
have on disk yet. Filenames come straight from the JSON's
image/audio/video fields -- this script never invents its own. (Renamed
from image_url/audio_url/video_url -- older lesson JSONs written before
this rename need those keys migrated, since this is a full cutover, not a
dual-format compatibility shim.)

Which activity types need media, and what gets generated:
    vocabulary      -> image from `word`; audio from `sentence` (spoken)
    image_selection -> one image per option, from options[].label -> options[].image
    mcq             -> text-only by default; if options are {"label", "image"}
                        objects instead of plain strings, one image per option
                        (same shape as image_selection's options)
    listening       -> audio from the *correct* option's text (that's what
                        the learner is supposed to hear)
    speaking        -> audio from `prompt_text` (the sentence to read aloud)
    video           -> a standalone watchable video, no question attached;
                        scene description comes from whatever descriptive
                        text the activity has (see _scene_description below)
true_false / fill_blank / ordering are text-only -- nothing to generate.

Video is also generic beyond the dedicated "video" type above: any activity
type can opt in to a short (~5s) scene video simply by adding a `video`
field to it in the JSON -- same "filename comes from the JSON" convention
as image/audio. The scene description fed to the video model is built from
whatever descriptive text the activity has, in priority order: `sentence`
-> `definition` + `word` -> `question` / `prompt_text` -> `word` -> `title`.
Video generation itself is Wan2.2-TI2V-5B, an HF-hosted model (via the
fal-ai Inference Provider), same as ai_automation.py.

Activities may also carry an optional `group` field (a plain string) --
purely a display hint for app.py's Activity picker to visually cluster
related activities together; it has no effect on what gets generated here.

Image style is "cartoon" or "natural". An activity/option can set its own
"style" field in the JSON to override the run's default (--style, itself
defaulting to "cartoon" -- age-appropriate for these grade-2 lessons).

Language for audio comes from the lesson's "medium" field:
    "sinhala" -> "si" -> gTTS
    anything else / absent -> "en" -> gTTS
gTTS supports "si" natively, with better voice quality than local
SpeechT5-based alternatives.

Image/audio/video generation calls are LangChain tools (generate_image,
generate_audio, generate_video). *Which* tool to call and *where* it saves
is still decided deterministically by collect_media_tasks() reading the
lesson JSON -- there's no LLM in that decision, only ai_automation.py's
freeform-teacher-prompt flow uses an LLM to decide which tool to call.

Optional: --teacher-profile points at a JSON file (see teacher_profile.json)
with age_group/gender_target/tone/comment. When given, personalization.py's
LangChain system+user prompt chain rewrites *what* gets generated for
images, video, and vocabulary narration audio to match that profile.
listening/speaking audio is deliberately never personalized -- see
personalization.py's docstring for why (that text is checked verbatim
against fixed answer strings; rewriting it would break grading).

Setup:
    pip install langchain-core langchain-huggingface huggingface_hub gtts python-dotenv
    Put your token in a .env file next to this script:
        HF_TOKEN=your-pro-token-here

Usage:
    python generate_lesson_media.py lessons-g2_maths_l1.json
    python generate_lesson_media.py lessons-g2_english_l1.json --out content
    python generate_lesson_media.py lessons-g2_maths_l1.json --dry-run
    python generate_lesson_media.py lessons-g2_maths_l1.json --style natural --skip-existing=false
    python generate_lesson_media.py lessons-g2_maths_l1.json --teacher-profile teacher_profile.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.tools import tool

from student.audio_tools import synthesize_audio
from student.image_processing import remove_background

from student.personalization import load_teacher_profile, personalize

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

LANGUAGE_MAP = {"sinhala": "si"}
DEFAULT_LANGUAGE = "en"


def resolve_language(medium):
    """Case/whitespace-tolerant lookup into LANGUAGE_MAP -- a lesson's "medium" field
    is free text wherever a human types it (courseware_portal.py's Medium field,
    or a hand-edited lesson JSON), and a plain LANGUAGE_MAP.get(medium, ...) silently
    falls back to English for anything not the exact lowercase string "sinhala"
    (e.g. "Sinhala", "SINHALA", trailing whitespace) -- confirmed as a real bug via
    user report, not hypothetical. Always resolve through this instead of touching
    LANGUAGE_MAP directly."""
    return LANGUAGE_MAP.get((medium or "").strip().lower(), DEFAULT_LANGUAGE)

STYLE_PROMPTS = {
    "cartoon": ("flat vector cartoon illustration, bright simple colors, clean shapes, "
                "children's educational app style, flat 2D design, no text -- not a 3D render, not CGI"),
    "natural": ("wildlife documentary photograph, shot on a DSLR camera with a telephoto lens, RAW "
                "unedited photo, National Geographic style, real nature/wildlife photography, genuine "
                "fur/feather/skin texture, shallow depth of field, natural outdoor lighting, sharp focus, "
                "high detail, no text -- absolutely not a 3D render, not CGI, not Pixar or DreamWorks "
                "style, not an illustration, no cute exaggerated cartoon features, not a baby animal "
                "unless explicitly described as one"),
}


# ---------------------------------------------------------------------------
# Step 1: walk the lesson JSON and collect exactly what needs generating.
# Each task carries everything execution needs: the seed text/subject, the
# target path (taken verbatim from the JSON), and which activity it's for.
# ---------------------------------------------------------------------------
def _scene_description(activity):
    if activity.get("sentence"):
        return activity["sentence"]
    if activity.get("definition") and activity.get("word"):
        return f"{activity['word']}: {activity['definition']}"
    if activity.get("question"):
        return activity["question"]
    if activity.get("prompt_text"):
        return activity["prompt_text"]
    # Last resort for types with no other descriptive field (e.g. a standalone
    # "video" activity) -- title/instructions are the only text guaranteed present.
    return activity.get("word") or activity.get("title") or activity.get("instructions", "")


def collect_media_tasks(lesson):
    image_tasks = []
    audio_tasks = []
    video_tasks = []

    for activity in lesson["activities"]:
        activity_id = activity["id"]
        activity_type = activity["type"]

        group = activity.get("group")

        if activity.get("video"):
            video_tasks.append({
                "activity_id": activity_id,
                "activity_type": activity_type,
                "scene_description": _scene_description(activity),
                "out_path": activity["video"],
                "group": group,
            })

        if activity_type == "vocabulary":
            if activity.get("image"):
                image_tasks.append({
                    "activity_id": activity_id,
                    "activity_type": activity_type,
                    "subject": activity["word"],
                    "style": activity.get("style"),
                    "out_path": activity["image"],
                    "group": group,
                })
            if activity.get("audio"):
                audio_tasks.append({
                    "activity_id": activity_id,
                    "activity_type": activity_type,
                    "text": activity["sentence"],
                    "personalizable": True,  # narration, not an answer -- safe to rewrite
                    "out_path": activity["audio"],
                    "group": group,
                })

        elif activity_type == "image_selection":
            # Options are expected to be {"label", "image"} objects for this type, but a
            # hand-authored (not Course-Creation-generated) lesson JSON isn't guaranteed to
            # follow that -- skip anything that isn't a dict rather than crashing the whole
            # page with an AttributeError on a plain string.
            for option in activity.get("options", []):
                if isinstance(option, dict) and option.get("image"):
                    image_tasks.append({
                        "activity_id": activity_id,
                        "activity_type": activity_type,
                        "subject": option["label"],
                        "style": option.get("style") or activity.get("style"),
                        "out_path": option["image"],
                        "group": group,
                    })

        elif activity_type == "mcq":
            # Options are normally plain strings (text-only, nothing to generate).
            # An mcq can opt into image answer choices the same way image_selection
            # does, by using {"label": ..., "image": ...} objects instead of strings.
            for option in activity.get("options", []):
                if isinstance(option, dict) and option.get("image"):
                    image_tasks.append({
                        "activity_id": activity_id,
                        "activity_type": activity_type,
                        "subject": option.get("label", ""),
                        "style": option.get("style") or activity.get("style"),
                        "out_path": option["image"],
                        "group": group,
                    })

        elif activity_type == "listening":
            if activity.get("audio"):
                options = activity.get("options", [])
                correct_indices = activity.get("correct_indices", [])
                # Bounds-check before indexing -- a hand-authored lesson JSON isn't guaranteed
                # to have a valid correct_indices (Course-Creation-generated ones are, via
                # validate_lesson(), but this script also runs standalone against any lesson
                # file), and an out-of-range index would otherwise crash the whole page.
                if options and correct_indices and 0 <= correct_indices[0] < len(options):
                    spoken_text = options[correct_indices[0]]
                    audio_tasks.append({
                        "activity_id": activity_id,
                        "activity_type": activity_type,
                        "text": spoken_text,
                        "personalizable": False,  # must say exactly one of `options`, verbatim
                        "out_path": activity["audio"],
                        "group": group,
                    })

        elif activity_type == "speaking":
            if activity.get("audio") and activity.get("prompt_text"):
                audio_tasks.append({
                    "activity_id": activity_id,
                    "activity_type": activity_type,
                    "text": activity["prompt_text"],
                    "personalizable": False,  # checked against acceptable_answers, verbatim
                    "out_path": activity["audio"],
                    "group": group,
                })

        # true_false / fill_blank / ordering: text-only, nothing to do.

    return image_tasks, audio_tasks, video_tasks


# Execution, as LangChain tools. HF_TOKEN comes from the module-level global
# (loaded from .env). These same tool objects are imported directly by
# ai_automation.py's freeform orchestrator, so both paths share one
# implementation of what each tool can do.
@tool
def generate_image(
    subject: str,
    style: Literal["cartoon", "natural"],
    out_path: str,
    width: int = 1024,
    height: int = 1024,
    remove_bg: bool = False,
) -> str:
    """Generate an illustration image for a lesson activity and save it to out_path.
    width/height control resolution/aspect ratio. remove_bg strips the background,
    leaving only the subject on a transparent background."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    client = InferenceClient(provider="auto", token=HF_TOKEN)
    prompt = f"{subject}, {STYLE_PROMPTS[style]}"
    image = client.text_to_image(prompt, model="black-forest-labs/FLUX.1-dev", width=width, height=height)
    image.save(target)
    if remove_bg:
        remove_background(target, subject_text=subject)
    return str(target)


@tool
def generate_audio(
    text: str,
    language: Literal["en", "si"],
    out_path: str,
    voice: Literal["us", "uk", "australia", "india", "canada", "ireland", "south_africa"] = "us",
    volume_db: float = 0.0,
) -> str:
    """Generate spoken audio (TTS) for a word or sentence and save it to out_path.
    voice picks an English accent (ignored for Sinhala, gTTS only has one voice there).
    volume_db adjusts loudness (positive = louder, negative = quieter, 0 = default)."""
    return synthesize_audio(text, language, out_path, voice=voice, volume_db=volume_db)


@tool
def generate_video(scene_description: str, out_path: str) -> str:
    """Generate a short (~5 second) video clip for a lesson scene and save it to out_path."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    client = InferenceClient(provider="fal-ai", token=HF_TOKEN)
    result = client.text_to_video(scene_description, model="Wan-AI/Wan2.2-TI2V-5B")
    with open(target, "wb") as f:
        f.write(result)
    return str(target)


@tool
def animate_image(image_path: str, motion_description: str, out_path: str) -> str:
    """Animate an existing static image into a short video clip, using it as the
    starting frame (image-to-video, not text-to-video) -- the output keeps the
    image's exact look/style, it doesn't reimagine the scene."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    client = InferenceClient(provider="fal-ai", token=HF_TOKEN)
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    result = client.image_to_video(image_bytes, model="Wan-AI/Wan2.2-I2V-A14B", prompt=motion_description)
    with open(target, "wb") as f:
        f.write(result)
    return str(target)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(lesson_path, out_dir, default_style, skip_existing, dry_run, teacher_profile_path=None):
    lesson = json.loads(Path(lesson_path).read_text(encoding="utf-8"))
    language = resolve_language(lesson.get("medium"))
    out_root = Path(out_dir) / lesson["id"]
    teacher_profile = load_teacher_profile(teacher_profile_path) if teacher_profile_path else None

    image_tasks, audio_tasks, video_tasks = collect_media_tasks(lesson)
    print(f"Lesson: {lesson['id']} ({lesson.get('subject')}, grade {lesson.get('grade')}, "
          f"language={language})")
    print(f"  {len(image_tasks)} image(s), {len(audio_tasks)} audio clip(s), "
          f"{len(video_tasks)} video(s) referenced.\n")

    if dry_run:
        for t in image_tasks:
            style = t["style"] or default_style
            print(f"  [image] {t['activity_id']}: \"{t['subject']}\" (style={style}) -> {t['out_path']}")
        for t in audio_tasks:
            print(f"  [audio] {t['activity_id']}: \"{t['text']}\" -> {t['out_path']}")
        for t in video_tasks:
            print(f"  [video] {t['activity_id']}: \"{t['scene_description']}\" -> {t['out_path']}")
        return []

    needs_hf_token = image_tasks or video_tasks or (audio_tasks and language != "si")
    if not HF_TOKEN and needs_hf_token:
        raise RuntimeError("Set HF_TOKEN environment variable (Pro account token) before running.")

    results = []

    for t in image_tasks:
        target = out_root / t["out_path"]
        if skip_existing and target.exists():
            print(f"  [image] {t['activity_id']}: SKIPPED (already exists) -> {target}")
            results.append({**t, "kind": "image", "status": "skipped_exists"})
            continue
        style = t["style"] or default_style
        prompt_subject = t["subject"]
        if teacher_profile:
            enriched = personalize(t["subject"], t["activity_type"], "image", style, teacher_profile)
            prompt_subject = f"{t['subject']} -- {enriched}"  # literal subject always kept, enrichment appended
        try:
            path = generate_image.invoke({"subject": prompt_subject, "style": style, "out_path": str(target)})
            print(f"  [image] {t['activity_id']}: -> {path}")
            results.append({**t, "style": style, "kind": "image", "status": "success", "path": path})
        except Exception as e:
            print(f"  [image] {t['activity_id']}: FAILED ({e})")
            results.append({**t, "style": style, "kind": "image", "status": "failed", "error": str(e)})

    for t in audio_tasks:
        target = out_root / t["out_path"]
        if skip_existing and target.exists():
            print(f"  [audio] {t['activity_id']}: SKIPPED (already exists) -> {target}")
            results.append({**t, "kind": "audio", "status": "skipped_exists"})
            continue
        text = t["text"]
        if teacher_profile and t["personalizable"]:
            text = personalize(t["text"], t["activity_type"], "audio", "narration", teacher_profile)
        try:
            path = generate_audio.invoke({"text": text, "language": language, "out_path": str(target)})
            print(f"  [audio] {t['activity_id']}: -> {path}")
            results.append({**t, "text": text, "kind": "audio", "status": "success", "path": path})
        except Exception as e:
            print(f"  [audio] {t['activity_id']}: FAILED ({e})")
            results.append({**t, "kind": "audio", "status": "failed", "error": str(e)})

    for t in video_tasks:
        target = out_root / t["out_path"]
        if skip_existing and target.exists():
            print(f"  [video] {t['activity_id']}: SKIPPED (already exists) -> {target}")
            results.append({**t, "kind": "video", "status": "skipped_exists"})
            continue
        scene_description = t["scene_description"]
        if teacher_profile:
            scene_description = personalize(t["scene_description"], "video_scene", "video", "cinematic", teacher_profile)
        try:
            path = generate_video.invoke({"scene_description": scene_description, "out_path": str(target)})
            print(f"  [video] {t['activity_id']}: -> {path}")
            results.append({**t, "scene_description": scene_description, "kind": "video", "status": "success", "path": path})
        except Exception as e:
            print(f"  [video] {t['activity_id']}: FAILED ({e})")
            results.append({**t, "kind": "video", "status": "failed", "error": str(e)})

    manifest_path = out_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"lesson_id": lesson["id"], "language": language, "results": results},
                   f, indent=2, ensure_ascii=False)
    print(f"\n=== Done. Manifest -> {manifest_path} ===")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lesson_json", help="Path to a lesson JSON file")
    parser.add_argument("--out", default="generated_content", help="Output root directory (default: generated_content)")
    parser.add_argument("--style", choices=["cartoon", "natural"], default="cartoon",
                         help="Default image style; an activity/option can override this via its own \"style\" field in the JSON (default: cartoon)")
    parser.add_argument("--dry-run", action="store_true", help="List what would be generated without generating it")
    parser.add_argument("--skip-existing", type=lambda v: v.lower() != "false", default=True,
                         help="Skip assets that already exist on disk (default: true)")
    parser.add_argument("--teacher-profile", default=None,
                         help="Path to a teacher profile JSON (age_group/gender_target/tone/comment) to personalize generation prompts")
    args = parser.parse_args()

    run(args.lesson_json, args.out, args.style, args.skip_existing, args.dry_run, args.teacher_profile)
