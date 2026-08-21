"""
Publishes a finished lesson (JSON + its generated media) from this pipeline into
Tutor-App's existing Cloudflare R2 content store, in the exact encrypted-Jinja2-
template format its R2LessonRepository already expects to fetch and decrypt.

Why this needs (almost) no format conversion -- confirmed by reading Tutor-App's
source directly (C:\\Users\\yathi\\Documents\\Tutor-App, 2026-08-21): its
Activity/Lesson Dart models (lib/shared/models/activity.dart, lesson.dart) parse
the exact same snake_case field names this pipeline's GeneratedLesson/Activity
already produce (correct_indices, correct_answer, correct_order, prompt_text,
acceptable_answers, image/audio/video, group), and its "Jinja2 template" fetch
(lib/shared/services/jinja_renderer_service.dart) just runs the decrypted text
through Jinja's Environment.render() -- a no-op when the content has no
{{ }}/{% %} tags, which our lesson JSON never does. So a finished lesson dict
from this pipeline IS already a valid Tutor-App lesson template; only
encryption + the right R2 path are needed.

Path convention (matches lib/shared/repositories/r2_lesson_repository.dart's
buildUrl() and lib/shared/providers/lessons_provider.dart's rewriteLessonAssets()
exactly): every current StudentProfile defaults to the same institute/agent/
teacher/category (the app's first-time setup wizard doesn't currently let a
student override them -- see lib/shared/models/student_profile.dart), so
DEFAULT_* below is effectively the one content partition all students read from
today, not a guess. Confirmed against the real bucket, 2026-08-21: the live
object for g2_english_l1 is at exactly this path, named
"lessons-g2_english_l1.json.enc" (the original filename -- fetchLessonTemplate()
tries that before its "lesson1.json.enc" rename fallback).

Setup:
    pip install boto3 pycryptodome
    Add to .env (these are NOT this project's own secrets -- get them from
    whoever administers Tutor-App's Cloudflare account):
        R2_ACCOUNT_ID=...
        R2_ACCESS_KEY_ID=...
        R2_SECRET_ACCESS_KEY=...
        R2_BUCKET_NAME=store                      # the real R2 bucket name (verified
                                                     # against the live bucket, 2026-08-21)
        R2_KEY_PREFIX=mentora-store                # NOT the bucket name -- a path prefix
                                                     # baked into Tutor-App's own AppConfig.
                                                     # r2BaseUrl, needed on every object key
        TUTOR_APP_R2_DECRYPTION_KEY=...            # must equal AppConfig.remoteDecryptionKey
                                                     # in app_config.dart EXACTLY (32 chars)

This module only ever WRITES to R2 when publish_lesson()/CLI is called with
dry_run=False (--live) -- importing it, building a plan, or running --help never
touches the network. Nothing in this project calls it automatically; publishing
is always a deliberate, explicitly-triggered action (a teacher clicking "Publish"
in app.py after previewing, or a person running --live by hand).
"""

import argparse
import base64
import json
import os
import re
from pathlib import Path

import boto3
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad
from dotenv import load_dotenv

from student.courseware_extraction import resolve_language

load_dotenv()

# Matches every current StudentProfile's defaults in Tutor-App's
# lib/shared/models/student_profile.dart exactly.
DEFAULT_INSTITUTE = "Institute|Agent"
DEFAULT_AGENT = "mento"
DEFAULT_TEACHER = "200422400660"
DEFAULT_CATEGORY = "local-syllabus"

# Path segment baked into Tutor-App's own AppConfig.r2BaseUrl -- the app always
# fetches from {r2BaseUrl}/... which already ends in "/mentora-store". This is
# NOT part of buildUrl()'s own path construction (r2_prefix() below mirrors
# that separately), it's specific to how the bucket's public dev URL was set
# up, so every object key this script writes needs it prepended.
DEFAULT_KEY_PREFIX = "mentora-store"

_MEDIA_KIND_SUBFOLDER = {"image": "images", "audio": "audios", "video": "videos"}
_UNIT_RE = re.compile(r"_l(\d+)$")


def _env(name):
    """Read a required env var, failing loudly at call time rather than silently
    publishing with an empty/wrong credential -- these are only read inside
    publish_lesson()/_s3_client(), never at import time, so this module can still
    be imported (e.g. for its path-building helpers) without them set."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set -- see this module's docstring for setup.")
    return value


def _s3_client():
    account_id = _env("R2_ACCOUNT_ID")
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=_env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=_env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
    )


def encrypt_for_tutor_app(plaintext: bytes) -> bytes:
    """AES-256-CBC, PKCS7-padded, random 16-byte IV prepended to the ciphertext,
    then base64-encoded -- byte-for-byte the same scheme as Tutor-App's
    lib/core/utils/encryption_utils.dart EncryptionUtils.encrypt() and
    lesson_encoder.html's encryptData(), so the app's EncryptionUtils.decrypt()
    can read the result unmodified."""
    key = _env("TUTOR_APP_R2_DECRYPTION_KEY").encode("utf-8")
    if len(key) != 32:
        raise ValueError(f"TUTOR_APP_R2_DECRYPTION_KEY must be exactly 32 bytes, got {len(key)}")
    iv = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))
    return base64.b64encode(iv + ciphertext)


def r2_prefix(grade, subject, unit, medium, institute=DEFAULT_INSTITUTE, agent=DEFAULT_AGENT,
              teacher=DEFAULT_TEACHER, category=DEFAULT_CATEGORY):
    """Mirrors buildUrl()'s path construction (and its medium-forcing rule) in
    r2_lesson_repository.dart exactly -- a wrong prefix here means the app looks
    in the wrong place and silently falls back to cached/no content, not an
    error, so this has to match byte-for-byte.

    `medium` here is this pipeline's own value ("english"/"sinhala", as typed into
    courseware_portal.py's Medium field) -- resolve_language() converts it to the
    short "en"/"si" code Tutor-App's StudentProfile.medium and R2 paths actually
    use (confirmed by reading lib/shared/models/student_profile.dart: its default
    is medium='en', never a full word). Passing the long-form word straight
    through here was a real bug caught by testing this function, not
    hypothetical -- it would have silently published to a path the app never
    looks at."""
    resolved_medium = "en" if subject == "english" else resolve_language(medium)
    return f"{institute}/{agent}/{teacher}/{category}/{resolved_medium}/{subject}/grade{grade}/unit{unit}"


def infer_unit(lesson_id: str) -> int:
    """Lesson ids from this pipeline always end "..._l<N>" (e.g. g2_english_l1) --
    same convention Tutor-App's own filteredLessonsProvider (lessons_provider.dart)
    already parses with this identical regex to sort lessons numerically."""
    match = _UNIT_RE.search(lesson_id)
    if not match:
        raise ValueError(f'Cannot infer a unit number from lesson id "{lesson_id}" (expected it to end "_lN").')
    return int(match.group(1))


def _collect_media_references(lesson: dict):
    """Walk a lesson's activities and return {(kind, filename)} for every
    image/audio/video this lesson actually references -- kind is "image"/"audio"/
    "video" (the R2 subfolder it belongs under regardless of what prefix, if any,
    the JSON field itself carries; see r2_prefix()'s docstring on why the app
    always resolves to images/audios/videos/ subfolders no matter what).

    Deliberately reference-driven rather than "upload everything in the folder":
    real generated_content/<lesson_id>/ folders (confirmed against the actual
    g2_english_l1 output on disk) also contain non-media files like
    approvals.json, and legacy hand-authored lessons store media flat
    (generated_content/<id>/cow.png) while assign_media_filenames() writes newer
    AI-generated lessons' JSON with an "images/"/"audios/"/"videos/" prefix
    already baked into the field value -- collecting by reference and stripping
    any such prefix handles both layouts without hardcoding either one."""
    refs = set()

    def add(kind, value):
        if not value or value.startswith(("http://", "https://", "assets/")):
            return
        subfolder = _MEDIA_KIND_SUBFOLDER[kind] + "/"
        filename = value[len(subfolder):] if value.startswith(subfolder) else value
        refs.add((kind, filename))

    for activity in lesson.get("activities", []):
        add("image", activity.get("image"))
        add("audio", activity.get("audio"))
        add("video", activity.get("video"))
        for option in activity.get("options") or []:
            if isinstance(option, dict):
                add("image", option.get("image"))
    return refs


def _find_media_file(media_dir: Path, kind: str, filename: str):
    """Real content on disk has been seen in both layouts (see
    _collect_media_references's docstring) -- try the flat legacy layout first
    since that's what's actually on disk for this project's existing lessons
    today, then the images/audios/videos-subfolder layout newer output may use."""
    flat = media_dir / filename
    if flat.exists():
        return flat
    nested = media_dir / _MEDIA_KIND_SUBFOLDER[kind] / filename
    return nested if nested.exists() else None


def build_publish_plan(lesson: dict, media_dir: Path, grade=None, subject=None, unit=None, medium=None,
                        bucket=None, key_prefix=None):
    """Figures out exactly what publish_lesson() would upload and where, without
    touching the network or requiring credentials beyond bucket/key_prefix
    defaults -- this is what powers both dry_run=True's printout and app.py's
    "Preview" step, so the two can never drift apart from what a real publish
    would actually do.

    grade/subject/unit/medium all default to the lesson dict's own fields (this
    pipeline always sets grade/subject/medium on GeneratedLesson, and unit is
    inferred from the id) -- only pass them explicitly to override."""
    grade = grade if grade is not None else lesson.get("grade")
    subject = subject or lesson.get("subject")
    medium = medium or lesson.get("medium", "english")
    unit = unit if unit is not None else infer_unit(lesson["id"])
    if grade is None or not subject:
        raise ValueError('Lesson is missing "grade"/"subject" and no override was given.')

    bucket = bucket or os.environ.get("R2_BUCKET_NAME", "")
    key_prefix = key_prefix or os.environ.get("R2_KEY_PREFIX", DEFAULT_KEY_PREFIX)
    prefix = f"{key_prefix}/{r2_prefix(grade, subject, unit, medium)}"
    lesson_filename = f"lessons-g{grade}_{subject}_l{unit}.json"

    uploads = [{"key": f"{prefix}/{lesson_filename}.enc", "kind": "lesson", "source": None, "label": lesson_filename}]
    missing = []
    for kind, filename in sorted(_collect_media_references(lesson)):
        found = _find_media_file(media_dir, kind, filename)
        if found is None:
            missing.append(f"{kind}: {filename}")
            continue
        uploads.append({"key": f"{prefix}/{_MEDIA_KIND_SUBFOLDER[kind]}/{filename}", "kind": "media",
                         "source": found, "label": filename})

    return {"bucket": bucket, "prefix": prefix, "uploads": uploads, "missing": missing}


def publish_lesson(lesson: dict, media_dir: Path, grade=None, subject=None, unit=None, medium=None,
                    bucket=None, key_prefix=None, dry_run=True):
    """lesson: this pipeline's finished lesson dict (already parsed -- e.g. what
    app.py holds in memory after "Upload lesson JSON", or json.loads() of a
    lessons-*.json file). media_dir: the matching
    student/generated_content/<lesson_id>/ folder -- every image/audio/video the
    lesson actually references gets located under it (flat or
    images/audios/videos-nested, see _find_media_file) and uploaded to the
    images/audios/videos R2 subfolders rewriteLessonAssets() always looks in.

    dry_run=True (the default) builds the plan and returns it without calling S3
    -- publishing real content to Tutor-App's production bucket is a deliberate,
    explicitly-requested action, not something this function should do by
    default just because it was called.

    Returns the plan dict from build_publish_plan(), plus "published": bool."""
    plan = build_publish_plan(lesson, media_dir, grade, subject, unit, medium, bucket, key_prefix)
    if not plan["bucket"]:
        raise RuntimeError('R2_BUCKET_NAME is not set -- see this module\'s docstring for setup.')

    if dry_run:
        return {**plan, "published": False}

    s3 = _s3_client()
    lesson_bytes = json.dumps(lesson, indent=2, ensure_ascii=False).encode("utf-8")
    for upload in plan["uploads"]:
        if upload["kind"] == "lesson":
            body = encrypt_for_tutor_app(lesson_bytes)
        else:
            body = upload["source"].read_bytes()
        s3.put_object(Bucket=plan["bucket"], Key=upload["key"], Body=body)
    return {**plan, "published": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish a finished lesson to Tutor-App's R2 store")
    parser.add_argument("lesson_json", type=Path, help="Path to a finished lessons-*.json file")
    parser.add_argument("media_dir", type=Path, help="Matching generated_content/<lesson_id>/ folder")
    parser.add_argument("--grade", type=int, help="Override the lesson JSON's own grade field")
    parser.add_argument("--subject", help="Override the lesson JSON's own subject field")
    parser.add_argument("--unit", type=int, help="Override the unit number inferred from the lesson id")
    parser.add_argument("--medium", help="Override the lesson JSON's own medium field")
    parser.add_argument("--live", action="store_true", help="Actually upload (default is a dry run print-only)")
    args = parser.parse_args()

    _lesson = json.loads(args.lesson_json.read_text(encoding="utf-8"))
    result = publish_lesson(
        _lesson, args.media_dir, args.grade, args.subject, args.unit, args.medium,
        dry_run=not args.live,
    )

    if result["missing"]:
        print(f"WARNING: {len(result['missing'])} referenced media file(s) not found under {args.media_dir} "
              f"-- the app will show broken media for these:")
        for m in result["missing"]:
            print(f"  {m}")

    verb = "Published" if result["published"] else "[dry run] Would publish"
    print(f"{verb} {len(result['uploads'])} file(s) to r2://{result['bucket']}/{result['prefix']}/ :")
    for u in result["uploads"]:
        print(f"  {u['label']} -> {u['key']}")
