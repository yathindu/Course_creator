"""
Pydantic models for the Course Creation (formerly "Courseware Design Portal") page's generated content --
shared between courseware_extraction.py (LLM generation) and
courseware_portal.py (teacher review/editing UI).

These mirror the lesson.json schema generate_lesson_media.py/app.py already
consume (see lessons-g2_maths_l1.json / lessons-g2_english_l1.json for
real examples) -- deliberately NOT a new schema. image/audio/video here
are still just relative filenames; nothing in this portal generates media
itself, that's still generate_lesson_media.py's job. (Renamed from
image_url/audio_url/video_url -- the media-URL field convention changed
across the project; see generate_lesson_media.py's docstring.)
"""

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

ActivityType = Literal[
    "content", "vocabulary", "mcq", "image_selection", "true_false",
    "fill_blank", "ordering", "listening", "speaking", "video",
]

# Which non-media fields each activity type requires -- used by validate_lesson()
# to catch an LLM generation/paraphrase leaving an activity structurally broken
# before a teacher ever sees it.
REQUIRED_FIELDS = {
    "content": ["body"],
    "vocabulary": ["word", "definition", "sentence"],
    "mcq": ["question", "options", "correct_indices"],
    "image_selection": ["question", "options", "correct_indices"],
    "true_false": ["question", "correct_answer"],
    "fill_blank": ["question", "options", "correct_indices"],
    "ordering": ["words", "correct_order"],
    "listening": ["question", "options", "correct_indices"],
    "speaking": ["prompt_text", "acceptable_answers"],
    "video": [],  # nothing beyond title/instructions -- the video itself is assigned, not authored
}


class ImageOption(BaseModel):
    label: str
    image: str = Field(description="Relative image filename, e.g. images/cow.png -- may be left blank, "
                                    "the portal assigns real filenames deterministically after generation")


class Activity(BaseModel):
    id: str
    type: ActivityType
    title: str
    instructions: str

    # Optional visual clustering only -- activities sharing the same group are shown
    # together in app.py's Activity picker. Portal-internal-ish like Grouping below,
    # but (unlike Grouping) this one DOES round-trip through the exported lesson.json,
    # since it's meaningful to the media-generation UI, not just this portal.
    group: Optional[str] = None

    # content -- a block of instructional/explanatory text the learner reads and
    # hears narrated, not a question. Also used verbatim as narration audio text,
    # same pattern as vocabulary's `sentence`.
    body: Optional[str] = None

    # vocabulary
    word: Optional[str] = None
    definition: Optional[str] = None
    sentence: Optional[str] = None

    # mcq / fill_blank / listening / image_selection
    question: Optional[str] = None
    options: Optional[list[Union[str, ImageOption]]] = None
    correct_indices: Optional[list[int]] = None

    # true_false
    correct_answer: Optional[bool] = None

    # ordering
    words: Optional[list[str]] = None
    correct_order: Optional[list[int]] = None

    # speaking
    prompt_text: Optional[str] = None
    acceptable_answers: Optional[list[str]] = None

    # media -- filenames only, assigned deterministically by
    # courseware_extraction.assign_media_filenames(), never invented by the
    # LLM (same "filename never invented by the scripts" rule generate_lesson_media.py
    # follows downstream). Left unset until that step runs.
    image: Optional[str] = None
    audio: Optional[str] = None
    video: Optional[str] = None


class LessonSkeleton(BaseModel):
    id: str
    title: str
    description: str


class LessonSkeletonsResult(BaseModel):
    lessons: list[LessonSkeleton]


class GeneratedLesson(BaseModel):
    id: str
    version: int = 1
    skill_area: Literal["technical", "soft_skills"]
    subject: str
    medium: Optional[str] = None
    title: str
    description: str
    pass_percentage: int = 70
    activities: list[Activity]


class Grouping(BaseModel):
    id: str
    title: str
    description: str = Field(description="What content from the source this group covers")


class GroupingsResult(BaseModel):
    groupings: list[Grouping]


BASE_FIELDS = {"id", "type", "title", "instructions", "group", "image", "audio", "video"}


def prune_activity_fields(activity: Activity):
    """Null out fields that don't belong to this activity's type. Structured-output
    generation fills in every optional field on the shared Activity model regardless
    of type (confirmed by real testing -- a "listening" activity came back also
    carrying word/definition/correct_answer/words/correct_order/prompt_text), since
    nothing stops an LLM from populating fields just because the schema allows it.
    Harmless for generate_lesson_media.py (it only reads fields for the matching
    activity type), but leaves messy JSON that doesn't match the hand-authored
    lessons' shape -- so strip it here instead of downstream."""
    allowed = BASE_FIELDS | set(REQUIRED_FIELDS.get(activity.type, []))
    for field in activity.model_fields:
        if field not in allowed:
            setattr(activity, field, None)
    # image/audio/video are legitimately optional on every type (any activity can opt
    # into a video per generate_lesson_media.py's convention), so they're always
    # "allowed" above -- but generation leaves unused ones as "" rather than actually
    # omitting them. Normalize so an unset media field reads as unset, matching the
    # hand-authored lesson JSONs (which never include a key for media that activity
    # doesn't use).
    for field in ("image", "audio", "video"):
        if getattr(activity, field) == "":
            setattr(activity, field, None)
    # An "ordering" activity with a missing/invalid correct_order used to just get
    # flagged by validate_lesson() and left for the teacher to fix by hand (setting
    # each word's position one by one in the review screen). Auto-repairing to the
    # words' own natural order here removes that friction for the common case --
    # matches courseware_portal.py's own _activity_editor() fallback logic (a
    # simple declarative sentence is often already correct in its given order),
    # and the teacher can still see and correct it in review if it's wrong, same
    # as any other generated content.
    if activity.type == "ordering" and activity.words:
        n = len(activity.words)
        if activity.correct_order is None or sorted(activity.correct_order) != list(range(n)):
            activity.correct_order = list(range(n))
    return activity


def prune_lesson_fields(lesson: GeneratedLesson):
    for activity in lesson.activities:
        prune_activity_fields(activity)
    return lesson


def validate_lesson(lesson: GeneratedLesson):
    """Structural invariant check -- pure Python, no LLM. Run after generation
    AND after any paraphrase pass, since an LLM rewrite can silently break
    correct_indices/correct_order/required fields without the JSON shape
    itself looking wrong. Returns a list of human-readable issue strings,
    one per activity problem found; empty list means the lesson is structurally
    sound (not a guarantee the *content* is pedagogically good -- that's the
    teacher review step's job)."""
    issues = []
    for activity in lesson.activities:
        prefix = f"{activity.id} ({activity.type})"
        for field in REQUIRED_FIELDS.get(activity.type, []):
            if getattr(activity, field, None) in (None, "", []):
                issues.append(f"{prefix}: missing required field '{field}'")

        if activity.correct_indices is not None and activity.options is not None:
            n = len(activity.options)
            for idx in activity.correct_indices:
                if not (0 <= idx < n):
                    issues.append(f"{prefix}: correct_indices has out-of-range index {idx} (options has {n})")

        if activity.type == "ordering" and activity.words is not None and activity.correct_order is not None:
            n = len(activity.words)
            if sorted(activity.correct_order) != list(range(n)):
                issues.append(f"{prefix}: correct_order is not a valid permutation of {n} word(s)")

        if activity.type == "image_selection" and activity.options:
            if not all(isinstance(o, ImageOption) for o in activity.options):
                issues.append(f"{prefix}: image_selection options must all be {{label, image}} objects")

    return issues
