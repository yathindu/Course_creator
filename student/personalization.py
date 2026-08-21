"""
LangChain prompt-personalization layer: rewrites generation content per a
teacher profile (age/gender/tone/comment) and lesson context (grade/subject/
medium), as a ChatPromptTemplate | ChatHuggingFace chain.

Scope, deliberately: this shapes *what* gets said to a generation tool, not
*which* tool gets called or *where* it saves -- that's still decided
deterministically by collect_media_tasks() in generate_lesson_media.py. Two
separate concerns, kept separate.

Correctness guardrail -- read this before wiring this into anything new:
some spoken text is checked verbatim against fixed answer strings
(`listening` activities: the audio must say exactly one of `options`;
`speaking` activities: the student's speech is checked against
`acceptable_answers`). Letting an LLM "personalize" that text would silently
break the activity's grading. So personalization is only applied to:
  - image generation prompts (style/mood descriptor -- the literal subject
    word is always kept as-is alongside it, never replaced)
  - video scene descriptions (illustrative, no correctness constraint)
  - vocabulary narration audio (the example sentence, not an answer)
generate_lesson_media.py enforces this by only calling personalize() for
those three cases -- listening/speaking audio tasks never reach this module.

Setup:
    pip install langchain-core langchain-huggingface python-dotenv
    Put your token in a .env file next to this script:
        HF_TOKEN=your-pro-token-here
"""

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from student.config import MODEL, MODEL_PROVIDER
from student.grade_bands import grade_band_context

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

SYSTEM_TEMPLATE = """You help generate lesson content for a children's education app.

Learning context:
- Grade: {grade}
- Subject: {subject}
- Medium (language): {medium}

{grade_band_context}

Mode (activity type): {activity_type}
Target media: {media_kind}
Desired behaviour/style: {behaviour}

Task: rewrite the given content into a short, vivid description suited to
the mode, media, and behaviour above. Preserve the core subject exactly --
never substitute, translate, or drop it. Match vocabulary and sentence
complexity to the grade-band guidance above.

If media is "image" and behaviour is "natural" (photorealistic): describe
an objective real-world photographic scene ONLY -- physical appearance,
pose, setting, lighting, like a wildlife-documentary caption. Do NOT use
storybook, fairy-tale, or anthropomorphic language: no personality, no
"happy"/"gentle"/"friendly", no talking, no props like flower crowns, no
whimsical scenery like butterflies unless literally present in the
original content. The teacher's tone/comment below describe how NARRATION
should sound to a child -- they say nothing about how the picture should
look. For behaviour="natural" ignore them completely, even if the tone is
"playful"; a photo of an adult animal is not less warm or appropriate for
a young learner than a picture of a baby animal.
If the subject is an animal, describe an ADULT of the species (unless the
original content specifically says baby/young/chick/calf/etc) and NEVER
use diminutive/cute wording: no "tiny", "little", "baby", "fluffy", "cute",
"adorable", and no baby-animal species names (duckling, chick, kitten,
puppy, cub, calf, foal, fawn) unless the original content used that word
itself. Image models have a strong bias toward cute baby-animal/3D-render
looks, and specifying the adult form in plain, neutral, technical language
is what actually breaks that bias -- not tone-appropriate wording. Name the
specific species where obvious (e.g. "mallard duck" not just "duck").

If media is "image" and behaviour is "cartoon" (or media is "audio"/
"video"), whimsical/warm language matching the teacher's tone is fine.

If media is "animation": the content describes a picture that ALREADY
exists and is about to be animated. Do not describe a new scene or add new
objects/props -- output a short, physically plausible motion description
for the existing subject only (e.g. gentle blinking, swaying, breathing,
a small wave), matching the teacher's tone. One short sentence.

Always write the rewritten text in the same language the original content is
written in (e.g. Sinhala in, Sinhala out) -- never translate it here. Any
translation needed for image/video generation models happens separately,
right before those calls, not in this rewrite step.

Output ONLY the rewritten text, no explanation, no quotes, no markdown."""

TRANSLATE_TEMPLATE = """Translate the following text to English so it can be used as an
image/video generation prompt. Output ONLY the English translation -- no
explanation, no quotes, no markdown.

Text: "{content}"

English translation:"""

_translate_prompt = ChatPromptTemplate.from_messages([("user", TRANSLATE_TEMPLATE)])

_SINHALA_RE = re.compile(r"[඀-෿]")  # Sinhala Unicode block

# Safety net for behaviour="natural": the system prompt tells the LLM to describe an
# adult animal in plain, non-diminutive language, but compliance isn't guaranteed on
# every call (confirmed by real testing -- e.g. "Duck" + a "playful" teacher tone still
# came back as "a bright yellow duckling, tiny wings... fluffy..." despite the explicit
# instruction). Rather than trust the first pass, detect this pattern and force one
# corrective rewrite before the text ever reaches an image model.
_BABY_CUE_RE = re.compile(
    r"\b(babies|baby|duckling|ducklings|chick|chicks|kitten|kittens|puppy|puppies|"
    r"cub|cubs|calf|calves|foal|foals|fawn|fawns|tiny|little|fluffy|cute|adorable)\b",
    re.IGNORECASE,
)

CORRECTION_TEMPLATE = """The following photorealistic image description uses baby-animal or cute/
diminutive language, which is not allowed for a "natural" (documentary-photo) style image.
Rewrite it to describe an ADULT of the species in plain, objective, technical photography
language -- no "tiny"/"little"/"fluffy"/"cute"/"adorable", and no baby-animal species names
(duckling, chick, kitten, puppy, cub, calf, foal, fawn). Keep the same setting, pose, and
lighting where reasonable. Output ONLY the corrected description, no explanation, no quotes.

Original description: "{text}"

Corrected description:"""

_correction_prompt = ChatPromptTemplate.from_messages([("user", CORRECTION_TEMPLATE)])


def _has_disallowed_baby_language(text, original_content):
    """True if text uses baby-animal/diminutive wording the original content never asked for."""
    if _BABY_CUE_RE.search(original_content or ""):
        return False  # the source content itself named a baby animal -- allowed
    return bool(_BABY_CUE_RE.search(text))

USER_TEMPLATE = """Teacher preference profile:
- Gender target: {gender_target}
- Tone: {tone}
- Teacher's comment: {comment}

Original content: "{content}"

Rewritten content:"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    ("user", USER_TEMPLATE),
])

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        endpoint = HuggingFaceEndpoint(
            repo_id=MODEL,
            huggingfacehub_api_token=HF_TOKEN,
            provider=MODEL_PROVIDER,
            max_new_tokens=200,
        )
        _llm = ChatHuggingFace(llm=endpoint)
    return _llm


def load_teacher_profile(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def is_sinhala(text):
    """True if text contains any Sinhala-script characters. Used to decide whether
    a content string needs translate_to_english() before it reaches an image/video
    model, and to skip that LLM call entirely for already-English content."""
    return bool(_SINHALA_RE.search(text or ""))


@lru_cache(maxsize=128)
def translate_to_english(text):
    """Translate content to English for image/video generation prompts -- those
    models (FLUX, Gemini, Veo) are trained almost entirely on English text and
    handle Sinhala prompts poorly or not at all. Audio generation never calls this:
    gTTS speaks Sinhala directly, so Sinhala text must reach it unchanged.
    Cached (by exact text) since the same content is often re-read across Streamlit
    reruns -- avoids redundant LLM calls for a value that won't change."""
    if not is_sinhala(text):
        return text
    chain = _translate_prompt | _get_llm()
    result = chain.invoke({"content": text})
    return result.content.strip()


def _safe_grade_band_context(grade):
    """grade_band_context() requires a real grade number -- learning_context's grade
    is "unspecified" (a string) whenever a caller doesn't pass one, e.g. generate_
    lesson_media.py's CLI path (see its --teacher-profile help text) can run without
    a lesson's grade wired through. Falls back to generic guidance rather than
    crashing personalize() over a missing optional field."""
    try:
        return grade_band_context(grade)
    except (TypeError, ValueError):
        return "No specific grade was given for this content -- use moderate, broadly age-appropriate vocabulary."


def personalize(content, activity_type, media_kind, behaviour, teacher_profile, learning_context=None):
    """Run the system+user prompt chain and return the personalized text.
    learning_context: optional dict with grade/subject/medium, normally lesson.get(...)
    values pulled straight from the lesson JSON (not the teacher profile -- these are
    facts about the lesson itself, distinct from the teacher's stylistic preferences).
    Grade-band vocabulary guidance is driven by the lesson's own grade here, not the
    teacher profile's "grade" preference -- the profile's grade is a UI default (see
    app.py's Teacher preference section and courseware_portal.py's Grade field
    default), while this rewrite needs to match the actual content being generated,
    which is always the lesson currently loaded, not a stale saved preference."""
    learning_context = learning_context or {}
    chain = _prompt | _get_llm()
    result = chain.invoke({
        "grade": learning_context.get("grade", "unspecified"),
        "subject": learning_context.get("subject", "unspecified"),
        "medium": learning_context.get("medium", "unspecified"),
        "grade_band_context": _safe_grade_band_context(learning_context.get("grade")),
        "activity_type": activity_type,
        "media_kind": media_kind,
        "behaviour": behaviour,
        "gender_target": teacher_profile.get("gender_target", "male and female students"),
        "tone": teacher_profile.get("tone", "neutral"),
        "comment": teacher_profile.get("comment", "none"),
        "content": content,
    })
    text = result.content.strip()

    if media_kind == "image" and behaviour == "natural" and _has_disallowed_baby_language(text, content):
        correction_chain = _correction_prompt | _get_llm()
        corrected = correction_chain.invoke({"text": text})
        text = corrected.content.strip()

    return text
