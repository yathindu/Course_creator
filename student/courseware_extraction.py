"""
Course Creation (formerly "Courseware Design Portal") -- LLM-driven pipeline that turns a raw book (PDF
or photographed/scanned page images) into lesson.json files matching this
project's existing schema exactly (see lessons-g2_maths_l1.json /
lessons-g2_english_l1.json). Nothing downstream changes: a finalized
lesson.json produced here is fed straight into generate_lesson_media.py /
app.py, unmodified -- the two projects connect only through that file
format, no shared code.

Pipeline (mirrors what was agreed with the stakeholder):
    1. extract_text()            -- PDF (pypdf) or page images (vision LLM read)
    2. generate_groupings()      -- chapters/modules; portal-internal
                                     organization only, never part of the
                                     exported lesson.json
    3. generate_lesson_skeletons() -- lesson title/description per grouping
    4. generate_lesson_activities() -- full activities for one lesson,
                                     matching the existing activity-type shapes
    5. assign_media_filenames()  -- deterministic (non-LLM) image/audio/
                                     video assignment
    6. paraphrase_lesson()       -- optional "Alter" pass: reword phrasing
                                     without changing subjects, correctness,
                                     or activity structure
    7. validate_lesson() (in courseware_schema.py) -- structural invariant
       check, run after step 4 AND after step 6, since an LLM rewrite can
       silently break correct_indices/correct_order/required fields.

All LLM calls go through OpenRouter (same OPENROUTER_API_KEY as
openrouter_pipeline.py) -- text generation uses this project's existing
orchestrator model (qwen/qwen3-235b-a22b-2507); image-page reading uses
google/gemini-2.5-flash, a vision-capable *chat* model -- distinct from
google/gemini-2.5-flash-image, which is the image-*generation* endpoint used
elsewhere in this project and cannot read/transcribe an input image.

Setup:
    pip install -r requirements-courseware.txt
    Put your key in .env next to this script (same .env as the rest of the
    project): OPENROUTER_API_KEY=sk-or-v1-...
"""

import base64
import os
import re
import time
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from openai import RateLimitError
from pypdf import PdfReader

from student.courseware_schema import (
    Activity,
    GeneratedLesson,
    Grouping,
    GroupingsResult,
    ImageOption,
    LessonSkeleton,
    LessonSkeletonsResult,
    prune_activity_fields,
    prune_lesson_fields,
)

load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

TEXT_MODEL = "qwen/qwen3-235b-a22b-2507"
VISION_MODEL = "google/gemini-2.5-flash"

LANGUAGE_MAP = {"sinhala": "si"}
DEFAULT_LANGUAGE = "en"

_slug_re = re.compile(r"[^a-z0-9]+")


def slugify(text):
    return _slug_re.sub("_", (text or "").lower()).strip("_") or "item"


def _normalized_medium(medium):
    return (medium or "").strip().lower()


def medium_language(medium):
    """Which language the LLM should *write* generated content in."""
    return "Sinhala" if _normalized_medium(medium) == "sinhala" else "English"


def resolve_language(medium):
    """Which gTTS language code audio for this medium should use -- case/whitespace-
    tolerant, same normalization as generate_lesson_media.py's resolve_language()
    (kept as a separate copy since this module has no import relationship with that
    one, but the two must stay in sync). A plain exact-match `medium == "sinhala"`
    silently fell back to English for anything not that exact lowercase string (e.g.
    "Sinhala", typed naturally into courseware_portal.py's Medium field) -- confirmed
    as a real bug via user report, not hypothetical."""
    return LANGUAGE_MAP.get(_normalized_medium(medium), DEFAULT_LANGUAGE)


def _llm(model=TEXT_MODEL, temperature=0.4, max_tokens=None):
    """max_tokens is a hard safety cap, not a target -- structured-output activity
    generation has been observed (real testing, Sinhala medium) to occasionally
    degenerate into a runaway response (thousands of near-blank lines) that never
    closes its JSON, which pydantic then rejects as invalid rather than truncated-
    but-usable. An explicit cap makes that failure fast and cheap instead of a
    multi-thousand-token generation that fails anyway -- callers with large
    expected output (activities, paraphrase) should set one.

    16000 (not the original 6000) for those large-output callers: real testing with
    a genuinely dense Sinhala-medium lesson (a photographed grade 5 exam paper) hit
    the OLD 6000 cap mid-generation -- Sinhala tokenizes far less efficiently than
    English, so a legitimate multi-activity Sinhala lesson can need well more than
    6000 tokens. That failure surfaced as a pydantic "EOF while parsing a string"
    error (valid-looking JSON truncated mid-string), not a rate-limit error, so
    _invoke_with_retry's retry loop didn't catch it -- confirmed via real user
    report, not hypothetical. 16000 still comfortably catches the original runaway-
    degenerate case (thousands of lines), just gives real content more headroom."""
    kwargs = dict(model=model, api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL, temperature=temperature)
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def _invoke_with_retry(fn, max_retries=4, base_delay=3):
    """OpenRouter's shared rate-limit pool can transiently 429 a model under load
    (confirmed via real user report: "qwen3-235b-a22b-2507 is temporarily
    rate-limited upstream" mid-generation) -- retrying after a short wait usually
    succeeds, so this shouldn't crash the whole Streamlit page the way an
    unhandled exception did. Exponential backoff; re-raises after exhausting
    retries so a genuinely persistent problem still surfaces to the caller
    instead of hanging silently. fn is a zero-arg callable (e.g.
    `lambda: chain.invoke(payload)`) so this works for both chain.invoke(dict)
    and llm.invoke([messages]) call shapes used across this module."""
    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


# ---------------------------------------------------------------------------
# Step 1: extraction
# ---------------------------------------------------------------------------
def _transcribe_image_bytes(img_bytes, ext):
    """Vision-LLM read of a photographed/scanned book page -- deliberately not
    OCR: a general vision-capable chat model handles skew/lighting/mixed
    text+diagram layouts better than a dedicated OCR library for this kind
    of source material. Shared by extract_text_from_image() (uploaded image
    files) and extract_text_from_pdf()'s scanned-page fallback (images
    extracted straight out of the PDF, no file on disk needed)."""
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    message = HumanMessage(content=[
        {"type": "text", "text": "Transcribe all readable text from this textbook page image, "
                                  "in reading order. Output only the transcribed text, no commentary. "
                                  "If the page has no readable text, output nothing."},
        {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
    ])
    result = _invoke_with_retry(lambda: _llm(VISION_MODEL).invoke([message]))
    return result.content.strip()


def extract_text_from_image(image_path):
    img_bytes = Path(image_path).read_bytes()
    ext = Path(image_path).suffix.lstrip(".").lower() or "png"
    return _transcribe_image_bytes(img_bytes, ext)


def extract_text_from_pdf(pdf_path, on_page=None):
    """Most PDFs have a real embedded text layer and pypdf's extract_text() just
    works. But a scanned book (very common for real textbooks -- confirmed on a
    real 78-page Sinhala grade-3 textbook, every single page had ZERO extractable
    text) has no text layer at all.

    A page with no text falls back to *rendering* that page with PyMuPDF and
    vision-transcribing the render -- deliberately not "extract this page's
    embedded image": on the real book above, every page's /Resources /XObject
    dictionary listed all 78 of the book's images identically (only one is
    actually painted per page, via that page's own content stream), so pulling
    "the" embedded image per page returned all 78 images on every single page --
    confirmed by real testing, would have been a ~78x duplicated, wildly wrong
    extraction. Rendering the page as it actually displays sidesteps that
    entirely, matching what a human (or poppler/pdftoppm, unavailable in this
    environment) would see.
    on_page(index, total): optional progress callback, called after each page
    finishes -- a scanned book means one real LLM call per page, which can take
    a while for a full book, so callers (courseware_portal.py) can show progress
    instead of one long silent spinner."""
    reader = PdfReader(pdf_path)
    total = len(reader.pages)
    mupdf_doc = None  # opened lazily -- only needed if some page has no text layer
    parts = []
    try:
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                if mupdf_doc is None:
                    mupdf_doc = pymupdf.open(pdf_path)
                pix = mupdf_doc[i].get_pixmap(dpi=150)
                text = _transcribe_image_bytes(pix.tobytes("png"), "png")
            parts.append(text)
            if on_page:
                on_page(i + 1, total)
    finally:
        if mupdf_doc is not None:
            mupdf_doc.close()
    return "\n\n".join(parts)


def extract_text(source_paths, on_page=None):
    """source_paths: list of file paths (PDF and/or images), in reading order.
    Returns the concatenated extracted text -- this is the raw material every
    later generation step reads from."""
    parts = []
    for path in source_paths:
        suffix = Path(path).suffix.lower()
        parts.append(extract_text_from_pdf(path, on_page=on_page) if suffix == ".pdf" else extract_text_from_image(path))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Step 2: groupings (chapters/modules) -- portal-internal organization only,
# never reaches the exported lesson.json.
# ---------------------------------------------------------------------------
GROUPINGS_SYSTEM = """You are organizing a textbook's content into teaching chapters/modules for a
grade {grade} {subject} course (medium: {medium}).

Read the source content and split it into logical groupings (chapters/modules) a teacher would
recognize -- group by topic, not by page boundaries. For each grouping give an id, a title, and a
description of exactly what content from the source it covers.
- Write the title and description in {medium_language} (match the source content's language).
- The id must always be plain ASCII: lowercase English letters, digits, and underscores only,
  regardless of what language the title/description are in.
Do not invent content that isn't in the source. Between all groupings together, cover the entire
source -- don't drop material."""

_groupings_prompt = ChatPromptTemplate.from_messages([
    ("system", GROUPINGS_SYSTEM),
    ("user", "Source content:\n\n{source_text}"),
])


def generate_groupings(source_text, grade, subject, medium):
    chain = _groupings_prompt | _llm().with_structured_output(GroupingsResult)
    result = _invoke_with_retry(lambda: chain.invoke({
        "source_text": source_text, "grade": grade, "subject": subject,
        "medium": medium, "medium_language": medium_language(medium),
    }))
    return result.groupings


# ---------------------------------------------------------------------------
# Step 3: lesson skeletons within one grouping
# ---------------------------------------------------------------------------
LESSON_SKELETONS_SYSTEM = """You are planning lessons for one chapter/module of a grade {grade}
{subject} course (medium: {medium}).

Chapter: {grouping_title}
Chapter content:
{grouping_text}

Split this chapter's content into one or more individual lessons -- each lesson should be
teachable in one sitting and focus on a coherent sub-topic. For each lesson give an id, a title,
and a one-paragraph description of what it teaches.
- Write the title and description in {medium_language} (match the source content's language).
- The id must always be plain ASCII (lowercase English letters, digits, underscores only),
  prefixed with "{grouping_id}_" -- regardless of what language the title/description are in.
Cover the whole chapter across the lessons you propose."""

_lesson_skeletons_prompt = ChatPromptTemplate.from_messages([("system", LESSON_SKELETONS_SYSTEM)])


def generate_lesson_skeletons(grouping: Grouping, grade, subject, medium):
    chain = _lesson_skeletons_prompt | _llm().with_structured_output(LessonSkeletonsResult)
    result = _invoke_with_retry(lambda: chain.invoke({
        "grouping_title": grouping.title,
        "grouping_text": grouping.description,
        "grouping_id": grouping.id,
        "grade": grade,
        "subject": subject,
        "medium": medium,
        "medium_language": medium_language(medium),
    }))
    return result.lessons


# ---------------------------------------------------------------------------
# Step 4: full activities for one lesson
# ---------------------------------------------------------------------------
# Single source of truth for each type's required fields -- used both in the
# full-lesson prompt below and in generate_single_activity()'s prompt, so the
# two can't drift out of sync on what a type needs.
ACTIVITY_TYPE_HINTS = {
    "vocabulary": "word, definition, sentence (one key term from the lesson, with an example "
                  "sentence that actually uses the word). All three are REQUIRED.",
    "mcq": "question, options (list of plain strings, 3-4 of them, all distinct), correct_indices "
           "(REQUIRED -- a non-empty list of indices into options, e.g. [0] or [0, 2]; every index "
           "must be within range 0..len(options)-1)",
    "image_selection": 'question, options (list of {label, image} -- leave "image" as an empty '
                        "string, it gets assigned later), correct_indices (REQUIRED -- a non-empty "
                        "list of indices into options, in range)",
    "true_false": "question, correct_answer (REQUIRED boolean -- true or false, never omitted)",
    "fill_blank": "question (containing a blank shown as ______ or __), options (plain strings, "
                  "one correct + distractors, all distinct), correct_indices (REQUIRED -- a "
                  "non-empty list of indices into options, in range)",
    "ordering": (
        "words: list the words/phrases in SCRAMBLED order -- do NOT list them already in the "
        "correct sequence, that defeats the exercise. correct_order: REQUIRED, never omit it -- "
        "a list the same length as words, where correct_order[i] is the index into words of "
        "whichever word belongs in position i of the correctly-ordered sentence; every index "
        "0..len(words)-1 must appear exactly once, no repeats, no gaps. Example: if words = "
        "[is, The, orange., carrot] and the correct sentence is The carrot is orange., then "
        "correct_order = [1, 3, 0, 2] -- position 0 is words[1] (The), position 1 is words[3] "
        "(carrot), position 2 is words[0] (is), position 3 is words[2] (orange.)."
    ),
    "listening": "question, options (plain strings), correct_indices (REQUIRED -- a non-empty "
                 "list of indices into options, in range) -- the *correct* option's text is what "
                 "gets read aloud to the learner, so make sure it is a short exact phrase to speak",
    "speaking": "prompt_text (a sentence for the learner to read aloud), acceptable_answers "
                "(REQUIRED -- a non-empty list of a few accepted variants/capitalization/"
                "punctuation of prompt_text)",
    "video": "no extra fields beyond title/instructions -- instructions should describe what the "
             "video shows, since that's what the video-generation step reads; the video file itself "
             "is assigned automatically, not authored by you",
}
# Escaped ({{ }}) for splicing directly into ACTIVITIES_SYSTEM's *template text* below --
# ChatPromptTemplate parses {..} as a substitution placeholder, and image_selection's hint
# contains a literal "{label, image}" that would otherwise crash .invoke() with a missing-key
# error. ACTIVITY_TYPE_HINTS itself stays unescaped, single-brace, for generate_single_activity()
# below, which passes a hint as an *input variable* (substituted, not re-parsed) instead.
_ACTIVITY_TYPE_HINTS_BLOCK = "\n".join(
    f"- {t}: {hint}".replace("{", "{{").replace("}", "}}") for t, hint in ACTIVITY_TYPE_HINTS.items()
)

ACTIVITIES_SYSTEM = """You are writing the activities for one lesson of a grade {grade} {subject}
course (medium: {medium}). Write all learner-facing text (title, instructions, question, word,
definition, sentence, options, prompt_text) in {medium_language}.

Lesson: {lesson_title}
Lesson description: {lesson_description}
Relevant source content (do not invent facts outside this):
{source_text}

Generate 6-14 activities covering this lesson, mixing activity types so the lesson isn't
repetitive. Available types and their required fields:
""" + _ACTIVITY_TYPE_HINTS_BLOCK + """

Give every activity an id prefixed with "{lesson_id}_a" plus a number (e.g. "{lesson_id}_a1") --
the id must always be plain ASCII (lowercase English letters, digits, underscores only), even
though the title/instructions/etc above are in {medium_language}. Also give a short title and
clear instructions. Match vocabulary and sentence complexity to grade {grade}.
Leave every image/audio/video field unset -- those are assigned separately, not by you."""
# A closing "every field marked REQUIRED must be filled in... double check before finishing"
# reinforcement paragraph was tried here and REMOVED -- real user-reported regression
# ("Generate activities" producing only 1 activity instead of 6-14). Confirmed via real LLM
# calls: the prompt with this paragraph produced just 1 activity on repeat calls; removing
# only this paragraph (keeping the REQUIRED/worked-example hints in ACTIVITY_TYPE_HINTS
# above, which are unaffected) produced 13-14 activities across 3 separate real trials.
# Genuine LLM run-to-run variance is real here (temperature=0.4, not 0) so a single sample
# either direction wouldn't be conclusive -- this was checked across multiple calls, not one.
# Don't re-add a "double check"/"before finishing" style instruction here without the same
# multi-trial check.

_activities_prompt = ChatPromptTemplate.from_messages([("system", ACTIVITIES_SYSTEM)])


def generate_lesson_activities(lesson: LessonSkeleton, source_text, grade, subject, medium):
    chain = _activities_prompt | _llm(max_tokens=16000).with_structured_output(GeneratedLesson)
    result = _invoke_with_retry(lambda: chain.invoke({
        "lesson_title": lesson.title,
        "lesson_description": lesson.description,
        "lesson_id": lesson.id,
        "source_text": source_text,
        "grade": grade,
        "subject": subject,
        "medium": medium,
        "medium_language": medium_language(medium),
    }))
    result.id = lesson.id
    result.title = lesson.title
    result.description = lesson.description
    result.grade = grade
    result.subject = subject
    result.medium = medium
    return prune_lesson_fields(result)


# ---------------------------------------------------------------------------
# Alternate step 4: verbatim transcription of a real paper's questions, instead of
# inventing new content inspired by source_text -- used by courseware_portal.py's
# "Course from an exam paper (verbatim)" mode, where a teacher has a real exam/worksheet
# and wants those exact questions turned into a course, not new AI-authored ones.
# ---------------------------------------------------------------------------
VERBATIM_ACTIVITIES_SYSTEM = """You are transcribing a real exam/worksheet paper into structured
activities for a grade {grade} {subject} course (medium: {medium}). All learner-facing text you
copy must stay in {medium_language} exactly as written in the source -- never translate it.

This is transcription, NOT content generation: every question you produce must be copied,
word-for-word, from the source text below. Do not invent a single new question, do not paraphrase,
do not reorder unrelated content into a question, and do not skip a question that's clearly
present. If the source text contains no legible questions at all, return zero activities rather
than inventing any.

Source text (the paper, exactly as extracted):
{source_text}

For each question in the source, in the order it appears, create one activity of the matching type:
- Multiple-choice question (a stem plus several lettered/numbered options) -> type "mcq".
  question = the stem itself, copied exactly. options = every option's exact text, one per list
  entry, in their original order.
- Fill-in-the-blank / short written-answer question with no options given in the source -> type
  "speaking". prompt_text = the sentence exactly as printed, including whatever blank marker the
  source uses (e.g. "...." or "______"), exactly as-is.
- True/false statement -> type "true_false". question = the exact statement, copied verbatim.
- An "arrange these words/phrases into the correct order" question -> type "ordering". words =
  the words/phrases from the source; correct_order = the grammatically correct sequence (this one
  you may work out yourself -- it's a matter of grammar, not subject-matter knowledge the source
  doesn't confirm).
- Anything else that doesn't cleanly fit one of the above (a definition/vocabulary prompt, an
  essay question, instructions-only text) -> skip it rather than forcing it into the wrong shape.

question/options/prompt_text/words are the entire point of this task -- get every one of them
filled in with the real copied text, never left blank. (Whatever you put in correct_indices/
correct_answer/acceptable_answers is discarded and overwritten afterward regardless of what you
write there, so don't spend effort on those -- put all of your attention on transcribing
question/options/prompt_text/words correctly instead.)

Worked example -- if the source contains:
    "3. What is 2 + 2?
    (a) 3   (b) 4   (c) 5   (d) 6"
the resulting activity must be: type "mcq", question "What is 2 + 2?", options
["3", "4", "5", "6"].

Give every activity an id prefixed with "{lesson_id}_a" plus a number (e.g. "{lesson_id}_a1") --
plain ASCII only, even though the copied text is in {medium_language}. Give each a short title
(e.g. "Question 3") and brief instructions (e.g. "Choose the correct answer" if the source implies
that). Leave every image/audio/video field unset."""

_verbatim_activities_prompt = ChatPromptTemplate.from_messages([("system", VERBATIM_ACTIVITIES_SYSTEM)])


def _clear_answer_keys(lesson: GeneratedLesson):
    """Enforced in code, not just prompted -- an LLM told "don't guess the answer" can still
    guess anyway. Exam/worksheet papers essentially never include an answer key (confirmed
    against a real photographed grade 5 exam paper), and a wrong guess would reach a teacher
    looking just as authoritative as a right one unless they independently re-solved every
    question themselves. This is the actual guarantee that answers are never fabricated; the
    prompt's instruction above is only the first line of defense, not the only one."""
    for activity in lesson.activities:
        if activity.type in ("mcq", "fill_blank", "listening", "image_selection"):
            activity.correct_indices = []
        elif activity.type == "true_false":
            activity.correct_answer = None
        elif activity.type == "speaking":
            activity.acceptable_answers = []
    return lesson


def generate_paper_lesson(source_text, grade, subject, medium, title):
    """Turns a real paper's questions directly into one lesson, preserving their exact wording --
    distinct from generate_lesson_activities() above (which invents fresh activities *inspired
    by* source_text rather than transcribing it). No grouping/lesson-skeleton steps needed first:
    one paper becomes one lesson directly.

    temperature=0, not this module's usual 0.4 default -- confirmed via real repeated testing
    that 0.4 made this specific call unreliable in a way generate_lesson_activities() (which
    also uses 0.4) isn't: across 3 real calls on the same real paper, one produced a
    correctly-filled lesson, one left every mcq's "question" field blank (options were fine --
    just the stem silently dropped), and one degenerated into the same "thousands of near-blank
    lines that never close the JSON" failure _llm()'s max_tokens docstring warns about.
    Content *generation* benefits from temperature's variety; this call has zero creative
    latitude at all (it's supposed to copy the source text exactly, not vary it), so there's no
    tradeoff to accept here -- temperature=0 is strictly better for a pure transcription task."""
    lesson_id = slugify(title)
    chain = _verbatim_activities_prompt | _llm(temperature=0, max_tokens=16000).with_structured_output(GeneratedLesson)
    result = _invoke_with_retry(lambda: chain.invoke({
        "source_text": source_text,
        "grade": grade,
        "subject": subject,
        "medium": medium,
        "medium_language": medium_language(medium),
        "lesson_id": lesson_id,
    }))
    result.id = lesson_id
    result.title = title
    result.description = f"Transcribed from an uploaded paper ({subject}, grade {grade})."
    result.grade = grade
    result.subject = subject
    result.medium = medium
    result = prune_lesson_fields(result)
    return _clear_answer_keys(result)


SINGLE_ACTIVITY_SYSTEM = """You are writing ONE new activity to add to an existing lesson of a grade
{grade} {subject} course (medium: {medium}). Write all learner-facing text (title, instructions,
question, word, definition, sentence, options, prompt_text) in {medium_language}.

Lesson: {lesson_title}
Lesson description: {lesson_description}
Relevant source content (do not invent facts outside this):
{source_text}

Generate exactly one activity of type "{activity_type}". Required fields for this type:
{type_hint}

Give it the id "{activity_id}" exactly, a short title, and clear instructions. Match vocabulary
and sentence complexity to grade {grade}. Leave every image/audio/video field unset --
those are assigned separately, not by you."""
# Same "double check before finishing" closing removed here as in ACTIVITIES_SYSTEM above --
# didn't reproduce the activity-count regression in a real test of this specific prompt, but
# it's the same instruction pattern that broke the other prompt, so it's not kept just because
# this one test happened not to show it.

_single_activity_prompt = ChatPromptTemplate.from_messages([("system", SINGLE_ACTIVITY_SYSTEM)])


def generate_single_activity(activity_type, activity_id, lesson_title, lesson_description, source_text, grade, subject, medium):
    """Used by courseware_portal.py's "Add a question" -- a teacher picks a type and gets one
    AI-drafted activity in that lesson's context, instead of a blank template to fill in by hand
    from scratch."""
    chain = _single_activity_prompt | _llm(max_tokens=1500).with_structured_output(Activity)
    result = _invoke_with_retry(lambda: chain.invoke({
        "activity_type": activity_type,
        "activity_id": activity_id,
        "lesson_title": lesson_title,
        "lesson_description": lesson_description,
        "source_text": source_text,
        "grade": grade,
        "subject": subject,
        "medium": medium,
        "medium_language": medium_language(medium),
        "type_hint": ACTIVITY_TYPE_HINTS[activity_type],
    }))
    result.id = activity_id
    result.type = activity_type
    return prune_activity_fields(result)


# ---------------------------------------------------------------------------
# Step 4b: teacher Q&A over the extracted source -- NotebookLM-style grounded
# chat. Deliberately full-context prompt-stuffing, not retrieval: the model's
# 262k-token context window comfortably fits a whole book's source_text
# (confirmed on a real 78-page scanned textbook), matching this module's
# existing pattern of never chunking source_text. No embeddings/vector store.
# ---------------------------------------------------------------------------
QA_SYSTEM = """You are answering a teacher's questions about a textbook they've uploaded, for a
grade {grade} {subject} course (medium: {medium}). Answer in {medium_language}, matching the
book's language.

Answer ONLY using the source content below -- do not use outside knowledge, and do not invent
facts, figures, or details that aren't in it. If the answer isn't in the source content, say so
plainly (e.g. "This isn't covered in the uploaded material") instead of guessing.

Keep answers concise -- a few sentences to a short paragraph, not an essay, unless the teacher's
question specifically asks for a list or detailed breakdown.

Source content:
{source_text}"""

_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM),
    MessagesPlaceholder("history"),
    ("user", "{question}"),
])

# Cost/latency guard only -- the 262k context window (see above) is nowhere near the
# limiting factor for one book's source_text, so this just caps how much history rides
# along on each turn. courseware_portal.py keeps the full transcript regardless.
MAX_QA_HISTORY_EXCHANGES = 6


def answer_question(question, source_text, history, grade, subject, medium):
    """history: list[tuple[str, str]] of (role, content), chronological -- kept as plain tuples
    rather than LangChain message objects so this module's session_state boundary holds
    (courseware_extraction.py never touches st.session_state; courseware_portal.py never
    touches LangChain types directly). Converted to Human/AIMessage here."""
    trimmed = history[-(MAX_QA_HISTORY_EXCHANGES * 2):]
    lc_history = [HumanMessage(content) if role == "user" else AIMessage(content) for role, content in trimmed]
    chain = _qa_prompt | _llm(temperature=0.2, max_tokens=700)
    result = _invoke_with_retry(lambda: chain.invoke({
        "question": question, "source_text": source_text, "history": lc_history,
        "grade": grade, "subject": subject, "medium": medium,
        "medium_language": medium_language(medium),
    }))
    return result.content.strip()


# ---------------------------------------------------------------------------
# Step 5: deterministic media filename assignment -- never left to the LLM,
# same "filename never invented by the scripts" discipline
# generate_lesson_media.py follows downstream of this step.
# ---------------------------------------------------------------------------
def _has_ascii_word_chars(text):
    return any(c.isascii() and c.isalnum() for c in (text or ""))


def _filename_slug(text, fallback):
    """slugify() strips every non-Latin character, so a Sinhala (or any non-ASCII
    script) word collapses to the same "item" fallback for every activity --
    confirmed by real testing, would silently overwrite every vocabulary
    image/audio in a Sinhala-medium lesson onto one shared filename. Fall back to
    a caller-supplied ASCII id (already enforced ASCII by the generation prompts)
    instead of the word itself whenever the word has no Latin/digit characters."""
    return slugify(text) if _has_ascii_word_chars(text) else slugify(fallback)


def assign_media_filenames(lesson: GeneratedLesson, language):
    for activity in lesson.activities:
        base = _filename_slug(activity.word, activity.id)
        if activity.type == "vocabulary":
            activity.image = f"images/{base}.png"
            activity.audio = f"audios/{base}_{language}.mp3"
        elif activity.type in ("image_selection", "mcq") and activity.options:
            for oi, option in enumerate(activity.options):
                if isinstance(option, ImageOption):
                    option.image = f"images/{_filename_slug(option.label, f'{activity.id}_opt{oi}')}.png"
        elif activity.type == "listening":
            activity.audio = f"audios/{activity.id}_listen.mp3"
        elif activity.type == "speaking":
            activity.audio = f"audios/{activity.id}_speak.mp3"
        elif activity.type == "video":
            activity.video = f"videos/{base}.mp4"
    return lesson


# ---------------------------------------------------------------------------
# Step 6: optional "Alter" paraphrase pass -- copyright/CW protection.
# Reword phrasing/instructions/order-of-presentation only. Deliberately does
# NOT touch: type, id, option counts, correct_indices/correct_answer/
# correct_order, or media filenames -- those must stay valid for the
# reworded content, and remapping them reliably via LLM was judged too
# fragile for a v1 (validate_lesson() catches it if a rewrite breaks this
# anyway, and the caller falls back to the pre-paraphrase activity).
# ---------------------------------------------------------------------------
PARAPHRASE_SYSTEM = """You are rewriting lesson content to avoid reproducing a source textbook's
exact wording, while still covering exactly the same material and staying factually identical.

Rules, strict:
- Reword sentence structure, instructions, and phrasing -- do not copy the original wording.
- Do NOT change: activity "type", "id", the number of items in "options"/"words", which
  option(s) are correct ("correct_indices"/"correct_answer"/"correct_order" must still be
  correct after your rewrite), or any image/audio/video field.
- For an "ordering" activity specifically: do NOT reword the individual "words" entries at all --
  only reword its title/instructions. Rewording the words themselves risks silently invalidating
  correct_order (which indexes into the exact original words), and the whole point of the
  exercise is reassembling those exact tokens.
- Do NOT change the core subject/vocabulary word of a "vocabulary" activity (its media depends
  on that exact word) -- reword its definition/sentence only.
- Keep the same language as the input.

Lesson to rewrite (JSON):
{lesson_json}

Output the rewritten lesson in the same structure."""

_paraphrase_prompt = ChatPromptTemplate.from_messages([("system", PARAPHRASE_SYSTEM)])


def paraphrase_lesson(lesson: GeneratedLesson):
    chain = _paraphrase_prompt | _llm(temperature=0.6, max_tokens=16000).with_structured_output(GeneratedLesson)
    result = _invoke_with_retry(lambda: chain.invoke({"lesson_json": lesson.model_dump_json(indent=2)}))
    result.id = lesson.id
    if len(result.activities) != len(lesson.activities):
        # The rewrite was instructed not to change activity count, but nothing enforces
        # that structurally -- if it drifted anyway, zipping by position would silently
        # pair the wrong original/rewritten activities and misassign media filenames.
        # Treat this the same as any other structural break: caller's validate_lesson()
        # call still runs, but raise here so the caller's except/fallback path (keep the
        # pre-paraphrase lesson) is guaranteed to trigger instead of shipping a
        # mismatched result that might coincidentally still "validate" on its own.
        raise ValueError(
            f"Alter pass changed the activity count ({len(lesson.activities)} -> "
            f"{len(result.activities)}) -- discarding rewrite."
        )
    for original, rewritten in zip(lesson.activities, result.activities):
        rewritten.image = original.image
        rewritten.audio = original.audio
        rewritten.video = original.video
        rewritten.group = original.group
        if rewritten.type in ("image_selection", "mcq") and rewritten.options and original.options:
            for new_opt, old_opt in zip(rewritten.options, original.options):
                if isinstance(new_opt, ImageOption) and isinstance(old_opt, ImageOption):
                    new_opt.image = old_opt.image
    return prune_lesson_fields(result)


def lesson_to_json_dict(lesson: GeneratedLesson):
    """Serialize to the exact plain-dict shape generate_lesson_media.py/app.py expect --
    drops None fields and unwraps ImageOption back into plain {"label","image"} dicts,
    matching lessons-g2_maths_l1.json's shape exactly."""
    return lesson.model_dump(exclude_none=True, mode="json")
