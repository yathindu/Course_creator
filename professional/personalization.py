"""
LangChain prompt-personalization layer: rewrites generation content to match
the lesson's context (skill_area/subject/medium) and target media/behaviour,
as a ChatPromptTemplate | ChatHuggingFace chain.

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
  - content-block narration audio (teaching text, not an answer)
generate_lesson_media.py enforces this by only calling personalize() for
those cases -- listening/speaking audio tasks never reach this module.

Setup:
    pip install langchain-core langchain-huggingface python-dotenv
    Put your token in a .env file next to this script:
        HF_TOKEN=your-pro-token-here
"""

import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from professional.config import MODEL, MODEL_PROVIDER

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

SYSTEM_TEMPLATE = """You help generate content for an adult workplace training course.

Learning context:
- Skill area: {skill_area}
- Subject: {subject}
- Medium (language): {medium}

Mode (activity type): {activity_type}
Target media: {media_kind}
Desired behaviour/style: {behaviour}

Task: rewrite the given content into a short, vivid description suited to
the mode, media, and behaviour above. Preserve the core subject exactly --
never substitute, translate, or drop it. Use clear, professional, practical
language suited to adult staff being trained for real work -- not academic
phrasing, not a childish or storybook tone.

If media is "image" and behaviour is "photographic": describe an objective,
realistic professional photograph ONLY -- physical setting, people, pose,
lighting, like a corporate/editorial photo caption. Do NOT use illustrative,
whimsical, or cartoon language: no "adorable", no exaggerated expressions,
no stock-photo cliches (forced grins, thumbs up, group high-fives).

If media is "image" and behaviour is "illustration" (or media is "audio"/
"video"), warmer, more encouraging language is fine, as long as it stays
professional.

If media is "animation": the content describes a picture that ALREADY
exists and is about to be animated. Do not describe a new scene or add new
objects/props -- output a short, physically plausible motion description
for the existing subject only (e.g. a hand reaching for an item, a nod, a
gesture). One short sentence.

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

USER_TEMPLATE = """Original content: "{content}"

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


def personalize(content, activity_type, media_kind, behaviour, learning_context=None):
    """Run the system+user prompt chain and return the personalized text.
    learning_context: optional dict with skill_area/subject/medium, normally lesson.get(...)
    values pulled straight from the lesson JSON."""
    learning_context = learning_context or {}
    chain = _prompt | _get_llm()
    result = chain.invoke({
        "skill_area": learning_context.get("skill_area", "unspecified"),
        "subject": learning_context.get("subject", "unspecified"),
        "medium": learning_context.get("medium", "unspecified"),
        "activity_type": activity_type,
        "media_kind": media_kind,
        "behaviour": behaviour,
        "content": content,
    })
    return result.content.strip()
