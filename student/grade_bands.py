"""
Sri Lankan school grade bands (Primary 1-5 / Junior Secondary 6-9 / Senior
Secondary G.C.E. O-Level 10-11 / Collegiate G.C.E. A-Level 12-13) and the
guidance derived from them.

Deliberately dependency-free -- no langchain/pymupdf/pypdf, stdlib only.
courseware_extraction.py's LLM prompts need this, and so does app.py's Media
Generation page (via personalization.py's Teacher preference UI). app.py
already carries the full torch/transformers/rembg weight for local media
generation; importing courseware_extraction.py's much heavier pipeline
(langchain_openai, pymupdf, pypdf) into app.py just to get these two pure
string-formatting functions is what caused a real Streamlit Cloud ImportError
in production (2026-08-21) -- almost certainly the same failure mode as the
SinhalaVITS incident documented in CLAUDE.md: too much import weight stacked
onto one already-heavy page. Keeping this module standalone/dependency-free is
the actual fix, not a workaround -- app.py and personalization.py must import
grade_band_context/suggested_teacher_preference from here, never from
courseware_extraction.py.
"""


def grade_band_context(grade):
    """One ready-to-splice guidance paragraph, keyed off which stage of the Sri
    Lankan school system `grade` falls into. Vocabulary, register, and assumed
    prior knowledge differ sharply across these stages, so every generation
    prompt that writes learner-facing text should use this instead of a bare
    "grade {grade}" number, which otherwise leaves the LLM to guess what that
    number implies."""
    grade = int(grade)
    if grade <= 5:
        stage, guidance = "Primary (grades 1-5)", (
            "Use very short, simple sentences and everyday vocabulary only. "
            "Explain any term a young child wouldn't already know. Favor "
            "concrete, familiar examples (home, family, animals, play) over "
            "abstract ones. Avoid subject-specific jargon unless the source "
            "text itself uses it, and define it in context if so."
        )
    elif grade <= 9:
        stage, guidance = "Junior Secondary (grades 6-9)", (
            "Use moderate sentence complexity. Subject-specific terms are fine "
            "but explain/introduce them the first time they're used rather than "
            "assuming prior knowledge. Balance concrete examples with simple "
            "abstract/conceptual reasoning."
        )
    elif grade <= 11:
        stage, guidance = "Senior Secondary / G.C.E. Ordinary Level (grades 10-11)", (
            "Use a formal academic register and exam-oriented phrasing, matching "
            "how real G.C.E. O-Level questions are worded. Subject terminology "
            "can be used freely without re-explaining basics, but keep "
            "explanations precise and unambiguous -- this is exam-prep content."
        )
    else:
        stage, guidance = "Collegiate / G.C.E. Advanced Level (grades 12-13)", (
            "Use an advanced, subject-specialist register appropriate to A-Level "
            "stream content (Science/Commerce/Arts/Technology). Assume strong "
            "foundational knowledge from earlier grades -- do not re-explain "
            "basic concepts. Use precise technical vocabulary and exam-board-"
            "style question phrasing, at a depth appropriate to pre-university "
            "study."
        )
    return f"Grade {grade} falls in Sri Lanka's {stage} stage. {guidance}"


def suggested_teacher_preference(grade):
    """A starting-point (tone, comment) pair for the Teacher preference fields in
    app.py, keyed off the same grade bands as grade_band_context() above -- these
    feed personalization.py's *style* rewrite (image/video mood, narration tone),
    a different concern from grade_band_context()'s *vocabulary complexity*
    guidance, so kept as a separate function rather than folded into it. Purely a
    suggested default a teacher can accept or overwrite via app.py's "Suggest for
    this grade" button -- never applied automatically, since a teacher's own typed
    preference should never be silently clobbered by a grade change."""
    grade = int(grade)
    if grade <= 5:
        return (
            "playful, warm, and encouraging",
            "Keep imagery bright and non-scary. Prefer simple, short sentences for "
            "anything read aloud.",
        )
    if grade <= 9:
        return (
            "friendly but a bit more grown-up -- encouraging without being childish",
            "Mild real-world detail and simple diagrams are fine. Avoid babyish "
            "imagery, but keep the overall feel upbeat and approachable.",
        )
    if grade <= 11:
        return (
            "clear, respectful, and exam-focused -- motivating without being "
            "patronizing",
            "Keep visuals realistic and straightforward, not cartoonish -- this age "
            "group tends to find childish styling off-putting. Favor clarity over "
            "decoration.",
        )
    return (
        "professional and direct -- treat the student as a young adult",
        "Avoid childish or cartoonish styling entirely. Imagery and narration "
        "should feel closer to how a textbook or lecture would present it.",
    )
