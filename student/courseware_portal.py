"""
Course Creation (formerly "Courseware Design Portal") -- upstream of the main app (app.py). Turns a raw
book (PDF or photographed/scanned page images) into lesson.json files, via:

    upload -> extract -> groupings (chapters/modules) -> lesson skeletons
    -> full activities -> optional "Alter" paraphrase pass -> teacher review
    & correction -> save

A saved lesson here is the exact same schema app.py already reads (see
lessons-g2_maths_l1.json / lessons-g2_english_l1.json) -- the two
apps connect only through that file, no shared code, no shared session
state. Chapter/module grouping is portal-internal organization only; it
never reaches the exported lesson.json.

Run: streamlit run courseware_portal.py --server.port 8503
Needs its own dependency set (requirements-courseware.txt) -- deliberately
does NOT depend on torch/transformers/CLIPSeg/rembg, none of which this
portal uses (no local media generation happens here).
"""

import json
import tempfile
from pathlib import Path

import streamlit as st

# Streamlit Cloud bridges secrets into os.environ the same way app.py does,
# so this portal can be deployed as its own Streamlit Cloud app pointed at
# a different main file in the same repo.
try:
    import os
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass

from student.courseware_extraction import (
    answer_question,
    assign_media_filenames,
    extract_text,
    generate_groupings,
    generate_lesson_activities,
    generate_lesson_skeletons,
    generate_single_activity,
    lesson_to_json_dict,
    paraphrase_lesson,
    resolve_language,
)
from student.courseware_schema import GeneratedLesson, prune_lesson_fields, validate_lesson

# Anchored to this file's own package dir, not cwd -- the unified app launches from
# unified_courses/ so both student/ and professional/ packages can coexist, which
# would otherwise resolve a bare relative path to the wrong (shared) location.
OUTPUT_DIR = Path(__file__).resolve().parent / "courseware_output"


def _generation_error_message(action, e):
    """Not every failure here is transient. A missing/misconfigured API key
    (e.g. OPENROUTER_API_KEY not set in this deployment's Secrets) raises at
    client-construction time, before any request is even sent -- telling a
    teacher to "just try again" for that is actively wrong, since retrying
    can never succeed until the key is actually fixed. Confirmed via a real
    deployment hitting exactly this (a fresh Streamlit Cloud app where the
    secret was never set)."""
    text = str(e).lower()
    if "credentials" in text or "api_key" in text or "api key" in text:
        return (f"{action} failed: {e}\n\nThis looks like a missing/misconfigured API key, not a transient "
                "error -- retrying won't help. If this is deployed on Streamlit Cloud, check the app's "
                "Settings -> Secrets for OPENROUTER_API_KEY. If running locally, check your .env file.")
    return (f"{action} failed: {e}\n\nThis is often transient (an upstream model was briefly overloaded) "
            "-- try the button again.")

# Autosave for in-progress work (extracted text, groupings, lessons, generated
# activities) -- session_state lives only in memory, so a server restart/crash
# (confirmed by real user report: lost all progress after restarting to pick up
# a code fix) previously wiped everything with no way to recover. Same discipline
# app.py already uses for teacher preference/content drafts. Deliberately a
# single file, not one per lesson: the portal authors one course at a time.
AUTOSAVE_PATH = OUTPUT_DIR / "_portal_progress.json"
AUTOSAVE_KEYS = ["source_text", "course_meta", "groupings", "lessons_by_group", "generated_lessons", "qa_messages"]


def _load_autosave():
    if not AUTOSAVE_PATH.exists():
        return
    data = json.loads(AUTOSAVE_PATH.read_text(encoding="utf-8"))
    for key in AUTOSAVE_KEYS:
        if key in data:
            st.session_state[key] = data[key]


def _save_autosave():
    OUTPUT_DIR.mkdir(exist_ok=True)
    snapshot = {key: st.session_state[key] for key in AUTOSAVE_KEYS}
    snapshot_json = json.dumps(snapshot, ensure_ascii=False)
    if snapshot_json != st.session_state.get("_last_autosave_snapshot"):
        AUTOSAVE_PATH.write_text(snapshot_json, encoding="utf-8")
        st.session_state["_last_autosave_snapshot"] = snapshot_json


try:
    st.set_page_config(page_title="Course Creation", page_icon="📘", layout="wide")
except st.errors.StreamlitAPIException:
    pass  # already set by main.py -- this page is running under its st.navigation()
st.title("📘 Course Creation")
st.caption("Book → chapters → lessons → activities → teacher review → lesson.json for the media pipeline")

for key, default in [
    ("source_text", None),
    ("groupings", []),          # list[dict]
    ("lessons_by_group", {}),   # group_id -> list[dict]
    ("generated_lessons", {}),  # lesson_id -> dict (post activity-generation)
    ("editing_lesson_id", None),
    ("course_meta", {}),
    ("qa_messages", []),        # list[dict] -- "Ask about this book" chat transcript
]:
    if key not in st.session_state:
        st.session_state[key] = default

if "_autosave_loaded" not in st.session_state:
    _load_autosave()
    st.session_state["_autosave_loaded"] = True
    st.session_state["_show_resume_banner"] = bool(st.session_state["source_text"])

# Deliberately NOT nested inside the one-time load block above: st.button() only
# reports True on the specific rerun immediately after it's clicked, which means
# the widget call itself must happen on *every* rerun, not just the first one.
# Gating the button behind "_autosave_loaded not in session_state" (true only on
# run 1) meant the click could never be detected -- confirmed by real testing,
# not a hypothetical: the button visibly did nothing, and the autosave file was
# never actually deleted on click.
if st.session_state.get("_show_resume_banner"):
    rcol1, rcol2 = st.columns([4, 1])
    rcol1.info("Resumed previously saved progress.")
    if rcol2.button("Clear & start over"):
        for key in AUTOSAVE_KEYS:
            st.session_state[key] = {} if isinstance(st.session_state[key], dict) else (
                [] if isinstance(st.session_state[key], list) else None)
        st.session_state["editing_lesson_id"] = None
        st.session_state["_show_resume_banner"] = False
        AUTOSAVE_PATH.unlink(missing_ok=True)
        st.session_state.pop("_last_autosave_snapshot", None)
        st.rerun()


# ---------------------------------------------------------------------------
# 1. Source upload + course metadata, alongside a "notebook-style" chat pane
# ---------------------------------------------------------------------------
# Two columns -- source material on the left, grounded Q&A chat on the right --
# rather than one long stacked section, per explicit user request for a more
# NotebookLM-like feel. The source column keeps its existing expander (collapses
# once there's already extracted text, so it doesn't keep taking up space above
# whatever the teacher is actually working on) while the chat column stays
# visible, since chat is the section's main working surface once a book exists.
col_source, col_chat = st.columns([2, 3])

with col_source:
    with st.expander("1. Source material", expanded=not st.session_state["source_text"]):
        col1, col2, col3 = st.columns(3)
        grade = col1.number_input("Grade", min_value=1, max_value=13, value=2)
        subject = col2.text_input("Subject", value="english")
        # Constrained to exactly what the pipeline supports (gTTS "en"/"si", per
        # generate_lesson_media.py's LANGUAGE_MAP) -- a free-text field let a
        # naturally-capitalized "Sinhala" silently fall back to English everywhere
        # downstream (exact-match string comparisons), a real bug found via user report.
        medium = col3.selectbox("Medium (language)", ["english", "sinhala"])

        uploads = st.file_uploader(
            "Book pages -- PDF and/or images", type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )

        if st.button("Extract content", disabled=not uploads):
            tmp_dir = Path(tempfile.mkdtemp())
            paths = []
            for f in uploads:
                p = tmp_dir / f.name
                p.write_bytes(f.getvalue())
                paths.append(str(p))
            # A scanned PDF (no embedded text layer -- common for real textbooks) needs
            # one real vision-LLM call per page, which can take a while for a full book.
            # Show real per-page progress instead of one long silent spinner.
            progress_bar = st.progress(0.0, text="Reading source material...")

            def _on_page(done, total):
                progress_bar.progress(done / total, text=f"Reading page {done}/{total}...")

            try:
                st.session_state["source_text"] = extract_text(paths, on_page=_on_page)
                st.session_state["course_meta"] = {"grade": int(grade), "subject": subject, "medium": medium}
                st.session_state["groupings"] = []
                st.session_state["lessons_by_group"] = {}
                st.session_state["generated_lessons"] = {}
                st.session_state["qa_messages"] = []  # new source -- fresh conversation
                st.success(f"Extracted {len(st.session_state['source_text'])} characters.")
            except Exception as e:
                st.error(_generation_error_message("Extraction", e))
            finally:
                progress_bar.empty()

        if st.session_state["source_text"]:
            with st.expander("Extracted text (preview)"):
                st.text_area("", st.session_state["source_text"], height=200, disabled=True, label_visibility="collapsed")

with col_chat:
    st.markdown("**💬 Ask about this book**")
    if not st.session_state["source_text"]:
        st.caption("Upload and extract a book on the left to start asking questions.")
    else:
        with st.container(border=True):
            history_box = st.container(height=380)
            with history_box:
                if not st.session_state["qa_messages"]:
                    st.caption("Ask anything about the uploaded book -- answers are grounded "
                               "only in its extracted text.")
                for msg in st.session_state["qa_messages"]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            question = st.chat_input("Ask a question about this book...")
            if question:
                st.session_state["qa_messages"].append({"role": "user", "content": question})
                meta = st.session_state["course_meta"]
                history_pairs = [(m["role"], m["content"]) for m in st.session_state["qa_messages"][:-1]]
                try:
                    answer = answer_question(
                        question, st.session_state["source_text"], history_pairs,
                        meta["grade"], meta["subject"], meta["medium"],
                    )
                    st.session_state["qa_messages"].append({"role": "assistant", "content": answer})
                    st.rerun()
                except Exception as e:
                    st.error(_generation_error_message("Answering", e))


# ---------------------------------------------------------------------------
# 2. Groupings (chapters/modules)
# ---------------------------------------------------------------------------
if st.session_state["source_text"]:
    # Same reasoning as Section 1's expander: collapses once a lesson is actively
    # being reviewed (Section 3 below), instead of the whole chapters/lessons tree
    # staying expanded above whatever the teacher is currently editing.
    with st.expander("2. Chapters / modules", expanded=not st.session_state.get("editing_lesson_id")):
        meta = st.session_state["course_meta"]

        if st.button("Generate groupings"):
            with st.spinner("Analyzing structure..."):
                try:
                    groupings = generate_groupings(st.session_state["source_text"], meta["grade"], meta["subject"], meta["medium"])
                    st.session_state["groupings"] = [g.model_dump() for g in groupings]
                    st.rerun()
                except Exception as e:
                    st.error(_generation_error_message("Generation", e))

        for gi, grouping in enumerate(st.session_state["groupings"]):
            with st.expander(f"📂 {grouping['title']}", expanded=False):
                grouping["title"] = st.text_input("Title", grouping["title"], key=f"g_title_{gi}")
                grouping["description"] = st.text_area("Covers", grouping["description"], key=f"g_desc_{gi}")

                gcol1, gcol2 = st.columns([1, 1])
                if gcol1.button("Generate lessons for this chapter", key=f"g_gen_{gi}"):
                    with st.spinner("Planning lessons..."):
                        try:
                            from student.courseware_schema import Grouping as GroupingModel
                            lessons = generate_lesson_skeletons(GroupingModel(**grouping), meta["grade"], meta["subject"], meta["medium"])
                            st.session_state["lessons_by_group"][grouping["id"]] = [l.model_dump() for l in lessons]
                            st.rerun()
                        except Exception as e:
                            st.error(_generation_error_message("Generation", e))
                if gcol2.button("Delete chapter", key=f"g_del_{gi}"):
                    st.session_state["groupings"].pop(gi)
                    st.session_state["lessons_by_group"].pop(grouping["id"], None)
                    st.rerun()

                lessons = st.session_state["lessons_by_group"].get(grouping["id"], [])
                for li, lesson in enumerate(lessons):
                    st.divider()
                    lesson["title"] = st.text_input("Lesson title", lesson["title"], key=f"l_title_{gi}_{li}")
                    lesson["description"] = st.text_area("Lesson covers", lesson["description"], key=f"l_desc_{gi}_{li}")
                    is_generated = lesson["id"] in st.session_state["generated_lessons"]
                    lcol1, lcol2, lcol3 = st.columns([1, 1, 1])
                    if lcol1.button("Generate activities" if not is_generated else "Regenerate activities", key=f"l_gen_{gi}_{li}"):
                        with st.spinner("Writing activities..."):
                            try:
                                from student.courseware_schema import LessonSkeleton
                                generated = generate_lesson_activities(
                                    LessonSkeleton(**lesson), st.session_state["source_text"],
                                    meta["grade"], meta["subject"], meta["medium"],
                                )
                                language = resolve_language(meta["medium"])
                                generated = assign_media_filenames(generated, language)
                                issues = validate_lesson(generated)
                                st.session_state["generated_lessons"][lesson["id"]] = {
                                    "lesson": lesson_to_json_dict(generated), "issues": issues, "altered": False,
                                }
                                st.rerun()
                            except Exception as e:
                                st.error(_generation_error_message("Generation", e))
                    if is_generated and lcol2.button("Review & edit", key=f"l_review_{gi}_{li}"):
                        st.session_state["editing_lesson_id"] = lesson["id"]
                        st.rerun()
                    if lcol3.button("Delete lesson", key=f"l_del_{gi}_{li}"):
                        lessons.pop(li)
                        st.session_state["generated_lessons"].pop(lesson["id"], None)
                        st.rerun()


# ---------------------------------------------------------------------------
# 3. Review & edit a generated lesson
# ---------------------------------------------------------------------------
ACTIVITY_TYPES = ["vocabulary", "mcq", "image_selection", "true_false", "fill_blank", "ordering", "listening", "speaking", "video"]


def _next_activity_id(lesson_id, activities):
    existing_ids = {a["id"] for a in activities}
    n = len(activities) + 1
    while f"{lesson_id}_a{n}" in existing_ids:
        n += 1
    return f"{lesson_id}_a{n}"


def _blank_activity(activity_type, lesson_id, activities):
    """A hand-authored starting point for a teacher-composed question -- not
    LLM-generated at all. Every field _activity_editor() renders for this type
    gets a sensible empty placeholder (2-3 blank options, not zero, so the
    teacher immediately has input boxes to fill rather than an empty list with
    no obvious way to add options)."""
    activity = {
        "id": _next_activity_id(lesson_id, activities),
        "type": activity_type,
        "title": "",
        "instructions": "",
    }
    if activity_type == "vocabulary":
        activity.update(word="", definition="", sentence="")
    elif activity_type in ("mcq", "fill_blank", "listening"):
        activity.update(question="", options=["", "", ""], correct_indices=[])
    elif activity_type == "image_selection":
        activity.update(question="", options=[{"label": "", "image": ""}, {"label": "", "image": ""}], correct_indices=[])
    elif activity_type == "true_false":
        activity.update(question="", correct_answer=True)
    elif activity_type == "ordering":
        activity.update(words=["", "", ""], correct_order=[0, 1, 2])
    elif activity_type == "speaking":
        activity.update(prompt_text="", acceptable_answers=[""])
    return activity


def _activity_editor(activity, key_prefix):
    activity["title"] = st.text_input("Title", activity.get("title", ""), key=f"{key_prefix}_title")
    activity["instructions"] = st.text_input("Instructions", activity.get("instructions", ""), key=f"{key_prefix}_instr")
    activity["group"] = st.text_input(
        "Group (optional -- clusters related activities together in the media app)",
        activity.get("group") or "", key=f"{key_prefix}_group",
    ) or None
    t = activity["type"]

    if t == "vocabulary":
        activity["word"] = st.text_input("Word", activity.get("word", ""), key=f"{key_prefix}_word")
        activity["definition"] = st.text_area("Definition", activity.get("definition", ""), key=f"{key_prefix}_def")
        activity["sentence"] = st.text_input("Example sentence", activity.get("sentence", ""), key=f"{key_prefix}_sent")
        st.caption(f"Media: {activity.get('image', '—')} / {activity.get('audio', '—')}")

    elif t in ("mcq", "fill_blank", "listening"):
        activity["question"] = st.text_area("Question", activity.get("question", ""), key=f"{key_prefix}_q")
        options = activity.get("options", [])
        new_options = []
        remove_idx = None
        for oi, opt in enumerate(options):
            ocol1, ocol2 = st.columns([6, 1])
            new_options.append(ocol1.text_input(f"Option {oi + 1}", opt, key=f"{key_prefix}_opt_{oi}"))
            if ocol2.button("✕", key=f"{key_prefix}_optdel_{oi}", help="Remove this option"):
                remove_idx = oi
        activity["options"] = new_options
        current_correct = [i for i in activity.get("correct_indices", []) if i < len(new_options)]
        if remove_idx is not None:
            activity["options"].pop(remove_idx)
            activity["correct_indices"] = [i if i < remove_idx else i - 1
                                            for i in current_correct if i != remove_idx]
            st.rerun()
        if st.button("+ Add option", key=f"{key_prefix}_optadd"):
            activity["options"].append("")
            st.rerun()
        # Keyed on option count (not just key_prefix) so this widget remounts fresh with a
        # correct `default=` whenever an option is added/removed -- otherwise Streamlit
        # matches its persisted selection by index value, which silently points at a
        # *different* option once the list has shifted (confirmed as a real risk, not
        # hypothetical, given correct_indices is itself a list of positions into options).
        correct = st.multiselect(
            "Correct option(s)", options=list(range(len(new_options))),
            default=current_correct,
            format_func=lambda i: new_options[i] if i < len(new_options) else str(i),
            key=f"{key_prefix}_correct_{len(new_options)}",
        )
        activity["correct_indices"] = correct

    elif t == "image_selection":
        activity["question"] = st.text_area("Question", activity.get("question", ""), key=f"{key_prefix}_q")
        options = activity.get("options", [])
        labels = []
        remove_idx = None
        for oi, opt in enumerate(options):
            ocol1, ocol2 = st.columns([6, 1])
            opt["label"] = ocol1.text_input(f"Option {oi + 1} label", opt.get("label", ""), key=f"{key_prefix}_optlbl_{oi}")
            ocol1.caption(f"Image: {opt.get('image', '—')}")
            if ocol2.button("✕", key=f"{key_prefix}_optdel_{oi}", help="Remove this option"):
                remove_idx = oi
            labels.append(opt["label"])
        current_correct = [i for i in activity.get("correct_indices", []) if i < len(labels)]
        if remove_idx is not None:
            activity["options"].pop(remove_idx)
            activity["correct_indices"] = [i if i < remove_idx else i - 1
                                            for i in current_correct if i != remove_idx]
            st.rerun()
        if st.button("+ Add option", key=f"{key_prefix}_optadd"):
            activity["options"].append({"label": "", "image": ""})
            st.rerun()
        correct = st.multiselect(
            "Correct option(s)", options=list(range(len(labels))),
            default=current_correct,
            format_func=lambda i: labels[i] if i < len(labels) else str(i),
            key=f"{key_prefix}_correct_{len(labels)}",
        )
        activity["correct_indices"] = correct

    elif t == "true_false":
        activity["question"] = st.text_area("Statement", activity.get("question", ""), key=f"{key_prefix}_q")
        activity["correct_answer"] = st.checkbox("Statement is true", activity.get("correct_answer", True), key=f"{key_prefix}_tf")

    elif t == "ordering":
        words = activity.get("words", [])
        new_words = [st.text_input(f"Word/phrase {wi + 1}", w, key=f"{key_prefix}_word_{wi}") for wi, w in enumerate(words)]
        activity["words"] = new_words

        # correct_order used to be a read-only caption -- there was no way to actually
        # fix a broken one (e.g. validate_lesson() flagging "not a valid permutation"),
        # confirmed as a real gap via user report: the review screen is supposed to be
        # the correction mechanism for exactly this kind of LLM miss, but the field
        # wasn't editable at all. Fixed with a 1-based "position in the correct
        # sentence" number per word instead of asking the teacher to hand-edit a raw
        # index list -- derived from the existing correct_order so an already-valid
        # one round-trips unchanged, and falls back to natural order if it's missing
        # or broken (giving a sane starting point to fix from).
        n = len(new_words)
        current_order = activity.get("correct_order") or []
        positions = [None] * n
        if sorted(current_order) == list(range(n)):
            for pos, word_idx in enumerate(current_order):
                positions[word_idx] = pos + 1
        else:
            positions = list(range(1, n + 1))

        st.caption("Set each word's position in the correctly-ordered sentence (1 = first). "
                   "Each number 1.." + str(n) + " must be used exactly once.")
        new_positions = []
        for wi, w in enumerate(new_words):
            p = st.number_input(f'Position of "{w}"', min_value=1, max_value=max(n, 1),
                                 value=positions[wi] or wi + 1, step=1, key=f"{key_prefix}_pos_{wi}")
            new_positions.append(int(p))

        if sorted(new_positions) == list(range(1, n + 1)):
            activity["correct_order"] = [new_positions.index(p) for p in range(1, n + 1)]
            preview = " ".join(new_words[i] for i in activity["correct_order"])
            st.caption(f'✅ Valid -- preview: "{preview}"')
        else:
            st.warning(f"Positions must use each number 1..{n} exactly once -- not a valid order yet.")

    elif t == "speaking":
        activity["prompt_text"] = st.text_input("Sentence to read aloud", activity.get("prompt_text", ""), key=f"{key_prefix}_prompt")
        # Real report: generation sometimes leaves acceptable_answers empty (validate_lesson()
        # flags "missing required field") -- seed the box with the prompt sentence itself
        # instead of leaving it totally blank, since prompt_text is always itself a valid
        # accepted answer. Same "give a sane starting point, not a blank/broken state"
        # philosophy as the ordering position editor above.
        existing_answers = activity.get("acceptable_answers") or []
        default_answers = existing_answers or ([activity["prompt_text"]] if activity["prompt_text"] else [])
        answers_text = st.text_area(
            "Acceptable answers (one per line)",
            "\n".join(default_answers), key=f"{key_prefix}_answers",
        )
        activity["acceptable_answers"] = [a for a in answers_text.split("\n") if a.strip()]

    elif t == "video":
        st.caption(f"Video: {activity.get('video', '—')}")


editing_id = st.session_state["editing_lesson_id"]
if editing_id and editing_id in st.session_state["generated_lessons"]:
    entry = st.session_state["generated_lessons"][editing_id]
    lesson = entry["lesson"]

    with st.container(border=True):
        st.subheader(f"3. Review: {lesson['title']}")

        if entry["issues"]:
            st.warning("Structural issues found -- fix before saving:\n\n" + "\n".join(f"- {i}" for i in entry["issues"]))
        else:
            st.success("Structurally valid.")

        alter = st.checkbox("Alter (paraphrase to avoid reproducing source wording)", value=entry["altered"], key=f"alter_{editing_id}")
        if alter and not entry["altered"]:
            if st.button("Apply Alter pass"):
                with st.spinner("Rewriting..."):
                    try:
                        rewritten = paraphrase_lesson(GeneratedLesson(**lesson))
                        rewritten_dict = lesson_to_json_dict(rewritten)
                        new_issues = validate_lesson(rewritten)
                        if new_issues:
                            st.error("Alter pass broke structural validity -- kept original wording instead:\n\n"
                                      + "\n".join(f"- {i}" for i in new_issues))
                        else:
                            entry["lesson"] = rewritten_dict
                            entry["altered"] = True
                            entry["issues"] = []
                            st.rerun()
                    except Exception as e:
                        st.error(_generation_error_message("Alter pass", e))

        st.divider()
        for ai, activity in enumerate(lesson["activities"]):
            with st.expander(f"{ai + 1}. [{activity['type']}] {activity.get('title', activity['id'])}"):
                _activity_editor(activity, f"act_{editing_id}_{ai}")
                if st.button("Delete activity", key=f"act_del_{editing_id}_{ai}"):
                    lesson["activities"].pop(ai)
                    st.rerun()

        st.divider()
        st.subheader("➕ Add a question")
        st.caption("Get an AI-drafted starting point in this lesson's context, or add a blank one to "
                   "compose entirely by hand.")
        new_col1, new_col2, new_col3 = st.columns([3, 1, 1])
        new_type = new_col1.selectbox("Type", ACTIVITY_TYPES, key=f"new_act_type_{editing_id}")

        def _add_activity(new_activity):
            lesson["activities"].append(new_activity)
            language = resolve_language(st.session_state["course_meta"].get("medium"))
            generated = assign_media_filenames(GeneratedLesson(**lesson), language)
            entry["lesson"] = lesson_to_json_dict(generated)
            st.rerun()

        if new_col2.button("✨ AI draft", key=f"add_act_ai_{editing_id}"):
            activity_id = _next_activity_id(lesson["id"], lesson["activities"])
            with st.spinner("Drafting a suggestion..."):
                try:
                    drafted = generate_single_activity(
                        new_type, activity_id, lesson["title"], lesson["description"],
                        st.session_state["source_text"] or "",
                        lesson["grade"], lesson["subject"], lesson["medium"],
                    )
                    _add_activity(drafted.model_dump(exclude_none=True, mode="json"))
                except Exception as e:
                    st.error(_generation_error_message("AI draft", e))
        if new_col3.button("Blank", key=f"add_act_blank_{editing_id}"):
            _add_activity(_blank_activity(new_type, lesson["id"], lesson["activities"]))

        st.divider()
        scol1, scol2 = st.columns([1, 3])

        def _reassign_and_validate():
            # A newly added option (image_selection's "+ Add option") has no filename yet --
            # assign_media_filenames() only ran at generation/add-question time, not on every
            # option edit. Re-running it here (idempotent for anything already named) catches
            # that right before it matters, instead of leaving a permanently blank image field.
            language = resolve_language(st.session_state["course_meta"].get("medium"))
            generated = assign_media_filenames(GeneratedLesson(**lesson), language)
            # prune_lesson_fields() also auto-repairs a missing/invalid "ordering"
            # correct_order to the words' natural order (see courseware_schema.py) --
            # only ran at generation/paraphrase time before, so an already-open lesson
            # with this problem stayed stuck on the warning until fixed by hand. Running
            # it here too means clicking Validate/Save now clears it automatically.
            generated = prune_lesson_fields(generated)
            entry["lesson"] = lesson_to_json_dict(generated)
            return validate_lesson(generated)

        if scol1.button("Validate", key=f"validate_{editing_id}"):
            entry["issues"] = _reassign_and_validate()
            st.rerun()

        if scol1.button("Save lesson", type="primary", key=f"save_{editing_id}"):
            issues = _reassign_and_validate()
            entry["issues"] = issues
            lesson = entry["lesson"]
            if issues:
                st.error("Fix structural issues before saving.")
            else:
                OUTPUT_DIR.mkdir(exist_ok=True)
                out_path = OUTPUT_DIR / f"lessons-{lesson['id']}.json"
                out_path.write_text(json.dumps(lesson, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                st.success(f"Saved to {out_path}")
                st.download_button(
                    "Download lesson.json", json.dumps(lesson, indent=2, ensure_ascii=False),
                    file_name=f"lessons-{lesson['id']}.json", mime="application/json", key=f"dl_{editing_id}",
                )
                st.info("Next: open the main app (`streamlit run app.py`) and upload this file there "
                        "to generate its supporting images/audio/video.")


# Autosave on every rerun -- write-only-on-change (see _save_autosave), so this
# is cheap even though it runs after every single widget interaction. Placed
# last so it captures whatever state this rerun ended up with.
_save_autosave()
