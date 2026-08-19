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
    lesson_to_json_dict,
    resolve_language,
)
from student.courseware_review_ui import _generation_error_message, render_lesson_review
from student.courseware_schema import validate_lesson

# Anchored to this file's own package dir, not cwd -- the unified app launches from
# unified_courses/ so both student/ and professional/ packages can coexist, which
# would otherwise resolve a bare relative path to the wrong (shared) location.
OUTPUT_DIR = Path(__file__).resolve().parent / "courseware_output"


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
editing_id = st.session_state["editing_lesson_id"]
if editing_id and editing_id in st.session_state["generated_lessons"]:
    render_lesson_review(
        st.session_state["generated_lessons"][editing_id], editing_id,
        st.session_state["source_text"], st.session_state["course_meta"], OUTPUT_DIR,
        heading_prefix="3. Review",
    )


# Autosave on every rerun -- write-only-on-change (see _save_autosave), so this
# is cheap even though it runs after every single widget interaction. Placed
# last so it captures whatever state this rerun ended up with.
_save_autosave()
