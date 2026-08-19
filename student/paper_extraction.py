"""
Paper to Course -- student-track-only page. For a teacher who has a real exam/worksheet
paper (photographed or scanned) and wants THOSE EXACT questions turned into a course,
not new AI-invented ones. Distinct from Course Creation (courseware_portal.py), which
generates fresh activities *inspired by* a source book's content across multiple
chapters/lessons -- this page transcribes one paper's questions verbatim into one lesson.

Real exam papers essentially never include an answer key (confirmed against a real
photographed grade 5 Dhamma-school paper), so every correct-answer field is deliberately
left blank after generation -- see generate_paper_lesson()/_clear_answer_keys() in
courseware_extraction.py. A teacher fills in real answers via the same Review & edit
screen Course Creation uses (shared, see courseware_review_ui.py), then saves --
producing the exact same lesson.json shape, in the same courseware_output/ folder, so
it flows into Media Generation like any other lesson.

Cheaper than a full Course Creation pass for this use case: no "Generate groupings" or
"Generate lesson skeletons" LLM calls needed (those exist to break a whole book into
multiple chapters/lessons, which doesn't apply to a single paper) -- just the vision
extraction (same cost either way) plus one structuring call.
"""

import json
import tempfile
from pathlib import Path

import streamlit as st

try:
    import os
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass

from student.courseware_extraction import assign_media_filenames, extract_text, generate_paper_lesson, lesson_to_json_dict, resolve_language
from student.courseware_review_ui import _generation_error_message, render_lesson_review
from student.courseware_schema import validate_lesson

OUTPUT_DIR = Path(__file__).resolve().parent / "courseware_output"

AUTOSAVE_PATH = OUTPUT_DIR / "_paper_progress.json"
AUTOSAVE_KEYS = ["paper_source_text", "paper_meta", "paper_lesson_entry", "paper_generation_id"]


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
    if snapshot_json != st.session_state.get("_paper_last_autosave_snapshot"):
        AUTOSAVE_PATH.write_text(snapshot_json, encoding="utf-8")
        st.session_state["_paper_last_autosave_snapshot"] = snapshot_json


try:
    st.set_page_config(page_title="Paper to Course", page_icon="📄", layout="wide")
except st.errors.StreamlitAPIException:
    pass  # already set by main.py -- this page runs under its st.navigation()
st.title("📄 Paper to Course")
st.caption("Upload a real exam/worksheet paper -- its exact questions become a course, ready for a "
           "teacher to fill in the correct answers (this paper's own answer key isn't extracted, "
           "since papers like this essentially never print one).")

for key, default in [
    ("paper_source_text", None),
    ("paper_meta", {}),
    ("paper_lesson_entry", None),  # {"lesson": dict, "issues": list, "altered": bool} once generated
    # A fixed "paper" editing_id passed to render_lesson_review() (shared with
    # courseware_portal.py) reuses the exact same st.* widget keys (e.g. "act_paper_0_q")
    # across every "Extract & structure" click. Streamlit widgets cache their OWN value
    # under that key independent of the value= argument passed on the next rerun -- so
    # re-extracting after a failed/retried attempt kept silently showing (and then
    # writing back over the freshly-generated data with) the FIRST attempt's stale
    # widget values. Confirmed as a real bug via real testing: a corrected re-run's
    # lesson had genuinely non-blank "question" fields in the returned data, but opening
    # the review screen's activity expander immediately overwrote them back to empty,
    # because _activity_editor() unconditionally writes the widget's (stale) value back
    # into the dict on every render. Fixed by incrementing this counter on every
    # successful generation and folding it into the editing_id below, so each attempt
    # gets fresh, never-before-used widget keys.
    ("paper_generation_id", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if "_paper_autosave_loaded" not in st.session_state:
    _load_autosave()
    st.session_state["_paper_autosave_loaded"] = True
    st.session_state["_paper_show_resume_banner"] = bool(st.session_state["paper_source_text"])

if st.session_state.get("_paper_show_resume_banner"):
    rcol1, rcol2 = st.columns([4, 1])
    rcol1.info("Resumed previously saved progress.")
    if rcol2.button("Clear & start over"):
        st.session_state["paper_source_text"] = None
        st.session_state["paper_meta"] = {}
        st.session_state["paper_lesson_entry"] = None
        st.session_state["paper_generation_id"] = 0
        st.session_state["_paper_show_resume_banner"] = False
        AUTOSAVE_PATH.unlink(missing_ok=True)
        st.session_state.pop("_paper_last_autosave_snapshot", None)
        st.rerun()


# ---------------------------------------------------------------------------
# 1. Upload the paper
# ---------------------------------------------------------------------------
with st.expander("1. Paper", expanded=not st.session_state["paper_source_text"]):
    col1, col2, col3 = st.columns(3)
    grade = col1.number_input("Grade", min_value=1, max_value=13, value=5)
    subject = col2.text_input("Subject", value="")
    medium = col3.selectbox("Medium (language)", ["english", "sinhala"])
    title = st.text_input("Lesson title", value="", placeholder="e.g. Grade 5 Dhamma Term Test")

    uploads = st.file_uploader(
        "Paper pages -- PDF and/or images", type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if st.button("Extract & structure into a course", disabled=not (uploads and subject and title)):
        tmp_dir = Path(tempfile.mkdtemp())
        paths = []
        for f in uploads:
            p = tmp_dir / f.name
            p.write_bytes(f.getvalue())
            paths.append(str(p))

        progress_bar = st.progress(0.0, text="Reading paper...")

        def _on_page(done, total):
            progress_bar.progress(done / total, text=f"Reading page {done}/{total}...")

        try:
            source_text = extract_text(paths, on_page=_on_page)
            progress_bar.progress(1.0, text="Transcribing questions into activities...")
            generated = generate_paper_lesson(source_text, int(grade), subject, medium, title)
            generated = assign_media_filenames(generated, resolve_language(medium))
            issues = validate_lesson(generated)
            st.session_state["paper_source_text"] = source_text
            st.session_state["paper_meta"] = {"grade": int(grade), "subject": subject, "medium": medium}
            st.session_state["paper_lesson_entry"] = {
                "lesson": lesson_to_json_dict(generated), "issues": issues, "altered": False,
            }
            # Fresh widget keys every generation -- see the paper_generation_id comment above.
            st.session_state["paper_generation_id"] += 1
            st.success(f"Extracted {len(source_text)} characters, transcribed "
                       f"{len(generated.activities)} question(s).")
        except Exception as e:
            st.error(_generation_error_message("Extraction", e))
        finally:
            progress_bar.empty()

    if st.session_state["paper_source_text"]:
        with st.expander("Extracted text (preview)"):
            st.text_area("", st.session_state["paper_source_text"], height=200, disabled=True,
                          label_visibility="collapsed")


# ---------------------------------------------------------------------------
# 2. Review & edit -- same screen Course Creation uses. Every question's correct
# answer starts blank on purpose (see generate_paper_lesson()'s docstring), so this
# will always open showing "Structural issues found" until a teacher fills them in.
# ---------------------------------------------------------------------------
if st.session_state["paper_lesson_entry"]:
    st.info("Every question's correct answer was deliberately left blank -- this paper didn't "
            "come with an answer key, so fill each one in below before saving.")
    render_lesson_review(
        st.session_state["paper_lesson_entry"], f"paper{st.session_state['paper_generation_id']}",
        st.session_state["paper_source_text"], st.session_state["paper_meta"], OUTPUT_DIR,
        heading_prefix="2. Review",
    )


_save_autosave()
