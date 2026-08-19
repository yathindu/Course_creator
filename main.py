"""
Unified entry point -- one Streamlit process, one URL, a role choice
(Student / Professional) up front that routes to that track's own
Media Generation + Course Creation pages.

Student = langchain_test's grades 1-5 English/Sinhala-medium pipeline (student/
package). Professional = training_courses' corporate/skill_area pipeline
(professional/ package, includes its own repo_persistence.py GitHub backup
module). Both packages are literal copies of the two source repos with
every cross-module import rewritten to be package-qualified
(student.x / professional.x instead of a bare import x) -- necessary
because both share 18 identically-named modules (app.py,
personalization.py, courseware_portal.py, etc.); without namespacing,
Python's sys.modules cache would let whichever side imports a shared
name first "win," and the other side would silently run the wrong code.
See CLAUDE.md for the full writeup.

st.Page()/st.navigation() (not st.tabs()) for the same reason both source
apps' own main.py already uses it: tabs re-run every tab's code on every
interaction, which would re-trigger the *other* track's
torch/transformers/rembg imports even while only using one track.

Run:
    streamlit run main.py
"""

import streamlit as st

st.set_page_config(page_title="Course Creator", page_icon="🎓", layout="centered")

STUDENT_PAGES = [
    st.Page("student/app.py", title="Media Generation", icon="🎬", default=True),
    st.Page("student/courseware_portal.py", title="Course Creation", icon="📘"),
    st.Page("student/paper_extraction.py", title="Paper to Course", icon="📄"),
]
PROFESSIONAL_PAGES = [
    st.Page("professional/app.py", title="Media Generation", icon="🎬", default=True),
    st.Page("professional/courseware_portal.py", title="Course Creation", icon="📘"),
]

ROLE_LABELS = {
    "student": "🧒 Student — Grades 1-5",
    "professional": "💼 Professional — Corporate Training",
}


def _set_role(role):
    # Both packages are literal copies and reuse the same session_state key
    # names internally (pending_jobs, groupings, lessons_by_group, qa_messages,
    # lesson_zip, approved_animation, etc.) -- clear everything on every role
    # change so the new track never sees the other track's in-memory state.
    # Each side's own file-based autosave (content_drafts/, courseware_output/
    # _portal_progress.json, teacher_profile.json) reloads correctly from disk
    # regardless, so nothing durable is lost by clearing in-memory state.
    st.session_state.clear()
    st.session_state["role"] = role


role = st.session_state.get("role")

if role is None:
    st.title("🎓 Course Creator")
    st.caption("Choose which track you're working in.")
    col1, col2 = st.columns(2)
    with col1:
        st.button(ROLE_LABELS["student"], use_container_width=True,
                   on_click=_set_role, args=("student",))
    with col2:
        st.button(ROLE_LABELS["professional"], use_container_width=True,
                   on_click=_set_role, args=("professional",))
else:
    with st.sidebar:
        st.caption(f"Track: {ROLE_LABELS[role]}")
        st.button("⬅ Switch role", on_click=_set_role, args=(None,))

    pg = st.navigation(STUDENT_PAGES if role == "student" else PROFESSIONAL_PAGES)
    pg.run()
