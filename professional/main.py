"""
Combined entry point -- Streamlit multi-page navigation between the media
generation studio (app.py) and Course Creation (formerly "Courseware Design
Portal", courseware_portal.py), one URL/one deployment, a sidebar page switcher.

st.navigation()/st.Page() was chosen over st.tabs() deliberately: tabs
re-run *every* tab's code on *every* interaction, which would re-trigger
app.py's torch/transformers/rembg imports (several seconds of cold-start
weight) even while only using the portal tab. Page navigation only executes
the currently selected page's script -- switching to the portal never pays
app.py's import cost, and vice versa.

app.py and courseware_portal.py both remain runnable standalone too
(`streamlit run app.py` / `streamlit run courseware_portal.py`) -- each
guards its own st.set_page_config() call so it works either way.

Run:
    streamlit run main.py
"""

import streamlit as st

st.set_page_config(page_title="Lesson Media Studio", page_icon="🎬", layout="centered")

pg = st.navigation([
    st.Page("app.py", title="Media Generation", icon="🎬", default=True),
    st.Page("courseware_portal.py", title="Course Creation", icon="📘"),
])
pg.run()
