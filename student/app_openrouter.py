"""
Earlier standalone UI driving only the OpenRouter pipeline (Gemini 2.5 Flash
Image / Veo 3.1 Lite): pick a lesson JSON + activity, set teacher preference,
Style (cartoon/natural -- image only), Generate, Play/Preview. Superseded by
app.py's unified HF+OpenRouter flow; kept as a lower-level testing tool.

Audio still uses gTTS from generate_lesson_media.py -- free either way, no
OpenRouter TTS worth switching to.

generate_video here has no Style control: it's fixed to Veo 3.1 Lite's
8s/720p/with-audio config in openrouter_pipeline.py (~$0.40/clip) -- unlike
generate_image, generate_video takes no style parameter at all.

Run:
    streamlit run app_openrouter.py
"""

import json
from pathlib import Path

import streamlit as st

from student.generate_lesson_media import collect_media_tasks, generate_audio, resolve_language
from student.openrouter_pipeline import generate_image, generate_video
from student.personalization import load_teacher_profile, personalize

st.set_page_config(page_title="Generate (OpenRouter): Images / Audio / Video", page_icon="\U0001F310")
st.title("Generate (OpenRouter): Images / Audio / Video")
st.caption("Same pipeline as app.py, routed through OpenRouter instead of Hugging Face. "
           "Image ≈ $0.04/image, Video ≈ $0.40/8s clip (with audio), Audio is free (gTTS either way).")

# --- 3/4: lesson + activity selection (Type + filename loaded from JSON) ---
lesson_path = st.text_input("Lesson JSON path", "lessons-g2_english_l1.json")
if not Path(lesson_path).exists():
    st.warning("File not found.")
    st.stop()

lesson = json.loads(Path(lesson_path).read_text(encoding="utf-8"))
language = resolve_language(lesson.get("medium"))
out_root = Path("generated_content_openrouter") / lesson["id"]  # separate from the HF pipeline's output
try:
    image_tasks, audio_tasks, video_tasks = collect_media_tasks(lesson)
except Exception as e:
    st.error(f"This lesson JSON has a structural problem collect_media_tasks() couldn't handle: {e}")
    st.stop()

all_tasks = (
    [("image", t) for t in image_tasks]
    + [("audio", t) for t in audio_tasks]
    + [("video", t) for t in video_tasks]
)
if not all_tasks:
    st.info("No media referenced by this lesson's activities.")
    st.stop()

labels = [f"[{t['activity_type']}/{kind}] {t['activity_id']} -> {t['out_path']}" for kind, t in all_tasks]
choice = st.selectbox("Activity / asset (Type loaded from JSON)", labels)
kind, task = all_tasks[labels.index(choice)]

col1, col2 = st.columns(2)
col1.text_input("Type (from JSON)", task["activity_type"], disabled=True)
col2.text_input("Filename (from JSON)", task["out_path"], disabled=True)

# --- 2: user prompt -- teacher preference profile ---
st.subheader("Teacher preference")
profile_path = st.text_input("Teacher profile JSON", "teacher_profile.json")
default_profile = load_teacher_profile(profile_path) if Path(profile_path).exists() else {}

pcol1, pcol2 = st.columns(2)
age_group = pcol1.text_input("Age group", default_profile.get("age_group", ""))
gender_target = pcol2.text_input("Gender target", default_profile.get("gender_target", "neutral"))
tone = st.text_input("Tone", default_profile.get("tone", ""))
comment = st.text_area("Comment", default_profile.get("comment", ""))

can_personalize = kind != "audio" or task.get("personalizable", False)
personalize_on = st.checkbox(
    "Personalize with LangChain (system + user prompt)",
    value=can_personalize,
    disabled=not can_personalize,
    help=None if can_personalize else "Disabled: this audio's text is checked verbatim against fixed answer "
                                       "options, so it's never personalized (would break grading).",
)
if not can_personalize:
    st.caption("This audio's text is an exact answer string (listening/speaking) -- never personalized.")

teacher_profile = {"age_group": age_group, "gender_target": gender_target, "tone": tone, "comment": comment}

# --- 1: system prompt (style/mood/cartoon-or-natural), image tasks only ---
style = task.get("style") if kind == "image" else None
if kind == "image":
    style = st.selectbox("Style", ["cartoon", "natural"], index=0 if (style or "cartoon") == "cartoon" else 1)
if kind == "video":
    st.caption("Video has no style control -- fixed to Veo 3.1 Lite's 8s/720p/with-audio config (~$0.40/clip).")

# --- 5: Generate ---
if st.button("Generate", type="primary"):
    out_path = str(out_root / task["out_path"])
    with st.spinner("Generating (video can take a few minutes)..."):
        try:
            if kind == "image":
                subject = task["subject"]
                if personalize_on:
                    enriched = personalize(subject, task["activity_type"], "image", style, teacher_profile)
                    subject = f"{task['subject']} -- {enriched}"
                result = generate_image.invoke({"subject": subject, "style": style, "out_path": out_path})
            elif kind == "audio":
                text = task["text"]
                if personalize_on:
                    text = personalize(text, task["activity_type"], "audio", "narration", teacher_profile)
                result = generate_audio.invoke({"text": text, "language": language, "out_path": out_path})
                result = f"{result} (cost: $0, gTTS)"
            else:  # video
                scene = task["scene_description"]
                if personalize_on:
                    scene = personalize(scene, "video_scene", "video", "cinematic", teacher_profile)
                result = generate_video.invoke({"scene_description": scene, "out_path": out_path})
            st.session_state["last_result"] = (kind, out_path, result)
            st.success(f"Done: {result}")
        except Exception as e:
            st.error(f"Generation failed: {e}")

# --- 6/7: Play / Preview ---
if "last_result" in st.session_state:
    st.subheader("Play / Preview")
    result_kind, result_path, result_detail = st.session_state["last_result"]
    if result_kind == "image":
        st.image(result_path)
    elif result_kind == "audio":
        st.audio(result_path)
    elif result_kind == "video":
        st.video(result_path)
