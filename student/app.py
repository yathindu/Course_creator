"""
Teacher-facing UI for generating and approving lesson media.

- HF ("LLM1") is tried first; a second button lets the teacher escalate the
  same content to OpenRouter ("LLM2") for comparison, showing both results
  and real costs side by side.
- The teacher only ever sees editable content text, never the underlying
  system/user prompt templates in personalization.py.
- Generation writes to a draft location (drafts_huggingface/ or
  drafts_openrouter/). Nothing is final until "Approve & Save", which copies
  the chosen draft into generated_content/<lesson_id>/ and appends a record
  to that lesson's approvals.json (source, cost, content, timestamp).
- personalize() receives the lesson's grade/subject/medium as learning
  context, so wording matches the actual lesson rather than only the teacher
  profile's preferences.

Run:
    streamlit run app.py
"""

import concurrent.futures
import io
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# Bridge Streamlit Cloud's st.secrets into os.environ before importing modules
# that read credentials via os.environ.get() at import time (local dev still
# works via .env, since this is a no-op when no secrets.toml exists).
try:
    for _key in ("HF_TOKEN", "OPENROUTER_API_KEY"):
        if _key in st.secrets:
            os.environ[_key] = st.secrets[_key]
except st.errors.StreamlitSecretNotFoundError:
    pass  # no secrets.toml at all (e.g. local dev) -- .env / real env vars still apply

import student.generate_lesson_media as hf
import student.openrouter_pipeline as orp
from student.image_processing import BACKGROUND_COLORS, flatten_on_background, has_transparency
from student.personalization import is_sinhala, load_teacher_profile, personalize, translate_to_english

# Data (drafts/, generated_content/, teacher_profile.json, etc.) lives next to this
# file inside the student/ package, not in the process's cwd -- the unified app is
# launched from unified_courses/ (so both student/ and professional/ packages can
# coexist), which would otherwise resolve these bare relative paths to the wrong
# (shared, ambiguous) location.
_BASE_DIR = Path(__file__).resolve().parent


def _show_media(kind, path, state_keys=(), **kwargs):
    """Defensively display an image/audio/video remembered in session_state. Streamlit
    Cloud's local disk is ephemeral -- a draft written earlier in the session can be gone
    after a container restart even though session_state still points at it. Without this
    check that raises an unhandled MediaFileStorageError and crashes the whole page."""
    if not Path(path).exists():
        st.warning(f"This file is no longer available (`{path}`) -- probably lost when the "
                   "app restarted. Please regenerate it.")
        for key in state_keys:
            st.session_state.pop(key, None)
        return False
    {"image": st.image, "audio": st.audio, "video": st.video}[kind](path, **kwargs)
    return True


# Autosave for in-progress, not-yet-approved Content/motion text edits -- a crash or restart
# before Approve & Save would otherwise silently lose wording that was tweaked but never used
# for a generation (it would just re-auto-compose fresh next time, discarding the edit).
_DRAFT_STATE_DIR = _BASE_DIR / "content_drafts"


def _load_draft_text(lesson_id, out_path, field):
    state_path = _DRAFT_STATE_DIR / f"{lesson_id}.json"
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return state.get(out_path, {}).get(field)


def _save_draft_text(lesson_id, out_path, field, text):
    _DRAFT_STATE_DIR.mkdir(exist_ok=True)
    state_path = _DRAFT_STATE_DIR / f"{lesson_id}.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state.setdefault(out_path, {})[field] = text
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _clear_draft_text(lesson_id, out_path):
    """Drop the autosaved draft for an asset once it's been approved -- no longer 'in progress'."""
    state_path = _DRAFT_STATE_DIR / f"{lesson_id}.json"
    if not state_path.exists():
        return
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.pop(out_path, None) is not None:
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


try:
    st.set_page_config(page_title="Lesson Media Studio", page_icon="🎬", layout="centered")
except st.errors.StreamlitAPIException:
    pass  # already set by main.py -- this page is running under its st.navigation()

# Cosmetic-only styling -- purely presentational, no effect on any widget
# behaviour or session state below.
st.markdown("""
<style>
    .block-container {padding-top: 2.5rem; padding-bottom: 3rem; max-width: 860px;}
    h1 {font-size: 1.9rem;}
    h3 {font-size: 1.1rem; margin-bottom: 0.5rem;}
    div.stButton > button {border-radius: 8px; font-weight: 600; padding: 0.4rem 1.1rem;}
    [data-testid="stImage"] img, video, audio {border-radius: 10px;}
    [data-testid="stExpander"] {border-radius: 10px;}
    div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.title("🎬 Lesson Media Studio")
st.caption("Generate images, audio, and video for a lesson. Hugging Face (LLM1) runs first -- "
           "escalate to OpenRouter (LLM2) only if you're not happy with the result.")

with st.container(border=True):
    st.subheader("🧑‍🏫 Teacher preference")
    _default_profile = load_teacher_profile(_BASE_DIR / "teacher_profile.json") if (_BASE_DIR / "teacher_profile.json").exists() else {}
    pcol1, pcol2 = st.columns(2)
    age_group = pcol1.text_input("Age group", _default_profile.get("age_group", ""))
    gender_target = pcol2.text_input("Gender target", _default_profile.get("gender_target", "neutral"))
    tone = st.text_input("Tone", _default_profile.get("tone", ""))
    comment = st.text_area("Comment", _default_profile.get("comment", ""))
    teacher_profile = {"age_group": age_group, "gender_target": gender_target, "tone": tone, "comment": comment}

    # Autosave: write back to teacher_profile.json on any change, so a crash/restart doesn't
    # lose what was typed here -- merge onto the existing file rather than overwrite it outright,
    # since the file can carry extra fields (e.g. teacher_name) this app never reads or edits.
    if teacher_profile != st.session_state.get("_last_saved_teacher_profile"):
        _profile_path = _BASE_DIR / "teacher_profile.json"
        _on_disk = load_teacher_profile(_profile_path) if _profile_path.exists() else {}
        _on_disk.update(teacher_profile)
        _profile_path.write_text(json.dumps(_on_disk, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        st.session_state["_last_saved_teacher_profile"] = teacher_profile

with st.container(border=True):
    st.subheader("📄 Lesson")
    uploaded_lesson = st.file_uploader("Upload lesson JSON", type=["json"])
    if not uploaded_lesson:
        st.info("Upload a lesson JSON file to begin.")
        st.stop()

    try:
        lesson = json.loads(uploaded_lesson.getvalue().decode("utf-8"))
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON file: {e}")
        st.stop()
    if "id" not in lesson or "activities" not in lesson:
        st.error("This JSON doesn't look like a lesson file (missing required \"id\"/\"activities\" fields).")
        st.stop()
    st.caption(f"Loaded **{lesson['id']}** -- grade {lesson.get('grade', '?')}, "
               f"{lesson.get('subject', '?')}, {lesson.get('medium', '?')} medium.")

language = hf.resolve_language(lesson.get("medium"))
learning_context = {"grade": lesson.get("grade"), "subject": lesson.get("subject"), "medium": lesson.get("medium")}
hf_draft_root = _BASE_DIR / "drafts_huggingface" / lesson["id"]
or_draft_root = _BASE_DIR / "drafts_openrouter" / lesson["id"]
final_root = _BASE_DIR / "generated_content" / lesson["id"]
try:
    image_tasks, audio_tasks, video_tasks = hf.collect_media_tasks(lesson)
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


@st.cache_resource
def _get_executor():
    # One shared, process-wide pool (not session-scoped) so background generations survive
    # across reruns instead of being recreated (and orphaned) on every script re-execution.
    return concurrent.futures.ThreadPoolExecutor(max_workers=4)


# Background generation bookkeeping. Jobs are keyed by "{out_path}::{backend}" so a HF job and
# an OpenRouter job for the same asset (or jobs for different assets) never collide, and so a
# job started while task X was selected still resolves correctly no matter what's selected when
# it finishes.
if "pending_jobs" not in st.session_state:
    st.session_state["pending_jobs"] = {}
if "ready_results" not in st.session_state:
    st.session_state["ready_results"] = {}

@st.fragment(run_every=2 if st.session_state["pending_jobs"] else None)
def _poll_pending_jobs():
    # Streamlit only re-executes the script on user interaction -- a background job
    # finishing on its own doesn't trigger anything by itself, so without this a
    # completed generation would just sit in ready_results, invisible, until the
    # teacher happened to click something (switch questions, etc.) that forced a
    # rerun. run_every=2 polls every 2s *only* while jobs are actually pending (this
    # decorator argument is re-evaluated fresh on every full rerun, since the whole
    # script -- including this function definition -- re-executes top to bottom).
    resolved_any = False
    for _job_key, _job in list(st.session_state["pending_jobs"].items()):
        if _job["future"].done():
            try:
                _detail = _job["future"].result()
                st.session_state["ready_results"][_job_key] = {
                    "kind": _job["kind"], "path": _job["result_path"], "detail": _detail, "error": None}
            except Exception as e:
                st.session_state["ready_results"][_job_key] = {
                    "kind": _job["kind"], "path": _job["result_path"], "detail": None, "error": str(e)}
            del st.session_state["pending_jobs"][_job_key]
            resolved_any = True

    if resolved_any:
        # scope="app": a plain st.rerun() here would only re-run this fragment, but the
        # Generate section further down (which reads ready_results for the *currently
        # selected* task) lives outside the fragment and needs the whole page to rerun
        # to pick the result up.
        st.rerun(scope="app")

    if st.session_state["pending_jobs"]:
        _pcol1, _pcol2 = st.columns([4, 1])
        _pcol1.info(f"\U0001F504 {len(st.session_state['pending_jobs'])} generation(s) running in the "
                    "background -- feel free to switch questions, they'll keep going.")
        if _pcol2.button("Refresh"):
            st.rerun(scope="app")


_poll_pending_jobs()

_approved_count = sum(1 for _kind, _t in all_tasks if (final_root / _t["out_path"]).exists())
st.progress(_approved_count / len(all_tasks),
            text=f"{_approved_count}/{len(all_tasks)} assets approved for this lesson")

if final_root.exists() and any(final_root.iterdir()):
    with st.container(border=True):
        st.subheader("⬇️ Download")
        st.caption("Download everything approved so far for this lesson -- images, audio, video, "
                   "and the approvals record -- as a ZIP for backup or use outside this app.")
        if st.button("Prepare ZIP of approved content"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in sorted(final_root.rglob("*")):
                    if f.is_file():
                        zf.write(f, arcname=f.relative_to(final_root))
            st.session_state["lesson_zip"] = buf.getvalue()
        if "lesson_zip" in st.session_state:
            st.download_button(
                f"Download {lesson['id']}_generated_content.zip",
                data=st.session_state["lesson_zip"],
                file_name=f"{lesson['id']}_generated_content.zip",
                mime="application/zip",
            )

with st.container(border=True):
    st.subheader("🎯 Activity")

    # Group by activity_id first -- a single question can have several assets (an mcq's N
    # image options, or a vocabulary word's image + audio), and a flat list repeats the same
    # activity_id back to back for each one, reading like unrelated entries. Picking the
    # question first, then which asset within it, makes that relationship visible instead.
    activity_order = []
    activity_groups = {}
    for _kind, _t in all_tasks:
        _aid = _t["activity_id"]
        if _aid not in activity_groups:
            activity_groups[_aid] = []
            activity_order.append(_aid)
        activity_groups[_aid].append((_kind, _t))

    def _asset_preview(t):
        return t.get("subject") or t.get("text") or t.get("scene_description") or t["out_path"]

    def _question_label(aid):
        group = activity_groups[aid]
        first_kind, first_t = group[0]
        preview = _asset_preview(first_t)
        suffix = f" ({len(group)} assets)" if len(group) > 1 else ""
        # Lesson-authored "group" (e.g. several activities under one "Vegetable Info"
        # theme) is purely a visual clustering hint here -- prefixing the label keeps
        # grouped questions visibly adjacent/labeled in the flat picker without
        # restructuring it into real nested selection.
        cluster = f"[{first_t['group']}] " if first_t.get("group") else ""
        return f"{cluster}{aid} [{first_t['activity_type']}] -- {preview}{suffix}"

    question_labels = [_question_label(aid) for aid in activity_order]
    chosen_question = st.selectbox("Question (loaded from JSON)", question_labels)
    chosen_aid = activity_order[question_labels.index(chosen_question)]
    group = activity_groups[chosen_aid]

    if len(group) > 1:
        def _asset_label(k, t):
            return f"[{k}] {t['out_path']} -- {_asset_preview(t)}"
        asset_labels = [_asset_label(k, t) for k, t in group]
        chosen_asset = st.selectbox("Asset for this question", asset_labels)
        kind, task = group[asset_labels.index(chosen_asset)]
    else:
        kind, task = group[0]
        st.caption(f"Only one asset for this question: [{kind}] {task['out_path']}")

    selection_key = f"{chosen_aid}::{kind}::{task['out_path']}"  # unique per (question, asset)

    col1, col2 = st.columns(2)
    col1.text_input("Type", task["activity_type"], disabled=True)
    col2.text_input("Filename", task["out_path"], disabled=True)

    can_personalize = kind != "audio" or task.get("personalizable", False)
    if not can_personalize:
        st.caption("This audio's text is an exact answer string (listening/speaking) -- wording "
                   "suggestions are disabled here since editing it would break grading. You can "
                   "still hand-edit it below, just keep the meaning/answer intact.")

style = task.get("style") or "cartoon"
remove_bg = False
if kind == "image":
    with st.container(border=True):
        st.subheader("🎨 Image options")
        icol1, icol2 = st.columns(2)
        style = icol1.selectbox("Style", ["cartoon", "natural"], index=0 if style == "cartoon" else 1,
                                 key=f"style_{selection_key}")
        aspect = icol2.selectbox("Aspect ratio", ["1:1 (square)", "16:9 (wide)", "9:16 (tall)", "4:3", "3:4"])
        remove_bg = st.checkbox("Remove background (isolate subject only)", value=False)

        _ASPECT_TO_WH = {
            "1:1 (square)": (1024, 1024), "16:9 (wide)": (1280, 720), "9:16 (tall)": (720, 1280),
            "4:3": (1024, 768), "3:4": (768, 1024),
        }
        _ASPECT_TO_OR = {
            "1:1 (square)": "1:1", "16:9 (wide)": "16:9", "9:16 (tall)": "9:16", "4:3": "4:3", "3:4": "3:4",
        }

        sibling_tasks = [t for t in image_tasks if t["activity_id"] == task["activity_id"]]
        if len(sibling_tasks) > 1:
            with st.expander(f"🖼️ Generate all {len(sibling_tasks)} options for this question"):
                st.caption(f"Options: {', '.join(t['subject'] for t in sibling_tasks)}. Generates all of "
                           "them at once with the Style/Aspect ratio/Remove background settings above.")

                # Each option gets its own composed prompt, editable individually -- previously
                # this was only ever computed silently inside _generate_batch() at generation
                # time, so the teacher could see/edit the single currently-selected task's prompt
                # in the Content box above but never the other N-1 options actually being generated.
                def _compose_batch_content():
                    for t in sibling_tasks:
                        st.session_state[f"batch_content_{t['out_path']}"] = personalize(
                            t["subject"], t["activity_type"], "image", style, teacher_profile, learning_context)

                batch_style_key = f"{task['activity_id']}::{style}"
                if st.session_state.get("batch_style_key") != batch_style_key:
                    st.session_state["batch_style_key"] = batch_style_key
                    with st.spinner(f"Composing {len(sibling_tasks)} suggested prompts..."):
                        _compose_batch_content()

                st.caption("Each option's prompt is auto-composed and editable below -- review before generating.")
                if st.button("Regenerate all suggestions (LangChain)"):
                    with st.spinner(f"Composing {len(sibling_tasks)} suggested prompts..."):
                        _compose_batch_content()

                for t in sibling_tasks:
                    st.text_area(f"{t['subject']} -> `{t['out_path']}`",
                                 key=f"batch_content_{t['out_path']}", height=80)

                def _generate_batch(module, out_root):
                    # sib_out_path (which we chose) is always the real file path. The tool's
                    # return value ("detail") is informational only -- OpenRouter's generate_image
                    # appends " (cost: $X)" to it, so it must never be used as a path (that was a
                    # real bug: passing it to st.image/shutil.copyfile broke every OpenRouter batch
                    # generation with a MediaFileStorageError, since HF's return value happens to be
                    # a clean path but OpenRouter's isn't).
                    width, height = _ASPECT_TO_WH[aspect]
                    results = {}
                    for t in sibling_tasks:
                        sib_content = st.session_state[f"batch_content_{t['out_path']}"]
                        prompt_text = translate_to_english(sib_content)
                        sib_out_path = str(out_root / t["out_path"])
                        if module is hf:
                            detail = module.generate_image.invoke(
                                {"subject": prompt_text, "style": style, "out_path": sib_out_path,
                                 "width": width, "height": height, "remove_bg": remove_bg})
                        else:
                            detail = module.generate_image.invoke(
                                {"subject": prompt_text, "style": style, "out_path": sib_out_path,
                                 "aspect_ratio": _ASPECT_TO_OR[aspect], "remove_bg": remove_bg})
                        results[t["out_path"]] = {
                            "label": t["subject"], "path": sib_out_path, "detail": detail, "content": sib_content}
                    return results

                bcol1, bcol2 = st.columns(2)
                if bcol1.button(f"Generate all {len(sibling_tasks)} with Hugging Face"):
                    with st.spinner(f"Generating {len(sibling_tasks)} images..."):
                        try:
                            st.session_state["batch_key"] = task["activity_id"]
                            st.session_state["batch_results"] = ("huggingface", _generate_batch(hf, hf_draft_root))
                            st.success("Batch generation complete -- review below, then Approve & Save.")
                        except Exception as e:
                            st.error(f"Batch generation failed: {e}")
                if bcol2.button(f"Generate all {len(sibling_tasks)} with OpenRouter"):
                    with st.spinner(f"Generating {len(sibling_tasks)} images..."):
                        try:
                            st.session_state["batch_key"] = task["activity_id"]
                            st.session_state["batch_results"] = ("openrouter", _generate_batch(orp, or_draft_root))
                            st.success("Batch generation complete -- review below, then Approve & Save.")
                        except Exception as e:
                            st.error(f"Batch generation failed: {e}")

                if st.session_state.get("batch_key") == task["activity_id"] and "batch_results" in st.session_state:
                    batch_source, batch_results = st.session_state["batch_results"]
                    st.caption(f"Batch results from {batch_source}:")
                    for out_path, info in batch_results.items():
                        bc1, bc2 = st.columns([1, 3])
                        with bc1:
                            _show_media("image", info["path"], width=120)
                        bc2.write(f"**{info['label']}** -> `{out_path}`")

                    if st.button(f"Approve & Save all {len(batch_results)} options", type="primary"):
                        approvals_path = final_root / "approvals.json"
                        approvals_path.parent.mkdir(parents=True, exist_ok=True)
                        approvals = (json.loads(approvals_path.read_text(encoding="utf-8"))
                                     if approvals_path.exists() else [])
                        for out_path, info in batch_results.items():
                            final_path = final_root / out_path
                            final_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(info["path"], final_path)
                            approvals.append({
                                "activity_id": task["activity_id"],
                                "kind": "image",
                                "out_path": out_path,
                                "approved_source": batch_source,
                                "detail": info["detail"],
                                "content": info["content"],
                                "approved_at": datetime.now(timezone.utc).isoformat(),
                            })
                        approvals_path.write_text(json.dumps(approvals, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
                        st.session_state.pop("batch_results", None)
                        st.session_state.pop("batch_key", None)
                        st.success(f"Approved and saved all {len(batch_results)} options.")

        approvals_path = final_root / "approvals.json"
        if approvals_path.exists():
            existing_approvals = json.loads(approvals_path.read_text(encoding="utf-8"))
            reusable = [a for a in existing_approvals
                        if a.get("kind") == "image" and a.get("out_path") != task["out_path"]]
            if reusable:
                with st.expander("🔁 Reuse an existing approved image"):
                    st.caption("Use an image already approved elsewhere in this lesson instead of "
                               "generating a new one -- useful when the same subject appears in "
                               "multiple questions (e.g. a Cow used for both a vocabulary card and "
                               "an mcq answer choice).")
                    reuse_labels = [f"{a['activity_id']} -> {a['out_path']}" for a in reusable]
                    reuse_choice = st.selectbox("Existing approved image", ["(none)"] + reuse_labels)
                    if reuse_choice != "(none)":
                        reuse_entry = reusable[reuse_labels.index(reuse_choice)]
                        reuse_path = final_root / reuse_entry["out_path"]
                        if _show_media("image", str(reuse_path), width=200):
                            if st.button("Use this image for the current question"):
                                st.session_state["hf_result"] = (
                                    "image", str(reuse_path),
                                    f"Reused from {reuse_entry['activity_id']} ({reuse_entry['out_path']})")
                                st.session_state.pop("or_result", None)
                                st.session_state.pop("approved", None)
                                st.success(f"Using {reuse_path} -- review below, then Approve & Save.")


def _behaviour():
    if kind == "image":
        return style
    return "cinematic" if kind == "video" else "narration"


raw_content = task["subject"] if kind == "image" else task.get("text") or task.get("scene_description")
# Include style in the key for images so switching cartoon<->natural re-composes the content
# automatically -- otherwise a leftover cartoon-flavored description could stay paired with a
# natural-style generation (or vice versa), the same mismatch that caused the earlier 3D-cartoon bug.
content_selection_key = f"{selection_key}::{style}" if kind == "image" else selection_key
if st.session_state.get("content_selection_key") != content_selection_key:
    st.session_state["content_selection_key"] = content_selection_key
    _saved_draft = _load_draft_text(lesson["id"], task["out_path"], "content_text")
    if _saved_draft is not None:
        st.session_state["content_text"] = _saved_draft
    elif can_personalize:
        with st.spinner("Composing a suggested prompt from teacher preferences..."):
            st.session_state["content_text"] = personalize(
                raw_content, task["activity_type"], kind, _behaviour(), teacher_profile, learning_context)
    else:
        st.session_state["content_text"] = raw_content
    st.session_state.pop("hf_result", None)
    st.session_state.pop("or_result", None)
    st.session_state.pop("approved", None)

with st.container(border=True):
    st.subheader("✍️ Content")
    if can_personalize:
        st.caption("Auto-composed from the teacher preferences above and this activity's type. "
                   "Edit freely, or regenerate after changing a preference.")
        if st.button("Regenerate suggestion (LangChain)"):
            with st.spinner("Asking the model for a suggestion..."):
                st.session_state["content_text"] = personalize(
                    raw_content, task["activity_type"], kind, _behaviour(), teacher_profile, learning_context)

    content_text = st.text_area("What should be generated (edit freely)", key="content_text", height=100)
    _content_cache_key = f"_last_saved_content_{task['out_path']}"
    if content_text != st.session_state.get(_content_cache_key):
        _save_draft_text(lesson["id"], task["out_path"], "content_text", content_text)
        st.session_state[_content_cache_key] = content_text
    if kind in ("image", "video") and is_sinhala(content_text):
        st.caption("Sinhala detected -- this will be auto-translated to English before it reaches "
                   "the image/video model (it doesn't understand Sinhala prompts well). Audio always "
                   "speaks the exact text you type here, unchanged.")

duration, with_audio = 8, True
if kind == "video":
    with st.container(border=True):
        st.subheader("🎥 Video options")
        st.caption("Duration and audio-in-video are OpenRouter-only -- HF's video model (Wan2.2) "
                   "has no such controls at all. These settings only take effect if you escalate "
                   "to OpenRouter below.")
        vcol1, vcol2 = st.columns(2)
        duration = vcol1.select_slider("Duration (seconds, OpenRouter only)", options=[4, 6, 8], value=8)
        with_audio = vcol2.checkbox("Include audio (OpenRouter only)", value=True)

voice, volume_db = "us", 0.0
if kind == "audio":
    with st.container(border=True):
        st.subheader("🔊 Audio options")
        acol1, acol2 = st.columns(2)
        voice = acol1.selectbox(
            "Voice / accent", ["us", "uk", "australia", "india", "canada", "ireland", "south_africa"],
            help="English accent via gTTS. Ignored for Sinhala (only one voice available).")
        volume_db = acol2.slider("Volume adjustment (dB)", -20.0, 20.0, 0.0, step=1.0)


def _build_generate_call(module, out_root, video_kwargs_supported):
    """Resolve which tool to call and with what args, synchronously (fast -- the only possible
    API call here is translate_to_english, a no-op for non-Sinhala text, cached otherwise).
    Returns (tool, kwargs) instead of calling .invoke() itself, so the actual slow generation
    can be handed to a background thread as a plain (callable, dict) pair. This matters: module-
    level functions look up free variables (task, content_text, style, ...) in the script's
    global namespace AT CALL TIME, not when defined -- and Streamlit reruns the script in that
    SAME namespace on every interaction. A background thread calling a closure that reads those
    names later could silently see a DIFFERENT question's data if the user had switched away by
    then. Building the concrete (tool, kwargs) snapshot here, before submitting, avoids that."""
    out_path = str(out_root / task["out_path"])
    if kind == "image":
        prompt_text = translate_to_english(content_text)
        width, height = _ASPECT_TO_WH[aspect]
        if module is hf:
            return module.generate_image, {"subject": prompt_text, "style": style, "out_path": out_path,
                                            "width": width, "height": height, "remove_bg": remove_bg}
        return module.generate_image, {"subject": prompt_text, "style": style, "out_path": out_path,
                                        "aspect_ratio": _ASPECT_TO_OR[aspect], "remove_bg": remove_bg}
    if kind == "audio":
        # Never translated -- gTTS must speak exactly what the teacher typed, in the
        # lesson's own medium (language, from LANGUAGE_MAP, already reflects that).
        return module.generate_audio, {"text": content_text, "language": language, "out_path": out_path,
                                        "voice": voice, "volume_db": volume_db}
    # video
    prompt_text = translate_to_english(content_text)
    if video_kwargs_supported:
        return module.generate_video, {"scene_description": prompt_text, "out_path": out_path,
                                        "duration": duration, "with_audio": with_audio}
    return module.generate_video, {"scene_description": prompt_text, "out_path": out_path}


_hf_job_key = f"{task['out_path']}::huggingface"
_or_job_key = f"{task['out_path']}::openrouter"

# Promote any background job that finished for THIS task into the normal hf_result/or_result
# slots the rest of the script already expects -- keeps Preview/Approve/Animate below completely
# unaware that generation happened asynchronously.
if _hf_job_key in st.session_state["ready_results"]:
    _job = st.session_state["ready_results"].pop(_hf_job_key)
    if _job["error"]:
        st.error(f"HF generation failed: {_job['error']}")
    else:
        st.session_state["hf_result"] = (_job["kind"], _job["path"], _job["detail"])
        st.session_state.pop("or_result", None)
        st.session_state.pop("approved", None)
if _or_job_key in st.session_state["ready_results"]:
    _job = st.session_state["ready_results"].pop(_or_job_key)
    if _job["error"]:
        st.error(f"OpenRouter generation failed: {_job['error']}")
    else:
        st.session_state["or_result"] = (_job["kind"], _job["path"], _job["detail"])
        st.session_state.pop("approved", None)

with st.container(border=True):
    st.subheader("⚡ Generate")
    if _hf_job_key in st.session_state["pending_jobs"]:
        st.info("\U0001F504 Generating with Hugging Face in the background -- switch to another "
                "question if you like, this keeps running.")
    elif st.button("Generate with Hugging Face (LLM1)", type="primary"):
        tool, kwargs = _build_generate_call(hf, hf_draft_root, video_kwargs_supported=False)
        future = _get_executor().submit(tool.invoke, kwargs)
        st.session_state["pending_jobs"][_hf_job_key] = {
            "future": future, "kind": kind, "result_path": str(hf_draft_root / task["out_path"])}
        st.rerun()

if "hf_result" in st.session_state:
    with st.container(border=True):
        st.subheader("▶️ Preview -- Hugging Face (LLM1) draft")
        r_kind, r_path, r_detail = st.session_state["hf_result"]
        _show_media(r_kind, r_path, state_keys=("hf_result", "or_result", "approved"))

        if _or_job_key in st.session_state["pending_jobs"]:
            st.info("\U0001F504 Generating with OpenRouter in the background...")
        elif st.button("Not satisfied? Try OpenRouter (LLM2)"):
            tool, kwargs = _build_generate_call(orp, or_draft_root, video_kwargs_supported=True)
            future = _get_executor().submit(tool.invoke, kwargs)
            st.session_state["pending_jobs"][_or_job_key] = {
                "future": future, "kind": kind, "result_path": str(or_draft_root / task["out_path"])}
            st.rerun()

if "or_result" in st.session_state:
    with st.container(border=True):
        st.subheader("▶️ Preview -- OpenRouter (LLM2) draft")
        r_kind, r_path, r_detail = st.session_state["or_result"]
        if _show_media(r_kind, r_path, state_keys=("or_result", "approved")):
            st.caption(f"Result: {r_detail}")

if "hf_result" in st.session_state:
    with st.container(border=True):
        st.subheader("✅ Approve & Save")
        options = {"Hugging Face (LLM1)": "hf_result"}
        if "or_result" in st.session_state:
            options["OpenRouter (LLM2)"] = "or_result"

        if len(options) > 1:
            chosen_label = st.radio("Which draft do you want to approve?", list(options.keys()))
        else:
            chosen_label = list(options.keys())[0]
        chosen_key = options[chosen_label]

        if st.button("Approve & Save", type="primary"):
            _, draft_path, draft_detail = st.session_state[chosen_key]
            final_path = final_root / task["out_path"]
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(draft_path, final_path)

            approvals_path = final_root / "approvals.json"
            approvals_path.parent.mkdir(parents=True, exist_ok=True)
            approvals = json.loads(approvals_path.read_text(encoding="utf-8")) if approvals_path.exists() else []
            approvals.append({
                "activity_id": task["activity_id"],
                "kind": kind,
                "out_path": task["out_path"],
                "approved_source": "huggingface" if chosen_key == "hf_result" else "openrouter",
                "detail": draft_detail,
                "content": content_text,
                "approved_at": datetime.now(timezone.utc).isoformat(),
            })
            approvals_path.write_text(json.dumps(approvals, indent=2, ensure_ascii=False), encoding="utf-8")
            _clear_draft_text(lesson["id"], task["out_path"])

            st.session_state["approved"] = str(final_path)
            st.success(f"Approved and saved to {final_path} (source: {chosen_label})")

if "approved" in st.session_state:
    st.caption(f"Currently approved for this activity: {st.session_state['approved']}")
    approved_path = Path(st.session_state["approved"])
    if approved_path.exists():
        st.download_button(
            f"Download {approved_path.name}", data=approved_path.read_bytes(),
            file_name=approved_path.name, key="dl_approved_single")

if kind == "image":
    # Each source carries its own "content" (the subject/description it was generated from)
    # alongside its path -- a sibling image (e.g. "sheep" from a batch of mcq options) needs
    # its OWN content for the motion suggestion, not the currently-selected task's content_text
    # (which would describe the wrong subject, e.g. "duck", when animating the sheep image).
    image_sources = {}
    if "hf_result" in st.session_state:
        image_sources["Hugging Face draft"] = {"path": st.session_state["hf_result"][1], "content": content_text}
    if "or_result" in st.session_state:
        image_sources["OpenRouter draft"] = {"path": st.session_state["or_result"][1], "content": content_text}
    final_image_path = final_root / task["out_path"]
    if final_image_path.exists():
        image_sources["Approved (saved)"] = {"path": str(final_image_path), "content": content_text}

    # Also offer every other approved image for this same question (e.g. the other options
    # from a batch generation of an mcq/image_selection with several images) -- otherwise only
    # whichever option happens to be selected in the Activity picker above can be animated,
    # even though all of them were just approved together.
    approvals_path = final_root / "approvals.json"
    if approvals_path.exists():
        existing_approvals = json.loads(approvals_path.read_text(encoding="utf-8"))
        for a in existing_approvals:
            if (a.get("kind") == "image" and a.get("activity_id") == task["activity_id"]
                    and a.get("out_path") != task["out_path"]):
                sib_path = final_root / a["out_path"]
                if sib_path.exists():
                    image_sources[f"Approved: {a['out_path']}"] = {
                        "path": str(sib_path), "content": a.get("content") or content_text}

    if image_sources:
        with st.container(border=True):
            st.subheader("🎞️ Animate this image")
            st.caption("Turns the exact image below into a short moving clip (image-to-video) -- "
                       "it keeps that image's look, it doesn't generate a new scene from scratch.")
            source_label = st.selectbox("Source image", list(image_sources.keys()))
            source_path = image_sources[source_label]["path"]
            source_content = image_sources[source_label]["content"]
            if not _show_media("image", source_path, width=200):
                st.stop()  # has_transparency() below would crash on a missing file

            anim_source_path = source_path
            if has_transparency(source_path):
                bg_color = st.selectbox(
                    "Background color for animation", list(BACKGROUND_COLORS),
                    help="This image has a removed/transparent background. Video models need a solid "
                         "backdrop, not transparency, so pick a color to fill it with before animating.")
                anim_source_path = str(Path(source_path).with_name(
                    f"{Path(source_path).stem}_on_{bg_color.lower()}{Path(source_path).suffix}"))
                flatten_on_background(source_path, bg_color, anim_source_path)
                st.image(anim_source_path, width=200, caption=f"Will animate this ({bg_color.lower()} background)")

            motion_draft_key = f"{task['out_path']}::motion::{source_label}"
            motion_selection_key = f"{selection_key}::{source_label}"
            if st.session_state.get("motion_selection_key") != motion_selection_key:
                st.session_state["motion_selection_key"] = motion_selection_key
                _saved_motion_draft = _load_draft_text(lesson["id"], motion_draft_key, "motion_text")
                if _saved_motion_draft is not None:
                    st.session_state["motion_text"] = _saved_motion_draft
                else:
                    with st.spinner("Composing a suggested motion description..."):
                        st.session_state["motion_text"] = personalize(
                            source_content, task["activity_type"], "animation", "animation",
                            teacher_profile, learning_context)
                st.session_state.pop("anim_result", None)
                st.session_state.pop("approved_animation", None)

            if st.button("Regenerate motion suggestion (LangChain)"):
                with st.spinner("Asking the model for a suggestion..."):
                    st.session_state["motion_text"] = personalize(
                        source_content, task["activity_type"], "animation", "animation",
                        teacher_profile, learning_context)

            motion_text = st.text_input("Motion description (edit freely)", key="motion_text")
            _motion_cache_key = f"_last_saved_motion_{motion_draft_key}"
            if motion_text != st.session_state.get(_motion_cache_key):
                _save_draft_text(lesson["id"], motion_draft_key, "motion_text", motion_text)
                st.session_state[_motion_cache_key] = motion_text
            if is_sinhala(motion_text):
                st.caption("Sinhala detected -- will be auto-translated to English before animating.")

            acol1, acol2 = st.columns(2)
            if acol1.button("Animate with Hugging Face"):
                with st.spinner("Animating (this can take a minute or two)..."):
                    try:
                        anim_out = str(hf_draft_root / (Path(task["out_path"]).stem + "_animated.mp4"))
                        motion_prompt = translate_to_english(motion_text)
                        result = hf.animate_image.invoke(
                            {"image_path": anim_source_path, "motion_description": motion_prompt,
                             "out_path": anim_out})
                        st.session_state["anim_result"] = ("huggingface", anim_out, result)
                        st.success(f"Animation ready: {result}")
                    except Exception as e:
                        st.error(f"Animation failed: {e}")
            if acol2.button("Animate with OpenRouter"):
                with st.spinner("Animating (this can take a minute or two)..."):
                    try:
                        anim_out = str(or_draft_root / (Path(task["out_path"]).stem + "_animated.mp4"))
                        motion_prompt = translate_to_english(motion_text)
                        result = orp.animate_image.invoke(
                            {"image_path": anim_source_path, "motion_description": motion_prompt,
                             "out_path": anim_out})
                        st.session_state["anim_result"] = ("openrouter", anim_out, result)
                        st.success(f"Animation ready: {result}")
                    except Exception as e:
                        st.error(f"Animation failed: {e}")

            if "anim_result" in st.session_state:
                anim_source, anim_path, anim_detail = st.session_state["anim_result"]
                if _show_media("video", anim_path, state_keys=("anim_result",)):
                    st.caption(f"From: {anim_source} -- {anim_detail}")

                    if st.button("Approve & Save animation"):
                        anim_out_name = Path(task["out_path"]).stem + "_animated.mp4"
                        final_anim_path = final_root / anim_out_name
                        final_anim_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(anim_path, final_anim_path)

                        approvals_path = final_root / "approvals.json"
                        approvals = (json.loads(approvals_path.read_text(encoding="utf-8"))
                                     if approvals_path.exists() else [])
                        approvals.append({
                            "activity_id": task["activity_id"],
                            "kind": "animation",
                            "out_path": anim_out_name,
                            "approved_source": anim_source,
                            "detail": anim_detail,
                            "content": motion_text,
                            "source_image": source_path,
                            "approved_at": datetime.now(timezone.utc).isoformat(),
                        })
                        approvals_path.write_text(
                            json.dumps(approvals, indent=2, ensure_ascii=False), encoding="utf-8")
                        _clear_draft_text(lesson["id"], motion_draft_key)
                        st.session_state["approved_animation"] = str(final_anim_path)
                        st.success(f"Approved and saved to {final_anim_path}")

            if "approved_animation" in st.session_state:
                approved_anim_path = Path(st.session_state["approved_animation"])
                if approved_anim_path.exists():
                    st.download_button(
                        f"Download {approved_anim_path.name}", data=approved_anim_path.read_bytes(),
                        file_name=approved_anim_path.name, key="dl_approved_animation")
