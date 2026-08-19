"""Shared "review & edit a generated lesson" UI -- factored out of courseware_portal.py so
student/paper_extraction.py can reuse the exact same review/validate/save screen instead of
duplicating ~150 lines of Streamlit widget code within this package. (Not shared with
professional/ -- that package keeps its own literal copy, same reasoning as everywhere else
in this project: no import relationship between the two packages, to avoid sys.modules
collisions on identically-named modules.)
"""

import json

import streamlit as st

from student.courseware_extraction import (
    assign_media_filenames,
    generate_single_activity,
    lesson_to_json_dict,
    paraphrase_lesson,
    resolve_language,
)
from student.courseware_schema import GeneratedLesson, prune_lesson_fields, validate_lesson

ACTIVITY_TYPES = ["vocabulary", "mcq", "image_selection", "true_false", "fill_blank", "ordering", "listening", "speaking", "video"]


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
        # philosophy as the ordering position editor above. Also the normal state for a
        # freshly-transcribed paper question (see student/paper_extraction.py) -- its answer
        # is deliberately left blank since the source paper never gave one.
        existing_answers = activity.get("acceptable_answers") or []
        default_answers = existing_answers or ([activity["prompt_text"]] if activity["prompt_text"] else [])
        answers_text = st.text_area(
            "Acceptable answers (one per line)",
            "\n".join(default_answers), key=f"{key_prefix}_answers",
        )
        activity["acceptable_answers"] = [a for a in answers_text.split("\n") if a.strip()]

    elif t == "video":
        st.caption(f"Video: {activity.get('video', '—')}")


def render_lesson_review(entry, editing_id, source_text, course_meta, output_dir, heading_prefix="Review"):
    """Renders the full review/validate/Alter/add-question/save screen for one generated
    lesson (`entry` is the {"lesson": dict, "issues": list, "altered": bool} shape both
    courseware_portal.py and paper_extraction.py store per lesson id). Mutates `entry` in
    place -- callers keep whatever dict/session_state structure holds it, this function
    doesn't own storage, just the widgets."""
    lesson = entry["lesson"]

    with st.container(border=True):
        st.subheader(f"{heading_prefix}: {lesson['title']}")

        if entry["issues"]:
            st.warning("Structural issues found -- fix before saving:\n\n" + "\n".join(f"- {i}" for i in entry["issues"]))
        else:
            st.success("Structurally valid.")

        alter = st.checkbox("Alter (paraphrase to avoid reproducing source wording)", value=entry["altered"], key=f"alter_{editing_id}")
        if alter and not entry["altered"]:
            if st.button("Apply Alter pass", key=f"alterbtn_{editing_id}"):
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
            language = resolve_language(course_meta.get("medium"))
            generated = assign_media_filenames(GeneratedLesson(**lesson), language)
            entry["lesson"] = lesson_to_json_dict(generated)
            st.rerun()

        if new_col2.button("✨ AI draft", key=f"add_act_ai_{editing_id}"):
            activity_id = _next_activity_id(lesson["id"], lesson["activities"])
            with st.spinner("Drafting a suggestion..."):
                try:
                    drafted = generate_single_activity(
                        new_type, activity_id, lesson["title"], lesson["description"],
                        source_text or "",
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
            language = resolve_language(course_meta.get("medium"))
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
                output_dir.mkdir(exist_ok=True)
                out_path = output_dir / f"lessons-{lesson['id']}.json"
                out_path.write_text(json.dumps(lesson, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                st.success(f"Saved to {out_path}")
                st.download_button(
                    "Download lesson.json", json.dumps(lesson, indent=2, ensure_ascii=False),
                    file_name=f"lessons-{lesson['id']}.json", mime="application/json", key=f"dl_{editing_id}",
                )
                st.info("Next: open the main app (`streamlit run app.py`) and upload this file there "
                        "to generate its supporting images/audio/video.")
