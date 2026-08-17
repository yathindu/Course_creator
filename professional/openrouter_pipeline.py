"""
Same LangChain pipeline as ai_automation.py / generate_lesson_media.py, but
backed by OpenRouter instead of Hugging Face Inference Providers.

Model picks (chosen for lowest verified cost, not necessarily highest
quality):

- Orchestrator: same Qwen/Qwen3-235B-A22B-Instruct-2507 used via HF, now
  through OpenRouter's OpenAI-compatible endpoint -- just ChatOpenAI pointed
  at a different base_url, no LangChain-specific OpenRouter package needed.
- Image: google/gemini-2.5-flash-image via OpenRouter's dedicated
  /api/v1/images endpoint (not chat completions, which is for image edits).
  Response includes real billed cost in usage.cost.
- Video: google/veo-3.1-lite via OpenRouter's async /api/v1/videos endpoint
  (submit -> poll polling_url -> download from unsigned_urls[0] once
  status="completed"). Cheapest Veo tier with native synchronized audio.
  Configured for 8s/720p/with-audio (~$0.40/clip) -- 8s is the max duration
  Veo supports (4/6/8 allowed, no 10).
- Audio: still gTTS (free either way -- OpenRouter's audio-output models are
  voice-chat models, not TTS). Registered as a tool here too for parity with
  ai_automation.py's image/audio/video/animation set.

Setup:
    pip install langchain-openai langchain-core requests python-dotenv
    Put your key in .env next to this script:
        OPENROUTER_API_KEY=sk-or-v1-...
"""

import base64
import os
import time
from pathlib import Path
from typing import Literal

import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from professional.audio_tools import synthesize_audio
from professional.orchestrator_prompt import ORCHESTRATOR_SYSTEM_PROMPT
from professional.image_processing import remove_background

load_dotenv()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

ORCHESTRATOR_MODEL = "qwen/qwen3-235b-a22b-2507"
IMAGE_MODEL = "google/gemini-2.5-flash-image"
VIDEO_MODEL = "google/veo-3.1-lite"

STYLE_PROMPTS = {
    "illustration": ("clean flat vector illustration, professional corporate/business style, simple "
                      "geometric shapes, muted modern color palette, subtle gradients, minimalist "
                      "flat 2D design, no text -- not a 3D render, not CGI, not a cartoon, not "
                      "childish or playful, suited to workplace training material"),
    "photographic": ("realistic professional photograph, shot on a DSLR camera, natural indoor "
                      "lighting, shallow depth of field, sharp focus, high detail, real workplace/"
                      "retail/office setting, professionally dressed adults, candid but polished "
                      "corporate photography style, no text -- absolutely not a 3D render, not CGI, "
                      "not an illustration or cartoon, not stock-photo cliche poses"),
}


def _headers():
    return {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Execution, as LangChain tools -- same shape as generate_lesson_media.py's
# tools, so the two pipelines are drop-in comparable.
# ---------------------------------------------------------------------------
@tool
def generate_image(
    subject: str,
    style: Literal["illustration", "photographic"],
    out_path: str,
    aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3", "3:4"] = "1:1",
    remove_bg: bool = False,
) -> str:
    """Generate an illustration image for a lesson activity and save it to out_path.
    aspect_ratio controls image shape. remove_bg strips the background, leaving only
    the subject on a transparent background. Returns the real billed cost too."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    prompt = f"{subject}, {STYLE_PROMPTS[style]}"

    response = requests.post(
        f"{OPENROUTER_BASE_URL}/images",
        headers=_headers(),
        json={"model": IMAGE_MODEL, "prompt": prompt, "aspect_ratio": aspect_ratio},
    )
    response.raise_for_status()
    result = response.json()

    image_bytes = base64.b64decode(result["data"][0]["b64_json"])
    target.write_bytes(image_bytes)
    if remove_bg:
        remove_background(target, subject_text=subject)
    cost = result.get("usage", {}).get("cost")
    return f"{target} (cost: ${cost})"


def _submit_and_poll_video(payload, out_path):
    """Shared submit->poll->download flow for /api/v1/videos, used by both
    generate_video (text-to-video) and animate_image (image-to-video) -- same
    async job lifecycle either way, only the request payload differs."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    response = requests.post(f"{OPENROUTER_BASE_URL}/videos", headers=_headers(), json=payload)
    response.raise_for_status()
    job = response.json()
    polling_url = job["polling_url"]

    elapsed = 0
    poll_interval = 10
    max_wait = 600
    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval
        poll_response = requests.get(polling_url, headers=_headers())
        poll_response.raise_for_status()
        status = poll_response.json()
        if status["status"] == "completed":
            video_url = status["unsigned_urls"][0]
            video_response = requests.get(video_url, headers=_headers())
            video_response.raise_for_status()
            content_type = video_response.headers.get("content-type", "")
            if "video" not in content_type:
                raise RuntimeError(
                    f"Download didn't return video content (content-type: {content_type!r}): "
                    f"{video_response.content[:300]!r}"
                )
            target.write_bytes(video_response.content)
            cost = status.get("usage", {}).get("cost")
            return f"{target} (cost: ${cost})"
        if status["status"] == "failed":
            raise RuntimeError(f"Video generation failed: {status.get('error')}")

    raise RuntimeError(f"Video generation timed out after {max_wait}s")


@tool
def generate_video(
    scene_description: str,
    out_path: str,
    duration: Literal[4, 6, 8] = 8,
    with_audio: bool = True,
) -> str:
    """Generate a short video clip for a lesson scene and save it to out_path. duration
    is in seconds (veo-3.1-lite only supports 4/6/8). with_audio toggles a native
    synchronized soundtrack. Returns the real billed cost too."""
    return _submit_and_poll_video({
        "model": VIDEO_MODEL,
        "prompt": scene_description,
        "duration": duration,
        "resolution": "720p",
        "aspect_ratio": "16:9",
        "generate_audio": with_audio,
    }, out_path)


@tool
def animate_image(
    image_path: str,
    motion_description: str,
    out_path: str,
    duration: Literal[4, 6, 8] = 4,
    with_audio: bool = False,
) -> str:
    """Animate an existing static image into a short video clip, using it as the
    starting frame (image-to-video, not text-to-video) -- the output keeps the
    image's exact look/style, it doesn't reimagine the scene. Returns the real
    billed cost too."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = Path(image_path).suffix.lstrip(".") or "png"
    data_url = f"data:image/{ext};base64,{b64}"

    return _submit_and_poll_video({
        "model": VIDEO_MODEL,
        "prompt": motion_description,
        "frame_images": [{"type": "image_url", "image_url": {"url": data_url}, "frame_type": "first_frame"}],
        "duration": duration,
        "resolution": "720p",
        "generate_audio": with_audio,
    }, out_path)


@tool
def generate_audio(
    text: str,
    language: Literal["en", "si"],
    out_path: str,
    voice: Literal["us", "uk", "australia", "india", "canada", "ireland", "south_africa"] = "us",
    volume_db: float = 0.0,
) -> str:
    """Generate spoken audio (TTS) for a word or sentence and save it to out_path.
    voice picks an English accent (ignored for Sinhala). volume_db adjusts loudness
    (positive = louder, negative = quieter). Free either way -- always gTTS."""
    path = synthesize_audio(text, language, out_path, voice=voice, volume_db=volume_db)
    return f"{path} (cost: $0, gTTS)"


TOOLS = [generate_image, generate_audio, generate_video, animate_image]


# Orchestrator: same LangChain tool-calling pattern as ai_automation.py, via
# OpenRouter's OpenAI-compatible endpoint.
def run_automation(teacher_prompt, out_dir="openrouter_output"):
    print(f'Teacher request: "{teacher_prompt}"\n')

    llm = ChatOpenAI(
        model=ORCHESTRATOR_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        max_tokens=800,
    )
    model_with_tools = llm.bind_tools(TOOLS)

    ai_message = model_with_tools.invoke([
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(content=teacher_prompt),
    ])
    tool_calls = ai_message.tool_calls

    if not tool_calls:
        print(f"No generation needed -- LLM replied directly:\n{ai_message.content}\n")
        return []

    print(f"Orchestrator decided on {len(tool_calls)} generation call(s). Executing for real:\n")
    results = []
    tool_map = {t.name: t for t in TOOLS}
    ext = {"generate_image": "png", "generate_audio": "mp3", "generate_video": "mp4", "animate_image": "mp4"}
    for i, tc in enumerate(tool_calls, start=1):
        name = tc["name"]
        args = dict(tc["args"])
        # The LLM often proposes its own out_path -- override with just its basename under
        # out_dir so results land somewhere predictable regardless of what it picks. This
        # controls *where* files land, not *what* the LLM decided to generate -- that
        # decision (tool + subject + style/text) is untouched.
        proposed_name = Path(args.get("out_path") or f"{i:02d}_{name}.{ext[name]}").name
        args["out_path"] = str(Path(out_dir) / proposed_name)
        print(f"[{i}/{len(tool_calls)}] {name}({args})")
        try:
            result = tool_map[name].invoke(args)
            print(f"    -> {result}")
            results.append({"tool": name, "args": args, "status": "success", "result": result})
        except Exception as e:
            print(f"    -> FAILED: {e}")
            results.append({"tool": name, "args": args, "status": "failed", "error": str(e)})
        print()
    return results


if __name__ == "__main__":
    import sys
    teacher_prompt = " ".join(sys.argv[1:]) or "I need an illustration of a cashier greeting a customer at checkout, for a customer service training module."
    run_automation(teacher_prompt)
