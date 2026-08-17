"""
Freeform-prompt automation: an LLM orchestrator decides which generation
tools to call from a teacher's natural-language request, via LangChain
tool-calling (bind_tools) rather than scripted keyword dispatch.

Uses the same tool implementations as generate_lesson_media.py and the
Streamlit app (generate_image/generate_audio/generate_video/animate_image),
so this has the same capabilities as the manual UI -- background removal,
accent/volume control, image-to-video animation -- available as things the
model can decide to use from plain English.

Orchestrator model: Qwen3-235B-A22B-Instruct-2507 (sparse MoE, 235B total /
22B active params) via HF's "novita" inference provider, pinned explicitly
since "auto" routing can land on a provider that only serves this model for
plain completion, not tool-calling.

Setup:
    pip install langchain langchain-huggingface huggingface_hub gtts python-dotenv
    Put your token in a .env file next to this script:
        HF_TOKEN=your-pro-token-here
    python ai_automation.py "your request"
"""

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from student.config import MODEL, MODEL_PROVIDER
from student.generate_lesson_media import animate_image, generate_audio, generate_image, generate_video
from student.orchestrator_prompt import ORCHESTRATOR_SYSTEM_PROMPT

load_dotenv()

if not os.environ.get("HF_TOKEN"):
    raise RuntimeError("Set HF_TOKEN environment variable (Pro account token) before running.")

HF_TOKEN = os.environ["HF_TOKEN"]
OUT_DIR = "automation_output"

TOOLS = [generate_image, generate_audio, generate_video, animate_image]
TOOL_MAP = {t.name: t for t in TOOLS}
_EXT = {"generate_image": "png", "generate_audio": "mp3", "generate_video": "mp4", "animate_image": "mp4"}


def run_automation(teacher_prompt, out_dir=OUT_DIR):
    print(f'Teacher request: "{teacher_prompt}"\n')
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    llm = HuggingFaceEndpoint(
        repo_id=MODEL,
        huggingfacehub_api_token=HF_TOKEN,
        provider=MODEL_PROVIDER,
        max_new_tokens=800,
    )
    chat_model = ChatHuggingFace(llm=llm)
    model_with_tools = chat_model.bind_tools(TOOLS)

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
    for i, tc in enumerate(tool_calls, start=1):
        name = tc["name"]
        args = dict(tc["args"])
        # The LLM often proposes its own out_path (or none) -- override with just its
        # basename under out_dir so results land somewhere predictable. This controls
        # *where* files land, not *what*/*how* it decided to generate.
        proposed_name = Path(args.get("out_path") or f"{i:02d}_{name}.{_EXT[name]}").name
        args["out_path"] = str(Path(out_dir) / proposed_name)
        print(f"[{i}/{len(tool_calls)}] {name}({args})")
        try:
            result = TOOL_MAP[name].invoke(args)
            print(f"    -> {result}")
            results.append({"tool": name, "args": args, "status": "success", "result": result})
        except Exception as e:
            print(f"    -> FAILED: {e}")
            results.append({"tool": name, "args": args, "status": "failed", "error": str(e)})
        print()
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        teacher_prompt = " ".join(sys.argv[1:])
    else:
        teacher_prompt = (
            "I need a cartoon picture of a cow, an English audio clip saying "
            "'moo', and a short video scene of a cow on a farm, for a lesson "
            "about farm animals."
        )

    all_results = run_automation(teacher_prompt)

    manifest_path = f"{OUT_DIR}/manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now().isoformat(),
                    "teacher_prompt": teacher_prompt,
                    "results": all_results}, f, indent=2, ensure_ascii=False)

    print(f"=== Done. {len(all_results)} generation(s) attempted. Manifest -> {manifest_path} ===")
