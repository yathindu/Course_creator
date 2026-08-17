"""
Standalone prompt-based video generation -- no lesson JSON required. Give it
a text scene description and it generates a video, auto-naming the output
from the prompt itself.

Reuses the same generate_video LangChain tool as generate_lesson_media.py
(Wan2.2-TI2V-5B via HF's fal-ai Inference Provider), for one-off generation
outside a lesson's asset structure.

Setup:
    pip install langchain-core huggingface_hub python-dotenv
    Put your token in a .env file next to this script:
        HF_TOKEN=your-pro-token-here

Usage:
    python generate_video.py "a cow says moo on a farm"
    python generate_video.py "a duck swimming in a pond" --out my_videos
"""

import argparse
import re
import time
from pathlib import Path

from student.generate_lesson_media import generate_video


def _slugify(text, max_len=40):
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len] or "video"


def run(prompt, out_dir):
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    filename = f"{_slugify(prompt)}_{int(time.time())}.mp4"
    out_path = out_root / filename

    print(f'Prompt: "{prompt}"')
    path = generate_video.invoke({"scene_description": prompt, "out_path": str(out_path)})
    print(f"-> {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", help="Scene description for the video")
    parser.add_argument("--out", default="generated_videos", help="Output directory (default: generated_videos)")
    args = parser.parse_args()

    run(args.prompt, args.out)
