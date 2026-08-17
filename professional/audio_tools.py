"""
Shared gTTS-based audio synthesis, used by both generate_lesson_media.py and
openrouter_pipeline.py's generate_audio tools (gTTS is free regardless of
platform, so this lives in one place instead of being duplicated).

Voice/accent control: gTTS has no distinct named voices, but its `tld`
parameter selects which Google Translate regional domain serves the request,
producing audibly different English accents (see VOICES below). Sinhala only
has one voice regardless of tld.

Volume control: gTTS has no volume parameter, so this renders at gTTS's
default level via a temp file, then applies a dB gain by shelling out to
ffmpeg directly -- not pydub, which depends on the stdlib `audioop` module
removed in Python 3.13+.

Setup:
    pip install gtts
    ffmpeg must be on PATH
"""

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from gtts import gTTS

VOICES = {
    "us": "com",       # United States
    "uk": "co.uk",      # United Kingdom
    "australia": "com.au",
    "india": "co.in",
    "canada": "ca",
    "ireland": "ie",
    "south_africa": "co.za",
}


def synthesize_audio(text, language, out_path, voice="us", volume_db=0.0, retries=3):
    """Generate speech for `text`, apply accent (`voice`, English only) and
    volume (`volume_db`, positive = louder / negative = quieter), save as MP3."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tld = VOICES.get(voice, "com") if language == "en" else "com"  # tld only affects English

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                gTTS(text=text, lang=language, tld=tld).save(tmp.name)
                tmp_path = tmp.name
            break
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(1.5)
    else:
        raise RuntimeError(f"gTTS failed after {retries} attempts: {last_error}")

    try:
        if volume_db:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_path,
                 "-filter:a", f"volume={volume_db}dB", str(target)],
                check=True,
            )
        else:
            # Path.replace()/os.rename() fails with EXDEV ("Invalid cross-device
            # link") when the OS temp dir and the target are on different
            # filesystems -- confirmed as a real failure on Streamlit Cloud, where
            # tempfile's default dir (/tmp) is a separate mount from the app's
            # working directory. shutil.move() falls back to copy+delete when a
            # plain rename isn't possible, so it works in both cases.
            shutil.move(tmp_path, target)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return str(target)
