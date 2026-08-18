"""
Shared gTTS-based audio synthesis, used by both generate_lesson_media.py and
openrouter_pipeline.py's generate_audio tools (gTTS is free regardless of
platform, so this lives in one place instead of being duplicated).

Voice/accent control: gTTS has no distinct named voices, but its `tld`
parameter selects which Google Translate regional domain serves the request,
producing audibly different English accents (see VOICES below). Sinhala only
has one voice regardless of tld.

A dedicated Sinhala model (SinhalaVITS-TTS-F1) was tried here instead of
gTTS's Sinhala voice, which real listening tests confirmed is genuinely
unintelligible. The dedicated model sounded "very good" in the same real
listening test -- but its dependency stack (torch/torchaudio/torchcodec/
coqui-tts, plus a ~950MB checkpoint) pushed this app over Streamlit
Community Cloud's free-tier resource limits in production and took the app
down. Reverted back to gTTS-only for Sinhala to restore service. The
integration is fully documented (code, real test results, exact dependency
list) in project memory if this gets revisited on a plan/host with more
headroom -- don't rebuild it from scratch.

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
            # Not Path.replace() -- that's a plain os.rename(), which raises
            # "Invalid cross-device link" (EXDEV) on Streamlit Cloud, where the
            # system temp dir and this app's working directory are different
            # filesystems/mounts. shutil.move() does the same fast rename when
            # possible but transparently falls back to copy+delete across a
            # filesystem boundary -- confirmed as a real failure in production,
            # not a hypothetical. This fix is independent of the Sinhala revert
            # above and stays regardless.
            shutil.move(tmp_path, target)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return str(target)
