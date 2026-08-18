"""
Shared audio synthesis, used by both generate_lesson_media.py and
openrouter_pipeline.py's generate_audio tools.

English audio uses gTTS (free, one call, "voice" picks an accent via its
`tld` param -- see VOICES below).

Sinhala audio uses a dedicated model, SinhalaVITS-TTS-F1 (see
sinhala_vits.py), not gTTS. gTTS's Sinhala voice was tested and found
genuinely unintelligible by a real listener -- not a style preference, an
actual comprehension failure -- so this is a hard swap for Sinhala only, not
an option, not something the `voice` parameter toggles. If SinhalaVITS fails
for any reason (its dependency stack is real added weight: torch/torchaudio/
torchcodec/coqui-tts, plus a lazy ~950MB checkpoint download on first use),
this falls back to gTTS automatically rather than failing the whole
generation -- gTTS's Sinhala is still technically speech, just not good
speech, so a fallback beats a hard error.

Volume control: applied uniformly after synthesis regardless of which engine
produced the audio, via ffmpeg (not pydub, which depends on the stdlib
`audioop` module removed in Python 3.13+).

Setup:
    pip install gtts
    ffmpeg must be on PATH
    (Sinhala path additionally needs coqui-tts[codec] and torchaudio -- see
    requirements.txt; downloads its own model weights on first real use, see
    sinhala_vits.py for why)
"""

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from gtts import gTTS

from student.sinhala_vits import synthesize_sinhala

VOICES = {
    "us": "com",       # United States
    "uk": "co.uk",      # United Kingdom
    "australia": "com.au",
    "india": "co.in",
    "canada": "ca",
    "ireland": "ie",
    "south_africa": "co.za",
}


def _synthesize_gtts(text, language, voice, retries=3):
    tld = VOICES.get(voice, "com") if language == "en" else "com"  # tld only affects English
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                gTTS(text=text, lang=language, tld=tld).save(tmp.name)
                return tmp.name
        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(1.5)
    raise RuntimeError(f"gTTS failed after {retries} attempts: {last_error}")


def synthesize_audio(text, language, out_path, voice="us", volume_db=0.0, retries=3):
    """Generate speech for `text`, apply accent (`voice`, English only) and
    volume (`volume_db`, positive = louder / negative = quieter), save as MP3.
    Sinhala routes through SinhalaVITS (see module docstring), with an
    automatic gTTS fallback if that fails."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = None
    if language == "si":
        try:
            tmp_path = synthesize_sinhala(text)
        except Exception as e:
            print(f"SinhalaVITS synthesis failed ({e!r}) -- falling back to gTTS for: {text!r}")

    if tmp_path is None:
        tmp_path = _synthesize_gtts(text, language, voice, retries=retries)

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
            # not a hypothetical.
            shutil.move(tmp_path, target)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return str(target)
