"""
SinhalaVITS-TTS-F1 -- a dedicated Sinhala text-to-speech model (Dialog Axiata
PLC + Dialog-UoM Research Lab, huggingface.co/dialoglk/SinhalaVITS-TTS-F1),
used in place of gTTS for Sinhala audio specifically. gTTS's Sinhala voice was
tested and found genuinely unintelligible (real user listening test, not
assumed); a Microsoft edge-tts alternative was also tested and rejected for
the same reason. SinhalaVITS was verified with real generated samples and
judged "very good" by the same listening test.

This is real added weight compared to gTTS -- a ~950MB checkpoint plus the
coqui-tts dependency tree (torch/torchaudio/torchcodec/transformers/librosa)
-- a deliberate, disclosed tradeoff accepted for Sinhala audio quality
specifically. English audio stays on gTTS entirely, unaffected by this file.

Real integration gotchas found getting this running on this project's actual
environment (Python 3.14, transformers==5.15.0 pinned for CLIPSeg elsewhere
in this project) -- kept here so they aren't rediscovered:
- The model's own pinned `TTS==0.21.1` (original Coqui TTS) requires Python
  <3.12 -- hard-incompatible here. Uses the actively-maintained `coqui-tts`
  fork instead (PyPI, currently 0.27.x) -- same import path (`TTS.*`), drop-in
  for what this module needs.
- Even that fork's TTS/__init__.py unconditionally imports XTTS code that
  calls transformers.pytorch_utils.isin_mps_friendly, which this project's
  transformers version doesn't have. _patch_transformers_compat() below
  papers over just that one missing attribute before import -- not needed by
  the VITS model this module actually uses, only needed to get past an
  unrelated import at TTS package-init time.
- Needs torchaudio and coqui-tts's [codec] extra (torchcodec) -- neither is
  mentioned on the model's own card; both surfaced as real ImportErrors
  during testing.

The checkpoint is NOT committed to this repo -- ~950MB is far past a
reasonable git diff and flirts with GitHub's file-size limits. It's
downloaded lazily on first real use via huggingface_hub (cached in HF's own
cache directory, so a warm container reuses it) and kept loaded in memory
as a module-level singleton for the rest of the process's lifetime -- reloading
a 950MB checkpoint per call would be far too slow for a background job.
"""

import subprocess
import tempfile
import threading
from pathlib import Path

_REPO_ID = "dialoglk/SinhalaVITS-TTS-F1"
_CHECKPOINT_FILE = "Nipunika_210000.pth"
_CONFIG_FILE = "Nipunika_config.json"

_synth = None
_synth_lock = threading.Lock()


def _patch_transformers_compat():
    """See module docstring -- papers over one missing attribute so importing
    TTS doesn't crash on an unrelated XTTS code path this module never uses."""
    import transformers.pytorch_utils as pu
    if not hasattr(pu, "isin_mps_friendly"):
        import torch
        pu.isin_mps_friendly = lambda elements, test_elements: torch.isin(elements, test_elements)


def _get_synthesizer():
    global _synth
    if _synth is None:
        with _synth_lock:
            if _synth is None:
                _patch_transformers_compat()
                from huggingface_hub import hf_hub_download
                from TTS.utils.synthesizer import Synthesizer
                import torch

                checkpoint_path = hf_hub_download(_REPO_ID, _CHECKPOINT_FILE)
                config_path = hf_hub_download(_REPO_ID, _CONFIG_FILE)
                _synth = Synthesizer(
                    tts_checkpoint=checkpoint_path, tts_config_path=config_path,
                    use_cuda=torch.cuda.is_available(),
                )
    return _synth


def synthesize_sinhala(text):
    """Generate Sinhala speech for `text` via SinhalaVITS. Returns the path to
    a temporary MP3 file -- caller (audio_tools.py) moves/volume-adjusts it
    into its final destination, same convention the gTTS path already uses.
    Raises on any failure; callers are expected to catch and fall back to
    gTTS, given this path's real fragility (heavy ML dependency stack, lazy
    first-use download) compared to gTTS's plain HTTP call."""
    from student.sinhala_romanizer import sinhala_to_roman

    synth = _get_synthesizer()
    roman_text = sinhala_to_roman(text)
    wav = synth.tts(roman_text)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        synth.save_wav(wav, tmp_wav.name)
        tmp_wav_path = tmp_wav.name

    tmp_mp3_path = str(Path(tmp_wav_path).with_suffix(".mp3"))
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_wav_path, tmp_mp3_path],
            check=True,
        )
    finally:
        Path(tmp_wav_path).unlink(missing_ok=True)

    return tmp_mp3_path
