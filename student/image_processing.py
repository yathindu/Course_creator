"""
Context-aware local background removal, shared by both pipelines
(generate_lesson_media.py and openrouter_pipeline.py).

Plain rembg (U2-Net) only keeps the single most salient object and strips
everything else -- including ground/water the subject is standing in, which
is usually part of the intended scene, not backdrop. This module combines
three techniques instead of relying on rembg alone:

- CLIPSeg (open-vocabulary, text-prompted segmentation) scores each region
  against the subject's description plus a fixed list of common ground
  materials (grass, water, sand, etc), identifying *which* regions belong
  with the subject. Its own output is a coarse, low-resolution heatmap.
- felzenszwalb superpixel segmentation supplies pixel-accurate region
  boundaries that CLIPSeg's scores get snapped to (keep a whole region if
  its mean CLIPSeg score clears a threshold).
- rembg's own mask is unioned in for the subject's own edges, since its
  dedicated matting model gives cleaner boundaries there than CLIPSeg would.
- A distance cap keeps only ground material within a fixed radius of the
  subject, regardless of how CLIPSeg/felzenszwalb score the rest of the frame.

Net effect: grass/water survive right around the subject, sky/distant backdrop
still gets removed, and a subject with no relevant ground material still gets
a plain, tight cutout.

Why the distance cap is load-bearing, not just tuning: felzenszwalb has no
internal edges to split on inside a smooth, low-contrast area (e.g. a blurred
bokeh pond in a shallow-depth-of-field "natural" style photo), so it can merge
the entire visible background into ONE giant superpixel. If that single region
happens to touch the subject and legitimately scores well for "water" (it IS
water, just far away too), a region-level-only threshold keeps the whole
thing -- confirmed by a real test: a duck-on-a-pond photo came back with 0%
removed, the entire frame kept, because one felzenszwalb segment covered 59%
of the image and scored 0.64 for "water". CLIPSeg's score alone can't
distinguish "the water at the duck's feet" from "the same water stretching to
the horizon" -- only physical distance from the subject can. The cap also
incidentally makes cartoon cutouts more correct too: it keeps the grass patch
a subject is actually standing on and drops distant scenery (e.g. bushes at
the frame edges), rather than keeping an entire background ground plane.

Setup:
    pip install rembg onnxruntime scikit-image scipy numpy torch transformers
First calls download rembg's ~176MB U2-Net model and CLIPSeg's ~600MB weights
(CIDAS/clipseg-rd64-refined) -- both one-time, cached locally after.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as tf
from PIL import Image
from rembg import remove
from scipy import ndimage
from skimage.segmentation import felzenszwalb
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

# Common ground/context materials to always check for alongside the subject
# itself -- these are what the subject is usually standing/swimming/sitting in,
# and the whole reason plain rembg fails: it only ever models "the subject",
# never what it's resting on. Deliberately no "sky"/"background"/"wall" here --
# those genuinely should be removed.
GROUND_PROMPTS = ["grass", "water", "sand", "snow", "dirt ground", "floor", "leaves", "rocks", "flowers"]

# felzenszwalb params -- larger scale/min_size merge grass/water into fewer,
# larger regions so a single CLIPSeg score decides a whole region's fate,
# instead of fragmenting it into many small islands that each need their own
# high score to survive.
_SEGMENT_SCALE = 300
_SEGMENT_SIGMA = 0.8
_SEGMENT_MIN_SIZE = 200
_REGION_SCORE_THRESHOLD = 0.3
_EDGE_SOFTEN_SIGMA = 1.5

# How far (as a fraction of the image diagonal) ground material is allowed to extend
# from the subject before it's cut off, regardless of its CLIPSeg score. See the module
# docstring -- without this, a single low-contrast felzenszwalb region (e.g. a whole
# blurred pond) can span most of the frame and get kept in full.
_MAX_GROUND_DISTANCE_FRAC = 0.10

_clipseg_processor = None
_clipseg_model = None


def _get_clipseg():
    global _clipseg_processor, _clipseg_model
    if _clipseg_model is None:
        _clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        _clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    return _clipseg_processor, _clipseg_model


def remove_background(image_path, subject_text=""):
    """Overwrite the image at image_path with a background-removed (RGBA) version
    that keeps whatever ground/water/etc the subject is standing in, and only
    strips regions genuinely disconnected from it. subject_text should be the
    same description used to generate the image (e.g. "a cow eating grass in a
    field") -- it's used as an extra CLIPSeg prompt alongside GROUND_PROMPTS so
    the subject's own region is recognized on top of rembg's mask. Falls back to
    a plain rembg-only cutout if subject_text isn't given."""
    path = Path(image_path)
    img = Image.open(path).convert("RGB")
    img_arr = np.array(img)
    w, h = img.size

    seed_alpha = np.array(remove(img))[:, :, 3]
    keep_mask = seed_alpha > 128

    processor, model = _get_clipseg()
    prompts = ([subject_text] if subject_text else []) + GROUND_PROMPTS
    inputs = processor(text=prompts, images=[img] * len(prompts), padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits)
    if probs.dim() == 2:
        probs = probs.unsqueeze(0)
    probs_resized = tf.interpolate(probs.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
    best_score = probs_resized.max(dim=0).values.numpy()

    # Distance from every pixel to the nearest subject pixel, in pixels -- caps how far a
    # well-scoring ground region is allowed to extend (see module docstring for why this
    # is necessary, not optional: a real duck-on-a-pond photo came back 0% removed without it).
    dist_from_subject = ndimage.distance_transform_edt(~keep_mask)
    max_ground_distance = _MAX_GROUND_DISTANCE_FRAC * np.hypot(h, w)

    segments = felzenszwalb(img_arr, scale=_SEGMENT_SCALE, sigma=_SEGMENT_SIGMA, min_size=_SEGMENT_MIN_SIZE)
    for seg_id in np.unique(segments):
        region = segments == seg_id
        if best_score[region].mean() > _REGION_SCORE_THRESHOLD:
            keep_mask[region & (dist_from_subject <= max_ground_distance)] = True

    alpha = ndimage.gaussian_filter(keep_mask.astype(np.float32) * 255, sigma=_EDGE_SOFTEN_SIGMA)
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)

    result = np.dstack([img_arr, alpha])
    Image.fromarray(result, mode="RGBA").save(path)
    return str(path)


# Video models take opaque images, not RGBA -- animating a background-removed
# (transparent) image needs a real backdrop composited in first, or the
# transparent regions render unpredictably depending on the model.
BACKGROUND_COLORS = {"Black": (0, 0, 0), "White": (255, 255, 255)}


def has_transparency(image_path):
    """True if the image has an alpha channel with any non-fully-opaque pixel."""
    img = Image.open(image_path)
    if img.mode != "RGBA":
        return False
    return img.getchannel("A").getextrema()[0] < 255


def flatten_on_background(image_path, color_name, out_path):
    """Composite a transparent image onto a solid color and save as opaque RGB."""
    img = Image.open(image_path).convert("RGBA")
    bg = Image.new("RGBA", img.size, BACKGROUND_COLORS[color_name] + (255,))
    Image.alpha_composite(bg, img).convert("RGB").save(out_path)
    return str(out_path)
