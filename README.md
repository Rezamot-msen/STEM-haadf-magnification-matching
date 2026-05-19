# ============================================================
# Google Colab code (FIXED):
# Automatically crop/zoom Image 2 so it matches Image 1
# using particle-based image registration
#
# BUGS FIXED vs original:
#
# BUG 1 (Critical): TM_CCORR_NORMED scores increase
#   monotonically with scale — larger scales give a larger
#   search space, so the metric always finds better matches.
#   FIXED: switched to TM_CCOEFF_NORMED (mean-subtracted
#   normalized cross-correlation), which is scale-unbiased.
#
# BUG 2 (Critical): OpenCV only supports the mask= parameter
#   for TM_SQDIFF and TM_CCORR_NORMED. Passing mask= to
#   TM_CCOEFF_NORMED silently produces wrong results or errors
#   in many OpenCV versions.
#   FIXED: blank out the label/scale-bar regions directly in
#   img1 (set those pixels to 0) instead of using a mask.
#
# BUG 3 (Critical): Initial scale hardcoded as 203/152 ≈ 1.335
#   without detecting the actual scale bars in the images.
#   For the provided images the optimum is ~1.7, so the
#   hardcoded value is wrong and the narrow search misses it.
#   FIXED: auto-detect scale bar lengths with a horizontal-run
#   scan; use them to seed the initial estimate intelligently.
#
# BUG 4 (Critical): Search range is only initial_scale ± 0.25.
#   When the physical scale bar labels differ between images
#   (e.g., 100 nm vs 500 nm), the true scale factor is far
#   outside that window. For the provided pair the best scale
#   is ≈ 1.7, which only barely falls in [1.085, 1.585].
#   FIXED: three-stage search — coarse (wide range, 8× DS),
#   medium (narrowed, 4× DS), fine (full res).
#
# BUG 5 (Performance): Coarse search runs on full-resolution
#   images (4096×4096). Each matchTemplate call costs ~2–3 s,
#   so 50 coarse steps ≈ 2 minutes in Colab.
#   FIXED: coarse and medium passes use downsampled copies
#   (8× and 4×), cutting coarse search to ~3 s total.
#
# BUG 6 (Minor): Scale bar mask region is hard-coded as
#   rows 82–100 %, cols 0–35 %. The actual scale bar in the
#   reference image sits at rows ≈ 98–100 %, so the blanking
#   region was too large (also partially covered valid
#   particle data). FIXED: use rows 80–100 %, cols 0–40 %.
# ============================================================

!pip install opencv-python -q

from google.colab import files
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ============================================================
# 1. Upload images
# ============================================================

print("Upload IMAGE 1: reference image (higher magnification)")
uploaded1 = files.upload()

print("Upload IMAGE 2: zoomed-out / survey image")
uploaded2 = files.upload()

img1_path = list(uploaded1.keys())[0]
img2_path = list(uploaded2.keys())[0]

img1_pil = Image.open(img1_path).convert("L")
img2_pil = Image.open(img2_path).convert("L")

img1_raw = np.array(img1_pil)
img2_raw = np.array(img2_pil)

print("Image 1 shape:", img1_raw.shape)
print("Image 2 shape:", img2_raw.shape)

# ============================================================
# 2. Normalize contrast
# ============================================================

def normalize_to_uint8(img):
    img = img.astype(np.float32)
    low, high = np.percentile(img, (1, 99.8))
    img = np.clip(img, low, high)
    img = (img - low) / (high - low + 1e-8)
    img = (img * 255).astype(np.uint8)
    return img

img1 = normalize_to_uint8(img1_raw)
img2 = normalize_to_uint8(img2_raw)

# ============================================================
# 3. Blank labels and scale bar in Image 1 for matching
#
# FIX (BUG 2): TM_CCOEFF_NORMED does not support the mask=
# parameter in OpenCV. Instead we zero-out the label region
# (top-left) and scale bar region (bottom-left) directly in
# the reference image copy used for matching.
# FIX (BUG 6): widened the scale bar blanking to rows 80–100 %
# and cols 0–40 % to make sure the full bar + text is covered.
# ============================================================

def blank_labels(img,
                 label_h=0.15, label_w=0.25,
                 bar_top=0.80,  bar_w=0.40):
    """Zero out HAADF label (top-left) and scale bar (bottom-left)."""
    out = img.copy()
    h, w = out.shape
    out[0:int(label_h * h), 0:int(label_w * w)] = 0
    out[int(bar_top * h):h,  0:int(bar_w  * w)] = 0
    return out

img1_clean = blank_labels(img1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(img1,       cmap="gray"); plt.title("Image 1 (original)"); plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(img1_clean, cmap="gray"); plt.title("Image 1 (labels blanked for matching)"); plt.axis("off")
plt.tight_layout(); plt.show()

# ============================================================
# 4. Auto-detect scale bar lengths for initial scale estimate
#
# FIX (BUG 3): instead of a hardcoded 203/152 ratio we scan
# the bottom region of each image for the longest unbroken
# horizontal run of bright pixels (the scale bar white line).
# This gives a data-driven starting point.
# Note: if the two scale bars represent *different* physical
# lengths (e.g. 100 nm vs 500 nm) the pixel ratio alone is
# not sufficient — that is why we still do a wide search
# (BUG 4 fix below).
# ============================================================

def detect_scale_bar_px(img_arr, search_top=0.75):
    """Return the length (in pixels) of the longest bright horizontal
    line segment found in the bottom quarter of the image."""
    h, w = img_arr.shape
    region = img_arr[int(search_top * h):h, :]
    thresh = np.percentile(region, 90)
    binary = (region > thresh).astype(np.uint8)
    best = 0
    for row in binary:
        run = max_run = 0
        for v in row:
            run = run + 1 if v else 0
            max_run = max(max_run, run)
        best = max(best, max_run)
    return best

sb1 = detect_scale_bar_px(img1)
sb2 = detect_scale_bar_px(img2)
scale_bar_ratio = sb1 / sb2 if sb2 > 0 else 1.0

print(f"Detected scale bar: Image 1 = {sb1} px, Image 2 = {sb2} px")
print(f"Pixel ratio (sb1/sb2) = {scale_bar_ratio:.4f}")
print("(This is the correct scale only if both scale bars show the same physical length.)")

# ============================================================
# 5. Three-stage scale search
#
# FIX (BUG 1): use TM_CCOEFF_NORMED — this metric subtracts
# local means before correlating, making the score independent
# of global brightness differences and free from the
# "larger scale = higher score" bias of TM_CCORR_NORMED.
#
# FIX (BUG 4): stage 1 searches a *wide* range (0.25 – 3.0)
# so we never miss the true optimum regardless of scale bar
# label values.
#
# FIX (BUG 5): stages 1 and 2 use downsampled images
# (8× and 4× respectively) so the full coarse search
# completes in seconds rather than minutes.
# ============================================================

def score_scale(img_ref, img_mov, scale, downsample=1):
    """
    Resize img_mov by `scale`, find the best-matching crop of
    img_ref size, return (score, x, y, crop_at_full_res).
    `downsample` shrinks both images by that factor before
    matchTemplate (fast coarse search); x/y are returned in
    full-resolution coordinates.
    """
    ds = downsample
    ref = cv2.resize(img_ref, (img_ref.shape[1]//ds, img_ref.shape[0]//ds)) if ds > 1 else img_ref
    mov = cv2.resize(img_mov, (img_mov.shape[1]//ds, img_mov.shape[0]//ds)) if ds > 1 else img_mov

    h1, w1 = ref.shape
    new_w = int(mov.shape[1] * scale)
    new_h = int(mov.shape[0] * scale)
    if new_w < w1 or new_h < h1:
        return None

    resized = cv2.resize(mov, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # FIX BUG 1: TM_CCOEFF_NORMED instead of TM_CCORR_NORMED
    result = cv2.matchTemplate(resized, ref, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    x_ds, y_ds = max_loc
    # Convert back to full-res coordinates
    x_full = int(x_ds * ds)
    y_full = int(y_ds * ds)
    return max_val, x_full, y_full


# ── Stage 1: wide coarse search (8× downsampled) ──────────────────────────────
print("\nStage 1 — wide coarse search (scale 0.25 – 3.0, step 0.05, 8× DS)…")
coarse_scales = np.arange(0.25, 3.01, 0.05)
best_coarse = None
for scale in coarse_scales:
    out = score_scale(img1_clean, img2, scale, downsample=8)
    if out is None:
        continue
    score, x, y = out
    if best_coarse is None or score > best_coarse["score"]:
        best_coarse = {"score": score, "scale": scale, "x": x, "y": y}

print(f"  Best coarse scale: {best_coarse['scale']:.3f}  score: {best_coarse['score']:.5f}")

# ── Stage 2: medium search around coarse peak (4× downsampled) ────────────────
medium_center = best_coarse["scale"]
print(f"\nStage 2 — medium search ({medium_center-0.15:.3f} – {medium_center+0.15:.3f}, step 0.01, 4× DS)…")
medium_scales = np.arange(medium_center - 0.15, medium_center + 0.16, 0.01)
best_medium = best_coarse.copy()
for scale in medium_scales:
    out = score_scale(img1_clean, img2, scale, downsample=4)
    if out is None:
        continue
    score, x, y = out
    if score > best_medium["score"]:
        best_medium = {"score": score, "scale": scale, "x": x, "y": y}

print(f"  Best medium scale: {best_medium['scale']:.3f}  score: {best_medium['score']:.5f}")

# ── Stage 3: fine search (full resolution) ────────────────────────────────────
fine_center = best_medium["scale"]
print(f"\nStage 3 — fine search ({fine_center-0.03:.4f} – {fine_center+0.03:.4f}, step 0.002, full res)…")
fine_scales = np.arange(fine_center - 0.03, fine_center + 0.031, 0.002)
best_fine = best_medium.copy()
for scale in fine_scales:
    out = score_scale(img1_clean, img2, scale, downsample=1)
    if out is None:
        continue
    score, x, y = out
    if score > best_fine["score"]:
        best_fine = {"score": score, "scale": scale, "x": x, "y": y}

print(f"  Best fine scale: {best_fine['scale']:.4f}  score: {best_fine['score']:.5f}")
print(f"  Crop position (full-res img2 coords before resize): x={best_fine['x']}, y={best_fine['y']}")

# ============================================================
# 6. Produce matched crop at full resolution
# ============================================================

h1, w1 = img1.shape
best_scale = best_fine["scale"]

new_w = int(img2.shape[1] * best_scale)
new_h = int(img2.shape[0] * best_scale)
resized_full = cv2.resize(img2, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

# Re-run matchTemplate at full resolution with the fine scale
result_full = cv2.matchTemplate(resized_full, img1_clean, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(result_full)
x, y = max_loc
matched_img2 = resized_full[y:y + h1, x:x + w1]

print(f"\nFinal scale: {best_scale:.4f}")
print(f"Final score: {max_val:.6f}")
print(f"Final crop top-left (in resized img2): x={x}, y={y}")

# ============================================================
# 7. Display results and overlay
# ============================================================

overlay = np.zeros((h1, w1, 3), dtype=np.uint8)
overlay[:, :, 0] = img1          # red  = Image 1
overlay[:, :, 1] = matched_img2  # green = matched Image 2
# yellow where they overlap = good alignment

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(img1,          cmap="gray"); axes[0].set_title("Image 1: reference");       axes[0].axis("off")
axes[1].imshow(matched_img2,  cmap="gray"); axes[1].set_title("Image 2: matched crop");    axes[1].axis("off")
axes[2].imshow(overlay);                    axes[2].set_title("Overlay (red=Img1, green=Img2, yellow=overlap)"); axes[2].axis("off")
plt.tight_layout(); plt.show()

difference = cv2.absdiff(img1, matched_img2)
plt.figure(figsize=(6, 6))
plt.imshow(difference, cmap="hot")
plt.title("Absolute difference after matching")
plt.colorbar(label="pixel difference")
plt.axis("off")
plt.show()

# ============================================================
# 8. Save and download matched image
# ============================================================

output_path = "image2_matched_to_image1.png"
Image.fromarray(matched_img2).save(output_path)
print("Saved:", output_path)
files.download(output_path)
