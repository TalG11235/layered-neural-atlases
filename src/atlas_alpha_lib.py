# atlas_alpha_lib.py
import cv2, numpy as np

def read_rgb(p):
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    if bgr is None: raise FileNotFoundError(p)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

def read_rgba_or_rgb(p):
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(p)
    if img.ndim == 3 and img.shape[2] == 4:
        rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255.0
        return rgba[..., :3], rgba[..., 3]
    if img.ndim == 3 and img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb, None
    g = img.astype(np.float32) / 255.0
    return np.stack([g, g, g], -1), None

def write_rgb(p, rgb):
    bgr = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(p, bgr)

def write_rgba(p, rgba):
    bgra = cv2.cvtColor((np.clip(rgba, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(p, bgra)

def key_black_make_alpha(rgb, threshold=12, blur_sigma=0.8):
    gray8 = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    a = (gray8 > threshold).astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), blur_sigma)
    return np.clip(a, 0, 1)

def color_bleed(
    rgb, alpha,
    bleed_px=12,
    binarize_thr=0.5,
    inner_offset_px=3,
    unpremultiply=True,
):
    """
    Extend colors around the silhouette edge.
    - Overwrites a ring OUTSIDE the mask (≈ bleed_px)
    - Overwrites a band INSIDE the mask (≈ inner_offset_px)
    Donor colors are sampled inner_offset_px pixels deeper inside the mask.

    Inputs
      rgb:   float32 HxWx3 in [0,1] (straight or premultiplied; we'll fix)
      alpha: float32 HxW    in [0,1]
    Output
      float32 HxWx3 in [0,1], STRAIGHT RGB with solid edge colors
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    assert alpha.ndim == 2 and alpha.shape == rgb.shape[:2]

    H, W = alpha.shape
    result = rgb.copy()

    # 1) Hard silhouette to avoid ambiguity from feathered alpha
    fg = (alpha >= binarize_thr).astype(np.uint8)
    if fg.sum() == 0 or (bleed_px <= 0 and inner_offset_px <= 0):
        return result

    # 2) If input might be premultiplied, unpremultiply INSIDE the mask
    work = rgb.copy()
    if unpremultiply:
        eps = 1e-6
        a3 = np.maximum(alpha, eps)[..., None]
        work = np.where(fg[..., None] == 1, work / a3, work)
        work = np.clip(work, 0.0, 1.0)

    # 3) Define regions
    inner_px = max(1, int(inner_offset_px))
    k_in  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*inner_px+1, 2*inner_px+1))
    fg_eroded = cv2.erode(fg, k_in, iterations=1)

    # Outside ring: dilate(fg, bleed_px) - fg
    if bleed_px > 0:
        k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*bleed_px+1, 2*bleed_px+1))
        fg_dil = cv2.dilate(fg, k_out, iterations=1)
        outside_ring = (fg_dil.astype(bool) & (~fg.astype(bool)))
    else:
        outside_ring = np.zeros_like(fg, dtype=bool)

    # Inside band: fg - erode(fg, inner_offset_px)
    inside_band = (fg.astype(bool) & (~fg_eroded.astype(bool))) if inner_px > 0 else np.zeros_like(fg, bool)

    # 4) Build donor set = eroded foreground (deeper interior). Fallback to fg if erosion wiped it out.
    donor = fg_eroded.copy()
    if donor.sum() == 0:
        donor = fg

    # 5) Nearest-donor lookup for *all* pixels (labels give nearest donor index)
    _, labels = cv2.distanceTransformWithLabels(
        1 - donor, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    labels -= 1
    ys, xs = np.nonzero(donor)
    if ys.size == 0:
        return result  # safety

    coords = np.stack([ys, xs], axis=1)         # (N, 2)
    nearest = np.clip(labels, 0, len(coords)-1) # (H, W)
    nyx = coords[nearest]                       # (H, W, 2)
    ny, nx = nyx[..., 0], nyx[..., 1]

    # 6) Overwrite outside ring and inside band with donor colors (full-strength)
    to_fill = outside_ring | inside_band
    if to_fill.any():
        result[to_fill] = work[ny[to_fill], nx[to_fill]]

    return np.clip(result, 0.0, 1.0)

# math helpers
def premultiply(rgb, alpha):              # use this AFTER compression if you need premul
    return rgb * alpha[..., None]

def unpremultiply_for_png(rgb_pm, alpha, eps=1e-6):  # if you want straight-alpha on disk
    a = np.clip(alpha, 0.0, 1.0)
    denom = np.maximum(a, eps)[..., None]
    return np.where(a[..., None] > eps, rgb_pm / denom, 0.0)
