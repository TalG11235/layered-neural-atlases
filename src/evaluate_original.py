#!/usr/bin/env python3
"""
Evaluate an experiment by comparing the MP4 input_video inside the experiment to
the ORIGINAL frames directory provided by the user.

This mirrors the outputs of evaluate_experiment.py but writes them under:
  <experiment_dir> / "evaluate original"/

Outputs:
  evaluate original/per_frame.csv
  evaluate original/summary.json
  evaluate original/frames_input/*.png
  evaluate original/frames_original/*.png

Metrics per frame:
  * PSNR
  * LPIPS (if 'lpips' is installed)
  * Flow EPE difference between consecutive frames (DIS optical flow)
  * CLIP cosine similarity (if CLIP is installed)
  * bpp computed from newest package ZIP (for parity with evaluate_experiment)
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

import torch

# Optional deps
try:
    import lpips  # type: ignore
except Exception:
    lpips = None

clip = None
try:
    import clip  # openai-clip
except Exception:
    try:
        import clip_anytorch as clip  # fallback if available
    except Exception:
        clip = None


# ----------------- Utilities -----------------

def find_input_video(exp_dir: Path) -> Path:
    path = exp_dir / "training results" / "input_video.mp4"
    if not path.exists():
        raise FileNotFoundError(f"Could not find input video: {path}")
    return path


def open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return cap, w, h, n, fps


def list_image_paths(frames_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = [p for p in frames_dir.iterdir() if p.suffix.lower() in exts and p.is_file()]
    # Natural sort by stem if numeric, else lexicographic
    def key_fn(p: Path):
        s = p.stem
        try:
            return (0, int(s))
        except Exception:
            return (1, s)
    files.sort(key=key_fn)
    if not files:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")
    return files


def ensure_out_dirs(exp_dir: Path) -> Tuple[Path, Path, Path]:
    base = exp_dir / "evaluate original"
    base.mkdir(parents=True, exist_ok=True)
    inp_dir = base / "frames_input"
    orig_dir = base / "frames_original"
    inp_dir.mkdir(parents=True, exist_ok=True)
    orig_dir.mkdir(parents=True, exist_ok=True)
    return base, inp_dir, orig_dir


def save_frame_rgb_png(rgb_img: np.ndarray, out_path: Path):
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr)


def ensure_same_size(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    return a, b


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.PSNR(a, b))


def to_torch_img(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img_rgb_uint8).permute(2, 0, 1).float() / 255.0
    t = t * 2.0 - 1.0
    return t.unsqueeze(0)


def make_dis_flow():
    return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def flow_epe(flow: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(flow ** 2, axis=2))


def compute_flow_EPE_diff(flow_alg, a_prev, a_cur, b_prev, b_cur) -> float:
    a_prev_g = cv2.cvtColor(a_prev, cv2.COLOR_RGB2GRAY)
    a_cur_g  = cv2.cvtColor(a_cur,  cv2.COLOR_RGB2GRAY)
    b_prev_g = cv2.cvtColor(b_prev, cv2.COLOR_RGB2GRAY)
    b_cur_g  = cv2.cvtColor(b_cur,  cv2.COLOR_RGB2GRAY)

    fa = flow_alg.calc(a_prev_g, a_cur_g, None)
    fb = flow_alg.calc(b_prev_g, b_cur_g, None)

    ea = flow_epe(fa)
    eb = flow_epe(fb)
    return float(np.mean(np.abs(ea - eb)))


def lowres_gray(img_rgb: np.ndarray, target_wh=(64, 36)) -> np.ndarray:
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, target_wh, interpolation=cv2.INTER_AREA)


def frame_distance(a_gray_lr: np.ndarray, b_gray_lr: np.ndarray) -> float:
    return float(np.mean(np.abs(a_gray_lr.astype(np.float32) - b_gray_lr.astype(np.float32))))


def read_frame_at_index(cap, idx: int) -> Optional[np.ndarray]:
    if idx < 0:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_clip(device: str = "cpu"):
    if clip is None:
        raise RuntimeError("CLIP not installed. Install either 'openai-clip' or 'clip-anytorch'.")
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    return model, preprocess


def clip_embed(model, preprocess, img_rgb_uint8: np.ndarray, device: str):
    from PIL import Image as PILImage
    pil = PILImage.fromarray(img_rgb_uint8)
    with torch.no_grad():
        t = preprocess(pil).unsqueeze(0).to(device)
        feat = model.encode_image(t)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu()


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.sum(x * y))


def find_latest_package_zip(exp_dir: Path) -> Optional[Path]:
    return None  # unused in this variant
    root = exp_dir / "compression results"
    if not root.exists():
        return None
    zips = list(root.rglob("*.zip"))
    if not zips:
        return None
    return max(zips, key=lambda p: p.stat().st_mtime)


# ------------- Alignment helpers -------------

def estimate_offset_frames(
    cap_inp, fps_i: float, orig_paths: List[Path],
    search_range_sec: float = 1.0, search_step_sec: float = 0.02
) -> int:
    """ Estimate a global offset in *frames* for the original sequence
        relative to the input video. Positive means: use later original frames. """
    if fps_i <= 0 or len(orig_paths) == 0:
        return 0

    # Sample some input indices
    n_i = int(cap_inp.get(cv2.CAP_PROP_FRAME_COUNT))
    samples = min(24, n_i)
    if samples <= 0:
        return 0
    inp_idxs = [int(round(i*(n_i-1)/(samples-1))) for i in range(samples)] if samples > 1 else [0]

    # Preload low-res gray for inputs
    inp_lr: Dict[int, np.ndarray] = {}
    for i in inp_idxs:
        f = read_frame_at_index(cap_inp, i)
        if f is not None:
            inp_lr[i] = lowres_gray(f)

    # Candidate offsets (in frames)
    offsets = np.arange(-search_range_sec, search_range_sec + 1e-9, search_step_sec)
    offsets_frames = [int(round(o * fps_i)) for o in offsets]

    best_off = 0
    best_score = float("inf")

    # Preload original lowres frames as well (up to used ones)
    def read_orig(j: int) -> Optional[np.ndarray]:
        if j < 0 or j >= len(orig_paths):
            return None
        img = cv2.cvtColor(cv2.imread(str(orig_paths[j])), cv2.COLOR_BGR2RGB)
        return img

    for off in offsets_frames:
        score_acc = 0.0
        cnt = 0
        for i in inp_lr.keys():
            j_nom = i + off
            best_local = float("inf")
            for dj in (-1, 0, 1):
                j = j_nom + dj
                fr = read_orig(j)
                if fr is None:
                    continue
                d = frame_distance(inp_lr[i], lowres_gray(fr))
                best_local = min(best_local, d)
            if best_local < float("inf"):
                score_acc += best_local
                cnt += 1
        if cnt > 0:
            score = score_acc / cnt
            if score < best_score:
                best_score = score
                best_off = off

    cap_inp.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return int(best_off)


# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate input_video.mp4 vs ORIGINAL frames directory.")
    ap.add_argument("--experiment_dir", required=True, help="Path to the experiment directory.")
    ap.add_argument("--original_frames_dir", required=True, help="Path to the directory with the source/original frames.")
    ap.add_argument("--align", choices=["index", "auto"], default="index",
                    help="Alignment mode. 'index' compares frame i to frame i. 'auto' searches a small global offset.")
    ap.add_argument("--auto_range_sec", type=float, default=1.0, help="Offset search range for 'auto' (±seconds).")
    ap.add_argument("--auto_step_sec", type=float, default=0.02, help="Offset search step (seconds).")
    ap.add_argument("--local_window", type=int, default=0, help="Optional local refinement window ±k around mapped index.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_frames", type=int, default=0, help="Optional cap on number of frames (0 = all).")
    args = ap.parse_args()

    exp_dir = Path(args.experiment_dir).resolve()
    orig_dir = Path(args.original_frames_dir).resolve()

    base_out, frames_inp_dir, frames_orig_dir = ensure_out_dirs(exp_dir)

    inp_path = find_input_video(exp_dir)
    cap_inp, w_i, h_i, n_i, fps_i = open_video(inp_path)

    if fps_i <= 0:
        print("[warn] Input FPS reported 0; assuming 30 FPS.")
        fps_i = 30.0

    orig_paths = list_image_paths(orig_dir)
    n_o = len(orig_paths)


    # bpp: based on size (in bits) of training results/input_video.mp4
    bits_total = Path(inp_path).stat().st_size * 8.0
    bpp_source = "input_video_mp4"
    # Choose evaluation size
    # If we will resize, use min resolution
    sample_orig = cv2.cvtColor(cv2.imread(str(orig_paths[0])), cv2.COLOR_BGR2RGB)
    w = min(w_i, sample_orig.shape[1])
    h = min(h_i, sample_orig.shape[0])

    # Alignment
    offset_frames = 0
    if args.align == "auto":
        print(f"[eval-orig] Searching offset in ±{args.auto_range_sec}s, step {args.auto_step_sec}s ...")
        offset_frames = estimate_offset_frames(cap_inp, fps_i, orig_paths, args.auto_range_sec, args.auto_step_sec)
        print(f"[eval-orig] Estimated offset (orig - input) in frames: {offset_frames:+d}")

    # Setup metrics
    use_lpips = lpips is not None
    if use_lpips:
        LossLPIPS = lpips.LPIPS(net='alex').to(args.device).eval()
    else:
        print("[warn] lpips not found; LPIPS will be NaN. pip install lpips")

    use_clip = True
    try:
        clip_model, clip_preproc = load_clip(args.device)
    except Exception as e:
        print(f"[warn] CLIP not available ({e}); CLIP similarity will be NaN.")
        use_clip = False
        clip_model = clip_preproc = None

    flow_alg = make_dis_flow()

    # Determine number of comparable frames
    if args.align == "index":
        n_frames = min(n_i, n_o)
        start_orig = 0
    else:
        # offset shifts original indices relative to input
        if offset_frames >= 0:
            start_orig = offset_frames
            n_frames = min(n_i, n_o - offset_frames)
        else:
            # negative offset means start input later
            start_orig = 0
            n_frames = min(n_i + offset_frames, n_o)  # offset_frames is negative
    n_frames = max(0, n_frames)
    if args.max_frames and args.max_frames > 0:
        n_frames = min(n_frames, args.max_frames)

    if n_frames == 0:
        print("[error] No overlapping frames to compare after alignment.")
        return

    # bpp denominator (for parity)
    if not math.isnan(bits_total):
        bpp = bits_total / float(w * h * n_frames)
    else:
        bpp = float("nan")

    rows = []
    prev_inp = prev_orig = None

    last_j_used = -1  # for monotonicity with local refinement

    pbar = tqdm(range(n_frames), desc="Per-frame metrics (orig)")
    for idx in pbar:
        # Input frame index
        i = idx if args.align == "index" or offset_frames >= 0 else idx - offset_frames
        a = read_frame_at_index(cap_inp, i)
        if a is None:
            break

        # Nominal original index
        j_nom = start_orig + idx if args.align == "index" or offset_frames >= 0 else start_orig + idx + offset_frames

        # Optional local refinement around j_nom
        best_j = None
        best_d = float("inf")
        a_lr = lowres_gray(a)

        for dj in range(-args.local_window, args.local_window + 1):
            j = j_nom + dj
            if j < 0 or j >= n_o:
                continue
            fr = cv2.cvtColor(cv2.imread(str(orig_paths[j])), cv2.COLOR_BGR2RGB)
            d = frame_distance(a_lr, lowres_gray(fr))
            if d < best_d and j >= last_j_used:
                best_d = d
                best_j = j

        if best_j is None:
            break

        b = cv2.cvtColor(cv2.imread(str(orig_paths[best_j])), cv2.COLOR_BGR2RGB)
        last_j_used = best_j

        a, b = ensure_same_size(a, b)

        # Save exact compared frames
        fname = f"{idx:06d}.png"
        save_frame_rgb_png(a, frames_inp_dir / fname)
        save_frame_rgb_png(b, frames_orig_dir / fname)

        # Metrics
        m_psnr = psnr(a, b)

        if use_lpips:
            with torch.no_grad():
                ta = to_torch_img(a).to(args.device)
                tb = to_torch_img(b).to(args.device)
                m_lpips = float(LossLPIPS(ta, tb).item())
        else:
            m_lpips = float("nan")

        if prev_inp is not None and prev_orig is not None:
            m_flow = compute_flow_EPE_diff(flow_alg, prev_inp, a, prev_orig, b)
        else:
            m_flow = float("nan")

        if use_clip:
            fa = clip_embed(clip_model, clip_preproc, a, args.device)
            fb = clip_embed(clip_model, clip_preproc, b, args.device)
            m_clip = cosine_similarity(fa, fb)
        else:
            m_clip = float("nan")

        rows.append({
            "frame": idx,
            "psnr": m_psnr,
            "lpips": m_lpips,
            "flow_epe_diff": m_flow,
            "clip_cosine": m_clip,
            "bpp": bpp,
            "input_index": int(i),
            "original_index": int(best_j),
            "t_input_sec": i / fps_i,
            "t_original_sec": (best_j) / fps_i,  # original has no fps; we map by input cadence for reference
        })

        prev_inp, prev_orig = a, b

    cap_inp.release()

    # Save per-frame CSV
    import csv
    csv_path = base_out / "per_frame.csv"
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.DictWriter(
            f,
            fieldnames=[
                "frame", "psnr", "lpips", "flow_epe_diff", "clip_cosine", "bpp",
                "input_index", "original_index", "t_input_sec", "t_original_sec"
            ]
        )
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow(r)

    # Averages
    def mean_of(key):
        vals = [r[key] for r in rows if not (isinstance(r[key], float) and math.isnan(r[key]))]
        return float(sum(vals) / max(len(vals), 1)) if vals else float("nan")

    summary = {
        "frames_used": len(rows),
        "resolution_used": {"width": int(w), "height": int(h)},
        "input_video": str(inp_path),
        "original_frames_dir": str(orig_dir),
        "frames_input_reported": int(n_i),
        "frames_original_reported": int(n_o),
        "local_refine_window": int(args.local_window),
        "bpp": rows[0]["bpp"] if rows else float("nan"),
        "bpp_source": bpp_source,
        "avg_psnr": mean_of("psnr"),
        "avg_lpips": mean_of("lpips"),
        "avg_flow_epe_diff": mean_of("flow_epe_diff"),
        "avg_clip_cosine": mean_of("clip_cosine"),
    }

    json_path = base_out / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[eval-orig] Frames written: {len(rows)} | Input FPS: {fps_i:.4g}")
    print(f"[eval-orig] Frame dumps -> input: {frames_inp_dir} | original: {frames_orig_dir}")
    print(f"[eval-orig] Wrote: {csv_path}")
    print(f"[eval-orig] Wrote: {json_path}")
    print("[eval-orig] Done.")


if __name__ == "__main__":
    main()
