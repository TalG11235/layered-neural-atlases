#!/usr/bin/env python3
"""
Evaluate a Layered Neural Atlases experiment.

Inputs (inside <EXPERIMENT_DIR>):
  training results/input_video.mp4
  compression results/reconstruction/<run_name>/*.mp4   # pick most recent run

Outputs:
  eval/per_frame.csv
  eval/summary.json

Metrics (per frame):
  - PSNR
  - LPIPS (Alex)
  - Flow wrapping error (EPE diff of DIS optical flow between t and t+1)
  - CLIP similarity (cosine; higher is better)
  - bpp (bits-per-pixel; constant per frame, derived from compressed .mp4)

Notes:
  * We align streams by the minimum frame count of the two videos.
  * bpp is computed from the compressed reconstruction video file size only.
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import numpy as np
from tqdm import tqdm

import torch

# ----- LPIPS -----
try:
    import lpips  # type: ignore
except Exception as e:
    lpips = None

# ----- CLIP (try both packages) -----
clip = None
try:
    import clip  # openai-clip
except Exception:
    try:
        import clip_anytorch as clip  # clip-anytorch as fallback
    except Exception:
        clip = None


def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def find_input_video(exp_dir: Path) -> Path:
    cand = exp_dir / "training results" / "input_video.mp4"
    if not cand.exists():
        raise FileNotFoundError(f"Could not find input video at {cand}")
    return cand


def find_latest_reconstruction_mp4(exp_dir: Path) -> Path:
    recon_root = exp_dir / "compression results" / "reconstruction"
    if not recon_root.exists():
        raise FileNotFoundError(f"Missing folder: {recon_root}")

    # collect candidate run folders that contain at least one mp4
    runs = []
    for p in recon_root.iterdir():
        if p.is_dir():
            mp4s = list(p.glob("*.mp4"))
            if mp4s:
                # choose newest file mtime inside the run as run mtime
                newest = max(mp4s, key=lambda x: x.stat().st_mtime)
                runs.append((p, newest.stat().st_mtime))

    if not runs:
        raise FileNotFoundError(f"No reconstruction runs with mp4 found under {recon_root}")

    # pick run with latest mtime
    runs.sort(key=lambda x: x[1], reverse=True)
    chosen_run = runs[0][0]

    # in that run, pick the newest mp4
    mp4s = list(chosen_run.glob("*.mp4"))
    mp4 = max(mp4s, key=lambda x: x.stat().st_mtime)
    return mp4


def open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    return cap, w, h, n, fps


def read_frame(cap) -> Optional[np.ndarray]:
    ok, frame = cap.read()
    if not ok:
        return None
    # BGR->RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def to_torch_im(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    """HWC uint8 [0,255] -> NCHW float32 in [-1,1]"""
    t = torch.from_numpy(img_rgb_uint8).permute(2, 0, 1).float() / 255.0
    t = t * 2.0 - 1.0
    return t.unsqueeze(0)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    # expects RGB uint8
    return float(cv2.PSNR(a, b))


def ensure_same_size(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    return a, b


def make_dis_flow():
    # DIS optical flow (fast and reasonably accurate)
    return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def flow_epe(f: np.ndarray) -> np.ndarray:
    # f: HxWx2
    return np.sqrt(np.sum(f ** 2, axis=2))


def compute_flow_EPE_diff(flow_alg, a_prev, a_cur, b_prev, b_cur) -> float:
    # images are RGB uint8
    a_prev_g = cv2.cvtColor(a_prev, cv2.COLOR_RGB2GRAY)
    a_cur_g  = cv2.cvtColor(a_cur,  cv2.COLOR_RGB2GRAY)
    b_prev_g = cv2.cvtColor(b_prev, cv2.COLOR_RGB2GRAY)
    b_cur_g  = cv2.cvtColor(b_cur,  cv2.COLOR_RGB2GRAY)

    fa = flow_alg.calc(a_prev_g, a_cur_g, None)   # HxWx2 float32
    fb = flow_alg.calc(b_prev_g, b_cur_g, None)

    # EPE maps
    ea = flow_epe(fa)
    eb = flow_epe(fb)

    # wrapping error = mean absolute difference of EPE maps
    return float(np.mean(np.abs(ea - eb)))


def load_clip(device: str = "cpu"):
    if clip is None:
        raise RuntimeError("CLIP not installed. Install either 'openai-clip' or 'clip-anytorch'.")
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    return model, preprocess


def clip_embed(model, preprocess, img_rgb_uint8: np.ndarray, device: str):
    # img expects RGB uint8 HxWx3
    import PIL.Image as Image
    pil = Image.fromarray(img_rgb_uint8)
    with torch.no_grad():
        t = preprocess(pil).unsqueeze(0).to(device)
        feat = model.encode_image(t)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu()


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.sum(x * y))


def ensure_eval_dir(exp_dir: Path) -> Path:
    out = exp_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate experiment metrics per frame and summarize.")
    ap.add_argument("--experiment_dir", required=True, help="Path to a single experiment directory.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_frames", type=int, default=0, help="Optional cap on frames (0 = all).")
    args = ap.parse_args()

    exp_dir = Path(args.experiment_dir).resolve()
    out_dir = ensure_eval_dir(exp_dir)

    inp_path = find_input_video(exp_dir)
    rec_path = find_latest_reconstruction_mp4(exp_dir)

    print(f"[eval] input_video:     {inp_path}")
    print(f"[eval] reconstruction:  {rec_path}")

    cap_inp, w_i, h_i, n_i, fps_i = open_video(inp_path)
    cap_rec, w_r, h_r, n_r, fps_r = open_video(rec_path)

    n_frames = min(n_i, n_r)
    if args.max_frames and args.max_frames > 0:
        n_frames = min(n_frames, args.max_frames)

    # bpp from compressed reconstruction file
    bits_total = rec_path.stat().st_size * 8.0
    # NOTE: we use the minimum width/height to be consistent with resizing step
    w = min(w_i, w_r)
    h = min(h_i, h_r)
    bpp = bits_total / (w * h * n_r if n_r > 0 else 1)

    # Metrics setup
    use_lpips = lpips is not None
    if use_lpips:
        LossLPIPS = lpips.LPIPS(net='alex')
        LossLPIPS = LossLPIPS.to(args.device).eval()
    else:
        print("[warn] lpips not found; LPIPS will be NaN. Install with: pip install lpips")

    use_clip = True
    try:
        clip_model, clip_preproc = load_clip(args.device)
    except Exception as e:
        print(f"[warn] CLIP not available ({e}); CLIP similarity will be NaN.")
        use_clip = False
        clip_model = clip_preproc = None

    flow_alg = make_dis_flow()

    rows = []
    prev_inp = prev_rec = None

    pbar = tqdm(range(n_frames), desc="Per-frame metrics")
    for idx in pbar:
        a = read_frame(cap_inp)
        b = read_frame(cap_rec)
        if a is None or b is None:
            break

        a, b = ensure_same_size(a, b)

        # PSNR
        m_psnr = psnr(a, b)

        # LPIPS
        if use_lpips:
            with torch.no_grad():
                ta = to_torch_im(a).to(args.device)
                tb = to_torch_im(b).to(args.device)
                m_lpips = float(LossLPIPS(ta, tb).item())
        else:
            m_lpips = float("nan")

        # Flow wrapping error (needs previous frames)
        if prev_inp is not None and prev_rec is not None:
            m_flow = compute_flow_EPE_diff(flow_alg, prev_inp, a, prev_rec, b)
        else:
            m_flow = float("nan")

        # CLIP similarity
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
        })

        prev_inp, prev_rec = a, b

    cap_inp.release()
    cap_rec.release()

    # Save per-frame CSV
    import csv
    csv_path = out_dir / "per_frame.csv"
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["frame", "psnr", "lpips", "flow_epe_diff", "clip_cosine", "bpp"])
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow(r)

    # Averages (ignore NaNs)
    def mean_of(key):
        vals = [r[key] for r in rows if not (isinstance(r[key], float) and math.isnan(r[key]))]
        return float(sum(vals) / max(len(vals), 1)) if vals else float("nan")

    summary = {
        "frames_used": len(rows),
        "resolution_used": {"width": int(w), "height": int(h)},
        "input_video": str(inp_path),
        "reconstruction_video": str(rec_path),
        "bpp": bpp,  # constant per frame
        "avg_psnr": mean_of("psnr"),
        "avg_lpips": mean_of("lpips"),
        "avg_flow_epe_diff": mean_of("flow_epe_diff"),
        "avg_clip_cosine": mean_of("clip_cosine"),
    }

    json_path = out_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[eval] Wrote: {csv_path}")
    print(f"[eval] Wrote: {json_path}")
    print("[eval] Done.")


if __name__ == "__main__":
    main()
