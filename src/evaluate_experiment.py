#!/usr/bin/env python3
"""
Evaluate a Layered Neural Atlases experiment with robust frame alignment.

Key features:
  * Auto-offset search to align reconstruction to input (fast, no heavy models).
  * Time-to-index mapping using FPS; per-frame local refinement over {j-1, j, j+1}.
  * Monotonic recon index progression to avoid back-and-forth mismatches.
  * Saves the exact resized frames compared (PNG) for visual verification.
  * bpp computed from the newest package ZIP under 'compression results' (fallback: recon .mp4).

Outputs:
  eval/per_frame.csv
  eval/summary.json
  eval/frames_input/*.png
  eval/frames_recon/*.png
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import cv2
import numpy as np
from tqdm import tqdm

import torch

# ----- LPIPS -----
try:
    import lpips  # type: ignore
except Exception:
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


# ----------------- Utilities & IO -----------------

def find_input_video(exp_dir: Path) -> Path:
    cand = exp_dir / "training results" / "input_video.mp4"
    if not cand.exists():
        raise FileNotFoundError(f"Could not find input video at {cand}")
    return cand


def find_latest_reconstruction_mp4(exp_dir: Path) -> Path:
    recon_root = exp_dir / "compression results" / "reconstruction"
    if not recon_root.exists():
        raise FileNotFoundError(f"Missing folder: {recon_root}")

    runs = []
    for p in recon_root.iterdir():
        if p.is_dir():
            mp4s = list(p.glob("*.mp4"))
            if mp4s:
                newest = max(mp4s, key=lambda x: x.stat().st_mtime)
                runs.append((p, newest.stat().st_mtime))
    if not runs:
        raise FileNotFoundError(f"No reconstruction runs with mp4 found under {recon_root}")

    runs.sort(key=lambda x: x[1], reverse=True)
    chosen_run = runs[0][0]
    mp4s = list(chosen_run.glob("*.mp4"))
    mp4 = max(mp4s, key=lambda x: x.stat().st_mtime)
    return mp4


def find_latest_package_zip(exp_dir: Path) -> Optional[Path]:
    root = exp_dir / "compression results"
    if not root.exists():
        return None
    zips = list(root.rglob("*.zip"))
    if not zips:
        return None
    return max(zips, key=lambda p: p.stat().st_mtime)


def open_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return cap, w, h, n, fps


def ensure_eval_dir(exp_dir: Path) -> Path:
    out = exp_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_frame_dirs(exp_dir: Path) -> Tuple[Path, Path]:
    base = ensure_eval_dir(exp_dir)
    inp_dir = base / "frames_input"
    rec_dir = base / "frames_recon"
    inp_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)
    return inp_dir, rec_dir


def save_frame_rgb_png(rgb_img: np.ndarray, out_path: Path):
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr)


# ----------------- Image ops & metrics -----------------

def to_torch_im(img_rgb_uint8: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img_rgb_uint8).permute(2, 0, 1).float() / 255.0
    t = t * 2.0 - 1.0
    return t.unsqueeze(0)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.PSNR(a, b))


def ensure_same_size(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    return a, b


def make_dis_flow():
    return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def flow_epe(f: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(f ** 2, axis=2))


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


def load_clip(device: str = "cpu"):
    if clip is None:
        raise RuntimeError("CLIP not installed. Install either 'openai-clip' or 'clip-anytorch'.")
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    return model, preprocess


def clip_embed(model, preprocess, img_rgb_uint8: np.ndarray, device: str):
    import PIL.Image as Image
    pil = Image.fromarray(img_rgb_uint8)
    with torch.no_grad():
        t = preprocess(pil).unsqueeze(0).to(device)
        feat = model.encode_image(t)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu()


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.sum(x * y))


# ----------------- Robust alignment helpers -----------------

def lowres_gray(img_rgb: np.ndarray, target_wh=(64, 36)) -> np.ndarray:
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, target_wh, interpolation=cv2.INTER_AREA)


def frame_distance(a_gray_lr: np.ndarray, b_gray_lr: np.ndarray) -> float:
    # Mean absolute difference on low-res gray
    return float(np.mean(np.abs(a_gray_lr.astype(np.float32) - b_gray_lr.astype(np.float32))))


def read_frame_at_index(cap, idx: int) -> Optional[np.ndarray]:
    if idx < 0:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def sample_indices(n: int, k: int) -> List[int]:
    if k <= 1:
        return [0] if n > 0 else []
    return [int(round(i*(n-1)/(k-1))) for i in range(k)]


def estimate_time_offset_seconds(
    cap_inp, cap_rec, n_i: int, n_r: int, fps_i: float, fps_r: float,
    search_range_sec: float = 1.0, search_step_sec: float = 0.02, samples: int = 24,
) -> float:
    """
    Estimate global time offset (recon lag vs input) by sampling 'samples' input frames,
    mapping to recon with candidate offsets, and picking offset that minimizes low-res MAD.
    Positive offset means: use later frames in recon (recon is 'ahead' and we shift it forward).
    """
    if fps_i <= 0 or fps_r <= 0 or n_i == 0 or n_r == 0:
        return 0.0

    inp_samples = sample_indices(n_i, min(samples, n_i))
    # Preload input low-res grayscale samples
    inp_lowres: Dict[int, np.ndarray] = {}
    for i in inp_samples:
        f = read_frame_at_index(cap_inp, i)
        if f is None:
            continue
        inp_lowres[i] = lowres_gray(f)

    # Candidate offsets
    offsets = np.arange(-search_range_sec, search_range_sec + 1e-9, search_step_sec)
    best_off = 0.0
    best_score = float("inf")

    for off in offsets:
        score_acc = 0.0
        cnt = 0
        for i in inp_lowres.keys():
            t_i = i / fps_i
            j_nom = int(round((t_i + off) * fps_r))
            # Compare among {j-1, j, j+1}
            best_local = float("inf")
            for j in (j_nom - 1, j_nom, j_nom + 1):
                fr = read_frame_at_index(cap_rec, j)
                if fr is None:
                    continue
                d = frame_distance(inp_lowres[i], lowres_gray(fr))
                if d < best_local:
                    best_local = d
            if best_local < float("inf"):
                score_acc += best_local
                cnt += 1
        if cnt > 0:
            score = score_acc / cnt
            if score < best_score:
                best_score = score
                best_off = off

    # Reset capture positions to start for the real pass
    cap_inp.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_rec.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return float(best_off)


# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser(description="Evaluate experiment metrics with robust alignment & frame dumps.")
    ap.add_argument("--experiment_dir", required=True, help="Path to a single experiment directory.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_frames", type=int, default=0, help="Optional cap on frames (0 = all).")
    ap.add_argument("--align", choices=["auto", "index"], default="auto",
                    help="Alignment mode: 'auto' (time map + auto offset + local refine) or 'index' (old behavior).")
    ap.add_argument("--offset_sec", type=float, default=None,
                    help="Manual override for offset seconds (recon time minus input time). If set, skips auto-search.")
    ap.add_argument("--auto_range_sec", type=float, default=1.0, help="Auto offset search range (±seconds).")
    ap.add_argument("--auto_step_sec", type=float, default=0.02, help="Auto offset search step (seconds).")
    ap.add_argument("--local_window", type=int, default=1, help="Local refinement ±window (frames) around mapped index.")
    args = ap.parse_args()

    exp_dir = Path(args.experiment_dir).resolve()
    out_dir = ensure_eval_dir(exp_dir)
    frames_inp_dir, frames_rec_dir = ensure_frame_dirs(exp_dir)

    inp_path = find_input_video(exp_dir)
    rec_path = find_latest_reconstruction_mp4(exp_dir)

    print(f"[eval] input_video:     {inp_path}")
    print(f"[eval] reconstruction:  {rec_path}")
    print(f"[eval] align mode: {args.align}")

    cap_inp, w_i, h_i, n_i, fps_i = open_video(inp_path)
    cap_rec, w_r, h_r, n_r, fps_r = open_video(rec_path)

    if fps_i <= 0:
        print("[warn] Input FPS reported as 0; falling back to 30 FPS.")
        fps_i = 30.0
    if fps_r <= 0:
        print("[warn] Recon FPS reported as 0; falling back to input FPS.")
        fps_r = fps_i

    # Resolution used (matches the resize step)
    w = min(w_i, w_r)
    h = min(h_i, h_r)

    # ---- bpp from PACKAGE ZIP if available, else fall back to recon .mp4 ----
    pkg_zip = find_latest_package_zip(exp_dir)
    if pkg_zip is not None:
        bits_total = pkg_zip.stat().st_size * 8.0
        bpp_source = "package_zip"
        print(f"[eval] bpp source: package_zip -> {pkg_zip}")
    else:
        bits_total = rec_path.stat().st_size * 8.0
        bpp_source = "reconstruction_mp4"
        print("[warn] No package ZIP found; using reconstruction .mp4 size for bpp.")

    # ---------- Alignment ----------
    offset_sec: float
    if args.align == "index":
        # Old behavior; no offset used
        offset_sec = 0.0
        n_frames = min(n_i, n_r)
    else:
        if args.offset_sec is not None:
            offset_sec = float(args.offset_sec)
            print(f"[eval] Using manual offset: {offset_sec:+.4f} s")
        else:
            print(f"[eval] Auto-searching offset in [{-args.auto_range_sec}, +{args.auto_range_sec}] step {args.auto_step_sec}s ...")
            offset_sec = estimate_time_offset_seconds(
                cap_inp, cap_rec, n_i, n_r, fps_i, fps_r,
                search_range_sec=args.auto_range_sec,
                search_step_sec=args.auto_step_sec,
                samples=24
            )
            print(f"[eval] Estimated offset (recon - input): {offset_sec:+.4f} s")

        # Time range we can safely evaluate
        dur_i = n_i / fps_i
        dur_r = n_r / fps_r
        usable_time = max(0.0, min(dur_i, dur_r - max(0.0, offset_sec)))
        n_frames = int(math.floor(usable_time * fps_i))
        if n_frames <= 0:
            print("[error] No overlapping time after offset. Try a different offset/range.")
            return

    if args.max_frames and args.max_frames > 0:
        n_frames = min(n_frames, args.max_frames)

    # Denominator for bpp matches frames we actually evaluate at resized resolution
    den_frames = max(n_frames, 1)
    bpp = bits_total / float(w * h * den_frames)

    # Metrics setup
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

    # Helpers for sequential-ish access with small forward jumps
    rec_cache: Dict[int, np.ndarray] = {}

    def get_recon_frame_at_index(j: int) -> Optional[np.ndarray]:
        if j in rec_cache:
            return rec_cache[j]
        f = read_frame_at_index(cap_rec, j)
        if f is not None:
            rec_cache[j] = f
        return f

    rows = []
    prev_inp = prev_rec = None

    # Main loop: step by input cadence; map time -> recon index; refine locally ±window
    pbar = tqdm(range(n_frames), desc="Per-frame metrics (aligned)")
    last_j_used = -1
    for idx in pbar:
        # Input frame index and time
        i = idx
        t_i = i / fps_i

        # Read input frame i
        a = read_frame_at_index(cap_inp, i)
        if a is None:
            break

        # Map to nominal recon index via (t_i + offset) * fps_r
        j_nom = int(round((t_i + (0.0 if args.align == "index" else offset_sec)) * fps_r))

        # Enforce monotonic progression
        j_nom = max(j_nom, last_j_used + 1)

        # Local refinement over ±window around j_nom
        best_j = None
        best_d = float("inf")
        a_lr = lowres_gray(a)

        for dj in range(-args.local_window, args.local_window + 1):
            j = j_nom + dj
            if j < 0 or j >= n_r:
                continue
            fr = get_recon_frame_at_index(j)
            if fr is None:
                continue
            d = frame_distance(a_lr, lowres_gray(fr))
            if d < best_d and j >= last_j_used:  # keep monotonic
                best_d = d
                best_j = j

        if best_j is None:
            # no valid recon frame -> stop
            break

        b = rec_cache[best_j]
        last_j_used = best_j

        # Resize to common evaluation size
        a, b = ensure_same_size(a, b)

        # Save exact frames used
        fname = f"{idx:06d}.png"
        save_frame_rgb_png(a, frames_inp_dir / fname)
        save_frame_rgb_png(b, frames_rec_dir / fname)

        # Metrics
        m_psnr = psnr(a, b)

        if use_lpips:
            with torch.no_grad():
                ta = to_torch_im(a).to(args.device)
                tb = to_torch_im(b).to(args.device)
                m_lpips = float(LossLPIPS(ta, tb).item())
        else:
            m_lpips = float("nan")

        if prev_inp is not None and prev_rec is not None:
            m_flow = compute_flow_EPE_diff(flow_alg, prev_inp, a, prev_rec, b)
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
            "input_index": i,
            "recon_index": best_j,
            "t_input_sec": t_i,
            "t_recon_sec": best_j / fps_r,
        })

        prev_inp, prev_rec = a, b

    # Release caps
    cap_inp.release()
    cap_rec.release()

    # Save per-frame CSV
    import csv
    csv_path = out_dir / "per_frame.csv"
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.DictWriter(
            f,
            fieldnames=[
                "frame", "psnr", "lpips", "flow_epe_diff", "clip_cosine", "bpp",
                "input_index", "recon_index", "t_input_sec", "t_recon_sec"
            ]
        )
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow(r)

    # Averages (ignore NaNs)
    def mean_of(key):
        vals = [r[key] for r in rows if not (isinstance(r[key], float) and math.isnan(r[key]))]
        return float(sum(vals) / max(len(vals), 1)) if vals else float("nan")

    summary = {
        "frames_used": len(rows),
        "frames_saved_input_dir": str(frames_inp_dir),
        "frames_saved_recon_dir": str(frames_rec_dir),
        "resolution_used": {"width": int(w), "height": int(h)},
        "input_video": str(inp_path),
        "reconstruction_video": str(rec_path),
        "fps_input": float(fps_i),
        "fps_recon": float(fps_r),
        "frames_input_reported": int(n_i),
        "frames_recon_reported": int(n_r),
        "alignment_mode": args.align,
        "offset_sec_used": float(0.0 if args.align == "index" else (args.offset_sec if args.offset_sec is not None else offset_sec)),
        "local_refine_window": int(args.local_window),
        "bpp": rows[0]["bpp"] if rows else float("nan"),
        "bpp_source": bpp_source,
        **({"package_zip": str(pkg_zip)} if pkg_zip is not None else {}),
        "avg_psnr": mean_of("psnr"),
        "avg_lpips": mean_of("lpips"),
        "avg_flow_epe_diff": mean_of("flow_epe_diff"),
        "avg_clip_cosine": mean_of("clip_cosine"),
    }

    json_path = out_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Helpful trace
    print(f"[eval] Used offset (recon - input): {summary['offset_sec_used']:+.4f} s | local window: ±{args.local_window} frames")
    print(f"[eval] Frames written: {len(rows)} | Input FPS: {fps_i:.4g} | Recon FPS: {fps_r:.4g}")
    print(f"[eval] Frames dirs -> input: {frames_inp_dir} | recon: {frames_rec_dir}")
    print(f"[eval] Wrote: {csv_path}")
    print(f"[eval] Wrote: {json_path}")
    print("[eval] Done.")


if __name__ == "__main__":
    main()
