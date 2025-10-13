#!/usr/bin/env python3
import os, argparse, numpy as np
from atlas_alpha_lib import read_rgb, write_rgba  # uses your existing helpers

def main():
    ap = argparse.ArgumentParser("Attach a uniform alpha to compressed RGB atlases -> RGBA")
    ap.add_argument("--comp_rgb_dir", required=True, help="Folder with compressed RGB atlases")
    ap.add_argument("--out_rgba_dir", required=True, help="Output folder for RGBA atlases")
    ap.add_argument("--names", nargs="+", default=["texture_orig1.png","texture_orig2.png"])
    ap.add_argument("--alpha", type=float, default=1.0, help="Uniform alpha value (0..1), default 1.0")
    args = ap.parse_args()

    os.makedirs(args.out_rgba_dir, exist_ok=True)

    a_val = float(np.clip(args.alpha, 0.0, 1.0))
    for name in args.names:
        print(f"[attach-uniform] {name} (alpha={a_val})")
        rgb = read_rgb(os.path.join(args.comp_rgb_dir, name))        # HxWx3, float32 [0,1], STRAIGHT
        alpha = np.full(rgb.shape[:2], a_val, dtype=np.float32)      # HxW
        rgba = np.dstack([rgb, alpha])                                # HxWx4
        write_rgba(os.path.join(args.out_rgba_dir, name), rgba)

if __name__ == "__main__":
    main()
