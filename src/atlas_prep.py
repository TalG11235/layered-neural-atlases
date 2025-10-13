#!/usr/bin/env python3
import os, argparse, cv2, numpy as np
from atlas_alpha_lib import *

def main():
    ap = argparse.ArgumentParser("Prep: BLEED outside alpha; write STRAIGHT RGB (no premultiply) + mask")
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out_rgb_dir", required=True)     # feed these to the compressor
    ap.add_argument("--out_mask_dir", required=True)    # sidecar lossless masks
    ap.add_argument("--names", nargs="+", default=["texture_orig1.png","texture_orig2.png"])
    ap.add_argument("--feather_sigma", type=float, default=0.6)
    ap.add_argument("--bleed_px", type=int, default=8)
    ap.add_argument("--key_thr", type=int, default=12)
    ap.add_argument("--debug_ring", action="store_true", help="save bleed visualization PNGs")
    args = ap.parse_args()

    os.makedirs(args.out_rgb_dir, exist_ok=True)
    os.makedirs(args.out_mask_dir, exist_ok=True)

    for name in args.names:
        print(f"[prep] {name}")
        eval_rgba = os.path.join(args.eval_dir, name.replace(".png","_alpha.png"))
        eval_rgb  = os.path.join(args.eval_dir, name)

        # 1) Get RGB + alpha (prefer *_alpha.png)
        if os.path.exists(eval_rgba):
            rgb_e, alpha = read_rgba_or_rgb(eval_rgba)
            print("  - using *_alpha.png")
        else:
            rgb_e = read_rgb(eval_rgb)
            alpha = key_black_make_alpha(rgb_e, threshold=args.key_thr, blur_sigma=args.feather_sigma)
            print(f"  - keyed alpha (thr={args.key_thr})")

        # 2) Optional small feather
        if args.feather_sigma > 0:
            alpha = cv2.GaussianBlur(alpha, (0,0), args.feather_sigma)
        alpha = np.clip(alpha, 0, 1)

        # 3) *** BLEED OUTSIDE ALPHA ***  (DO NOT premultiply in PREP)
        rgb_bled = color_bleed(
            rgb_e, alpha,
            bleed_px=args.bleed_px,          # how wide to pad outside
            inner_offset_px=2,    # how far inside to sample & overwrite
            binarize_thr=0.5,
            unpremultiply=True
        )

        # 4) Write straight RGB for compression + lossless mask
        write_rgb(os.path.join(args.out_rgb_dir, name), rgb_bled)
        cv2.imwrite(os.path.join(args.out_mask_dir, name.replace(".png","_mask.png")),
                    (alpha * 255).astype(np.uint8))

if __name__ == "__main__":
    main()
