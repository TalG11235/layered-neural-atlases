#!/usr/bin/env python3
import os, argparse
import numpy as np
import cv2

# expected helpers in atlas_alpha_lib.py:
#   read_rgba_or_rgb(path) -> (rgb, alpha_or_None), float32 in [0,1]
#   write_rgb(path, rgb_float01)
#   color_bleed(rgb, alpha, bleed_px=..., inner_offset_px=..., binarize_thr=0.5, unpremultiply=True)
from atlas_alpha_lib import read_rgba_or_rgb, write_rgb, color_bleed

def main():
    ap = argparse.ArgumentParser(
        "Prep: optionally BLEED outside alpha; write STRAIGHT RGB (no premultiply) + mask"
    )
    ap.add_argument("--eval_dir", required=True, help="Folder with evaluation atlases")
    ap.add_argument("--out_rgb_dir", required=True, help="Output folder for bled STRAIGHT RGB (feed to compressor)")
    ap.add_argument("--out_mask_dir", required=True, help="Output folder for sidecar alpha masks (*.png)")
    ap.add_argument("--names", nargs="+", default=["texture_orig1.png","texture_orig2.png"])

    # bleed controls
    ap.add_argument("--bleed_px", type=int, default=0,
                    help="Width (px) of OUTSIDE color pad. Set 0 to disable bleeding.")
    ap.add_argument("--inner_offset_px", type=int, default=0,
                    help="How many px INSIDE the mask to sample donor colors from AND overwrite that inner band. 0 disables inner overwrite.")
    ap.add_argument("--binarize_thr", type=float, default=0.5,
                    help="Threshold for hard silhouette used by the bleed op (0..1).")

    # small optional softening for the saved alpha (does not affect bleed unless you want it to)
    ap.add_argument("--feather_sigma", type=float, default=0.0,
                    help="Gaussian feather (px) applied to the alpha BEFORE saving. 0 keeps alpha exactly as provided.")

    ap.add_argument("--debug_ring", action="store_true",
                    help="Also write *_bleed_out.png and *_bleed_in.png visualizations.")
    args = ap.parse_args()

    os.makedirs(args.out_rgb_dir, exist_ok=True)
    os.makedirs(args.out_mask_dir, exist_ok=True)

    for name in args.names:
        print(f"[prep] {name}")

        # Prefer *_alpha.png; else try the base file (may be RGBA)
        eval_rgba_path = os.path.join(args.eval_dir, name.replace(".png", "_alpha.png"))
        eval_base_path = os.path.join(args.eval_dir, name)

        if os.path.exists(eval_rgba_path):
            rgb_e, alpha = read_rgba_or_rgb(eval_rgba_path)
            print("  - using *_alpha.png")
        else:
            rgb_e, alpha = read_rgba_or_rgb(eval_base_path)
            if alpha is None:
                raise FileNotFoundError(
                    f"No alpha found for {name}. Expected {eval_rgba_path} or an RGBA {eval_base_path}."
                )
            print("  - using alpha from source RGBA")

        # optional feather for the SAVED alpha (not required for bleeding)
        alpha_save = alpha.copy()
        if args.feather_sigma and args.feather_sigma > 0.0:
            alpha_save = cv2.GaussianBlur(alpha_save, (0, 0), args.feather_sigma)
        alpha_save = np.clip(alpha_save, 0.0, 1.0)

        # decide whether to bleed
        do_bleed = (args.bleed_px > 0) or (args.inner_offset_px > 0)

        if do_bleed:
            # for the bleed op we want a hard silhouette (avoid ambiguous soft edges)
            alpha_for_bleed = (alpha >= args.binarize_thr).astype(np.float32)

            rgb_out = color_bleed(
                rgb=rgb_e,
                alpha=alpha_for_bleed,
                bleed_px=int(args.bleed_px),
                inner_offset_px=int(args.inner_offset_px),
                unpremultiply=True,              # use full-strength donor colors
                # binarize_thr is handled by the hard mask above
            )
            print(f"  - bleed applied (bleed_px={args.bleed_px}, inner_offset_px={args.inner_offset_px})")
        else:
            # no bleed requested: just pass the RGB through untouched
            rgb_out = rgb_e
            print("  - bleed disabled (bleed_px=0 and inner_offset_px=0)")

        # write outputs
        out_rgb_path  = os.path.join(args.out_rgb_dir, name)
        out_mask_path = os.path.join(args.out_mask_dir, name.replace(".png", "_mask.png"))

        write_rgb(out_rgb_path, rgb_out)
        cv2.imwrite(out_mask_path, (alpha_save * 255).astype(np.uint8))
        print(f"  - wrote RGB  -> {out_rgb_path}")
        print(f"  - wrote mask -> {out_mask_path}")

        # optional debug images to visualize what changed
        if args.debug_ring and do_bleed:
            # show outside ring (zero interior)
            dbg_out = rgb_out.copy()
            dbg_out[alpha_save >= 0.99] = 0.0
            write_rgb(os.path.join(args.out_rgb_dir, name.replace(".png", "_bleed_out.png")), dbg_out)

            # show inside band region only
            fg = (alpha >= args.binarize_thr).astype(np.uint8)
            if args.inner_offset_px > 0:
                k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*args.inner_offset_px+1, 2*args.inner_offset_px+1))
                fg_eroded = cv2.erode(fg, k_in, iterations=1)
                inside_band = (fg.astype(bool) & (~fg_eroded.astype(bool)))
                dbg_in = rgb_out.copy()
                dbg_in[~inside_band] = 0.0
                write_rgb(os.path.join(args.out_rgb_dir, name.replace(".png", "_bleed_in.png")), dbg_in)

    print("Done.")

if __name__ == "__main__":
    main()
