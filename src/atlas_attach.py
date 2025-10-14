#!/usr/bin/env python3
import os, argparse
import numpy as np
import cv2

# from your shared helpers
from atlas_alpha_lib import read_rgb, write_rgba

def main():
    ap = argparse.ArgumentParser("Attach saved alpha masks to compressed RGB atlases -> RGBA")
    ap.add_argument("--comp_rgb_dir", required=True, help="Folder with compressed RGB atlases")
    # mask_dir is optional now; required only if no constant alpha is provided
    ap.add_argument("--mask_dir", default=None, help="Folder with sidecar *_mask.png (from PREP)")
    ap.add_argument("--out_rgba_dir", required=True, help="Output folder for RGBA atlases")
    ap.add_argument("--names", nargs="+", default=["texture_orig1.png","texture_orig2.png"])
    ap.add_argument("--premultiply", action="store_true",
                    help="If set, premultiply RGB by alpha before writing (renderer-friendly; some viewers may show fringes).")

    # NEW: constant alpha options
    ap.add_argument("--alpha-constant", type=int, default=None,
                    help="If set, attach this constant alpha (0-255) to all outputs and ignore masks.")
    ap.add_argument("--alpha-one", action="store_true",
                    help="Shortcut for --alpha-constant 255 (full opacity).")

    args = ap.parse_args()

    # Resolve shortcut / validate
    if args.alpha_one and args.alpha_constant is None:
        args.alpha_constant = 255
    if args.alpha_constant is not None and not (0 <= args.alpha_constant <= 255):
        raise ValueError("--alpha-constant must be in [0,255]")

    # Require mask_dir only when we are NOT using a constant alpha
    if args.alpha_constant is None and args.mask_dir is None:
        raise ValueError("When --alpha-constant is not provided, --mask_dir must be specified.")

    os.makedirs(args.out_rgba_dir, exist_ok=True)

    for name in args.names:
        print(f"[attach] {name}")

        # 1) Read compressed RGB (STRAIGHT)
        rgb_path = os.path.join(args.comp_rgb_dir, name)
        rgb = read_rgb(rgb_path)  # HxWx3, float32 [0,1]
        if rgb is None:
            raise FileNotFoundError(f"RGB not found or unreadable: {rgb_path}")
        H, W = rgb.shape[:2]

        # 2) Get alpha as float [0,1]
        if args.alpha_constant is not None:
            # Constant alpha path (ignore mask_dir)
            a = np.full((H, W), args.alpha_constant / 255.0, dtype=np.float32)
            print(f"  - using constant alpha = {args.alpha_constant} (ignored mask_dir)")
        else:
            # Read alpha mask (lossless PNG written in PREP)
            mask_path = os.path.join(args.mask_dir, name.replace(".png", "_mask.png"))
            a_u8 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if a_u8 is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            a = (a_u8.astype(np.float32) / 255.0)  # HxW, [0,1]

            # 3) Size-guard: resize alpha to match RGB if needed
            if (H, W) != a.shape[:2]:
                a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
                print("  - resized mask to match RGB")

        # 4) Optional premultiply
        if args.premultiply:
            rgb_to_save = rgb * a[..., None]
            print("  - premultiplied RGB by alpha")
        else:
            rgb_to_save = rgb  # save as straight RGBA (viewer-friendly)

        # 5) Write RGBA
        out_path = os.path.join(args.out_rgba_dir, name)
        rgba = np.dstack([rgb_to_save, a])  # HxWx4, float32 [0,1]
        write_rgba(out_path, rgba)
        print(f"  -> wrote {out_path}")

if __name__ == "__main__":
    main()
