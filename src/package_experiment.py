#!/usr/bin/env python3
import argparse, sys, zipfile
from pathlib import Path

def add_tree(zf: zipfile.ZipFile, src_root: Path, arc_root: str) -> int:
    """Add all files under src_root to zip under arc_root. Returns count of files added."""
    if not src_root.is_dir():
        return 0
    n = 0
    for p in src_root.rglob("*"):
        if p.is_file():
            arcname = Path(arc_root) / p.relative_to(src_root)
            zf.write(p, arcname.as_posix()); n += 1
    return n

def main():
    ap = argparse.ArgumentParser("Package checkpoint + compressed data + config.json into a ZIP")
    ap.add_argument("experiment", help="Path to the experiment directory")
    args = ap.parse_args()

    exp = Path(args.experiment).resolve()
    if not exp.is_dir():
        sys.exit(f"Experiment path not found: {exp}")

    tmp = exp / "_tmp"
    comp_res = exp / "compression results"

    # Source checkpoint from compression results
    ckpt_file = comp_res / "checkpoint"       # single file case
    ckpt_dir  = comp_res / "checkpoint"       # directory case (same name)

    # compressed bitstreams (still from _tmp)
    rgb_data_dir    = tmp / "compressed_atlases_rgb" / "data"
    masks_gray_data = tmp / "compressed_masks_rgb" / "data"  # optional

    cfg = exp / "config.json"
    if not cfg.is_file():
        found = next(exp.rglob("config.json"), None)
        if not found:
            sys.exit("config.json not found under the experiment.")
        cfg = found

    # Validations
    if not comp_res.is_dir():
        sys.exit(f"'compression results' folder not found at: {comp_res}")
    if not ckpt_file.exists():
        # allow directory form too
        if not ckpt_dir.is_dir():
            sys.exit(
                "Checkpoint not found in 'compression results'. "
                f"Tried as file: {ckpt_file} and as directory: {ckpt_dir}"
            )
    if not rgb_data_dir.is_dir():
        sys.exit(f"Missing _tmp/compressed_atlases_rgb/data at: {rgb_data_dir}")

    # ZIP goes to compression results/ if it exists, else to exp root
    out_dir = comp_res if comp_res.is_dir() else exp
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{exp.name}_package.zip"

    files_added = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        # config.json at root of zip
        zf.write(cfg, "config.json"); files_added += 1

        # ✅ Add checkpoint
        if ckpt_file.is_file():
            # single file -> compression results/checkpoint
            zf.write(ckpt_file, "compression results/checkpoint"); files_added += 1
        else:
            # directory -> compression results/checkpoint/...
            files_added += add_tree(zf, ckpt_dir, "compression results/checkpoint")

        # compressed atlas bitstreams -> compressed_atlases_rgb/data/<name>/...
        files_added += add_tree(zf, rgb_data_dir, "compressed_atlases_rgb/data")

        # compressed mask bitstreams (if present) -> compressed_masks_rgb/data/<name>/...
        if masks_gray_data.is_dir():
            files_added += add_tree(zf, masks_gray_data, "compressed_masks_rgb/data")

        # small manifest
        manifest = (
            "Contents:\n"
            "- compression results/checkpoint      [file or directory from 'compression results']\n"
            "- config.json                         [auto-located]\n"
            "- compressed_atlases_rgb/data/*       [from _tmp/...]\n"
            "- compressed_masks_rgb/data/*         [from _tmp/... if present]\n"
        )
        zf.writestr("manifest.txt", manifest)

    print(f"[package] Created: {zip_path}")
    print(f"[package] Files added: {files_added}")

if __name__ == "__main__":
    main()
