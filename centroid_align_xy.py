"""centroid_align_xy.py — align embryo movies by XY centroid tracking.

Reads a threshold YAML produced by find_threshold.py, loads the full ND2
file, shifts each embryo to the frame centre at every timepoint, and writes
one OME-TIFF per embryo position and one MP4 per channel.

"XY" in the script name is intentional: only lateral (XY) drift is corrected.
Z-axis (focus) drift correction is a separate problem not handled here.

All image processing functions are defined in embryo_tools.py. This script
contains only argument parsing, file I/O orchestration, and progress reporting.

Usage:
    python centroid_align_xy.py nd1188_Venus_threshold.yaml
    python centroid_align_xy.py nd1188_Venus_threshold.yaml --enlarge_canvas
"""

import argparse
import os
import sys

import imageio
import numpy as np
import yaml
from scipy.ndimage import shift
from tqdm import tqdm

from useful_functions import (
    align_frame_xy,
    auto_contrast,
    compute_shift_xy,
    load_nd2,
    make_grid_frame,
    save_ome_tiff,
)


def main():
    parser = argparse.ArgumentParser(description="Centroid-align embryo movies from ND2 data.")
    parser.add_argument("yaml_file", help="Threshold YAML from find_threshold.py")
    parser.add_argument("--fps", type=float, default=2, help="Frames per second (default: 2)")
    parser.add_argument(
        "--enlarge_canvas",
        action="store_true",
        help=(
            "Precompute all shifts, print them, then expand the canvas "
            "asymmetrically before aligning so no edge data is lost."
        ),
    )
    args = parser.parse_args()

    # ── 1. Load YAML config ───────────────────────────────────────────────────

    with open(args.yaml_file) as f:
        cfg = yaml.safe_load(f)

    sigma      = cfg["parameters"]["sigma"]
    percentile = cfg["parameters"]["percentile"]
    ch_idx     = cfg["parameters"]["channel_index"]
    nd2_path   = cfg["source"]["file"]

    if not os.path.isfile(nd2_path):
        print(f"Error: ND2 file not found: {nd2_path}", file=sys.stderr)
        sys.exit(1)

    # ── 2. Load ND2 ───────────────────────────────────────────────────────────

    print(f"Loading {nd2_path} ...")
    data, channel_names, vox, period_s = load_nd2(nd2_path)
    T, P, Z, C, Y, X = data.shape
    print(f"  Shape: T={T}, P={P}, Z={Z}, C={C}, Y={Y}, X={X}")
    print(f"  Channels: {channel_names}")
    print(f"  Threshold channel: {channel_names[ch_idx]} (index {ch_idx})")
    print(f"  sigma={sigma}, percentile={percentile}")

    # ── 3. Prepare output directory ───────────────────────────────────────────

    base    = os.path.splitext(os.path.basename(nd2_path))[0]
    out_dir = os.path.join(os.path.dirname(nd2_path) or ".", f"aligned_{base}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Output: {out_dir}/")

    # Collect max-projected, contrast-normalised frames for MP4 encoding.
    # Structure: aligned_for_mp4[channel_index][timepoint] = list of (Y,X) uint8 images,
    # one per embryo position. This is populated identically by both the
    # single-pass and two-pass (--enlarge_canvas) paths so that MP4 writing
    # below is shared between them.
    aligned_for_mp4 = [[[] for _ in range(T)] for _ in range(C)]

    # ── 4a. Two-pass path: --enlarge_canvas ───────────────────────────────────

    if args.enlarge_canvas:
        # Pass 1 — precompute all (dy, dx) shifts without modifying data.
        # Printing each value as it arrives lets the user spot directional
        # drift patterns before committing to any canvas expansion.
        print(f"\n--- Pass 1: precomputing shifts ---")
        shifts = np.zeros((P, T, 2))  # shifts[p, t] = [dy, dx]

        for p in range(P):
            print(f"\n  Embryo {p}/{P-1}:")
            for t in range(T):
                # Reorder from ND2 axis order (Z, C, Y, X) to pipeline order (C, Z, Y, X).
                frame        = data[t, p].transpose(1, 0, 2, 3)
                dy, dx       = compute_shift_xy(frame, sigma, percentile, ch_idx)
                shifts[p, t] = [dy, dx]

        dy_all, dx_all = shifts[:, :, 0], shifts[:, :, 1]
        print(f"\nShift summary (pixels):")
        print(f"  dy  min={dy_all.min():+.1f}  max={dy_all.max():+.1f}  mean={dy_all.mean():+.1f}")
        print(f"  dx  min={dx_all.min():+.1f}  max={dx_all.max():+.1f}  mean={dx_all.mean():+.1f}")

        # Compute asymmetric padding so that no original data is clipped.
        # scipy shift with +dy moves content downward → bottom rows are lost → pad bottom.
        # scipy shift with -dy moves content upward  → top rows are lost    → pad top.
        # Same logic applies to dx / left / right.
        pad_top    = int(np.ceil(max(0, -dy_all.min())))
        pad_bottom = int(np.ceil(max(0,  dy_all.max())))
        pad_left   = int(np.ceil(max(0, -dx_all.min())))
        pad_right  = int(np.ceil(max(0,  dx_all.max())))
        print(f"\nCanvas padding:  top={pad_top}  bottom={pad_bottom}  left={pad_left}  right={pad_right}")

        # Pad only Y and X (axes 4 and 5); all other axes are unchanged.
        # The precomputed shifts remain valid on the larger canvas — each
        # centroid will land at (pad_top + Y_orig/2, pad_left + X_orig/2),
        # i.e. stabilised but not re-centred in the new frame. This is
        # intentional: the embryo position is consistent across timepoints,
        # which is the goal of alignment.
        data = np.pad(
            data,
            ((0, 0), (0, 0), (0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        )
        T, P, Z, C, Y, X = data.shape
        print(f"  Expanded canvas: Y={Y}  X={X}")

        # Pass 2 — apply precomputed shifts on the expanded canvas.
        # We call scipy.ndimage.shift directly here rather than align_frame_xy
        # because the shifts are already computed and we must not recompute
        # centroids on the padded canvas (the padding zeros would shift the
        # percentile threshold and produce incorrect centroids).
        print(f"\n--- Pass 2: aligning {P} embryo(s) × {T} timepoints ---")
        for p in range(P):
            print(f"\n  Embryo {p}/{P-1}")
            volume = np.zeros((T, C, Z, Y, X), dtype=np.float32)
            for t in tqdm(range(T), desc="    Aligning", unit="frame", leave=True):
                dy, dx = shifts[p, t]
                frame   = data[t, p].transpose(1, 0, 2, 3)
                shifted = shift(frame, (0, 0, dy, dx), order=1, mode="constant", cval=0)
                volume[t] = shifted
                for c in range(C):
                    aligned_for_mp4[c][t].append(auto_contrast(shifted[c].max(axis=0)))

            fpath = os.path.join(out_dir, f"{base}_P{p}.ome.tif")
            print(f"    Saving {os.path.basename(fpath)} ...")
            save_ome_tiff(fpath, volume, channel_names, vox, period_s)
            print(f"    Saved {os.path.basename(fpath)}")

    # ── 4b. Single-pass path (default) ───────────────────────────────────────

    else:
        print(f"\nAligning {P} embryo(s) × {T} timepoints ...")
        for p in range(P):
            print(f"\n  Embryo {p}/{P-1}")
            volume = np.zeros((T, C, Z, Y, X), dtype=np.float32)
            for t in tqdm(range(T), desc="    Aligning timepoints", unit="frame", leave=True):
                frame           = data[t, p].transpose(1, 0, 2, 3)
                # align_frame_xy calls compute_shift_xy internally, which
                # prints dy/dx. No separate logging needed here.
                shifted, dy, dx = align_frame_xy(frame, sigma, percentile, ch_idx)
                volume[t] = shifted
                for c in range(C):
                    aligned_for_mp4[c][t].append(auto_contrast(shifted[c].max(axis=0)))

            fpath = os.path.join(out_dir, f"{base}_P{p}.ome.tif")
            print(f"    Saving {os.path.basename(fpath)} ...")
            save_ome_tiff(fpath, volume, channel_names, vox, period_s)
            print(f"    Saved {os.path.basename(fpath)}")

    # ── 5. Write MP4s ─────────────────────────────────────────────────────────

    # One MP4 per channel, showing all embryos as a 2×2 grid over time.
    # This is shared between both alignment paths because aligned_for_mp4
    # is populated identically above.
    print(f"\nWriting MP4s for {C} channel(s) ...")
    for c in range(C):
        ch_name  = channel_names[c]
        mp4_path = os.path.join(out_dir, f"{base}_{ch_name}_aligned.mp4")
        print(f"  Channel '{ch_name}': {mp4_path}")

        writer = imageio.get_writer(mp4_path, fps=args.fps)
        for t in tqdm(range(T), desc="    Encoding frames", unit="frame", leave=True):
            grid = make_grid_frame(aligned_for_mp4[c][t])
            # imageio expects RGB frames; stack the grayscale grid three times
            # to produce a greyscale-looking RGB image.
            rgb = np.stack([grid, grid, grid], axis=-1)
            writer.append_data(rgb)
        writer.close()
        print(f"  Saved {os.path.basename(mp4_path)}")

    print("\nDone.")



if __name__ == "__main__":
    main()
