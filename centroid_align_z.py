"""centroid_align_z.py — correct Z-axis drift in XY-aligned OME-TIFF stacks.

Reads the per-embryo OME-TIFFs produced by centroid_align_xy.py, detects
Z-axis drift at each timepoint using the intensity centroid of a chosen
fluorescent channel along Z, applies an integer slice shift to compensate,
and writes Z-corrected OME-TIFFs to the same directory.

The detection channel defaults to index 0 (Venus). The z_diagnostic.py
analysis established that the Venus intensity centroid is the most reliable
Z drift indicator across all embryo positions: the signal has a clear
Z-dependent structure that shifts consistently with drift, while mCherry
(nuclear marker) is concentrated at the top of the Z range and shows no
useful variation.

Two values are printed per frame: the integer slice shift applied to the
image data, and the equivalent physical distance in micrometres. The µm
value is the un-rounded centroid difference and is provided for future
automated acquisition scripts that will command the stage by an exact
physical amount rather than a rounded slice count.

Usage:
    python centroid_align_z.py nd1188_Venus_threshold.yaml
    python centroid_align_z.py nd1188_Venus_threshold.yaml --ch_idx 1
"""

import argparse
import glob
import os
import sys

import numpy as np
import tifffile
import yaml
from tqdm import tqdm

from useful_functions import (
    align_frame_z,
    compute_centroid_z,
    compute_shift_z,
    compute_z_profile,
    load_nd2_metadata,
    save_ome_tiff,
)


def main():
    parser = argparse.ArgumentParser(
        description="Correct Z-axis drift in XY-aligned OME-TIFF stacks."
    )
    parser.add_argument(
        "yaml_file",
        help="Threshold YAML produced by find_threshold.py. Used to locate "
             "the ND2 file (for physical metadata) and the aligned directory.",
    )
    parser.add_argument(
        "--ch_idx",
        type=int,
        default=0,
        help="Channel index to use for Z drift detection (default: 0 = Venus). "
             "z_diagnostic.py established Venus as the most reliable Z indicator "
             "for this dataset.",
    )
    args = parser.parse_args()

    # ── 1. Load YAML config ───────────────────────────────────────────────────

    with open(args.yaml_file) as f:
        cfg = yaml.safe_load(f)

    nd2_path = cfg["source"]["file"]

    if not os.path.isfile(nd2_path):
        print(f"Error: ND2 file not found: {nd2_path}", file=sys.stderr)
        sys.exit(1)

    # ── 2. Load ND2 metadata ──────────────────────────────────────────────────
    # We use load_nd2_metadata rather than load_nd2 because image data is read
    # from the already-written OME-TIFFs below. Loading the full ND2 array
    # (up to several GB) just to get channel names and voxel size would waste
    # memory and time.

    print(f"Loading metadata from {nd2_path} ...")
    channel_names, vox, period_s = load_nd2_metadata(nd2_path)
    print(f"  Channels: {channel_names}")
    print(f"  Voxel z:  {vox.z:.3f} µm/slice")
    print(f"  Z detection channel: {channel_names[args.ch_idx]} (index {args.ch_idx})")

    # ── 3. Find XY-aligned TIFFs ──────────────────────────────────────────────

    base        = os.path.splitext(os.path.basename(nd2_path))[0]
    aligned_dir = os.path.join(os.path.dirname(nd2_path) or ".", f"aligned_{base}")

    if not os.path.isdir(aligned_dir):
        print(f"Error: aligned directory not found: {aligned_dir}", file=sys.stderr)
        print("Run centroid_align_xy.py first to produce XY-aligned TIFFs.",
              file=sys.stderr)
        sys.exit(1)

    # glob collects all per-embryo TIFFs matching the naming convention written
    # by centroid_align_xy.py. Files ending in _z.ome.tif are excluded so that
    # re-running this script on the same directory does not process its own
    # previous output.
    tiff_paths = sorted(glob.glob(os.path.join(aligned_dir, f"{base}_P*.ome.tif")))
    tiff_paths = [p for p in tiff_paths if not p.endswith("_z.ome.tif")]

    if not tiff_paths:
        print(f"Error: no XY-aligned TIFFs found in {aligned_dir}", file=sys.stderr)
        sys.exit(1)

    P = len(tiff_paths)
    print(f"\nFound {P} XY-aligned TIFF(s) in {aligned_dir}/")

    # ── 4. Align each embryo ──────────────────────────────────────────────────

    print(f"\nAligning {P} embryo(s) ...")
    for tiff_path in tiff_paths:
        p_label = os.path.basename(tiff_path)
        print(f"\n  {p_label}")

        # tifffile.imread reads the OME-TIFF and respects the TCZYX axis order
        # stored in the OME-XML metadata written by save_ome_tiff. The result
        # is cast to float32 to match the dtype produced by load_nd2, ensuring
        # that downstream arithmetic (centroid computation, shift) behaves
        # identically regardless of which entry point produced the data.
        volume      = tifffile.imread(tiff_path).astype(np.float32)
        T, C, Z, Y, X = volume.shape
        print(f"    Shape: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")

        # Compute the reference Z centroid from t=0. Every subsequent timepoint
        # is shifted to match this value so the same anatomical Z position
        # stays at the same slice index throughout the timelapse — the goal
        # that motivates the whole Z-alignment step.
        profile_0          = compute_z_profile(volume[0], args.ch_idx)
        reference_centroid = compute_centroid_z(profile_0)
        print(f"    Reference Z centroid (t=0): {reference_centroid:.3f} slices"
              f"  ({reference_centroid * vox.z:.2f} µm)")

        corrected = np.zeros_like(volume)

        for t in tqdm(range(T), desc="    Aligning Z", unit="frame", leave=True):
            # compute_shift_z computes the current centroid, compares it to
            # reference_centroid, and prints the dz for this frame. Both the
            # integer slice shift (for the image) and the µm value (for future
            # stage control) are returned, though only dz_slices is used here.
            dz_slices, dz_um = compute_shift_z(
                volume[t], args.ch_idx, reference_centroid, vox.z
            )
            corrected[t] = align_frame_z(volume[t], dz_slices)

        # Insert _z before .ome.tif to produce the output filename, making it
        # clear this file has had both XY and Z correction applied.
        out_path = tiff_path.replace(".ome.tif", "_z.ome.tif")
        print(f"    Saving {os.path.basename(out_path)} ...")
        save_ome_tiff(out_path, corrected, channel_names, vox, period_s)
        print(f"    Saved {os.path.basename(out_path)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
