"""centroid_align_z.py -- correct Z-axis drift in XY-aligned OME-TIFF stacks.

Reads the per-embryo OME-TIFFs produced by centroid_align_xy.py, detects
Z-axis drift at each timepoint using the intensity centroid of a chosen
fluorescent channel along Z, applies an integer slice shift to compensate,
and writes Z-corrected OME-TIFFs to the same directory.

The user selects a reference timepoint (--ref_t) where the PSM looks how
they want it to for analysis. The Z centroid at that timepoint becomes the
target; all other timepoints are shifted so their centroids match, keeping
every Z slice at the same anatomical depth across T.

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

--enlarge_canvas
    By default, shifting the Z stack discards slices that move outside the
    original Z range. With --enlarge_canvas the script runs two passes:
    the first computes all shifts without touching the data; the second
    pads the Z dimension asymmetrically (matching the --enlarge_canvas
    behaviour of centroid_align_xy.py) so that no original slice is ever
    overwritten by the zero-fill. The output stack has more Z slices than
    the input. Downstream tools (Fiji, napari) handle the larger stack
    correctly provided the OME-TIFF physical metadata is read.

Usage:
    python centroid_align_z.py nd1188_Venus_threshold.yaml
    python centroid_align_z.py nd1188_Venus_threshold.yaml --ref_t 5
    python centroid_align_z.py nd1188_Venus_threshold.yaml --ch_idx 1
    python centroid_align_z.py nd1188_Venus_threshold.yaml --enlarge_canvas
"""

import argparse
import sys
from pathlib import Path

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
    parser.add_argument(
        "--ref_t",
        type=int,
        default=0,
        help="Reference timepoint for Z alignment. Choose a frame where the "
             "PSM looks how you want it to for analysis. All other timepoints "
             "are shifted to match this frame's Z centroid so the anatomy stays "
             "at the same Z slice across T. (default: 0)",
    )
    parser.add_argument(
        "--enlarge_canvas",
        action="store_true",
        help=(
            "Precompute all Z shifts, then expand the Z dimension "
            "asymmetrically before aligning so no slice data is lost. "
            "Mirrors the --enlarge_canvas behaviour of centroid_align_xy.py. "
            "The output stack will have more Z slices than the input."
        ),
    )
    args = parser.parse_args()

    # ── 1. Load YAML config ───────────────────────────────────────────────────

    with open(args.yaml_file) as f:
        cfg = yaml.safe_load(f)

    nd2_path = Path(cfg["source"]["file"])

    if not nd2_path.is_file():
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
    print(f"  Reference timepoint: t={args.ref_t}")

    # ── 3. Find XY-aligned TIFFs ──────────────────────────────────────────────

    base        = nd2_path.stem
    aligned_dir = nd2_path.parent / f"aligned_{base}"

    if not aligned_dir.is_dir():
        print(f"Error: aligned directory not found: {aligned_dir}", file=sys.stderr)
        print("Run centroid_align_xy.py first to produce XY-aligned TIFFs.",
              file=sys.stderr)
        sys.exit(1)

    # glob collects all per-embryo TIFFs matching the naming convention written
    # by centroid_align_xy.py. Files ending in _z.ome.tif are excluded so that
    # re-running this script on the same directory does not process its own
    # previous output.
    tiff_paths = sorted(aligned_dir.glob(f"{base}_P*.ome.tif"))
    tiff_paths = [p for p in tiff_paths if not p.name.endswith("_z.ome.tif")]

    if not tiff_paths:
        print(f"Error: no XY-aligned TIFFs found in {aligned_dir}", file=sys.stderr)
        sys.exit(1)

    P = len(tiff_paths)
    print(f"\nFound {P} XY-aligned TIFF(s) in {aligned_dir}/")

    # ── 4. Align each embryo ──────────────────────────────────────────────────

    print(f"\nAligning {P} embryo(s) ...")
    for tiff_path in tiff_paths:
        p_label = tiff_path.name
        print(f"\n  {p_label}")

        # tifffile.imread reads the OME-TIFF and respects the TCZYX axis order
        # stored in the OME-XML metadata written by save_ome_tiff. The result
        # is cast to float32 to match the dtype produced by load_nd2, ensuring
        # that downstream arithmetic (centroid computation, shift) behaves
        # identically regardless of which entry point produced the data.
        volume      = tifffile.imread(tiff_path).astype(np.float32)
        T, C, Z, Y, X = volume.shape
        print(f"    Shape: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")

        # Compute the reference Z centroid from the user-chosen timepoint.
        # Every other timepoint is shifted to match this value so that
        # whatever anatomy is visible at a given Z slice in the reference
        # frame stays at that same slice index across the entire timelapse.
        ref_t = args.ref_t
        if ref_t < 0 or ref_t >= T:
            print(f"    Error: --ref_t {ref_t} is out of range [0, {T-1}]",
                  file=sys.stderr)
            sys.exit(1)
        profile_ref        = compute_z_profile(volume[ref_t], args.ch_idx)
        reference_centroid = compute_centroid_z(profile_ref)
        print(f"    Reference Z centroid (t={ref_t}): {reference_centroid:.3f} slices"
              f"  ({reference_centroid * vox.z:.2f} µm)")

        if args.enlarge_canvas:
            # ── Two-pass path: enlarge canvas ─────────────────────────────────
            # Pass 1: compute all shifts without modifying the volume.
            # Printing each shift as it arrives lets the user see the drift
            # pattern before any data is written, matching the --enlarge_canvas
            # behaviour of centroid_align_xy.py.
            print(f"\n--- Pass 1: precomputing Z shifts ---")
            all_dz = np.zeros(T, dtype=int)
            for t in range(T):
                dz_slices, _ = compute_shift_z(
                    volume[t], args.ch_idx, reference_centroid, vox.z
                )
                all_dz[t] = dz_slices

            print(f"\n    Shift summary (slices):")
            print(f"      min={all_dz.min():+d}  max={all_dz.max():+d}  "
                  f"mean={all_dz.mean():+.1f}")

            # Asymmetric padding: positive shifts move content toward higher Z
            # indices, so the high end of the stack would be lost without extra
            # slices there. Negative shifts move content toward lower Z indices,
            # so the low end would be lost. This mirrors the top/bottom padding
            # logic in centroid_align_xy.py --enlarge_canvas.
            pad_high = int(np.ceil(max(0,  all_dz.max())))
            pad_low  = int(np.ceil(max(0, -all_dz.min())))
            print(f"    Z canvas padding: low={pad_low}  high={pad_high}")

            # np.pad with (pad_low, pad_high) on the Z axis (axis 2 in
            # (T, C, Z, Y, X)). All other axes are unchanged.
            volume = np.pad(
                volume,
                ((0, 0), (0, 0), (pad_low, pad_high), (0, 0), (0, 0)),
            )
            _, _, Z_padded, _, _ = volume.shape
            print(f"    Expanded Z: {Z} → {Z_padded} slices")

            # Pass 2: apply precomputed shifts on the expanded volume.
            # We reuse align_frame_z with the already-computed all_dz values
            # rather than calling compute_shift_z again, for the same reason
            # centroid_align_xy.py avoids recomputing centroids on the padded
            # canvas: the padding zeros would shift the percentile threshold
            # and produce incorrect results.
            print(f"\n--- Pass 2: applying Z shifts ---")
            corrected = np.zeros_like(volume)
            for t in tqdm(range(T), desc="    Aligning Z", unit="frame", leave=True):
                corrected[t] = align_frame_z(volume[t], all_dz[t])

        else:
            # ── Single-pass path (default) ────────────────────────────────────
            # Shifts are computed and applied in one pass. Slices that move
            # outside the original Z range are replaced with zeros, so some
            # data is lost when the shift is large. Use --enlarge_canvas to
            # avoid this.
            corrected = np.zeros_like(volume)
            for t in tqdm(range(T), desc="    Aligning Z", unit="frame", leave=True):
                # compute_shift_z computes the current centroid, compares it to
                # reference_centroid, and prints the dz for this frame. Both the
                # integer slice shift (for the image) and the µm value (for
                # future stage control) are returned, though only dz_slices is
                # used here.
                dz_slices, _ = compute_shift_z(
                    volume[t], args.ch_idx, reference_centroid, vox.z
                )
                corrected[t] = align_frame_z(volume[t], dz_slices)

        # Insert _z before .ome.tif to produce the output filename, making it
        # clear this file has had both XY and Z correction applied.
        # with_name operates only on the filename component, never touching
        # directory separators — unlike a bare str.replace on the full path,
        # which would corrupt the path if ".ome.tif" appeared in a directory name.
        out_path = tiff_path.with_name(tiff_path.name.replace(".ome.tif", "_z.ome.tif"))
        print(f"    Saving {out_path.name} ...")
        save_ome_tiff(out_path, corrected, channel_names, vox, period_s)
        print(f"    Saved {out_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
