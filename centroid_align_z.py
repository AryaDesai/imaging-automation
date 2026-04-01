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
    python centroid_align_z.py aligned_nd1188/nd1188_P0_xy.ome.tif
    python centroid_align_z.py aligned_nd1188/nd1188_P0_xy.ome.tif --ref_t 5
    python centroid_align_z.py aligned_nd1188/ --ref_t 5
    python centroid_align_z.py aligned_nd1188/ --ch_idx 1 --enlarge_canvas
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from useful_functions import (
    LazyTifReader,
    align_frame_z,
    compute_centroid_z,
    compute_shift_z,
    compute_z_profile,
    load_tif_metadata,
    save_ome_tiff,
)


def main():
    parser = argparse.ArgumentParser(
        description="Correct Z-axis drift in XY-aligned OME-TIFF stacks."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more OME-TIFF files or directories. Directories are "
             "searched for *.ome.tif files.",
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
        "--recursive",
        action="store_true",
        help="Search directories recursively for *.ome.tif files.",
    )
    parser.add_argument(
        "--no_pad_z",
        action="store_false",
        dest="pad_z",
        help=(
            "Disable Z padding. By default the Z dimension is expanded "
            "asymmetrically so no slice data is lost. With this flag, "
            "slices that shift outside the original Z range are discarded."
        ),
    )
    args = parser.parse_args()

    # ── 1. Resolve input TIFFs ─────────────────────────────────────────────────
    # Each argument can be a TIF file or a directory. Directories are searched
    # for *.ome.tif files.

    tiff_paths = []
    for arg in args.inputs:
        p = Path(arg)
        if p.is_dir():
            glob_fn = p.rglob if args.recursive else p.glob
            tiff_paths.extend(sorted(glob_fn("*.ome.tif")))
        elif p.is_file():
            tiff_paths.append(p)
        else:
            print(f"Error: not a file or directory: {p}", file=sys.stderr)
            sys.exit(1)

    if not tiff_paths:
        print("Error: no OME-TIFF files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(tiff_paths)} OME-TIFF(s) to process.")

    # ── 2. Align each TIF ────────────────────────────────────────────────────

    for tiff_path in tiff_paths:
        print(f"\n  {tiff_path.name}")

        channel_names, vox, period_s = load_tif_metadata(tiff_path)
        reader = LazyTifReader(tiff_path)
        T, C, Z, Y, X = reader.T, reader.C, reader.Z, reader.Y, reader.X
        print(f"    Shape: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
        print(f"    Channels: {channel_names}")
        print(f"    Voxel z:  {vox.z:.3f} µm/slice")
        print(f"    Z detection channel: {channel_names[args.ch_idx]} (index {args.ch_idx})")

        # Compute the reference Z centroid from the user-chosen timepoint.
        # Every other timepoint is shifted to match this value so that
        # whatever anatomy is visible at a given Z slice in the reference
        # frame stays at that same slice index across the entire timelapse.
        ref_t = args.ref_t
        if ref_t < 0 or ref_t >= T:
            print(f"    Error: --ref_t {ref_t} is out of range [0, {T-1}]",
                  file=sys.stderr)
            sys.exit(1)
        profile_ref        = compute_z_profile(reader.read_frame(ref_t), args.ch_idx)
        reference_centroid = compute_centroid_z(profile_ref)
        print(f"    Reference Z centroid (t={ref_t}): {reference_centroid:.3f} slices"
              f"  ({reference_centroid * vox.z:.2f} µm)")

        if args.pad_z:
            # ── Two-pass path: enlarge canvas/pad Z ─────────────────────────────────
            # Pass 1: compute all shifts without modifying the volume.
            # Printing each shift as it arrives lets the user see the drift
            # pattern before any data is written, matching the --enlarge_canvas
            # behaviour of centroid_align_xy.py.
            print(f"\n--- Pass 1: precomputing Z shifts ---")
            all_dz = np.zeros(T, dtype=int)
            for t in range(T):
                dz_slices, _ = compute_shift_z(
                    reader.read_frame(t), args.ch_idx, reference_centroid, vox.z
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
            Z_padded = Z + pad_low + pad_high
            print(f"    Z canvas padding: low={pad_low}  high={pad_high}")
            print(f"    Expanded Z: {Z} → {Z_padded} slices")

            # Pass 2: re-read each frame, pad the Z axis, and apply the
            # precomputed shift. We reuse align_frame_z with the already-
            # computed all_dz values rather than calling compute_shift_z
            # again, for the same reason centroid_align_xy.py avoids
            # recomputing centroids on the padded canvas: the padding zeros
            # would corrupt the percentile threshold.
            print(f"\n--- Pass 2: applying Z shifts ---")
            def _aligned_frames_padded():
                for t in tqdm(range(T), desc="    Aligning Z", unit="frame", leave=True):
                    frame = reader.read_frame(t)
                    # Pad Z axis (axis 1 in (C, Z, Y, X)) per frame.
                    frame = np.pad(
                        frame,
                        ((0, 0), (pad_low, pad_high), (0, 0), (0, 0)),
                    )
                    yield align_frame_z(frame, all_dz[t])

            out_shape = (T, C, Z_padded, Y, X)

        else:
            # ── Single-pass path ────────────────────────────────────
            # Shifts are computed and applied in one pass. Slices that move
            # outside the original Z range are replaced with zeros, so some
            # data is lost when the shift is large. Use --no_pad_z to do this
            def _aligned_frames():
                for t in tqdm(range(T), desc="    Aligning Z", unit="frame", leave=True):
                    frame = reader.read_frame(t)
                    dz_slices, _ = compute_shift_z(
                        frame, args.ch_idx, reference_centroid, vox.z
                    )
                    yield align_frame_z(frame, dz_slices)

            out_shape = (T, C, Z, Y, X)

        # Insert _z before .ome.tif to produce the output filename, making it
        # clear this file has had both XY and Z correction applied.
        # with_name operates only on the filename component, never touching
        # directory separators — unlike a bare str.replace on the full path,
        # which would corrupt the path if ".ome.tif" appeared in a directory name.
        out_path = tiff_path.with_name(tiff_path.name.replace(".ome.tif", "_z.ome.tif"))
        print(f"    Saving {out_path.name} ...")
        frames = _aligned_frames_padded() if args.pad_z else _aligned_frames()
        save_ome_tiff(out_path, frames, channel_names, vox, period_s,
                      shape=out_shape, dtype=reader.dtype)
        reader.close()
        print(f"    Saved {out_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
