"""Make 2x2 grid MP4s from a raw ND2 file (no alignment).

Usage: python movie_from_nd2.py nd1188.nd2
"""

import argparse
import os
import sys

import imageio
import numpy as np

from useful_functions import auto_contrast, load_nd2, make_grid_frame, max_project_z


def main():
    parser = argparse.ArgumentParser(description="Make 2x2 grid MP4s from a raw ND2 file.")
    parser.add_argument("nd2_file", help="Path to ND2 file")
    parser.add_argument("--fps", type=float, default=2, help="Frames per second (default: 2)")
    args = parser.parse_args()

    nd2_path = args.nd2_file
    if not os.path.isfile(nd2_path):
        print(f"Error: file not found: {nd2_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {nd2_path} ...")
    # load_nd2 returns the full 6-D array; vox and period_s are not needed
    # for a raw movie so we discard them with _.
    data, channel_names, _, _ = load_nd2(nd2_path)

    # Collapse Z by max-projection to get a 2-D image per timepoint/position/channel.
    max_proj = max_project_z(data)  # (T, P, C, Y, X)
    T, P, C, Y, X = max_proj.shape
    print(f"  Shape: T={T}, P={P}, C={C}, Y={Y}, X={X}")
    print(f"  Channels: {channel_names}")

    base    = os.path.splitext(os.path.basename(nd2_path))[0]
    out_dir = os.path.join(os.path.dirname(nd2_path) or ".", f"raw_{base}")
    os.makedirs(out_dir, exist_ok=True)

    for c in range(C):
        ch_name  = channel_names[c]
        mp4_path = os.path.join(out_dir, f"{base}_{ch_name}_raw.mp4")
        print(f"  Writing {mp4_path} ...")

        writer = imageio.get_writer(mp4_path, fps=args.fps)
        for t in range(T):
            images = [auto_contrast(max_proj[t, p, c]) for p in range(P)]
            grid   = make_grid_frame(images)
            # imageio expects RGB frames; stack the grayscale grid three times
            # to produce a greyscale-looking RGB image.
            rgb = np.stack([grid, grid, grid], axis=-1)
            writer.append_data(rgb)
        writer.close()

    print("Done.")


if __name__ == "__main__":
    main()
